import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from .backend.errors import BackendError
from .backend.selector import select_backend
from .backend.session import get_session
from .backend.types import BackendType, DataType, OperationProperties, OperationType

if TYPE_CHECKING:
    from .buffer import BufferView

logger = logging.getLogger("corepy.array")

from .broadcasting import get_c_strides


def _map_to_rust_dtype(dtype: Any) -> Any:
    """Map a Python DataType / string / dtype object to _RustDType."""
    try:
        from ._corepy_rust import _RustDType
    except ImportError:
        return None

    if isinstance(dtype, _RustDType):
        return dtype

    if hasattr(dtype, "value"):
        dtype_str = str(dtype.value).lower()
    else:
        dtype_str = str(dtype).lower()

    mapping = {
        "float32": _RustDType.Float32,
        "float64": _RustDType.Float64,
        "int32": _RustDType.Int32,
        "int64": _RustDType.Int64,
        "bool": _RustDType.Bool,
        "string": _RustDType.String,
        "str": _RustDType.String,
    }


def _unflatten_list(flat: list, shape: tuple) -> list:
    if len(shape) <= 1:
        return flat
    rows = shape[0]
    sub_shape = shape[1:]
    chunk_size = 1
    for s in sub_shape:
        chunk_size *= s
    if len(flat) != rows * chunk_size:
        return flat
    return [
        _unflatten_list(flat[i * chunk_size : (i + 1) * chunk_size], sub_shape)
        for i in range(rows)
    ]


class ndarray:
    """
    A multi-dimensional array object (NumPy-compatible naming).

    This is the primary array type in CorePy, designed to be a drop-in
    replacement for NumPy's ndarray where applicable. It automatically
    selects the best execution backend (CPU/GPU) based on data size
    and operation complexity.

    Attributes:
        shape: Tuple of array dimensions.
        dtype: Data type of array elements.
        ndim: Number of dimensions.
        size: Total number of elements.

    Example:
        >>> import corepy as cp
        >>> arr = cp.array([1.0, 2.0, 3.0])
        >>> arr.shape
        (3,)
        >>> arr.dtype
        DataType.FLOAT32
    """

    def __init__(
        self,
        data: Union[Sequence[Any], "ndarray", Any],
        dtype: DataType = DataType.FLOAT32,
        backend: Optional[Union[str, BackendType]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize an array.

        Args:
            data: Input data (list, tuple, or another array).
            dtype: Data type (default: float32).
            backend: Explicitly requested backend ('cpu', 'gpu').
            device: Explicit device string (e.g. 'cuda:0', 'cpu').
                    If provided, overrides 'backend'.
        """
        self._dtype = dtype

        # Determine shape and element count (simplified for this implementation)
        # In a real impl, we'd recursively check list depth/lengths
        # Determine shape and element count
        if isinstance(data, (list, tuple)):
            # Recursive shape detection
            if len(data) == 0:
                self._shape = (0,)
                self._element_count = 0
                self._backing_data = data
            else:
                shape = []
                curr = data
                while isinstance(curr, (list, tuple)) and len(curr) > 0:
                    shape.append(len(curr))
                    curr = curr[0]

                self._shape = tuple(shape)
            # Calculate element count
            count = 1
            for dim in self._shape:
                count *= dim
            self._element_count = count
            self._backing_data = data
        elif isinstance(data, (bytes, bytearray)):
            self._shape = (len(data),)
            self._element_count = len(data)
            self._backing_data = data  # type: ignore[assignment]
        elif isinstance(data, memoryview):
            self._shape = data.shape if data.shape else (len(data),)
            # Calculate size for ND memoryview
            size = 1
            for dim in self._shape:
                size *= dim
            self._element_count = size
            self._backing_data = data  # type: ignore[assignment]
        elif hasattr(data, "shape") and hasattr(
            data, "size"
        ):  # numpy or array protocol
            self._shape = tuple(int(d) for d in data.shape)
            self._element_count = int(data.size)
            self._backing_data = data  # type: ignore[assignment]
        else:
            # scalar or error
            self._shape = (1,)
            self._element_count = 1
            self._backing_data = [data]

        # Resolve requested backend/device
        requested_backend = None
        if device:
            if "cuda" in device:
                requested_backend = BackendType.CUDA
            elif "metal" in device:
                requested_backend = BackendType.METAL
            elif "gpu" in device:
                requested_backend = BackendType.GPU
            elif "cpu" in device:
                requested_backend = BackendType.CPU
        elif backend:
            if isinstance(backend, str):
                requested_backend = BackendType(backend.lower())
            else:
                requested_backend = backend

        # Select Backend
        # We classify Creation as MEMORY_BOUND or SCALAR usually,
        # but the meaningful decision happens for subsequent ops.
        # However, we must decide where to allocations *now*.
        # Determine where the data should live initially
        if device is not None:
            if "cuda" in device or "gpu" in device or "metal" in device:
                self._device = "metal"  # Normalize to metal for now
            else:
                self._device = "cpu"
        elif requested_backend == BackendType.GPU:
            self._device = "metal"
        else:
            self._device = "cpu"

        # Phase 2: GPU Persistent Buffer Tracking
        # Track where data currently resides to minimize transfers
        self._cpu_data: Any = None
        self._gpu_data = None  # Metal buffer handle (if GPU-resident)
        self._data_location = "cpu"  # "cpu", "gpu", or "both"

        # Initialize CPU data from backing_data
        self._cpu_data = self._backing_data
        self._data_location = "cpu"

        # Select backend through policy
        context = OperationProperties(
            element_count=self._element_count,
            shape=self._shape,
            # Approximating bytes: len * 4 bytes for float32
            dtype_bytes=4,
        )
        # We treat 'allocation' as a memory operation.
        # However, for 'Correctness-First', we usually default to CPU for storage
        # unless explicitly told otherwise or if we are consuming GPU data.
        # BUT, the goal is "ndarray(data)" usually implies "Ready for compute".
        # So we should check if this data is "large enough" to justify GPU storage?
        # Usually, just storing is not compute. So auto-placement should default CPU
        # unless immediate heavy compute is expected?

        # For now, let's just classify it as MEMORY_BOUND. Backend selection defaults to CPU for construction.
        # This can be revised.
        session = get_session()
        self._backend_type = select_backend(
            OperationType.MEMORY_BOUND, context, session.device_info, requested_backend
        )

        # Store original device intent
        self._device_intent = self._device
        self._core_array: Any = None

        logger.debug(f"Array created on {self._backend_type}. Shape={self._shape}")

    @classmethod
    def _wrap_core_array(
        cls,
        core_array: Any,
        dtype: DataType = DataType.FLOAT32,
        device: str = "cpu",
        backend: Optional[Union[str, BackendType]] = None,
    ) -> "ndarray":
        """Wrap a Rust _RustCoreArray object into a Python ndarray."""
        data = core_array.to_list() if hasattr(core_array, "to_list") else []
        arr = cls(data, dtype=dtype, device=device, backend=backend)
        if hasattr(core_array, "shape"):
            raw_shape = (
                core_array.shape() if callable(core_array.shape) else core_array.shape
            )
            arr._shape = tuple(raw_shape)
            count = 1
            for d in arr._shape:
                count *= d
            arr._element_count = count
            if arr._cpu_data is not None and hasattr(arr._cpu_data, "reshape"):
                arr._cpu_data = arr._cpu_data.reshape(arr._shape)
        arr._core_array = core_array
        return arr

    def to_list(self) -> list:
        """Convert array to a Python list matching its shape."""
        raw_flat = []
        if self._core_array is not None:
            raw_flat = self._core_array.to_list()
        elif hasattr(self, "_cpu_data") and self._cpu_data is not None:
            if isinstance(self._cpu_data, list):
                raw_flat = self._cpu_data
            elif hasattr(self._cpu_data, "tolist"):
                raw_flat = self._cpu_data.tolist()
            else:
                raw_flat = list(self._cpu_data)
        elif hasattr(self, "_backing_data") and self._backing_data is not None:
            if isinstance(self._backing_data, list):
                raw_flat = self._backing_data
            elif hasattr(self._backing_data, "tolist"):
                raw_flat = self._backing_data.tolist()
            else:
                raw_flat = list(self._backing_data)

        if len(self.shape) > 1 and isinstance(raw_flat, list):
            if raw_flat and not isinstance(raw_flat[0], list):
                return _unflatten_list(raw_flat, self.shape)

        return raw_flat

    def tolist(self) -> list:
        """NumPy-compatible alias for to_list()."""
        return self.to_list()

    def sin(self) -> "ndarray":
        from .ops.trigonometry import sin

        return sin(self)

    def cos(self) -> "ndarray":
        from .ops.trigonometry import cos

        return cos(self)

    def tan(self) -> "ndarray":
        from .ops.trigonometry import tan

        return tan(self)

    def arcsin(self) -> "ndarray":
        from .ops.trigonometry import arcsin

        return arcsin(self)

    def arccos(self) -> "ndarray":
        from .ops.trigonometry import arccos

        return arccos(self)

    def arctan(self) -> "ndarray":
        from .ops.trigonometry import arctan

        return arctan(self)

    def sinh(self) -> "ndarray":
        from .ops.trigonometry import sinh

        return sinh(self)

    def cosh(self) -> "ndarray":
        from .ops.trigonometry import cosh

        return cosh(self)

    def tanh(self) -> "ndarray":
        from .ops.trigonometry import tanh

        return tanh(self)

    def exp(self) -> "ndarray":
        from .ops.exponential import exp

        return exp(self)

    def log(self) -> "ndarray":
        from .ops.exponential import log

        return log(self)

    def sqrt(self) -> "ndarray":
        from .ops.exponential import sqrt

        return sqrt(self)

    def prod(self) -> float:
        from .ops.reduction import prod

        return prod(self)

    def cumsum(self) -> "ndarray":
        from .ops.reduction import cumsum

        return cumsum(self)

    def square(self) -> "ndarray":
        from .ops.special import square

        return square(self)

    def floor(self) -> "ndarray":
        from .ops.rounding import floor

        return floor(self)

    def ceil(self) -> "ndarray":
        from .ops.rounding import ceil

        return ceil(self)

    def sort(self, descending: bool = False) -> "ndarray":
        from .ops.sorting import sort

        return sort(self, descending)

    def argsort(self, descending: bool = False) -> "ndarray":
        from .ops.sorting import argsort

        return argsort(self, descending)

    def argmax(self) -> int:
        from .ops.searching import argmax

        return argmax(self)

    def argmin(self) -> int:
        from .ops.searching import argmin

        return argmin(self)

    def split(self, indices_or_sections: Any, axis: int = 0) -> list:
        from .ops.stacking import split

        return split(self, indices_or_sections, axis)

    def is_even(self) -> "ndarray":
        from .ops.ufunc_engine import ufunc_unary

        return ufunc_unary("is_even", self)

    def is_odd(self) -> "ndarray":
        from .ops.ufunc_engine import ufunc_unary

        return ufunc_unary("is_odd", self)

    def abs(self) -> "ndarray":
        from .ops.ufunc_engine import ufunc_unary

        return ufunc_unary("abs", self)

    def clamp(self, min_val: float, max_val: float) -> "ndarray":
        from .ops.rounding import clamp

        return clamp(self, min_val, max_val)

    def clip(self, min_val: float, max_val: float) -> "ndarray":
        from .ops.rounding import clip

        return clip(self, min_val, max_val)

    def take(self, indices: Any) -> "ndarray":
        from .ops.indexing import take

        return take(self, indices)

    def boolean_index(self, mask: Any) -> "ndarray":
        from .ops.indexing import boolean_index

        return boolean_index(self, mask)

    @property
    def backend(self) -> BackendType:
        return self._backend_type

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DataType:
        """Data type of the array elements (NumPy-compatible)."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of array dimensions (NumPy-compatible)."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements in the array (NumPy-compatible)."""
        return self._element_count

    def _should_use_fast_path(self) -> bool:
        """
        Determine if numpy fast path should be used to avoid FFI overhead.

        Returns True for small CPU arrays where NumPy is faster than FFI call.


        """
        from .config import SMALL_ARRAY_THRESHOLD

        return self.size < SMALL_ARRAY_THRESHOLD and self._device == "cpu"

    @property
    def T(self) -> "ndarray":
        """Transpose of the array (NumPy-compatible). For 2D arrays only."""
        return self.transpose()

    def reshape(self, *shape) -> "ndarray":
        """
        Return array reshaped to given dimensions.

        Args:
            shape: New shape as multiple arguments or single tuple/list.

        Returns:
            Reshaped array with same data.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        new_arr = ndarray(
            self.to_list(), dtype=self._dtype, device=self._device, backend=self.backend
        )
        new_arr._shape = target_shape
        count = 1
        for d in target_shape:
            count *= d
        new_arr._element_count = count
        if self._core_array is not None and hasattr(self._core_array, "reshape"):
            try:
                new_arr._core_array = self._core_array.reshape(list(target_shape))
            except Exception:
                pass
        return new_arr

    def transpose(self, *axes) -> "ndarray":
        """
        Return transposed array.

        Optimized for 2D Metal matrices. Falls back to NumPy for others.
        """
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()

        # Check for simple 2D transpose (0,1) -> (1,0)
        is_simple_transpose = False
        if len(self.shape) == 2:
            if not axes or axes == (1, 0) or axes == [1, 0]:
                is_simple_transpose = True

        # Priority 1: Metal GPU
        if is_simple_transpose and self.backend == BackendType.GPU:
            view = self._get_buffer_view()
            if view.device.is_metal() and self._element_count >= 1024:
                try:
                    import numpy as np

                    from . import _corepy_rust as ffi  # type: ignore[import-untyped]

                    m, n = self.shape
                    # Allocate output buffer (N x M)
                    final_np = np.empty((n, m), dtype=np.float32)
                    ptr_out = final_np.__array_interface__["data"][0]

                    ffi.metal_transpose_f32(view.data_ptr, ptr_out, m, n)

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    record_op("transpose", elapsed_ms, "Metal")
                    return ndarray(final_np, dtype=self._dtype, backend=self.backend)

                except ImportError:
                    pass

        # Native 2D transpose fallback
        if len(self.shape) == 2:
            rows, cols = self.shape
            grid = self.to_list()
            if grid and isinstance(grid[0], list):
                transposed_grid = [
                    [grid[r][c] for r in range(rows)] for c in range(cols)
                ]
            else:
                # 1D flat list representation
                transposed_grid = [
                    [grid[r * cols + c] for r in range(rows)] for c in range(cols)
                ]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("transpose", elapsed_ms, "Native-CPU")
            return ndarray(transposed_grid, dtype=self._dtype, backend=self.backend)

        # Multi-dimensional fallback via lazy to_numpy
        np_arr = self.to_numpy()
        if axes:
            np_arr = np_arr.transpose(*axes)
        else:
            np_arr = np_arr.T

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op("transpose", elapsed_ms, "NumPy")
        return ndarray(np_arr, dtype=self._dtype, backend=self.backend)

    def to(self, device: str) -> "ndarray":
        """
        Explicitly move array to a device (deprecated - use to_device).

        Arguments:
            device: 'cpu' or 'gpu'
        """
        return self.to_device(device)

    def to_device(self, device: str) -> "ndarray":
        """
        Move array to specified device with persistent buffer tracking.

        Phase 2: Lazy transfer - only moves data when needed, caches on both sides.

        Args:
            device: Target device ('cpu', 'metal', 'gpu')

        Returns:
            Self (modified in-place for efficiency)
        """
        # Normalize device name
        if device in ("gpu", "metal"):
            device = "metal"  # Metal is macOS GPU

        # Already on target device?
        if device == "cpu" and self._data_location in ("cpu", "both"):
            self._device = "cpu"
            return self
        elif device == "metal" and self._data_location in ("gpu", "both"):
            self._device = "metal"
            return self

        # Need to transfer
        if device == "metal":
            # TODO: Implement actual GPU transfer when Metal backend ready
            # For now, mark intention but keep data on CPU
            self._device = "metal"
            # In production: self._gpu_data = transfer_to_metal(self._cpu_data)
            # self._data_location = "both"
        elif device == "cpu":
            # GPU → CPU transfer
            if self._gpu_data is not None:
                # TODO: Transfer from GPU
                # self._cpu_data = transfer_from_metal(self._gpu_data)
                pass
            self._device = "cpu"
            self._data_location = "cpu" if self._cpu_data is not None else "cpu"

        return self

    def _ensure_on_device(self, device: str):
        """
        Internal method: Ensure data is available on specified device.

        Lazy transfer - only transfers if not already present.
        """
        if device == "metal":
            if self._data_location not in ("gpu", "both"):
                # Need to transfer to GPU
                self.to_device("metal")
        elif device == "cpu":
            if self._data_location not in ("cpu", "both"):
                # Need to transfer to CPU
                self.to_device("cpu")

    def __getitem__(self, item) -> Union["ndarray", float, int, bool]:
        """Access element or slice of the array."""
        grid = self.to_list()
        try:
            if isinstance(item, tuple):
                sub = grid
                for idx in item:
                    sub = sub[idx]
                result = sub
            else:
                result = grid[item]
        except (TypeError, IndexError):
            # Fallback for nested indexing
            sub = grid
            if isinstance(item, tuple):
                for idx in item:
                    sub = sub[idx]
                result = sub
            else:
                result = sub[item]

        if isinstance(result, list):
            return ndarray(result, dtype=self._dtype, backend=self.backend)
        elif hasattr(result, "tolist"):
            return result.tolist()
        else:
            return result.item() if hasattr(result, "item") else result

    def __repr__(self):
        return f"ndarray({self._backing_data}, backend='{self._backend_type.value}')"

    def __add__(self, other: Any) -> "ndarray":
        """Element-wise addition."""
        return self._binary_op("add", other)

    def __radd__(self, other: Any) -> "ndarray":
        """Reverse element-wise addition."""
        return self._binary_op("add", other)

    def __sub__(self, other: Any) -> "ndarray":
        """Element-wise subtraction."""
        return self._binary_op("sub", other)

    def __rsub__(self, other: Any) -> "ndarray":
        """Reverse element-wise subtraction."""
        return ndarray(other, backend=self.backend)._binary_op("sub", self)

    def __mul__(self, other: Any) -> "ndarray":
        """Element-wise multiplication."""
        return self._binary_op("mul", other)

    def __rmul__(self, other: Any) -> "ndarray":
        """Reverse element-wise multiplication."""
        return self._binary_op("mul", other)

    def __truediv__(self, other: Any) -> "ndarray":
        """Element-wise division."""
        return self._binary_op("div", other)

    def __rtruediv__(self, other: Any) -> "ndarray":
        """Reverse element-wise division."""
        return ndarray(other, backend=self.backend)._binary_op("div", self)

    def __pow__(self, other: Any) -> "ndarray":
        """Element-wise exponentiation."""
        return self._binary_op("power", other)

    def __rpow__(self, other: Any) -> "ndarray":
        """Reverse element-wise exponentiation."""
        return ndarray(other, backend=self.backend)._binary_op("power", self)

    def __mod__(self, other: Any) -> "ndarray":
        """Element-wise modulo."""
        return self._binary_op("mod", other)

    def __rmod__(self, other: Any) -> "ndarray":
        """Reverse element-wise modulo."""
        return ndarray(other, backend=self.backend)._binary_op("mod", self)

    def __floordiv__(self, other: Any) -> "ndarray":
        """Element-wise floor division."""
        return self._binary_op("floor_div", other)

    def __rfloordiv__(self, other: Any) -> "ndarray":
        """Reverse element-wise floor division."""
        return ndarray(other, backend=self.backend)._binary_op("floor_div", self)

    def __neg__(self) -> "ndarray":
        """Unary negation."""
        from .ops.ufunc_engine import ufunc_unary

        return ufunc_unary("neg", self)

    def __abs__(self) -> "ndarray":
        """Unary absolute value."""
        from .ops.ufunc_engine import ufunc_unary

        return ufunc_unary("abs", self)

    def matmul(self, other: Any) -> "ndarray":
        """Matrix multiplication (@ operator)."""
        # Ensure other is ndarray
        if not isinstance(other, ndarray):
            other = ndarray(other, backend=self.backend)

        # Phase 1: GPU fallback - if we're on GPU but operation not supported, use CPU
        if self.backend == BackendType.GPU:
            from .backend.dispatch import Dispatcher
            from .backend.errors import OperationNotSupportedError

            # Check if GPU kernel exists
            try:
                Dispatcher.get_kernel("matmul", BackendType.GPU)
            except OperationNotSupportedError:
                # GPU doesn't support matmul, fall back to CPU
                logger.info("GPU matmul not supported, falling back to CPU")
                cpu_self = ndarray(self.to_list(), backend=BackendType.CPU)
                cpu_other = ndarray(other.to_list(), backend=BackendType.CPU)
                return cpu_self.matmul(cpu_other)

        # Phase 2: Automatic GPU Switching for large operations
        # If operation is large enough, promoting to GPU might be faster
        if self.backend == BackendType.CPU:
            # Basic heuristic: if both dims >= 2048
            if (
                len(self.shape) >= 2
                and self.shape[-1] >= 2048
                and self.shape[-2] >= 2048
            ):
                # Check if Metal is available
                from .backend import get_system_capabilities

                caps = get_system_capabilities()
                if caps.get("gpu", {}).get("metal_available", False):
                    # Promote to GPU for large matmul
                    logger.info("Auto-switching large matmul to GPU based on size")
                    gpu_self = ndarray(self.to_list(), device="metal")
                    gpu_other = ndarray(other.to_list(), device="metal")
                    return gpu_self.matmul(gpu_other)

        # Delegate to backend implementation
        # Start timing
        import time

        from .backend.dispatch import dispatch_kernel
        from .profiler.core import record_op

        start_time = time.perf_counter()

        result = dispatch_kernel(
            "matmul", self.backend, self, other, shape_a=self.shape, shape_b=other.shape
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op("matmul", elapsed_ms, self.backend.name)

        return ndarray(result, dtype=self._dtype, backend=self.backend)

    def __matmul__(self, other: Any) -> "ndarray":
        """Matrix multiplication (@ operator)."""
        return self.matmul(other)

    def _get_scalar_value(self) -> float:
        """Extract scalar value from single-element array."""
        if self._element_count != 1:
            raise ValueError("Cannot compare non-scalar array")
        if isinstance(self._backing_data, list):
            return float(self._backing_data[0])
        return float(self._backing_data)  # type: ignore[arg-type]

    def __lt__(self, other: Any) -> bool:
        """Less than comparison (for scalar arrays)."""
        return self._get_scalar_value() < (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison (for scalar arrays)."""
        return self._get_scalar_value() <= (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison (for scalar arrays)."""
        return self._get_scalar_value() > (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison (for scalar arrays)."""
        return self._get_scalar_value() >= (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def _get_buffer_view(self) -> "BufferView":
        """
        Extract BufferView for zero-copy FFI.

        Returns a BufferView abstraction that provides:
        - Stride awareness (is_contiguous check)
        - Device tracking (CPU/GPU)
        - Explicit ownership (keeps data alive)
        - Cleaner dispatch paths

        Returns:
            BufferView wrapping the backing data

        Raises:
            ValueError: If backing data cannot be wrapped
        """
        from .buffer import CPU, METAL, BufferView, from_buffer, from_numpy

        # Determine target device
        target_device = CPU
        if self.backend == BackendType.GPU:
            import platform

            if platform.system() == "Darwin":
                target_device = METAL
            # Future: CUDA support

        # Case 1: NumPy / Array interface array (fastest path)
        if hasattr(self._backing_data, "__array_interface__"):
            return from_numpy(
                self._backing_data, device=target_device, dtype=self._dtype
            )

        # Case 2: bytes/bytearray/memoryview (buffer protocol)
        elif isinstance(self._backing_data, (bytes, bytearray, memoryview)):
            return from_buffer(
                self._backing_data,
                dtype=self._dtype,
                shape=self._shape,
                device=target_device,
            )

        # Case 3: List (convert to bytearray buffer)
        elif isinstance(self._backing_data, list):
            import struct

            from .ops.math import _flatten

            flat = _flatten(self._backing_data)
            if self._dtype == DataType.INT32:
                buf = bytearray()
                for val in flat:
                    buf.extend(struct.pack("i", int(val)))
            else:
                buf = bytearray()
                for val in flat:
                    buf.extend(struct.pack("f", float(val)))

            return from_buffer(
                buf,
                dtype=self._dtype,
                shape=self._shape,
                device=target_device,
            )

        # Case 4: Unknown type
        else:
            raise ValueError(
                f"Cannot create BufferView from {type(self._backing_data)}. "
                "Object must be NumPy array, list, or support buffer protocol."
            )

    def _get_buffer_pointer(self, dtype_char="u1") -> Tuple[int, int, Any]:
        """
        Extract raw buffer pointer for zero-copy FFI.

        Args:
            dtype_char: NumPy dtype character code ('u1'=uint8, 'f4'=float32, 'i4'=int32)

        Returns:
            tuple: (pointer: int, count: int, buffer_ref: Any)
                   buffer_ref must be kept alive during FFI call

        Raises:
            ValueError: If backing data cannot be converted to buffer
        """
        import ctypes

        # Case 1: Array with buffer interface (e.g. NumPy/ctypes array)
        if hasattr(self._backing_data, "__array_interface__"):
            array = self._backing_data
            ptr = array.__array_interface__["data"][0]
            count = getattr(array, "size", self._element_count)
            return (ptr, count, array)

        # Case 2: bytes/bytearray/memoryview (buffer protocol)
        elif isinstance(self._backing_data, (bytes, bytearray, memoryview)):
            mv = memoryview(self._backing_data)

            # Get pointer via ctypes
            c_type: type
            if dtype_char == "u1":
                c_type = ctypes.c_uint8
            elif dtype_char == "f4":
                c_type = ctypes.c_float
            elif dtype_char == "i4":
                c_type = ctypes.c_int32
            else:
                raise ValueError(f"Unsupported dtype: {dtype_char}")

            # Cast memoryview to c_type array
            c_buffer = (c_type * len(mv)).from_buffer(mv)
            ptr = ctypes.addressof(c_buffer)
            count = len(mv)

            return (ptr, count, (mv, c_buffer))

        # Case 3: List (convert to bytearray)
        elif isinstance(self._backing_data, list):
            import struct

            if dtype_char == "u1":
                # Simplified flattening for POC
                def flatten(items):
                    for x in items:
                        if isinstance(x, (list, tuple)):
                            yield from flatten(x)
                        else:
                            yield x

                buffer = bytearray(int(x) for x in flatten(self._backing_data))
                count = len(buffer)
            elif dtype_char == "f4":
                import struct

                def flatten(items):
                    for x in items:
                        if isinstance(x, (list, tuple)):
                            yield from flatten(x)
                        else:
                            yield x

                buffer = bytearray()
                for x in flatten(self._backing_data):
                    buffer.extend(struct.pack("f", float(x)))
                count = self._element_count
            elif dtype_char == "i4":
                import struct

                def flatten(items):
                    for x in items:
                        if isinstance(x, (list, tuple)):
                            yield from flatten(x)
                        else:
                            yield x

                buffer = bytearray()
                for x in flatten(self._backing_data):
                    buffer.extend(struct.pack("i", int(x)))
                count = self._element_count
            else:
                raise ValueError(f"Unsupported dtype: {dtype_char}")

            c_buffer = (ctypes.c_uint8 * len(buffer)).from_buffer(buffer)
            ptr = ctypes.addressof(c_buffer)

            return (ptr, count, (buffer, c_buffer))

        # Case 4: Generic buffer protocol fallback
        else:
            try:
                mv = memoryview(self._backing_data)  # type: ignore[arg-type]
                # Recursive(ish) logic for memoryview path
                # Just fail for now to keep simple, as case 2 handles mv
                ptr = ctypes.addressof((ctypes.c_byte * len(mv)).from_buffer(mv))
                return (ptr, len(mv), mv)
            except TypeError:
                raise ValueError(
                    f"Cannot extract buffer from {type(self._backing_data)}. "
                    "Object must support buffer protocol or be a list."
                ) from None

    def all(self) -> "ndarray":
        """Returns True if all elements evaluate to True."""
        try:
            from ._corepy_rust import array_all  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = array_all(ptr, count)

            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)

        except ImportError:
            logger.warning("Rust extension not available.")
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("all", self.backend, self._backing_data)
            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)

    def any(self) -> "ndarray":
        """Returns True if any element evaluates to True."""
        try:
            from ._corepy_rust import array_any  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = array_any(ptr, count)

            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("any", self.backend, self._backing_data)
            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)

    def sum(self) -> "ndarray":
        """Returns sum of all elements."""
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()

        # Fast path: Native Python sum for small arrays
        if self._should_use_fast_path():
            import builtins

            from .ops.math import _flatten

            result = float(builtins.sum(_flatten(self.to_list())))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("sum", elapsed_ms, "Native-Fast")
            return ndarray([result], dtype=self._dtype, backend=self.backend)

        try:
            from ._corepy_rust import (  # type: ignore[import-untyped]
                array_mean_f32,
                array_mean_f32_strided,
                array_sum_f32,
                array_sum_f32_strided,
                array_sum_i32,
                metal_sum_f32,
            )

            # Use BufferView for dispatch decision
            view = self._get_buffer_view()

            if self._dtype == DataType.INT32:
                # INT32 path (contiguous only for now)
                ptr, count, _ref = self._get_buffer_pointer("i4")
                result = array_sum_i32(ptr, count)
            elif view.device.is_metal():
                # Metal GPU path
                result = metal_sum_f32(view.data_ptr, view.element_count)
            elif view.is_contiguous():
                # Fast path: contiguous f32
                result = array_sum_f32(view.data_ptr, view.element_count)
            else:
                # Zero-copy path: strided f32
                result = array_sum_f32_strided(
                    view.data_ptr,
                    list(view.shape),
                    list(view.strides)
                    if view.strides is not None
                    else [s * view.dtype.itemsize for s in get_c_strides(view.shape)],  # type: ignore[arg-type]
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op(
                "sum", elapsed_ms, "CPU" if not view.device.is_metal() else "Metal"
            )
            return ndarray([result], dtype=self._dtype, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("sum", self.backend, self._backing_data)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("sum", elapsed_ms, self.backend.name)
            return ndarray(result, dtype=self._dtype, backend=self.backend)

    def mean(self) -> "ndarray":
        """Returns arithmetic mean of all elements."""
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()

        # Fast path: Native Python mean for small arrays
        if self._should_use_fast_path():
            import builtins

            from .ops.math import _flatten

            flat = _flatten(self.to_list())
            result = float(builtins.sum(flat) / len(flat)) if flat else 0.0
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("mean", elapsed_ms, "Native-Fast")
            return ndarray([result], dtype=DataType.FLOAT32, backend=self.backend)

        try:
            from ._corepy_rust import (  # type: ignore[import-untyped]
                array_mean_f32,
                array_mean_f32_strided,
                metal_mean_f32,
            )

            # Use BufferView for dispatch decision
            view = self._get_buffer_view()

            if view.device.is_metal():
                # Metal GPU path
                result = metal_mean_f32(view.data_ptr, view.element_count)
            elif view.is_contiguous():
                # Fast path: contiguous f32
                result = array_mean_f32(view.data_ptr, view.element_count)
            else:
                # Zero-copy path: strided f32
                result = array_mean_f32_strided(
                    view.data_ptr,
                    list(view.shape),
                    list(view.strides)
                    if view.strides is not None
                    else [s * view.dtype.itemsize for s in get_c_strides(view.shape)],  # type: ignore[arg-type]
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op(
                "mean", elapsed_ms, "CPU" if not view.device.is_metal() else "Metal"
            )
            return ndarray([result], dtype=DataType.FLOAT32, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("mean", self.backend, self._backing_data)
            return ndarray(result, dtype=DataType.FLOAT32, backend=self.backend)

    def std(self) -> "ndarray":
        """Returns standard deviation of all elements."""
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("std", self.backend, self._backing_data)
        return ndarray(result, dtype=DataType.FLOAT32, backend=self.backend)

    def max(self) -> "ndarray":
        """Returns maximum value of all elements."""
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()

        # Priority 1: Metal GPU
        target_view = self._get_buffer_view()
        if (
            self.backend == BackendType.GPU
            and target_view.device.is_metal()
            and self._element_count >= 1024
        ):
            try:
                from . import _corepy_rust as ffi

                result = ffi.metal_max_f32(target_view.data_ptr, self._element_count)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                record_op("max", elapsed_ms, "Metal")
                return ndarray([result], dtype=self._dtype, backend=self.backend)
            except ImportError:
                pass

        # Priority 2: CPU Rust (Strided) - Fallback for now because contiguous not exposed?
        # Actually strided optimization works for contiguous too if strides are correct.
        try:
            from . import _corepy_rust as ffi

            # Use strided version for generality
            view = self._get_buffer_view()
            result = ffi.array_max_f32_strided(
                view.data_ptr,
                list(view.shape),
                list(view.strides)
                if view.strides is not None
                else [s * view.dtype.itemsize for s in get_c_strides(view.shape)],  # type: ignore[arg-type]
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("max", elapsed_ms, "Rust-CPU")
            return ndarray([result], dtype=self._dtype, backend=self.backend)
        except ImportError:
            pass

        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("max", self.backend, self._backing_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op("max", elapsed_ms, self.backend.name)
        return ndarray(result, dtype=self._dtype, backend=self.backend)

    def min(self) -> "ndarray":
        """Returns minimum value of all elements."""
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()

        # Priority 1: Metal GPU
        target_view = self._get_buffer_view()
        if (
            self.backend == BackendType.GPU
            and target_view.device.is_metal()
            and self._element_count >= 1024
        ):
            try:
                from . import _corepy_rust as ffi

                result = ffi.metal_min_f32(target_view.data_ptr, self._element_count)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                record_op("min", elapsed_ms, "Metal")
                return ndarray([result], dtype=self._dtype, backend=self.backend)
            except ImportError:
                pass

        # Priority 2: CPU Rust (Strided)
        try:
            from . import _corepy_rust as ffi

            view = self._get_buffer_view()
            result = ffi.array_min_f32_strided(
                view.data_ptr,
                list(view.shape),
                list(view.strides)
                if view.strides is not None
                else [s * view.dtype.itemsize for s in get_c_strides(view.shape)],  # type: ignore[arg-type]
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("min", elapsed_ms, "Rust-CPU")
            return ndarray([result], dtype=self._dtype, backend=self.backend)
        except ImportError:
            pass

        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("min", self.backend, self._backing_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op("min", elapsed_ms, self.backend.name)
        return ndarray(result, dtype=self._dtype, backend=self.backend)

    def copy(self) -> "ndarray":
        """Returns a copy of the array."""
        if isinstance(self._backing_data, list):
            new_data = list(self._backing_data)
        else:
            new_data = self._backing_data  # type: ignore[assignment]
        return ndarray(new_data, dtype=self._dtype, backend=self.backend)

    def _to_numpy_internal(self) -> Any:
        """Internal helper to convert backing_data to NumPy (used during init)."""
        try:
            import numpy as np
        except ImportError:
            return self._backing_data

        if isinstance(self._backing_data, np.ndarray):
            return self._backing_data
        elif isinstance(self._backing_data, list):
            from .ops.math import _flatten

            dtype_map = {
                DataType.FLOAT32: np.float32,
                DataType.FLOAT64: np.float64,
                DataType.INT32: np.int32,
                DataType.INT64: np.int64,
                DataType.BOOL: bool,
            }
            np_dtype = dtype_map.get(self._dtype, np.float32)
            return np.array(_flatten(self._backing_data), dtype=np_dtype).reshape(
                self.shape
            )
        else:
            return np.array(self._backing_data).reshape(self.shape)

    def to_numpy(self) -> Any:
        """
        Convert the array to a NumPy array.
        Returns:
            NDArray: NumPy array containing the data.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to export corepy.ndarray to numpy format (to_numpy()). "
                "Please install numpy: pip install numpy"
            ) from None

        # Phase 2: Check if we already have CPU data cached
        if self._cpu_data is not None and isinstance(self._cpu_data, np.ndarray):
            return self._cpu_data

        if isinstance(self._backing_data, np.ndarray):
            return self._backing_data
        elif isinstance(self._backing_data, list):
            from .ops.math import _flatten

            dtype_map = {
                DataType.FLOAT32: np.float32,
                DataType.FLOAT64: np.float64,
                DataType.INT32: np.int32,
                DataType.INT64: np.int64,
                DataType.BOOL: bool,
            }
            np_dtype = dtype_map.get(self._dtype, np.float32)
            return np.array(_flatten(self._backing_data), dtype=np_dtype).reshape(
                self.shape
            )
        else:
            return np.array(self._backing_data).reshape(self.shape)

    def __len__(self) -> int:
        """Returns the number of elements in the array."""
        return self._element_count

    def _binary_op(self, op: str, other: Any) -> "ndarray":
        """Helper for binary operations via Rust FFI."""
        import time

        from .profiler.core import record_op

        start_time = time.perf_counter()
        if isinstance(other, (int, float)):
            other = ndarray([float(other)] * self._element_count, device=self._device)
        elif isinstance(other, ndarray) and other._element_count == 1:
            # Broadcasting: single-element array to match self's size
            scalar_val = (
                other._backing_data[0]
                if isinstance(other._backing_data, list)
                else float(other._backing_data)  # type: ignore[arg-type]
            )
            other = ndarray(
                [float(scalar_val)] * self._element_count, device=self._device
            )  # type: ignore[arg-type]

        if not isinstance(other, ndarray):
            raise ValueError("Binary ops require array or scalar")

        if other.backend != self.backend:
            raise BackendError(f"Backend mismatch: {self.backend} vs {other.backend}")

        if self.shape != other.shape:
            from .ops.ufunc_engine import _broadcast_pair

            a_bc, b_bc = _broadcast_pair(self, other)
            return a_bc._binary_op(op, b_bc)

        # Fast path: Native Python for small CPU arrays to avoid FFI overhead
        if (
            self._should_use_fast_path()
            and other._should_use_fast_path()
            and self.shape == other.shape
        ):
            from .ops.math import _flatten

            flat_a = _flatten(self.to_list())
            flat_b = _flatten(other.to_list())
            res = None
            if op == "add":
                res = [x + y for x, y in zip(flat_a, flat_b)]
            elif op == "sub":
                res = [x - y for x, y in zip(flat_a, flat_b)]
            elif op == "mul":
                res = [x * y for x, y in zip(flat_a, flat_b)]
            elif op == "div":
                res = [x / y for x, y in zip(flat_a, flat_b)]
            elif op == "power":
                res = [x**y for x, y in zip(flat_a, flat_b)]
            elif op == "mod":
                res = [x % y for x, y in zip(flat_a, flat_b)]
            elif op == "floor_div":
                res = [x // y for x, y in zip(flat_a, flat_b)]

            if res is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                record_op(op, elapsed_ms, "Native-Fast")
                return ndarray(res, dtype=self._dtype, backend=self.backend).reshape(
                    self.shape
                )

        # Priority 1: Metal GPU (if available and large enough OR if broadcasting needed)
        # We generally avoid GPU overhead for very small arrays (< 1024 elements)
        # but allow it for broadcasting since CPU fallback doesn't support it yet
        view = self._get_buffer_view()
        is_metal = view.device.is_metal()
        use_metal = (
            self.backend == BackendType.GPU
            and is_metal
            and (self._element_count >= 1024 or self.shape != other.shape)
        )
        # print(f"DEBUG: use_metal={use_metal} is_metal={is_metal} count={self._element_count}")

        if use_metal:
            # ...
            try:
                # Get input buffers (f32 hardcoded for now)
                # Note: buffer pointers on Metal are actually encoded offsets/handles
                # handled by the Rust/Metal bridge.
                # Allocate output buffer (Metal-compatible via NumPy empty for now)
                # In a real implementation, we'd allocate a MetalBuffer directly.
                import numpy as np

                from . import _corepy_rust as ffi  # type: ignore[import-untyped]

                final_np = np.empty(self.shape, dtype=np.float32)
                ptr_out = final_np.__array_interface__["data"][0]

                # Dispatch
                # Dispatch
                success = False

                # Check for broadcasting scenarios
                if self.shape != other.shape:
                    try:
                        from .broadcasting import (
                            broadcast_shapes,
                            compute_broadcast_strides,
                            get_c_strides,
                        )

                        target_shape = broadcast_shapes(self.shape, other.shape)
                        size = 1
                        for d in target_shape:
                            size *= d

                        # Allocate output
                        final_np = np.empty(target_shape, dtype=np.float32)
                        ptr_out = final_np.__array_interface__["data"][0]

                        # Compute strides
                        strides_a = compute_broadcast_strides(
                            self.shape, target_shape, get_c_strides(self.shape)
                        )
                        strides_b = compute_broadcast_strides(
                            other.shape, target_shape, get_c_strides(other.shape)
                        )

                        op_map = {"add": 0, "sub": 1, "mul": 2, "div": 3}
                        op_code = op_map.get(op)

                        if op_code is not None:
                            ffi.metal_broadcast_op(
                                op_code,
                                view.data_ptr,
                                other._get_buffer_view().data_ptr,
                                ptr_out,
                                list(target_shape),
                                strides_a,
                                strides_b,
                                size,
                                self._element_count,
                                other._element_count,
                            )
                            success = True
                    except (ValueError, AttributeError, ImportError):
                        # Fallback to Rust/CPU if broadcasting fails or not implemented
                        pass

                if not success and self.shape == other.shape:
                    if op == "add":
                        ffi.metal_add_f32(
                            view.data_ptr,
                            other._get_buffer_view().data_ptr,
                            ptr_out,
                            self._element_count,
                        )
                        success = True
                    elif op == "sub":
                        ffi.metal_sub_f32(
                            view.data_ptr,
                            other._get_buffer_view().data_ptr,
                            ptr_out,
                            self._element_count,
                        )
                        success = True
                    elif op == "mul":
                        ffi.metal_mul_f32(
                            view.data_ptr,
                            other._get_buffer_view().data_ptr,
                            ptr_out,
                            self._element_count,
                        )
                        success = True
                    elif op == "div":
                        ffi.metal_div_f32(
                            view.data_ptr,
                            other._get_buffer_view().data_ptr,
                            ptr_out,
                            self._element_count,
                        )
                        success = True

                if success:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    record_op(op, elapsed_ms, "Metal")
                    # Result is currently on CPU (transferred back by metal_backend implementation logic
                    # which copies to output buffer).
                    # Future optimization: Keep on GPU.
                    return ndarray(final_np, dtype=self._dtype, backend=self.backend)

            except ImportError:
                pass  # Fallback to Rust/CPU

        # Priority 2: Try Rust CPU kernels (fastest for medium/large arrays)
        if self.backend == BackendType.CPU:  # Priority 2: Rust CPU kernels
            try:
                from . import _corepy_rust as ffi  # type: ignore[import-untyped]

                # Get input buffers (f32 hardcoded for now)
                ptr_a, count_a, _ref_a = self._get_buffer_pointer("f4")
                ptr_b, count_b, _ref_b = other._get_buffer_pointer("f4")

                if count_a != count_b:
                    raise ValueError(f"Shape mismatch: {count_a} vs {count_b}")

                # Prepare output buffer
                import ctypes
                import struct

                # Allocate output buffer (bytearray for now, zero-init)
                # size = count * 4 bytes
                out_size = count_a * 4
                buf_out = bytearray(out_size)

                c_out = (ctypes.c_char * out_size).from_buffer(buf_out)
                ptr_out = ctypes.addressof(c_out)

                # Dispatch to Rust kernels
                if op == "add":
                    ffi.array_add_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "sub":
                    ffi.array_sub_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "mul":
                    ffi.array_mul_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "div":
                    ffi.array_div_f32(ptr_a, ptr_b, ptr_out, count_a)

                out_floats = [
                    struct.unpack("f", buf_out[i : i + 4])[0]
                    for i in range(0, len(buf_out), 4)
                ]
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                record_op(op, elapsed_ms, "Rust-CPU")
                return ndarray(out_floats, dtype=self._dtype, backend=self.backend)

            except ImportError:
                pass  # Fall through to C++ backend

        # Priority 3: Fallback to C++ dispatch (via backend system)
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel(
            op, self.backend, self._backing_data, other._backing_data
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op(op, elapsed_ms, self.backend.name)
        return ndarray(result, dtype=self._dtype, backend=self.backend)


#     def matmul(self, other: "ndarray") -> "ndarray":
#         """Matrix multiplication (handles 1D dot product and 2D matmul)."""
#         import time
#
#         from .profiler.core import record_op
#
#         start_time = time.perf_counter()
#         if not isinstance(other, ndarray):
#             raise ValueError("matmul requires array")
#
# Fast path: Use NumPy for small arrays to avoid FFI overhead
#         if self._should_use_fast_path() and other._should_use_fast_path():
#             import numpy as np
#
#             np_a = self.to_numpy()
#             np_b = other.to_numpy()
#             result = np.matmul(np_a, np_b)
#             elapsed_ms = (time.perf_counter() - start_time) * 1000
#             record_op("matmul", elapsed_ms, "NumPy-Fast")
#             return ndarray(result, dtype=self._dtype, backend=self.backend)
#
#         if self.backend == BackendType.CPU or (
#             self.backend == BackendType.GPU
#             and "metal" in str(self._get_buffer_view().device)
#         ):
#             try:
#                 from . import _corepy_rust as ffi  # type: ignore[import-untyped]
#
# Check for Metal dispatch
#                 view = self._get_buffer_view()
#                 is_metal = view.device.is_metal()
#
#                 if is_metal and len(self.shape) == 2 and len(other.shape) == 2:
# Metal Matrix Multiplication
#                     m, k1 = self.shape
#                     k2, n = other.shape
#
#                     if k1 != k2:
#                         raise ValueError(
#                             f"Matrix dimension mismatch: ({m}, {k1}) @ ({k2}, {n})"
#                         )
#
#                     view_b = other._get_buffer_view()
#
# Allocate output
#                     import numpy as np
#
#                     final_np = np.empty((m, n), dtype=np.float32)
#                     ptr_out = final_np.__array_interface__["data"][0]
#
#                     ffi.metal_matmul_f32(
#                         view.data_ptr, view_b.data_ptr, ptr_out, m, k1, n
#                     )
#
#                     elapsed_ms = (time.perf_counter() - start_time) * 1000
#                     record_op("matmul", elapsed_ms, "Metal")
#                     return ndarray(final_np, dtype=self._dtype, backend=self.backend)
#
# Case 1: Dot Product (1D @ 1D)
#                 if len(self.shape) == 1 and len(other.shape) == 1 and not is_metal:
#                     ptr_a, count_a, _ref_a = self._get_buffer_pointer("f4")
#                     ptr_b, count_b, _ref_b = other._get_buffer_pointer("f4")
#
#                     if count_a != count_b:
#                         raise ValueError(
#                             f"Dot product size mismatch: {count_a} vs {count_b}"
#                         )
#
#                     result = ffi.tensor_matmul_f32(ptr_a, ptr_b, count_a)
#                     elapsed_ms = (time.perf_counter() - start_time) * 1000
#                     record_op("matmul", elapsed_ms, "CPU")
#                     return ndarray(result, dtype=self._dtype, backend=self.backend)
#
# Case 2: Matrix Multiplication (2D @ 2D) - CPU
#                 elif len(self.shape) == 2 and len(other.shape) == 2 and not is_metal:
#                     m, k1 = self.shape
#                     k2, n = other.shape
#
#                     if k1 != k2:
#                         raise ValueError(
#                             f"Matrix dimension mismatch: ({m}, {k1}) @ ({k2}, {n})"
#                         )
#
#                     ptr_a, _c_a, _ref_a = self._get_buffer_pointer("f4")
#                     ptr_b, _c_b, _ref_b = other._get_buffer_pointer("f4")
#
# Prepare output buffer (Zero-Copy Optimization)
#                     import numpy as np
#
# Allocate uninitialized memory directly (fastest)
# Note: C++ kernels (AVX2/OpenBLAS) will initialize this (beta=0.0)
#                     final_np = np.empty((m, n), dtype=np.float32)
#
# Get raw pointer to the numpy array's data
#                     ptr_out = final_np.__array_interface__["data"][0]
#
# Dispatch 2D kernel
#                     ffi.tensor_matmul_2d_f32(ptr_a, ptr_b, ptr_out, m, k1, n)
#
# Return wrapped array
#                     elapsed_ms = (time.perf_counter() - start_time) * 1000
#                     record_op("matmul", elapsed_ms, "CPU")
#                     return ndarray(final_np, dtype=self._dtype, backend=self.backend)
#
#                 else:
# Generic shapes not supported in optimized kernel yet
#                     pass
#
#             except ImportError:
#                 pass  # Fallback
#
#         from .backend.dispatch import dispatch_kernel
#
#         result = dispatch_kernel(
#             "matmul",
#             self.backend,
#             self._backing_data,
#             other._backing_data,
#             shape_a=self.shape,
#             shape_b=other.shape,
#         )
#         elapsed_ms = (time.perf_counter() - start_time) * 1000
#         record_op("matmul", elapsed_ms, self.backend.name)
#         return ndarray(result, dtype=self._dtype, backend=self.backend)
#

# Backward compatibility alias (deprecated)
import warnings


def _deprecated_tensor_class(*args, **kwargs):
    """Deprecated: Use ndarray instead."""
    warnings.warn(
        "corepy.Tensor is deprecated, use corepy.array() or corepy.ndarray instead",
        DeprecationWarning,
        stacklevel=2,
    )


# Keep Tensor as deprecated alias for backward compatibility
Tensor = ndarray
