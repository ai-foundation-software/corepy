import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .backend.errors import BackendError
from .backend.selector import select_backend
from .backend.session import get_session
from .backend.types import BackendType, DataType, OperationProperties, OperationType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .buffer import BufferView

logger = logging.getLogger("corepy.array")

from .broadcasting import get_c_strides


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
        data: Union[Sequence[Any], "ndarray", "NDArray[Any]"],
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
        elif isinstance(data, np.ndarray):  # numpy array
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
            if "cuda" in device or "gpu" in device or "metal" in device:
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
        self._cpu_data: Optional[np.ndarray] = None
        self._gpu_data = None  # Metal buffer handle (if GPU-resident)
        self._data_location = "cpu"  # "cpu", "gpu", or "both"

        # Initialize CPU data from backing_data
        if self._device == "cpu":
            self._cpu_data = self._to_numpy_internal()
            self._data_location = "cpu"
        else:
            # GPU device - prepare for lazy transfer
            self._cpu_data = self._to_numpy_internal()
            self._data_location = "cpu"  # Start on CPU, transfer on first use

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

        logger.debug(f"Array created on {self._backend_type}. Shape={self._shape}")

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
        Return array reshaped to given dimensions (NumPy-compatible).

        Args:
            shape: New shape as multiple arguments or single tuple/list.

        Returns:
            Reshaped array with same data.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        np_arr = self.to_numpy().reshape(shape)
        return ndarray(np_arr, dtype=self._dtype, backend=self.backend)

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

        # Fallback to NumPy
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
        Explicitly move tensor to a device (deprecated - use to_device).

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
        """
        Access element or slice of the array.

        Args:
            item: Index or slice.

        Returns:
            ndarray or scalar: Result of indexing.
        """
        # Delegate to backing data
        if isinstance(self._backing_data, list):
            # List slicing returns list, indexing returns element
            try:
                result = self._backing_data[item]
            except TypeError:
                # Handle tuple indexing for lists (not supported natively)
                # Fallback to numpy conversion for advanced indexing
                return self.to_numpy()[item]

            if isinstance(result, list):
                return ndarray(result, dtype=self._dtype, backend=self.backend)
            else:
                return result  # scalar
        elif isinstance(self._backing_data, np.ndarray):
            # NumPy slicing returns view or scalar
            result = self._backing_data[item]
            if isinstance(result, np.ndarray):
                # View (keeps backing data)
                return ndarray(result, dtype=self._dtype, backend=self.backend)
            else:
                # Scalar (numpy scalar)
                return result.item() if hasattr(result, "item") else result
        else:
            # Fallback for other types
            result = self.to_numpy()[item]
            if isinstance(result, np.ndarray):
                return ndarray(result, dtype=self._dtype, backend=self.backend)
            else:
                return result.item() if hasattr(result, "item") else result

    def __repr__(self):
        return f"ndarray({self._backing_data}, backend='{self._backend_type.value}')"

    def __add__(self, other: Any) -> "ndarray":
        """Element-wise addition."""
        return self._binary_op("add", other)

    def __sub__(self, other: Any) -> "ndarray":
        """Element-wise subtraction."""
        return self._binary_op("sub", other)

    def __mul__(self, other: Any) -> "ndarray":
        """Element-wise multiplication."""
        return self._binary_op("mul", other)

    def __truediv__(self, other: Any) -> "ndarray":
        """Element-wise division."""
        return self._binary_op("div", other)

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
                cpu_self = ndarray(self.to_numpy(), backend=BackendType.CPU)
                cpu_other = ndarray(other.to_numpy(), backend=BackendType.CPU)
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
                    gpu_self = ndarray(self.to_numpy(), device="metal")
                    gpu_other = ndarray(other.to_numpy(), device="metal")
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
        """Extract scalar value from single-element tensor."""
        if self._element_count != 1:
            raise ValueError("Cannot compare non-scalar tensor")
        if isinstance(self._backing_data, list):
            return float(self._backing_data[0])
        return float(self._backing_data)  # type: ignore[arg-type]

    def __lt__(self, other: Any) -> bool:
        """Less than comparison (for scalar tensors)."""
        return self._get_scalar_value() < (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison (for scalar tensors)."""
        return self._get_scalar_value() <= (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison (for scalar tensors)."""
        return self._get_scalar_value() > (
            other._get_scalar_value() if isinstance(other, ndarray) else float(other)
        )

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison (for scalar tensors)."""
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

        # Case 1: NumPy array (fastest path)
        if isinstance(self._backing_data, np.ndarray):
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

        # Case 3: List (convert to NumPy first)
        elif isinstance(self._backing_data, list):
            import struct

            # Flatten nested lists
            def flatten(items):
                for x in items:
                    if isinstance(x, (list, tuple)):
                        yield from flatten(x)
                    else:
                        yield x

            # Convert to numpy based on dtype
            if self._dtype == DataType.FLOAT32:
                arr = np.array(list(flatten(self._backing_data)), dtype=np.float32)
            elif self._dtype == DataType.INT32:
                arr = np.array(list(flatten(self._backing_data)), dtype=np.int32)
            elif self._dtype == DataType.FLOAT64:
                arr = np.array(list(flatten(self._backing_data)), dtype=np.float64)
            elif self._dtype == DataType.INT64:
                arr = np.array(list(flatten(self._backing_data)), dtype=np.int64)
            else:
                arr = np.array(list(flatten(self._backing_data)), dtype=np.float32)

            return from_numpy(arr, device=target_device, dtype=self._dtype)

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

        import numpy as np

        # Case 1: NumPy array (fastest path)
        if isinstance(self._backing_data, np.ndarray):
            array = self._backing_data

            # SYSTEMS SAFETY CHECK:
            # We must ensure the array is C-contiguous before extracting the raw pointer.
            # Passing non-contiguous strides to a dense kernel results in data corruption.
            if not array.flags["C_CONTIGUOUS"]:
                # Safe Copy (Performance Penalty, but Correct)
                # Log at WARNING level so hidden copies are visible in production
                logger.warning(
                    f"Copying non-contiguous array (shape={array.shape}) for kernel; "
                    "consider using contiguous arrays for better performance"
                )

                array = np.ascontiguousarray(array)

            # Validate dtype matches request?
            if dtype_char == "f4" and array.dtype != np.float32:
                pass  # TODO: Handle dtype mismatch or define strict rules

            # __array_interface__ provides (ptr, readonly)
            ptr = array.__array_interface__["data"][0]
            count = array.size
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
            from ._corepy_rust import tensor_all  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = tensor_all(ptr, count)

            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)

        except ImportError:
            logger.warning("Rust extension not available.")
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("all", self.backend, self._backing_data)
            return ndarray(result, dtype=DataType.BOOL, backend=self.backend)

    def any(self) -> "ndarray":
        """Returns True if any element evaluates to True."""
        try:
            from ._corepy_rust import tensor_any  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = tensor_any(ptr, count)

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

        # Fast path: Use NumPy for small arrays to avoid FFI overhead
        if self._should_use_fast_path():
            result = float(np.sum(self.to_numpy()))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("sum", elapsed_ms, "NumPy-Fast")
            return ndarray([result], dtype=self._dtype, backend=self.backend)

        try:
            from ._corepy_rust import (  # type: ignore[import-untyped]
                metal_sum_f32,
                tensor_sum_f32,
                tensor_sum_f32_strided,
                tensor_sum_i32,
            )

            # Use BufferView for dispatch decision
            view = self._get_buffer_view()

            if self._dtype == DataType.INT32:
                # INT32 path (contiguous only for now)
                ptr, count, _ref = self._get_buffer_pointer("i4")
                result = tensor_sum_i32(ptr, count)
            elif view.device.is_metal():
                # Metal GPU path
                result = metal_sum_f32(view.data_ptr, view.element_count)
            elif view.is_contiguous():
                # Fast path: contiguous f32
                result = tensor_sum_f32(view.data_ptr, view.element_count)
            else:
                # Zero-copy path: strided f32
                result = tensor_sum_f32_strided(
                    view.data_ptr,
                    list(view.shape),
                    list(view.shape),  # type: ignore[arg-type]
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

        # Fast path: Use NumPy for small arrays to avoid FFI overhead
        if self._should_use_fast_path():
            result = float(np.mean(self.to_numpy()))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("mean", elapsed_ms, "NumPy-Fast")
            return ndarray([result], dtype=DataType.FLOAT32, backend=self.backend)

        try:
            from ._corepy_rust import (  # type: ignore[import-untyped]
                metal_mean_f32,
                tensor_mean_f32,
                tensor_mean_f32_strided,
            )

            # Use BufferView for dispatch decision
            view = self._get_buffer_view()

            if view.device.is_metal():
                # Metal GPU path
                result = metal_mean_f32(view.data_ptr, view.element_count)
            elif view.is_contiguous():
                # Fast path: contiguous f32
                result = tensor_mean_f32(view.data_ptr, view.element_count)
            else:
                # Zero-copy path: strided f32
                result = tensor_mean_f32_strided(
                    view.data_ptr,
                    list(view.shape),
                    list(view.shape),  # type: ignore[arg-type]
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
            result = ffi.tensor_max_f32_strided(
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
            result = ffi.tensor_min_f32_strided(
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

    def _to_numpy_internal(self) -> "NDArray[Any]":
        """Internal helper to convert backing_data to NumPy (used during init)."""
        if isinstance(self._backing_data, np.ndarray):
            return self._backing_data
        elif isinstance(self._backing_data, list):
            import struct

            def flatten(items):
                for x in items:
                    if isinstance(x, (list, tuple)):
                        yield from flatten(x)
                    else:
                        yield x

            dtype_map = {
                DataType.FLOAT32: np.float32,
                DataType.FLOAT64: np.float64,
                DataType.INT32: np.int32,
                DataType.INT64: np.int64,
                DataType.BOOL: bool,
            }
            np_dtype = dtype_map.get(self._dtype, np.float32)
            return np.array(list(flatten(self._backing_data)), dtype=np_dtype).reshape(
                self.shape
            )
        else:
            return np.array(self._backing_data).reshape(self.shape)

    def to_numpy(self) -> "NDArray[Any]":
        """
        Convert the array to a NumPy array.
        Returns:
            NDArray: NumPy array containing the data.
        """
        # Phase 2: Check if we already have CPU data cached
        if self._cpu_data is not None:
            return self._cpu_data

        # Original logic for backward compatibility
        if isinstance(self._backing_data, np.ndarray):
            return self._backing_data
        elif isinstance(self._backing_data, list):
            # Same logic as _get_buffer_view for consistency
            import struct

            def flatten(items):
                for x in items:
                    if isinstance(x, (list, tuple)):
                        yield from flatten(x)
                    else:
                        yield x

            dtype_map = {
                DataType.FLOAT32: np.float32,
                DataType.FLOAT64: np.float64,
                DataType.INT32: np.int32,
                DataType.INT64: np.int64,
                DataType.BOOL: bool,
            }
            np_dtype = dtype_map.get(self._dtype, np.float32)
            return np.array(list(flatten(self._backing_data)), dtype=np_dtype).reshape(
                self.shape
            )
        else:
            # Fallback for buffer protocol objects
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
            # Broadcasting: single-element tensor to match self's size
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

        # Fast path: Use NumPy for small arrays to avoid FFI overhead
        # print(f"DEBUG: op={op} device={self._device} backend={self.backend} fast={self._should_use_fast_path()}")
        if self._should_use_fast_path() and other._should_use_fast_path():
            # print("DEBUG: Using Fast Path")
            np_a = self.to_numpy()
            np_b = other.to_numpy()
            # ... (rest of fast path)

            if op == "add":
                result = np_a + np_b
            elif op == "sub":
                result = np_a - np_b
            elif op == "mul":
                result = np_a * np_b
            elif op == "div":
                result = np_a / np_b
            else:
                raise ValueError(f"Unknown operation: {op}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op(op, elapsed_ms, "NumPy-Fast")
            return ndarray(result, dtype=self._dtype, backend=self.backend)

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
                    ffi.tensor_add_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "sub":
                    ffi.tensor_sub_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "mul":
                    ffi.tensor_mul_f32(ptr_a, ptr_b, ptr_out, count_a)
                elif op == "div":
                    ffi.tensor_div_f32(ptr_a, ptr_b, ptr_out, count_a)

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
