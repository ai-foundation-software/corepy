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

logger = logging.getLogger("corepy.tensor")


class Tensor:
    """
    A multi-dimensional array object that automatically selects the best
    execution backend (CPU/GPU) based on data size and operation complexity.
    """

    _backing_data: Any
    _shape: Tuple[int, ...]

    def __init__(
        self,
        data: Union[Sequence[Any], "Tensor", "NDArray[Any]"],
        dtype: DataType = DataType.FLOAT32,
        backend: Optional[Union[str, BackendType]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize a Tensor.

        Args:
            data: Input data (list, tuple, or another Tensor).
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
            self._backing_data = data
        elif isinstance(data, memoryview):
            self._shape = data.shape if data.shape else (len(data),)
            # Calculate size for ND memoryview
            size = 1
            for dim in self._shape:
                size *= dim
            self._element_count = size
            self._backing_data = data
        elif isinstance(data, np.ndarray):  # numpy array
            self._shape = tuple(int(d) for d in data.shape)
            self._element_count = int(data.size)
            self._backing_data = data
        else:
            # scalar or error
            self._shape = (1,)
            self._element_count = 1
            self._backing_data = [data]

        # Resolve requested backend/device
        requested_backend = None
        if device:
            if "cuda" in device or "gpu" in device:
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
        # Let's assume creation is MEMORY_BOUND.
        op_props = OperationProperties(
            element_count=self._element_count,
            shape=self._shape,
            # Approximating bytes: len * 4 bytes for float32
            dtype_bytes=4,
        )

        # We treat 'allocation' as a memory operation.
        # However, for 'Correctness-First', we usually default to CPU for storage
        # unless explicitly told otherwise or if we are consuming GPU data.
        # BUT, the goal is "Tensor(data) # auto".
        # So we should check if this data is "large enough" to justify GPU storage?
        # Usually, just storing is not compute. So auto-placement should default CPU
        # unless immediate heavy compute is expected?
        # Actually, "Tensor(data)" usually implies "Ready for compute".
        # Let's use COMPUTE_VECTOR as a proxy for "Will I use this for compute?"
        # to see if it qualifies for GPU memory residence.
        # This is a heuristic. Stronger approach: Default CPU, move on demand.
        # But per requirements: "Core Principles: Small data -> CPU always wins".

        session = get_session()
        self._backend_type = select_backend(
            OperationType.COMPUTE_VECTOR,  # Probe: "If I treated this as a compute vector, where would it go?"
            op_props,
            session.device_info,
            requested_backend=requested_backend,
        )
        self._device = device

        logger.debug(f"Tensor created on {self._backend_type}. Shape={self._shape}")

    @property
    def backend(self) -> BackendType:
        return self._backend_type

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def to(self, device: str) -> "Tensor":
        """
        Explicitly move tensor to a device.
        Arguments:
            device: 'cpu' or 'gpu'
        """
        # Create a new Tensor with explicit backend
        # In real impl, we would copy data buffer
        return Tensor(self._backing_data, dtype=self._dtype, device=device)

    def __repr__(self):
        return f"Tensor({self._backing_data}, backend='{self._backend_type.value}')"

    def __add__(self, other: Any) -> "Tensor":
        """Element-wise addition."""
        return self._binary_op("add", other)

    def __sub__(self, other: Any) -> "Tensor":
        """Element-wise subtraction."""
        return self._binary_op("sub", other)

    def __mul__(self, other: Any) -> "Tensor":
        """Element-wise multiplication."""
        return self._binary_op("mul", other)

    def __truediv__(self, other: Any) -> "Tensor":
        """Element-wise division."""
        return self._binary_op("div", other)

    def _get_scalar_value(self) -> float:
        """Extract scalar value from single-element tensor."""
        if self._element_count != 1:
            raise ValueError("Cannot compare non-scalar tensor")
        if isinstance(self._backing_data, list):
            return float(self._backing_data[0])
        return float(self._backing_data)

    def __lt__(self, other: Any) -> bool:
        """Less than comparison (for scalar tensors)."""
        return self._get_scalar_value() < (other._get_scalar_value() if isinstance(other, Tensor) else float(other))

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison (for scalar tensors)."""
        return self._get_scalar_value() <= (other._get_scalar_value() if isinstance(other, Tensor) else float(other))

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison (for scalar tensors)."""
        return self._get_scalar_value() > (other._get_scalar_value() if isinstance(other, Tensor) else float(other))

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison (for scalar tensors)."""
        return self._get_scalar_value() >= (other._get_scalar_value() if isinstance(other, Tensor) else float(other))

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
                # Option A: Error
                # raise ValueError("Non-contiguous arrays not supported yet")

                # Option B: Safe Copy (Performance Penalty, but Correct)
                # We log this because hidden copies are a performance pitfall.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Copying non-contiguous array (shape={array.shape}) for zero-copy access"
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

    def all(self) -> "Tensor":
        """Returns True if all elements evaluate to True."""
        try:
            from ._corepy_rust import tensor_all  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = tensor_all(ptr, count)

            return Tensor(result, dtype=DataType.BOOL, backend=self.backend)

        except ImportError:
            logger.warning("Rust extension not available.")
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("all", self.backend, self._backing_data)
            return Tensor(result, dtype=DataType.BOOL, backend=self.backend)

    def any(self) -> "Tensor":
        """Returns True if any element evaluates to True."""
        try:
            from ._corepy_rust import tensor_any  # type: ignore[import-untyped]

            ptr, count, _ref = self._get_buffer_pointer("u1")
            result = tensor_any(ptr, count)

            return Tensor(result, dtype=DataType.BOOL, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("any", self.backend, self._backing_data)
            return Tensor(result, dtype=DataType.BOOL, backend=self.backend)

    def sum(self) -> "Tensor":
        """Returns sum of all elements."""
        import time
        from .profiler.core import record_op
        
        start_time = time.perf_counter()
        try:
            from ._corepy_rust import (  # type: ignore[import-untyped]
                tensor_sum_f32,
                tensor_sum_i32,
            )

            if self._dtype == DataType.INT32:
                ptr, count, _ref = self._get_buffer_pointer("i4")
                result = tensor_sum_i32(ptr, count)
            else:
                # Default to float32
                ptr, count, _ref = self._get_buffer_pointer("f4")
                result = tensor_sum_f32(ptr, count)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("sum", elapsed_ms, "CPU")
            return Tensor([result], dtype=self._dtype, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("sum", self.backend, self._backing_data)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op("sum", elapsed_ms, self.backend.name)
            return Tensor(result, dtype=self._dtype, backend=self.backend)

    def mean(self) -> "Tensor":
        """Returns arithmetic mean of all elements."""
        try:
            from ._corepy_rust import tensor_mean_f32  # type: ignore[import-untyped]

            # Mean implies float result usually
            ptr, count, _ref = self._get_buffer_pointer("f4")
            result = tensor_mean_f32(ptr, count)

            return Tensor(result, dtype=DataType.FLOAT32, backend=self.backend)
        except ImportError:
            from .backend.dispatch import dispatch_kernel

            result = dispatch_kernel("mean", self.backend, self._backing_data)
            return Tensor(result, dtype=DataType.FLOAT32, backend=self.backend)

    def std(self) -> "Tensor":
        """Returns standard deviation of all elements."""
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("std", self.backend, self._backing_data)
        return Tensor(result, dtype=DataType.FLOAT32, backend=self.backend)

    def max(self) -> "Tensor":
        """Returns maximum value of all elements."""
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("max", self.backend, self._backing_data)
        return Tensor(result, dtype=self._dtype, backend=self.backend)

    def min(self) -> "Tensor":
        """Returns minimum value of all elements."""
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel("min", self.backend, self._backing_data)
        return Tensor(result, dtype=self._dtype, backend=self.backend)

    def copy(self) -> "Tensor":
        """Returns a copy of the tensor."""
        if isinstance(self._backing_data, list):
            new_data = list(self._backing_data)
        else:
            new_data = self._backing_data
        return Tensor(new_data, dtype=self._dtype, backend=self.backend)

    def __len__(self) -> int:
        """Returns the number of elements in the tensor."""
        return self._element_count

    def _binary_op(self, op: str, other: Any) -> "Tensor":
        """Helper for binary operations via Rust FFI."""
        import time
        from .profiler.core import record_op
        
        start_time = time.perf_counter()
        if isinstance(other, (int, float)):
            other = Tensor([float(other)] * self._element_count, device=self._device)
        elif isinstance(other, Tensor) and other._element_count == 1:
            # Broadcasting: single-element tensor to match self's size
            scalar_val = other._backing_data[0] if isinstance(other._backing_data, list) else float(other._backing_data)
            other = Tensor([float(scalar_val)] * self._element_count, device=self._device)

        if not isinstance(other, Tensor):
            raise ValueError("Binary ops require Tensor or scalar")

        if other.backend != self.backend:
            raise BackendError(f"Backend mismatch: {self.backend} vs {other.backend}")

        # Check if we can use optimized CPU Runtime
        if self.backend == BackendType.CPU:
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

                # Dispatch
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
                record_op(op, elapsed_ms, "CPU")
                return Tensor(out_floats, dtype=self._dtype, backend=self.backend)

            except ImportError:
                pass  # Fallback to dispatch

        # Fallback to general dispatch (GPU, Custom, or if Rust missing)
        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel(
            op, self.backend, self._backing_data, other._backing_data
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op(op, elapsed_ms, self.backend.name)
        return Tensor(result, dtype=self._dtype, backend=self.backend)

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication (handles 1D dot product and 2D matmul)."""
        import time
        from .profiler.core import record_op
        
        start_time = time.perf_counter()
        if not isinstance(other, Tensor):
            raise ValueError("matmul requires Tensor")

        if self.backend == BackendType.CPU:
            try:
                from . import _corepy_rust as ffi  # type: ignore[import-untyped]

                # Case 1: Dot Product (1D @ 1D)
                if len(self.shape) == 1 and len(other.shape) == 1:
                    ptr_a, count_a, _ref_a = self._get_buffer_pointer("f4")
                    ptr_b, count_b, _ref_b = other._get_buffer_pointer("f4")

                    if count_a != count_b:
                        raise ValueError(
                            f"Dot product size mismatch: {count_a} vs {count_b}"
                        )

                    result = ffi.tensor_matmul_f32(ptr_a, ptr_b, count_a)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    record_op("matmul", elapsed_ms, "CPU")
                    return Tensor(result, dtype=self._dtype, backend=self.backend)

                # Case 2: Matrix Multiplication (2D @ 2D)
                elif len(self.shape) == 2 and len(other.shape) == 2:
                    m, k1 = self.shape
                    k2, n = other.shape

                    if k1 != k2:
                        raise ValueError(
                            f"Matrix dimension mismatch: ({m}, {k1}) @ ({k2}, {n})"
                        )

                    ptr_a, _c_a, _ref_a = self._get_buffer_pointer("f4")
                    ptr_b, _c_b, _ref_b = other._get_buffer_pointer("f4")

                    # Prepare output buffer (Zero-Copy Optimization)
                    import numpy as np

                    # Allocate uninitialized memory directly (fastest)
                    # Note: C++ kernels (AVX2/OpenBLAS) will initialize this (beta=0.0)
                    final_np = np.empty((m, n), dtype=np.float32)

                    # Get raw pointer to the numpy array's data
                    ptr_out = final_np.__array_interface__["data"][0]

                    # Dispatch 2D kernel
                    ffi.tensor_matmul_2d_f32(ptr_a, ptr_b, ptr_out, m, k1, n)

                    # Return wrapped Tensor
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    record_op("matmul", elapsed_ms, "CPU")
                    return Tensor(final_np, dtype=self._dtype, backend=self.backend)

                else:
                    # Generic shapes not supported in optimized kernel yet
                    pass

            except ImportError:
                pass  # Fallback

        from .backend.dispatch import dispatch_kernel

        result = dispatch_kernel(
            "matmul",
            self.backend,
            self._backing_data,
            other._backing_data,
            shape_a=self.shape,
            shape_b=other.shape,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        record_op("matmul", elapsed_ms, self.backend.name)
        return Tensor(result, dtype=self._dtype, backend=self.backend)
