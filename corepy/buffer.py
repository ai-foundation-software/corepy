"""
BufferView: Unified buffer abstraction for CPU/GPU memory.

This module provides the core abstractions for Phase 5 of the Corepy architecture:
- Zero-copy when possible
- Explicit about copies
- Stride-aware
- Device-agnostic dispatch

See: docs/03_architecture/PHASE_5_BUFFER_INTERFACE.md
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .backend.types import DataType

logger = logging.getLogger("corepy.buffer")


class DeviceType(Enum):
    """Supported device types for tensor execution."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"


class MemoryType(Enum):
    """
    Memory allocation types.

    CPU Memory Types:
    - NORMAL: Standard pageable memory (malloc)

    GPU Memory Types (CUDA):
    - PINNED: Page-locked host memory (cudaHostAlloc)
              → Enables DMA, faster H2D/D2H transfers
              → Limited resource, use sparingly

    - UNIFIED: Managed memory (cudaMallocManaged)
               → Automatic migration
               → Convenient but unpredictable latency

    - DEVICE: GPU VRAM (cudaMalloc)
              → Fastest access from GPU kernels
              → No CPU access (segfault)
    """

    NORMAL = "normal"
    PINNED = "pinned"
    UNIFIED = "unified"
    DEVICE = "device"


@dataclass
class Device:
    """
    Device abstraction for CPU/GPU dispatch.

    Examples:
        Device(DeviceType.CPU)           # cpu
        Device(DeviceType.CUDA, 0)       # cuda:0
        Device(DeviceType.CUDA, 1)       # cuda:1
    """

    type: DeviceType
    index: int = 0

    def is_cpu(self) -> bool:
        """Check if this is a CPU device."""
        return self.type == DeviceType.CPU

    def is_cuda(self) -> bool:
        """Check if this is a CUDA GPU device."""
        return self.type == DeviceType.CUDA

    def is_metal(self) -> bool:
        """Check if this is a Metal GPU device."""
        return self.type == DeviceType.METAL

    def __str__(self) -> str:
        if self.is_cpu():
            return "cpu"
        return f"{self.type.value}:{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return False
        return self.type == other.type and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.type, self.index))


# Default CPU device singleton
CPU = Device(DeviceType.CPU)
METAL = Device(DeviceType.METAL, 0)


@dataclass
class BufferView:
    """
    Unified buffer abstraction for CPU/GPU memory.

    Design Principles:
    - Zero-copy when possible
    - Explicit about copies
    - Stride-aware
    - Device-agnostic dispatch

    Attributes:
        data_ptr: Raw pointer to memory (usize in Rust FFI)
        shape: Logical dimensions of the tensor
        strides: Byte strides per dimension (None = C-contiguous)
        dtype: Element data type
        device: CPU or GPU device
        memory_type: Type of memory allocation
        owner: Keep-alive reference to prevent GC
        writable: Whether buffer can be mutated
    """

    data_ptr: int
    shape: Tuple[int, ...]
    strides: Optional[Tuple[int, ...]]
    dtype: DataType
    device: Device
    memory_type: MemoryType
    owner: Any
    writable: bool

    def is_contiguous(self) -> bool:
        """
        Check if buffer is C-contiguous (dense row-major layout).

        Returns:
            True if strides match expected C-contiguous layout.
        """
        if self.strides is None:
            return True

        if len(self.shape) == 0:
            return True

        # Calculate expected C-contiguous strides
        expected_stride = self.dtype.itemsize
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]

        return True

    def ensure_contiguous(self) -> "BufferView":
        """
        Return a contiguous view, copying if needed.

        If already contiguous, returns self (zero-copy).
        If non-contiguous, creates a contiguous copy.

        Returns:
            BufferView that is guaranteed to be contiguous.
        """
        if self.is_contiguous():
            return self

        # Currently only CPU contiguous copy is supported
        if self.device.is_cpu():
            return self._cpu_contiguous_copy()

        # Future: GPU contiguous copy
        raise NotImplementedError(
            f"Contiguous copy not implemented for device: {self.device}"
        )

    def _cpu_contiguous_copy(self) -> "BufferView":
        """Create a contiguous CPU copy of this buffer."""
        import ctypes

        # Reconstruct array from pointer and create contiguous copy
        if isinstance(self.owner, np.ndarray):
            contiguous = np.ascontiguousarray(self.owner)
            ptr = contiguous.__array_interface__["data"][0]

            logger.warning(
                f"Copying non-contiguous array (shape={self.shape}) for kernel; "
                "consider using contiguous arrays for better performance"
            )

            return BufferView(
                data_ptr=ptr,
                shape=self.shape,
                strides=None,  # Now contiguous
                dtype=self.dtype,
                device=self.device,
                memory_type=self.memory_type,
                owner=contiguous,  # New owner
                writable=contiguous.flags["WRITEABLE"],
            )

        raise ValueError(f"Cannot create contiguous copy from owner: {type(self.owner)}")

    @property
    def element_count(self) -> int:
        """Total number of elements in the buffer."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        """Total size in bytes (for contiguous layout)."""
        return self.element_count * self.dtype.itemsize


def from_numpy(
    arr: "NDArray[Any]",
    device: Device = CPU,
    dtype: Optional[DataType] = None,
) -> BufferView:
    """
    Create a BufferView from a NumPy array.

    This is zero-copy for contiguous arrays. Non-contiguous arrays
    are marked with their actual strides for later handling.

    Args:
        arr: NumPy array to wrap
        device: Target device (default: CPU)
        dtype: Override dtype (default: infer from array)

    Returns:
        BufferView wrapping the array

    Safety:
        The returned BufferView keeps a reference to `arr` in the
        `owner` field, preventing garbage collection during use.
    """
    # Infer dtype from array if not provided
    if dtype is None:
        dtype = DataType.from_numpy(arr.dtype) if hasattr(DataType, "from_numpy") else _infer_dtype(arr.dtype)

    # Extract pointer from array interface
    ptr = arr.__array_interface__["data"][0]

    # Extract strides (None if C-contiguous)
    strides: Optional[Tuple[int, ...]] = None
    if not arr.flags["C_CONTIGUOUS"]:
        strides = tuple(arr.strides)

    return BufferView(
        data_ptr=ptr,
        shape=tuple(arr.shape),
        strides=strides,
        dtype=dtype,
        device=device,
        memory_type=MemoryType.NORMAL,
        owner=arr,
        writable=arr.flags["WRITEABLE"],
    )


def _infer_dtype(np_dtype: np.dtype) -> DataType:
    """Infer DataType from NumPy dtype."""
    dtype_map = {
        np.float32: DataType.FLOAT32,
        np.float64: DataType.FLOAT64,
        np.int32: DataType.INT32,
        np.int64: DataType.INT64,
        np.bool_: DataType.BOOL,
    }
    return dtype_map.get(np_dtype.type, DataType.FLOAT32)


def from_buffer(
    obj: Any,
    dtype: DataType,
    shape: Tuple[int, ...],
    device: Device = CPU,
) -> BufferView:
    """
    Create a BufferView from any buffer protocol object.

    Args:
        obj: Object supporting buffer protocol (bytes, bytearray, memoryview)
        dtype: Data type of elements
        shape: Shape of the tensor
        device: Target device (default: CPU)

    Returns:
        BufferView wrapping the buffer
    """
    import ctypes

    mv = memoryview(obj)
    
    # Handle read-only vs writable buffers differently
    if mv.readonly:
        # For immutable buffers (bytes), we must copy
        c_buffer = (ctypes.c_uint8 * len(mv)).from_buffer_copy(mv)
    else:
        # For mutable buffers (bytearray), zero-copy
        c_buffer = (ctypes.c_uint8 * len(mv)).from_buffer(mv)
    
    ptr = ctypes.addressof(c_buffer)

    return BufferView(
        data_ptr=ptr,
        shape=shape,
        strides=None,  # Assume contiguous for raw buffers
        dtype=dtype,
        device=device,
        memory_type=MemoryType.NORMAL,
        owner=(mv, c_buffer),  # Keep both alive
        writable=not mv.readonly,
    )
