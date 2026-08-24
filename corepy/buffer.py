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
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from .backend.types import DataType

logger = logging.getLogger("corepy.buffer")


class DeviceType(Enum):
    """Supported device types for array execution."""

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
        shape: Logical dimensions of the array
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

        raise ValueError(
            f"Cannot create contiguous copy from owner: {type(self.owner)}. "
            "Please explicitly copy external arrays to contiguous memory before wrapping."
        )

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
        shape: Shape of the array
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


def from_numpy(
    arr: Any,
    device: Device = CPU,
    dtype: Optional[DataType] = None,
) -> BufferView:
    """
    Create a BufferView from a NumPy ndarray.

    Args:
        arr: NumPy ndarray
        device: Target device (default: CPU)
        dtype: Data type override (optional)

    Returns:
        BufferView wrapping the numpy array memory
    """
    if not hasattr(arr, "ctypes"):
        raise TypeError("from_numpy expected an object with ctypes attribute")

    ptr = arr.ctypes.data
    shape = tuple(int(d) for d in arr.shape)
    strides = tuple(int(s) for s in arr.strides) if hasattr(arr, "strides") else None

    if dtype is None:
        dtype_str = str(arr.dtype)
        if "float32" in dtype_str:
            dtype = DataType.FLOAT32
        elif "float64" in dtype_str:
            dtype = DataType.FLOAT64
        elif "int32" in dtype_str:
            dtype = DataType.INT32
        elif "int64" in dtype_str:
            dtype = DataType.INT64
        elif "bool" in dtype_str:
            dtype = DataType.BOOL
        else:
            dtype = DataType.FLOAT32

    return BufferView(
        data_ptr=ptr,
        shape=shape,
        strides=strides,
        dtype=dtype,
        device=device,
        memory_type=MemoryType.NORMAL,
        owner=arr,
        writable=arr.flags.writeable if hasattr(arr, "flags") else True,
    )


class GPUBuffer:
    """
    High-level abstraction for GPU memory buffers.
    Wraps raw pointers/addresses with automatic cleanup and CPU fallback.
    """

    def __init__(
        self,
        ptr: int,
        size_bytes: int,
        device: Union[Device, str] = "metal",
        dealloc_fn: Optional[Any] = None,
    ):
        self.ptr = ptr
        self.size_bytes = size_bytes
        self.device = (
            Device(DeviceType.METAL)
            if isinstance(device, str) and "metal" in device
            else device
        )
        self._dealloc_fn = dealloc_fn
        self._freed = False

    def free(self):
        """Release GPU memory buffer."""
        if not self._freed and self.ptr != 0:
            if self._dealloc_fn is not None:
                try:
                    self._dealloc_fn(self.ptr)
                except Exception as e:
                    logger.warning(
                        f"Failed to deallocate GPU buffer at {hex(self.ptr)}: {e}"
                    )
            self._freed = True
            self.ptr = 0

    def __del__(self):
        self.free()

    def to_cpu(self) -> bytes:
        """Transfer buffer contents to CPU memory (fallback)."""
        if self.ptr == 0:
            return b""
        import ctypes

        return ctypes.string_at(self.ptr, self.size_bytes)
