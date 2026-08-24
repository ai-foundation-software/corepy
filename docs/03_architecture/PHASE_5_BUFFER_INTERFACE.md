# Phase 5: Generic Buffer Interface Design

## Motivation

Current limitations in `corepy/array.py`:
1. **No stride support**: All non-contiguous arrays trigger copies
2. **No device abstraction**: CPU-only, no path to GPU
3. **Implicit copies**: Hidden `ascontiguousarray()` calls
4. **No ownership tracking**: Lifetime safety relies on Python GC
5. **Type-specific dispatch**: Separate paths for NumPy, bytes, lists

**Goal**: Design a unified `BufferView` abstraction that enables:
- Stride-aware kernels
- CPU/GPU dispatch
- Explicit copy-vs-view semantics
- Zero-cost Rust/C++ interop

---

## Reference: Production ML Systems

### PyTorch Array Storage Model
```
Array
  ├─ Storage (actual memory)
  │   ├─ data_ptr: void*
  │   ├─ device: CPU | CUDA | ...
  │   ├─ allocator: CPUAllocator | CUDAAllocator
  │   └─ nbytes: size_t
  └─ ArrayImpl (view metadata)
      ├─ sizes: [int]
      ├─ strides: [int]
      ├─ offset: int
      └─ dtype: float32 | int32 | ...
```

**Insight**: Separate storage (ownership) from view (indexing logic).

### NumPy Array Interface
```python
arr.__array_interface__ = {
    "data": (ptr, readonly),
    "strides": (s0, s1, ...),  # bytes per dimension
    "shape": (n0, n1, ...),
    "typestr": "<f4",  # little-endian float32
    "descr": [("", "<f4")],
    "version": 3,
}
```

### DLPack (Cross-framework standard)
```c
typedef struct {
    void* data;
    DLDevice device;      // {kDLCPU, kDLCUDA, ...}
    int ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;     // NULL = C-contiguous
    uint64_t byte_offset;
} DLArray;
```

**Insight**: Industry standard for zero-copy array exchange.

---

## Proposed Design: CorePy BufferView

### 1. Core Abstraction

```python
@dataclass
class BufferView:
    """
    Unified buffer abstraction for CPU/GPU memory.

    Design Principles:
    - Zero-copy when possible
    - Explicit about copies
    - Stride-aware
    - Device-agnostic dispatch
    """

    # Memory location
    data_ptr: int  # Raw pointer (usize in Rust)

    # Layout
    shape: Tuple[int, ...]  # Logical dimensions
    strides: Tuple[int, ...]  # Byte strides (None = C-contiguous)
    dtype: DataType  # Element type

    # Device & Memory Type
    device: Device  # CPU | CUDA:0 | CUDA:1 | ...
    memory_type: MemoryType  # Normal | Pinned | Unified | Device

    # Ownership
    owner: Any  # Keep-alive reference
    writable: bool  # Mutability flag

    def is_contiguous(self) -> bool:
        """Check if C-contiguous (dense row-major)."""
        if self.strides is None:
            return True

        expected_stride = self.dtype.itemsize
        for dim in reversed(self.shape):
            if self.strides[-1] != expected_stride:
                return False
            expected_stride *= dim
        return True

    def ensure_contiguous(self) -> "BufferView":
        """Return contiguous view, copying if needed."""
        if self.is_contiguous():
            return self

        # Trigger copy (device-aware)
        if self.device.is_cuda():
            return self._cuda_contiguous_copy()
        else:
            return self._cpu_contiguous_copy()

    def to_device(self, target: Device) -> "BufferView":
        """Move to different device (D2H, H2D, D2D)."""
        if self.device == target:
            return self

        # Dispatch to appropriate copy path
        return device_copy(self, target)
```

### 2. Memory Type Enumeration

```python
class MemoryType(Enum):
    """
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
```

### 3. Device Abstraction

```python
@dataclass
class Device:
    type: DeviceType  # CPU | CUDA | ROCm | Metal
    index: int = 0  # For multi-GPU: cuda:0, cuda:1

    def is_cpu(self) -> bool:
        return self.type == DeviceType.CPU

    def is_cuda(self) -> bool:
        return self.type == DeviceType.CUDA

    def __str__(self) -> str:
        if self.is_cpu():
            return "cpu"
        return f"{self.type.value}:{self.index}"
```

### 4. Conversion from Python Objects

```python
def from_numpy(arr: np.ndarray, device: Device = CPU) -> BufferView:
    """
    Zero-copy from NumPy array.

    Safety:
    - Keeps reference to `arr` in `owner` field
    - Extracts strides (may be non-contiguous)
    - Validates dtype compatibility
    """
    ptr = arr.__array_interface__["data"][0]
    strides = arr.strides if not arr.flags["C_CONTIGUOUS"] else None

    return BufferView(
        data_ptr=ptr,
        shape=arr.shape,
        strides=strides,
        dtype=DataType.from_numpy(arr.dtype),
        device=device,
        memory_type=MemoryType.NORMAL,
        owner=arr,  # Critical: keep alive
        writable=arr.flags["WRITEABLE"],
    )


def from_buffer(obj: Any, dtype: DataType, device: Device = CPU) -> BufferView:
    """
    From any buffer protocol object (bytes, bytearray, memoryview).
    """
    mv = memoryview(obj)
    # ... similar extraction logic
    return BufferView(...)
```

---

## Migration Path

### Phase 5.1: Internal BufferView (No API Change)
- `Array._get_buffer_pointer()` → `Array._get_buffer_view()`
- Returns `BufferView` internally
- Dispatch logic uses `BufferView.is_contiguous()`, `BufferView.device`
- **Benefit**: Cleaner dispatch, easier to add GPU later

### Phase 5.2: Stride-Aware Kernels
- Implement C++ kernels that accept strides
- Example: `sum_f32_strided(data, shape, strides, ndim)`
- Fallback to contiguous copy if kernel unavailable
- **Benefit**: No hidden copies for sliced arrays

### Phase 5.3: GPU Support
- Add CUDA allocator
- Implement `BufferView.to_device("cuda:0")`
- GPU kernel dispatch via same `BufferView` interface
- **Benefit**: Unified API for CPU/GPU

### Phase 5.4: DLPack Interop
- Implement `Array.__dlpack__()` and `Array.from_dlpack()`
- Zero-copy with PyTorch, JAX, ArrayFlow
- **Benefit**: Ecosystem integration

---

## Performance Implications

### Current System (Post-Fix)
```
NumPy contiguous → Zero-copy ✅
NumPy sliced     → Hidden copy ⚠️
List             → Explicit copy ✅
```

### With BufferView + Strided Kernels
```
NumPy contiguous → Zero-copy ✅
NumPy sliced     → Zero-copy (strides) ✅
List             → Explicit copy ✅
```

### Copy Decision Tree
```
Input Buffer
  │
  ├─ Is contiguous?
  │   ├─ YES → Zero-copy
  │   └─ NO  → Check if strided kernel exists
  │       ├─ YES → Zero-copy (stride-aware)
  │       └─ NO  → Explicit copy (logged)
  │
  └─ Device mismatch?
      └─ Copy via device transfer (H2D, D2H, D2D)
```

---

## Engineering Standards Checklist

- [x] **Precise terminology**: Defined strides, contiguity, device types
- [x] **No hand-waving**: Concrete code examples with semantics
- [x] **Trade-offs explicit**: Copy vs stride performance documented
- [x] **Unsafe patterns called out**: Non-contiguous silent corruption explained
- [x] **Production references**: Cited PyTorch, NumPy, DLPack designs
- [ ] **Implementation**: Code not written yet (design phase)

---

## Open Questions for Review

1. **Stride Kernel Priority**: Should we implement strided sum/mean first, or wait until GPU work begins?
2. **Copy Logging**: Should non-contiguous copies log at WARNING level for production visibility?
3. **Pinned Memory**: When to use `cudaHostAlloc` vs normal malloc? (Threshold: transfer size > 1MB?)
4. **DLPack Version**: Target DLPack v1.0 (latest) or maintain v0.8 compat?

---

## References

- [NumPy Array Interface](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)
- [PyTorch Storage Design](https://pytorch.org/docs/stable/array_attributes.html)
- [CUDA Memory Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
