# Metal GPU Acceleration (macOS)

Corepy provides native support for Apple Silicon GPU acceleration using the **Metal** API. This allows for significantly faster array operations on M1, M2, and M3 chips compared to CPU execution.

## Requirements

*   **Hardware**: Apple Silicon Mac (M1 or newer). Intel Macs are supported but will fall back to OpenBLAS/AVX2.
*   **OS**: macOS 12.0 (Monterey) or newer.
*   **Dependencies**: The `Metal` and `Foundation` frameworks (included with macOS).

## Enabling Metal

To use the Metal backend, simply specify `device="metal"` when creating an array:

```python
import corepy as cp

# Allocate directly on Metal GPU
# Note: You can also use cp.array([...], device="metal")
t = cp.array([1.0, 2.0, 3.0, 4.0], device="metal")

# Perform operations (executed largely on GPU)
result = t.sum() 
print(result) # 10.0
```

### Automatic Fallback
If you request `device="metal"` on a non-Mac or an Intel Mac without Metal support, Corepy will log a warning and automatically fall back to the **CPU** backend to prevent crashes.

## Supported Operations

As of v0.3.0, the following operations are optimized for Metal:

*   **Reduction**: `sum`, `mean`, `max`, `min`
*   **Matrix Multiplication**: `matmul` (@) for 2D arrays
*   **Element-wise**: `add`, `sub`, `mul`, `div` (Basic kernels)

## Performance considerations

1.  **Transfer Overhead**: Moving data between CPU (Python/NumPy) and GPU (Metal) has a cost. For small arrays (< 1M elements), the CPU is often faster due to lower latency.
2.  **Shared Memory**: Corepy uses `MTLResourceStorageModeShared` on Apple Silicon, which allows the CPU and GPU to share system memory without explicit copies in some cases, but cache coherency must be managed.
3.  **Compilation**: Metal kernels are compiled at runtime (or loaded from `.metallib` if pre-compiled). The first operation might be slightly slower due to pipeline state creation.

## Troubleshooting

### "Metal framework not found"
If you see this warning during build, ensure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

### Verification
You can verify Metal is active by checking the array's device:

```python
t = cp.array([1.0], device="metal")
# Internal buffer check (advanced)
print(t._get_buffer_view().device.is_metal()) # True
```
