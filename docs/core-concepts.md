# Core Concepts

This document explains the mental model you should have when working with Corepy.

## 1. The Tensor

The atom of Corepy is the `Tensor`.

### Mental Model
Think of a Tensor not just as a container for numbers, but as a **handle** to memory that might live anywhere (Main RAM, GPU, Accelerator).

- **Data**: The actual numbers.
- **Shape**: Dimensions (e.g., `(3, 4)`).
- **Dtype**: The type of data (Strictly `Float32`, `Int32`, etc.).
- **Backend/Device**: Where the data lives and who computes on it.

### Zero-Copy Interop
Corepy is designed to play nice with others. It supports zero-copy views from objects supporting the buffer protocol (like NumPy arrays or byte buffers).

```python
import numpy as np
import corepy as cp

# Zero-copy if data is contiguous and typed correctly
np_array = np.array([1, 2, 3], dtype=np.float32)
tensor = cp.Tensor(np_array) 
```

## 2. Backends & Policies

Corepy uses a hybrid Runtime.

- **Python**: Orchestration, API, high-level logic.
- **Rust**: Safety, Dispatching, Parallelism (Rayon), Memory Management.
- **C++/Kernels**: The raw compute (AVX2, OpenBLAS, CUDA).

### Backend Policy
You can control how Corepy dispatches operations.

```python
from corepy.backend import set_backend_policy, BackendPolicy

# Default: Checks for AVX2, then falls back to generic
set_backend_policy(BackendPolicy.DEFAULT)

# Force OpenBLAS (good for huge matrices)
set_backend_policy(BackendPolicy.OPENBLAS)
```

## 3. Profiling

Performance is opacity's enemy. Corepy treats observability as a first-class feature.

### Recommendations
The profiler doesn't just show numbers; it understands your hardware.

```python
import corepy as cp

cp.enable_profiling()
# ... code ...
print(cp.profile_report())
```

**Common Bottlenecks detected:**
- **OS Jitter**: Sudden spikes in operation time.
- **Memory Bandwidth**: Operations that are simple math but huge data (linear scan).
- **Backend Thrashing**: Moving data between devices (CPU <-> GPU) too often.

## 4. Correctness First

Corepy prefers to crash (safely) rather than give a wrong answer.

- **Type Checking**: No silent casting of `Float64` down to `Float32` if precision loss is significant.
- **Shape Broadcasting**: Strict rules on how shapes align.
- **FFI Safety**: The Rust barrier ensures that pointers passed to C++ are valid, aligned, and bounded.
