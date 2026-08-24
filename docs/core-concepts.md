# Core Concepts

This document explains the mental model you should have when working with Corepy.

## 1. The Array

The atom of Corepy is the `Array`.

### Mental Model
Think of a Array not just as a container for numbers, but as a **handle** to memory that might live anywhere (Main RAM, GPU, Accelerator).

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
array = cp.Array(np_array)
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

# Default: Checks for AVX2/NEON, detects Cache sizes, falls back to generic
set_backend_policy(BackendPolicy.DEFAULT)

# Force Reference Backend (educational)
set_backend_policy(BackendPolicy.REFERENCE)
```

## 3. Advanced Features (DataFrames & Random Ops)

Corepy is not just numbers; it includes higher-level structures and probability functions deeply integrated with Rust.

### High-Performance Random Numbers
Corepy includes multi-threaded PRNG implementations such as **PCG64** and **Xoshiro256++** for uniform and normal distribution generation directly into Arrays without GIL overhead.

```python
import corepy as cp

# Generates 10 million random floats instantly across all cores
uniform_data = cp.rand(10_000_000, algo="xoshiro")
```

### Relational Engine (DataFrame)
Corepy includes a `pandas`-like columnar DataFrame engine optimized for data processing prior to array execution.

```python
import corepy as cp

df = cp.DataFrame()
df.add_int_column("id", [1, 2, 3])
df.add_float_column("score", [99.5, 45.0, 88.0])

good_scores = df.filter("score", ">", 50.0)
```

## 4. Profiling

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

## 5. Correctness First

Corepy prefers to crash (safely) rather than give a wrong answer.

- **Type Checking**: No silent casting of `Float64` down to `Float32` if precision loss is significant.
- **Shape Broadcasting**: Strict rules on how shapes align.
- **FFI Safety**: The Rust barrier ensures that pointers passed to C++ are valid, aligned, and bounded.
