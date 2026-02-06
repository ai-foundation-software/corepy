# Architecture

Corepy is built as a layered system to maximize performance while retaining Pythonic usability.

## The Stack

```mermaid
graph TD
    A[Python API] --> B[Rust Runtime]
    B --> C[Memory Arena]
    B --> D[Dispatcher]
    D --> E[C++ Kernels (AVX2)]
    D --> F[OpenBLAS / Accelerate]
    D --> G[Metal / CUDA (Future)]
```

### 1. Python Layer (`corepy/`)
- **Duty**: Exposes the user-facing API (`Tensor`, `profiler`).
- **State**: Holds the `PyCapsule` or bindings to the underlying Rust objects.
- **Overhead**: Minimal. Most operations immediately drop into Rust.

### 2. Rust Runtime (`rust/core/`)
- **Duty**: The "Brain" of the operation.
- **Safety**: Uses Rust's borrow checker to ensure threads don't race on memory.
- **Parallelism**: Uses `rayon` to parallelize operations over large tensors (automatically trigger > 1M elements).
- **FFI Boundary**: Marshals pointers and shapes before handing them to raw C functions.

### 3. Compute Kernels (`csrc/`)
- **Duty**: The "Muscle".
- **Implementation**: Pure C/C++ or Assembly.
- **Optimization**: Hand-tuned SIMD intrinsics (AVX2, AVX-512, NEON).
- **Zero-Allocation**: Kernels do not allocate. They only operate on pointers provided by the Rust runtime.

## Dispatch Mechanism

When you call `tensor.sum()`, the following happens:

1.  **Python**: `Tensor.sum()` calls `corepy._corepy_rust.tensor_sum_f32`.
2.  **Rust**:
    -   Receives the pointer to the data.
    -   Checks the size.
    -   If size < Threshold: Runs purely on the calling thread.
    -   If size > Threshold: Splits the data into chunks and uses `rayon` to reduce them in parallel.
3.  **Kernel**: Each thread executes a tight C++ loop with SIMD instructions.
4.  **Result**: Aggregated back in Rust and returned to Python as a float or scalar Tensor.

## Memory Management

- **Initialization**: Default allocator is the system allocator (aligned to 64 bytes for AVX-512 compatibility).
- **Lifetime**: Managed by Python's reference counting (ARC on the Rust side if shared).
- **View Semantics**: Creating a Tensor from a list *copies* data. Creating from a NumPy array *views* data (zero-copy) if possible.
