# CorePy 2026 Upgrade Plan: From Parity to Supremacy

**Current Status**: CorePy v0.3.2 has achieved **CPU Parity** with NumPy for large matrices and **1.5x Speedup** for mid-sized matrices via optimized OpenBLAS dispatch, Rayon parallelism, PyO3 type unwrapping, and zero-copy architecture.

---

## Phase 1: Completed Foundation (v0.3.2)

### 1.1 Universal Broadcasting & Comparison (Completed)
- [x] Implicit scalar expansion and 2D matrix broadcasting.
- [x] Multi-dimensional scalar comparison operators in Rust `elementwise_cmp`.
- [x] Boolean matrix mask indexing (`arr[arr > 2]`).

### 1.2 Element-wise Parallelism (Completed)
- [x] Activated `rayon` parallel iterator for elementwise UFunc operations.
- [x] Multi-threaded Rayon PRNG generators (`rand`, `randn`).

## Phase 2: Advanced Capabilities (Q2 2026)

### 2.1 Advanced Slicing & Vista Views
**Problem**: Slicing logic is basic.
**Goal**: Full support for strided views without copying.
- **Tasks**:
    - [ ] Implement `__getitem__` with step support (`arr[::2]`).
    - [ ] Ensure all kernels respect `strides` (remove `ascontiguousarray` fallback).

### 2.2 Autograd Engine (differentiation)
**Problem**: CorePy is valid for inference but not training.
**Goal**: Backpropagation support.
- **Tasks**:
    - [ ] Add `requires_grad=bool` to Array.
    - [ ] Implement DAG (Directed Acyclic Graph) to track operation history.
    - [ ] Implement `backward()` for MatMul, Add, and ReLU.

## Phase 3: Hardware Acceleration (Q3 2026)

### 3.1 CUDA Backend
**Problem**: CPU parity is the limit. to go 10x faster, we need GPU.
**Goal**: Seamless CUDA integration.
- **Tasks**:
    - [ ] Integrate `cudarc` or raw CUDA FFI.
    - [ ] Add `Backend.CUDA` policy.
    - [ ] Implement Zero-Copy CPU->GPU transfer (Pinned Memory).
    - [ ] Map `matmul` to `cuBLAS`.

## Upgrade Roadmap Summary

| Component | Current State | Target State | Impact |
| :--- | :--- | :--- | :--- |
| **MatMul** | Parity/Faster (CPU) | GPU Accelerated | 10x-50x Faster |
| **Broadcasting** | None | Full NumPy Algo | Usability Parity |
| **Element-wise** | Sequential AVX2 | Parallel AVX2 | 4x Faster (on 8-core) |
| **Autograd** | None | Reverse Mode | ML Training Capable |

## Immediate "Next Steps" (Task List)
1.  **Implement Broadcasting**: This is the single biggest blocker for replacing NumPy in real code.
2.  **Enable Rayon for Add/Sub**: Low hanging fruit for an easy 4x-8x speedup on large arrays.
