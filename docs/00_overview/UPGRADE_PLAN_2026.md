# CorePy 2026 Upgrade Plan: From Parity to Supremacy

**Current Status**: CorePy has achieved **CPU Parity** with NumPy for large matrices and **1.5x Speedup** for mid-sized matrices via optimized OpenBLAS dispatch and Zero-Copy architecture.

**Vision**: Move beyond simple matrix multiplication into full tensor capabilities (Broadcasting, Autograd) and Hardware Acceleration (CUDA).

## Phase 1: Completing the Foundation (Q1 2026)

### 1.1 Universal Broadcasting (Critical)
**Problem**: Currently, shapes must match exactly (e.g., `(10, 10) + (10, 10)` works, but `(10, 10) + (10,)` fails).
**Goal**: Implement NumPy-style broadcasting rules.
- **Tasks**:
    - [ ] Implement `broadcast_to(shape)` in `array.py`.
    - [ ] Update `elementwise.rs` to handle strides != 1 (virtual expansion).
    - [ ] Support implicit expansion in binary ops (`add`, `sub`, `mul`).

### 1.2 Element-wise Parallelism
**Problem**: `matmul` is parallel (via OpenBLAS), but `add/sub` are sequential AVX2.
**Goal**: Use Rayon to parallelize element-wise ops for N > 100k.
- **Tasks**:
    - [ ] Activate `rayon` thread pool in `elementwise.rs`.
    - [ ] Implement chunked parallel iterator for `tensor_add_f32`, etc.
    - [ ] Benchmark vs NumPy for 10M element arrays (Target: 2-4x speedup).

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
    - [ ] Add `requires_grad=bool` to Tensor.
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
