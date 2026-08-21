# Corepy Roadmap

This document outlines the development milestones for Corepy.

## 🏆 Current Status: v0.3.2 (Current)
- **Feature Focus**: Backend Dispatch, PyO3 Type Unwrapping & Remediation
- **Key Deliverables**:
  - ✅ PyO3 `_RustCoreArray` type unwrapping and stacking
  - ✅ 2D scalar comparison broadcasting & boolean mask indexing (`arr2d[arr2d > 2]`)
  - ✅ Full Series & DataFrame statistical methods (`mean`, `std`, `var`, `sum`, `min`, `max`, `apply`, `map`, `filter`, `describe`)
  - ✅ Math domain error guards returning NaN with RuntimeWarning
  - ✅ High-level GPUBuffer memory abstraction
  - ✅ Hardware-aware Rust Backend (`optimizer.rs`, `device.rs`)
  - ✅ Intel/AMD specific optimized dispatch logic
  - ✅ Professional workload analysis (`cp.analyse_workload`)
  - ✅ Zero-config Profiling System (`corepy.profiler`)
  - ✅ Array Operations (50+ UFuncs)
  - ✅ Rust FFI Integration with PyO3


---

## 📅 Milestones

### v0.3.0: The "Array Completeness" Release (Q2 2026)
**Goal**: Make Corepy usable for basic ML algorithms.
- **Features**:
  - [ ] Full Broadcast support for binary operations
  - [ ] Advanced Reduction ops (`max`, `min`, `argmax`)
  - [ ] Slicing and Indexing support (`array[0:5]`)
  - [ ] Save/Load arrays to disk
- **Tech Stack**: 
  - Migrate C++ fallback kernels to Rust/SIMD (Active).

### v0.4.0: The "GPU Prototype" Release (Q3 2026)
**Goal**: First working GPU acceleration on consumer hardware.
- **Features**:
  - [ ] CUDA Backend integration
  - [ ] `array.to("cuda")` implementation
  - [ ] Basic memory management for GPU
- **Tech Stack**:
  - CUDA kernels via C++ layer
  - Rust managing GPU streams

### v1.0.0: The "Production Ready" Release (2027)
**Goal**: Stable API and performance parity with NumPy for supported ops.
- **Features**:
  - [ ] Stable public API (SemVer guarantees)
  - [ ] Comprehensive documentation
  - [ ] Wheels for separate GPU/CPU builds
  - [ ] <1% overhead vs raw C++
- **Tech Stack**:
  - Full Work-stealing Scheduler (Rust)

### v2.0.0: Distributed & Advanced (Future)
- **Features**:
  - [ ] Distributed Arrays (Multi-node)
  - [ ] Lazy Evaluation Graph
  - [ ] Auto-differentiation (Autograd)
  - [ ] **Investigation**: Optional Polars integration for zero-copy data loading.

---

## 🧪 Experiments
We are currently exploring:
- **JIT Compilation**: Compiling operation graphs to fused kernels.
- **WebAssembly**: Running Corepy models in the browser.
