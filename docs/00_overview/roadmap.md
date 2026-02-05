# Corepy Roadmap

This document outlines the development milestones for Corepy.

## 🏆 Current Status: v0.2.0 (Released)
- **Feature Focus**: Performance Profiling & Foundation
- **Key Deliverables**:
  - ✅ Hybrid Python/Rust/C++ Architecture
  - ✅ Zero-config Profiling System (`corepy.profiler`)
  - ✅ Basic Tensor Operations (add, sub, mul, matmul)
  - ✅ Rust FFI Integration (`corepy-runtime`) - **Active**
  - ✅ Python API (`corepy.tensor`, explicit Backend selection)

---

## 📅 Milestones

### v0.3.0: The "Tensor Completeness" Release (Q2 2026)
**Goal**: Make Corepy usable for basic ML algorithms.
- **Features**:
  - [ ] Full Broadcast support for binary operations
  - [ ] Advanced Reduction ops (`max`, `min`, `argmax`)
  - [ ] Slicing and Indexing support (`tensor[0:5]`)
  - [ ] Save/Load tensors to disk
- **Tech Stack**: 
  - Migrate C++ fallback kernels to Rust/SIMD (Active).

### v0.4.0: The "GPU Prototype" Release (Q3 2026)
**Goal**: First working GPU acceleration on consumer hardware.
- **Features**:
  - [ ] CUDA Backend integration
  - [ ] `tensor.to("cuda")` implementation
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
  - [ ] Distributed Tensors (Multi-node)
  - [ ] Lazy Evaluation Graph
  - [ ] Auto-differentiation (Autograd)

---

## 🧪 Experiments
We are currently exploring:
- **JIT Compilation**: Compiling operation graphs to fused kernels.
- **WebAssembly**: Running Corepy models in the browser.
