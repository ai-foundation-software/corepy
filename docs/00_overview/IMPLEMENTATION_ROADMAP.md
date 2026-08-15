# Implementation Roadmap: Performance & Operations

**Status**: Phase 6 Complete, Phase 7 Planned  
**Priority**: P0 (Critical) → P3 (Nice-to-have)

---

## ✅ Completed (Phase 1-4.5)

### Infrastructure
- [x] Rust-C++ FFI boundary with zero-copy semantics
- [x] PyO3 Python bindings
- [x] Thread-local arena allocators (implemented but unused)
- [x] Rayon work-stealing scheduler (implemented but unused)
- [x] Universal buffer protocol support (NumPy, bytes, memoryview)
- [x] Memory safety hardening (C_CONTIGUOUS checks)

### Operations
- [x] Reductions: `all()`, `any()`, `sum()`, `mean()`
- [x] Element-wise: `add`, `sub`, `mul`, `div` (AVX2 SIMD)
- [x] Basic matmul: 1D dot product (pure Rust implementation)

### Testing
- [x] Comprehensive test suite for all endpoints (`make ci`)
- [x] Memory safety verification via robust CI
- [x] SIMD performance native validation

### 🚀 Phase 5 & 6: DataFrames, Lazy Eval & Performance (Completed)
- [x] DataFrame Engine (CSV, GroupBy, Pivot, Merge)
- [x] Lazy Evaluation Engine / Operation Fusion
- [x] Rayon Multi-threading actively scheduled
- [x] Global LRU Buffer Pool (No-OS allocations)
- [x] Rayon-backed PRNG (Xoshiro/PCG64)

---

## 🚀 Phase 7: Optimization & GPU Selection (Next Sprint)

**Objective**: Finish SIMD coverage and start preparing for the GPU backend.

### P0: Arena Integration for Reductions
**Why**: Eliminate heap allocations in hot paths  
**Impact**: ~10-20% speedup for small-medium arrays

**Tasks**:
- [ ] **6.1.1**: Add arena parameter to C++ reduction kernels
  - Signature: `float sum_f32_cpu(const float* data, size_t count, ArenaAllocator* arena)`
  - Use case: Intermediate buffers for Kahan summation
  
- [ ] **6.1.2**: Wire Rust `with_arena()` wrapper in dispatch
  ```rust
  pub unsafe fn sum_f32_cpu_dispatch(data: *const f32, count: usize) -> f32 {
      use crate::scheduler::arena::with_arena;
      with_arena(|arena| {
          // Pass arena to C++ if needed
          sum_f32_cpu(data, count)
      })
  }
  ```

- [ ] **6.1.3**: Benchmark arena vs heap
  - Measure: allocation count, time per operation
  - Target: <1% overhead for arena path

**Estimated Effort**: 4-6 hours

---

### P0: Parallel Dispatch for Large Arrays
**Why**: Utilize multi-core CPUs for large reductions  
**Impact**: 2-4x speedup for arrays >100K elements

**Tasks**:
- [ ] **6.2.1**: Add size threshold for parallelization
  ```rust
  const PARALLEL_THRESHOLD: usize = 100_000;
  
  pub unsafe fn sum_f32_cpu_dispatch(data: *const f32, count: usize) -> f32 {
      if count < PARALLEL_THRESHOLD {
          sum_f32_cpu(data, count)  // Sequential
      } else {
          parallel_sum_f32_cpu(data, count)  // Rayon
      }
  }
  ```

- [ ] **6.2.2**: Implement parallel reduction with Rayon
  ```rust
  use rayon::prelude::*;
  
  fn parallel_sum_f32_cpu(data: *const f32, count: usize) -> f32 {
      let slice = unsafe { std::slice::from_raw_parts(data, count) };
      let chunk_size = count / num_cpus::get();
      
      slice.par_chunks(chunk_size)
           .map(|chunk| chunk.iter().sum::<f32>())
           .sum()
  }
  ```

- [ ] **6.2.3**: Benchmark sequential vs parallel
  - Test sizes: 10K, 100K, 1M, 10M elements
  - Measure: throughput (GB/s), speedup factor
  - Tune: `PARALLEL_THRESHOLD` based on results

**Estimated Effort**: 6-8 hours

---

### P1: SIMD-Optimized Dot Product (matmul)
**Why**: Current implementation is scalar, slow  
**Impact**: 8-16x speedup with AVX2

**Tasks**:
- [ ] **6.3.1**: Implement C++ dot product kernel
  ```cpp
  extern "C" float dot_product_f32_cpu(const float* a, const float* b, size_t count) {
      #ifdef __AVX2__
      __m256 sum_vec = _mm256_setzero_ps();
      size_t avx_count = count / 8;
      
      for (size_t i = 0; i < avx_count; ++i) {
          __m256 va = _mm256_loadu_ps(a + i * 8);
          __m256 vb = _mm256_loadu_ps(b + i * 8);
          __m256 prod = _mm256_mul_ps(va, vb);
          sum_vec = _mm256_add_ps(sum_vec, prod);
      }
      
      // Horizontal sum + scalar remainder
      // ...
      #else
      // Scalar fallback
      #endif
  }
  ```

- [ ] **6.3.2**: Update Rust FFI to call C++ kernel
  ```rust
  extern "C" {
      pub fn dot_product_f32_cpu(a: *const f32, b: *const f32, count: usize) -> f32;
  }
  
  pub unsafe fn dot_product_f32_cpu_dispatch(a: *const f32, b: *const f32, count: usize) -> f32 {
      dot_product_f32_cpu(a, b, count)
  }
  ```

- [ ] **6.3.3**: Add to CMakeLists.txt
  ```cmake
  add_library(corepy_kernels STATIC
      src/cpu/reduce.cpp
      src/cpu/elementwise.cpp
      src/cpu/matmul.cpp  # NEW
  )
  ```

**Estimated Effort**: 4-6 hours

---

### P1: Full Matrix Multiplication (2D matmul)
**Why**: Dot product only handles 1D; need 2D for real ML  
**Impact**: Enables matrix ops, foundation for neural networks

**Tasks**:
- [ ] **6.4.1**: Design matmul API with shape validation
  ```python
  # Python API
  result = array_a.matmul(array_b)  # (m, k) @ (k, n) -> (m, n)
  ```

- [x] **6.4.2**: Implement naive C++ kernel (row-major)
  ```cpp
  void matmul_f32_cpu(
      const float* a, const float* b, float* c,
      size_t m, size_t k, size_t n
  ) {
      for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < n; ++j) {
              float sum = 0.0;
              for (size_t p = 0; p < k; ++p) {
                  sum += a[i * k + p] * b[p * n + j];
              }
              c[i * n + j] = sum;
          }
      }
  }
  ```

- [x] **6.4.3**: SIMD optimization (blocked algorithm)
  - Use AVX2 for inner loop
  - Cache-blocking for large matrices
  - Reference: BLIS design

- [x] **6.4.4**: Integrate with Python layer
  - Shape validation: `(m, k) @ (k, n)`
  - Zero-copy dispatch via `BufferView`

**Status**: Basic working dot-product and matmul completed via Rust `nano-gemm` crates and parallel dot-products.

---

## 🧪 Phase 8: Benchmarking Suite

**Objective**: Quantify performance vs NumPy/PyTorch

### P2: Baseline Benchmarks
**Tasks**:
- [ ] **7.1.1**: Create `benchmarks/bench_reductions.py`
  ```python
  import numpy as np
  import corepy as cp
  import torch
  import pytest
  
  @pytest.mark.parametrize("size", [1000, 10_000, 100_000, 1_000_000])
  def bench_sum(benchmark, size):
      data = np.random.rand(size).astype(np.float32)
      
      # Corepy
      cp_array = cp.Array(data)
      result = benchmark(cp_array.sum)
      
      # Compare with NumPy
      np_result = np.sum(data)
      assert abs(result._backing_data[0] - np_result) < 1e-5
  ```

- [ ] **7.1.2**: Benchmark element-wise ops
  - `add`, `sub`, `mul`, `div`
  - Compare: Corepy vs NumPy vs PyTorch (CPU)

- [ ] **7.1.3**: Benchmark matmul
  - Sizes: (100x100), (1000x1000), (5000x5000)
  - Compare: Corepy vs NumPy (OpenBLAS) vs PyTorch (MKL)

**Estimated Effort**: 8-10 hours

---

### P3: Advanced Benchmarks
**Tasks**:
- [ ] **7.2.1**: Memory bandwidth benchmarks
  - Measure: GB/s for sequential access
  - Target: >80% of theoretical peak

- [ ] **7.2.2**: Cache efficiency analysis
  - Use `perf stat` for cache miss rates
  - Optimize: blocked algorithms, prefetching

- [ ] **7.2.3**: Profiling integration
  - Enable profiler for benchmarks
  - Generate flamegraphs with `py-spy`

**Estimated Effort**: 12-16 hours

---

## 📊 Success Metrics

### Performance Targets
| Operation | Size | Target vs NumPy | Target vs PyTorch |
|-----------|------|----------------|-------------------|
| `sum()` | 1M | 0.8-1.2x | 0.9-1.1x |
| `add()` | 1M | 0.9-1.1x | 0.9-1.1x |
| `dot()` | 100K | 1.0-1.5x | 0.8-1.2x |
| `matmul()` (1Kx1K) | - | 0.5-0.8x* | 0.4-0.7x* |

*Note: NumPy/PyTorch use highly optimized BLAS (OpenBLAS, MKL). Initial implementation won't match, but should be within 2x.

### Quality Targets
- **Test Coverage**: >80% for core operations
- **Zero-Copy Rate**: >95% for contiguous inputs
- **Arena Usage**: 100% of reductions use arenas
- **Parallel Threshold**: Auto-tuned per operation

---

## 🗓️ Sprint Planning

### Sprint 1 (Week 1): Arena + Parallel
- Days 1-2: Arena integration (6.1)
- Days 3-5: Parallel dispatch (6.2)
- **Deliverable**: 2-4x speedup for large reductions

### Sprint 2 (Week 2): Matmul
- Days 1-3: SIMD dot product (6.3)
- Days 4-5: 2D matmul design (6.4.1-6.4.2)
- **Deliverable**: Working matmul, basic benchmarks

### Sprint 3 (Week 3): Optimization + Benchmarking
- Days 1-2: Matmul SIMD optimization (6.4.3)
- Days 3-5: Comprehensive benchmarks (7.1)
- **Deliverable**: Performance report vs NumPy/PyTorch

---

## 🔧 Implementation Notes

### Arena Best Practices
```rust
// ✅ CORRECT: Arena lifetime managed
use crate::scheduler::arena::with_arena;

pub fn compute() {
    with_arena(|arena| {
        let temp_buf = arena.alloc::<f32>(1024)?;
        // Use temp_buf
        // Auto-freed on scope exit
    });
}

// ❌ WRONG: Arena escapes scope
let leaked = with_arena(|arena| {
    arena.alloc::<f32>(1024)  // Dangling pointer!
});
```

### Rayon Best Practices
```rust
// ✅ CORRECT: GIL released for parallel work
use pyo3::prelude::*;
use crate::scheduler::rayon_pool::execute_parallel;

#[pyfunction]
fn parallel_sum(py: Python, data: &[f32]) -> f32 {
    execute_parallel(py, || {
        data.par_iter().sum()  // GIL released here
    })
}

// ❌ WRONG: GIL held during parallel work
fn bad_parallel(data: &[f32]) -> f32 {
    data.par_iter().sum()  // GIL still held, no true parallelism!
}
```

---

## 📚 References

- **SIMD**: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- **Rayon**: [Rayon Documentation](https://docs.rs/rayon/)
- **Benchmarking**: [criterion.rs](https://github.com/bheisler/criterion.rs)
- **BLAS**: [GotoBLAS2 Algorithm](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
