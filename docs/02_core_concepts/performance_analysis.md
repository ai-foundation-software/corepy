# Analysis Report: Backend Performance Diagnosis & Selection Strategy

**Date**: 2026-01-27  
**Context**: Corepy vs. NumPy (OpenBLAS) Benchmarking  
**Status**: DRAFT

---

## 1. Reference Benchmarks (Ground Truth)

| Size | Corepy (ms) | NumPy (ms) | Speedup | Dominant Regime |
| :--- | :--- | :--- | :--- | :--- |
| **100×100** | 0.984 | 1.745 | **1.77×** | Overhead-Bound |
| **512×512** | 5.520 | 9.655 | **1.75×** | Cache-Bound |
| **1024×1024** | 28.035 | 18.737 | **0.67×** | Bandwidth/Scaling-Bound |

---

## 2. Performance Diagnosis

### Why Corepy Wins (Small & Mid Sizes)
Corepy's dominance in the 100-500 range is attributed to its **lean orchestration path**:
*   **Reduced Dispatch Cost**: NumPy's `np.matmul` travels through a thick layer of CPython type-checking, ndarray safety logic, and multiple levels of internal dispatch before hitting CBLAS. Corepy's Rust "Brain" provides a more direct FFI path.
*   **Avoidance of Packing Overhead**: High-performance BLAS libraries (OpenBLAS/MKL) often "pack" matrices into temporary buffers to ensure cache-line alignment and optimal SIMD loading. For 100x100 matrices, the **$O(N^2)$ packing time** outweighs the $O(N^3)$ compute gain. Corepy's "Muscle" (AVX2-optimized C++) operates directly on the pointers, skipping this step.
*   **Predictable AVX2 Execution**: At mid-scales (512x512), the working set often fits in L2/L3 cache. Corepy's custom unrolled kernels maximize instruction-level parallelism (ILP) without the overhead of sophisticated blocking headers.

### Why NumPy/BLAS Wins (Large Sizes)
At 1024x1024 and beyond, the math complexity ($N^3$) becomes the dominant factor:
*   **Superior Multi-thread Scheduling**: OpenBLAS uses highly tuned internal threading (pthreads/OpenMP) with **NUMA-aware pinning**. For 1M+ elements, the overhead of the thread pool is amortized.
*   **Cache Tiling & Blocking**: While Corepy uses naive unrolling, OpenBLAS uses multi-level tiling (L1/L2/L3) to maximize memory bandwidth utilization. As matrices stop fitting in cache, these micro-architectural tunings prevent stall cycles.

### Bottleneck Classification
*   **100-256**: **Overhead-Bound** (Dispatch and FFI latency dominate).
*   **512-768**: **Cache-Bound** (Corepy's unrolling excels when data fits in L3).
*   **1024+**: **Thread-Scaling / Bandwidth-Bound** (Tiling and scheduling become critical).

---

## 3. Interpretation of the "Closing Gap"

The transition at 1024×1024 represents the **convergence of compute and bandwidth**. 
*   **Expected Behavior**: This is not a failure of Corepy, but the meeting point where specialized BLAS libraries' years of micro-architecture tuning start to yield fruit.
*   **Bandwidth Ceiling**: As throughput increases, both seekers reach the DRAM bandwidth limit. BLAS wins here by being more efficient with cache-line reloads via sophisticated **Packing Kernels**.

---

## 4. Backend Selection Strategy (Proposed)

We will implement an **Architecture-Aware Dispatcher** with the following policy:

| Matrix Dimension ($N$) | Recommended Backend | Rationale |
| :--- | :--- | :--- |
| **$N \le 256$** | `COREPY_AVX2` | Lowest dispatch latency; no packing overhead. |
| **$256 < N \le 768$** | `AUTO_BENCH` | Hardware-dependent flip point (cached at startup). |
| **$N > 768$** | `OPENBLAS` | Superior tiling and multi-threading for large data. |

### API Design
```python
# User-level control
corepy.set_backend_policy("auto")  # default
corepy.set_backend_policy("corepy") # Force native AVX2
corepy.set_backend_policy("blas")   # Force OpenBLAS

# Introspection
info = corepy.explain_last_dispatch()
# Returns: {"op": "matmul", "size": (1024, 1024), "backend": "OpenBLAS", "reason": "Size threshold exceeded"}
```

---

## 5. CODE-LEVEL RECOMMENDATIONS

### Immediate (P1)
*   **Zero-Copy Handover**: Ensure that when switching to OpenBLAS, we use the `cblas_sgemm` interface directly on the `ndarray` buffer pointer to avoid *any* data movement.
*   **Threading Control**: In the "Large" regime, we must call `openblas_set_num_threads()` to match Corepy's overall session settings, preventing oversubscription.

### Long-Term (P2)
*   **Packing Kernels**: If we wish to compete at >1024, Corepy needs an automated "packing" step for A and B. This is a significant task and should be deferred until P3.
*   **L3-Aware Tiling**: Implement a macro-kernel that tiles by the size of the L3 cache.

---

## 6. Memory & Parallelism Analysis (Arena Integration)

**Date**: 2026-02-05
**Context**: Phase 6.1/6.2 Implementation

### 6.1 Arena Allocator Performance
We implemented a thread-local bump allocator (Arena) to replace `malloc` for temporary buffers.

**Results (1M elements)**:
- **Bandwidth**: ~13-18 GB/s (Sum/Mean)
- **Overhead**: <1% (No measurable difference vs raw pointer arithmetic)
- **Benefit**: eliminated `malloc` contention in multi-threaded contexts.

### 6.2 Parallel Dispatch Scaling
For reductions (`sum`, `mean`), we switch to Rayon parallel iterators at a threshold of **100k elements**.

**Scaling Profile**:
- **<100k**: Sequential AVX2 is faster (due to thread startup overhead).
- **100k-500k**: Transition zone (Parallel is comparable or slightly slower).
- **>1M**: Parallel wins (Scales with core count).

**Conclusion**: The current threshold of 100k is conservative. Future tuning may raise this to 500k for optimal throughput on high-core-count machines.

---

## 7. Next Steps (Ranked)

1.  **Implement the Dispatcher**: Add the size-based logic to `rust/corepy-runtime/src/ops/matmul.rs`.
2.  **Expose `set_backend_policy`**: Build the Python/Rust bridge for policy control.
3.  **Startup Auto-bench**: Add a one-time micro-benchmark at `import corepy` to find the exact flip-point for the current CPU.
4.  **Refactor C++ Micro-kernels**: Move AVX2 logic into its own internal module to simplify the BLAS vs. Native toggle.
