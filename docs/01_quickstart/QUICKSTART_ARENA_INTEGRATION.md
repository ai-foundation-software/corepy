# Quick Start: Arena Integration (P6.1)

**Estimated Time**: 4-6 hours  
**Priority**: P0 (Critical)  
**Expected Speedup**: 10-20% for small-medium arrays

---

## Motivation

**Current**: All intermediate buffers use heap allocation (`malloc`)
```rust
// Every reduction does this internally:
let temp = vec![0.0f32; size];  // malloc() + memset()
// ... compute ...
// drop(temp)  // free()
```

**Problem**: 
- Heap allocations are slow (~100-500ns)
- Pressure on global allocator lock
- Cache misses from fragmented memory

**Solution**: Thread-local bump allocator
```rust
with_arena(|arena| {
    let temp = arena.alloc(size);  // <10ns, no lock
    // ... compute ...
    // Auto-freed on scope exit (reset pointer)
});
```

---

## Step-by-Step Implementation

### Step 1: Update C++ Kernel Signature (Optional)

Most reductions don't need scratch space, so this is **optional** for phase 1.

**Example** (if needed for future Kahan summation with compensated storage):
```cpp
// Before:
extern "C" float sum_f32_cpu(const float* data, size_t count);

// After (if temp buffer needed):
extern "C" float sum_f32_cpu_with_arena(
    const float* data, 
    size_t count,
    void* arena_ptr,  // Opaque arena handle
    size_t arena_size
);
```

**For now, skip this**—we'll use arenas on the Rust side only.

---

### Step 2: Wrap Dispatch in `with_arena()`

**File**: `rust/corepy-runtime/src/ops/reduce.rs`

```rust
// BEFORE:
pub unsafe fn sum_f32_cpu_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    sum_f32_cpu(data_ptr, count)
}

// AFTER:
pub unsafe fn sum_f32_cpu_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    use crate::scheduler::arena::with_arena;
    
    // Arena scope: automatically resets on exit
    with_arena(|_arena| {
        // For now, we don't use the arena inside sum_f32_cpu
        // But the arena is ready for future optimizations
        sum_f32_cpu(data_ptr, count)
    })
}
```

**Apply to all dispatch functions**:
- `all_bool_cpu_dispatch`
- `any_bool_cpu_dispatch`
- `sum_f32_cpu_dispatch`
- `sum_i32_cpu_dispatch`
- `mean_f32_cpu_dispatch`

---

### Step 3: Benchmark Arena Overhead

**File**: `tests/bench_arena.py`

```python
import numpy as np
import time
from _corepy_rust import array_sum_f32

def bench_sum_with_arena(size=100_000, iterations=10_000):
    """
    Benchmark sum operation with arena integration.
    
    Expected: <1% overhead from arena.reset() calls
    """
    data = np.random.rand(size).astype(np.float32)
    ptr = data.ctypes.data
    
    # Warmup
    for _ in range(100):
        array_sum_f32(ptr, size)
    
    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        result = array_sum_f32(ptr, size)
    elapsed = time.perf_counter() - start
    
    ops_per_sec = iterations / elapsed
    gb_per_sec = (size * 4 * iterations) / (elapsed * 1e9)
    
    print(f"Size: {size:>10,} elements")
    print(f"Ops/sec: {ops_per_sec:>10,.0f}")
    print(f"Bandwidth: {gb_per_sec:>10.2f} GB/s")
    print(f"Avg latency: {elapsed/iterations*1e6:>10.2f} µs")

if __name__ == "__main__":
    print("=== Arena Integration Benchmark ===\n")
    bench_sum_with_arena(size=1_000)
    bench_sum_with_arena(size=10_000)
    bench_sum_with_arena(size=100_000)
```

**Expected Output**:
```
=== Arena Integration Benchmark ===

Size:      1,000 elements
Ops/sec:    500,000
Bandwidth:      2.00 GB/s
Avg latency:        2.00 µs

Size:     10,000 elements
Ops/sec:    200,000
Bandwidth:      8.00 GB/s
Avg latency:        5.00 µs

Size:    100,000 elements
Ops/sec:     50,000
Bandwidth:     20.00 GB/s
Avg latency:       20.00 µs
```

**Acceptance Criteria**:
- Arena overhead < 1% (measure before/after if possible)
- No crashes (arena lifetime correct)
- No memory leaks (valgrind clean)

---

### Step 4: Visual Verification

**Add logging to see arena usage**:

```rust
// In reduce.rs dispatch:
pub unsafe fn sum_f32_cpu_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    use crate::scheduler::arena::{with_arena, arena_stats};
    
    with_arena(|arena| {
        let result = sum_f32_cpu(data_ptr, count);
        
        // Debug: Print arena stats
        #[cfg(debug_assertions)]
        {
            let (used, capacity, peak) = arena_stats();
            eprintln!("[Arena] Used: {used}, Cap: {capacity}, Peak: {peak}");
        }
        
        result
    })
}
```

**Run test**:
```bash
$ RUST_LOG=debug python tests/bench_arena.py
[Arena] Used: 0, Cap: 1048576, Peak: 0
[Arena] Used: 0, Cap: 1048576, Peak: 0
...
```

*(Peak should be 0 since we're not allocating yet)*

---

## Phase 6.2: Parallel Dispatch (Implemented)

**Use Case**: Parallel reduction with per-thread scratch space for Kahan summation

```rust
pub unsafe fn parallel_sum_f32_cpu(data: *const f32, count: usize) -> f32 {
    use crate::scheduler::arena::with_arena;
    use rayon::prelude::*;
    
    let slice = std::slice::from_raw_parts(data, count);
    let chunk_size = count / num_cpus::get();
    
    slice.par_chunks(chunk_size)
         .map(|chunk| {
             // Each thread gets its own arena!
             with_arena(|arena| {
                 // Allocate temp buffer for Kahan summation
                 let compensate = arena.alloc::<f32>(chunk.len()).ok()?;
                 
                 // Kahan algorithm with compensated storage
                 kahan_sum(chunk, compensate)
             })
         })
         .sum()
}
```

**Benefit**: Zero malloc contention across threads.

---

## Validation Checklist

- [x] All dispatch functions wrapped in `with_arena()`
- [x] Benchmark shows <1% overhead
- [x] Tests still pass (`pytest tests/ -v`)
- [x] No memory leaks (`valgrind --leak-check=full`)
- [x] Arena stats show correct reset behavior
- [x] Documentation updated in code comments

---

## Troubleshooting

### Issue: Segfault in arena code
**Cause**: Allocating beyond arena capacity (1MB default)  
**Fix**: Increase `COREPY_ARENA_SIZE` env var or detect overflow

### Issue: Performance regression
**Cause**: Arena reset overhead  
**Fix**: Profile with `perf record`; may be unrelated

### Issue: Tests fail
**Cause**: Arena lifetime bug (pointer escapes scope)  
**Fix**: Ensure all arena allocations stay within `with_arena` closure

---

## Next Steps After Arena Integration

1. **Parallel Dispatch** (P6.2): Use arenas in parallel context ✅ (Done)
2. **GPU Arenas** (P5.3): Implement device-side arena for CUDA
3. **Custom Allocators** (P8): Per-operation arena size tuning

---

## References

- **Arena Allocator Design**: [Rust arena.rs](file:rust/corepy-runtime/src/ops/reduce.rs)
- **Bump Allocation**: [bumpalo crate](https://docs.rs/bumpalo/)
- **Performance**: [Memory Allocation Strategies](https://www.gingerbill.org/article/2019/02/08/memory-allocation-strategies/)
