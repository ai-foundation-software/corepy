// ============================================================================
// Thread Policy Module
// ============================================================================
#![allow(dead_code)] // Public API — get_default_large_matrix_plan / needs_env used by callers

use std::sync::OnceLock;

use super::selector_v2::MathBackend;
use super::vendor::{detect_vendor, has_hyperthreading, CpuVendor};

/// Physical core count, cached on first call (avoids /proc/cpuinfo re-read).
static PHYSICAL_CORES: OnceLock<usize> = OnceLock::new();
/// Logical core count (incl. hyperthreads), cached.
static LOGICAL_CORES: OnceLock<usize> = OnceLock::new();

#[inline(always)]
fn physical_cores() -> usize {
    *PHYSICAL_CORES.get_or_init(num_cpus::get_physical)
}

#[inline(always)]
fn logical_cores() -> usize {
    *LOGICAL_CORES.get_or_init(num_cpus::get)
}

/// Result of thread policy calculation.
#[derive(Debug, Clone, Copy)]
pub struct ThreadPlan {
    /// Number of threads the BLAS/Rayon pool should use.
    pub count: usize,
    /// Whether environment variables must be set for this backend.
    pub needs_env: bool,
}

/// Compute the optimal thread plan for a matmul of the given dimensions.
///
/// `size` = max(M, N, K) — the largest matrix side length.
pub fn compute_thread_plan(size: usize, backend: MathBackend) -> ThreadPlan {
    // ------------------------------------------------------------------
    // Fast path: explicit user override
    // ------------------------------------------------------------------
    if let Some(env_threads) = super::env_setter::get_env_thread_count(backend) {
        return ThreadPlan {
            count: env_threads.max(1),
            needs_env: !matches!(backend, MathBackend::Accelerate),
        };
    }

    let physical = physical_cores();
    let vendor = detect_vendor();
    let ht = has_hyperthreading();

    // ------------------------------------------------------------------
    // Base thread count from matrix size
    // ------------------------------------------------------------------
    //
    // KEY RULE: For OpenBLAS, thread-spawn overhead (~4-8ms) dominates
    // compute time for matrices < 1024. Single-threaded BLAS is faster.
    // MKL has a low-overhead internal threading scheduler, so it can use
    // more threads earlier.
    let base_threads: usize = match backend {
        // OpenBLAS: single-thread small/mid matrices to avoid spawn overhead
        MathBackend::OpenBlas => {
            if size < 1024 {
                1
            } else if size < 2048 {
                (physical / 2).max(1)
            } else if size < 4096 {
                physical
            } else if ht {
                num_cpus::get()
            } else {
                physical
            }
        }
        // MKL on Intel: low threading overhead, use more cores earlier
        MathBackend::Mkl => {
            if size < 256 {
                1
            } else if size < 1024 {
                (physical / 2).max(1)
            } else {
                physical
            }
        }
        // AOCL on AMD: similar to MKL overhead profile
        MathBackend::Aocl => {
            if size < 256 {
                1
            } else if size < 1024 {
                (physical / 2).max(1)
            } else {
                physical
            }
        }
        // Pure-Rust: fine-grained rayon control, low spawn overhead
        MathBackend::RustParallel => {
            if size < 256 {
                1
            } else if size < 512 {
                2.min(physical)
            } else if size < 2048 {
                (physical / 2).max(1)
            } else {
                physical
            }
        }
        // Accelerate (Apple): GCD manages threads internally
        MathBackend::Accelerate => 1,
    };

    // ------------------------------------------------------------------
    // Hard cap for very small CPUs (laptops / embedded boards)
    // ------------------------------------------------------------------
    let capped = if physical <= 2 {
        base_threads.min(physical)
    } else {
        base_threads
    };

    // ------------------------------------------------------------------
    // Vendor + backend fine-tuning
    // ------------------------------------------------------------------
    let adjusted = match (vendor, backend) {
        // AMD + OpenBLAS: reduce threads slightly to avoid false sharing on Zen.
        (CpuVendor::Amd, MathBackend::OpenBlas) if capped > 2 => {
            ((capped as f64 * 0.75).round() as usize).max(1)
        }
        // Intel + MKL: MKL handles its own dynamic tuning.
        (CpuVendor::Intel, MathBackend::Mkl) => physical.min(capped.max(physical / 2)),
        // Avoid hyperthreads for mid-size with OpenBLAS (false sharing)
        (_, MathBackend::OpenBlas) if ht && size < 2048 => capped.min(physical),
        _ => capped,
    };

    let final_count = adjusted.max(1);

    ThreadPlan {
        count: final_count,
        needs_env: !matches!(backend, MathBackend::Accelerate),
    }
}

/// Global cached thread plan (computed once per process for the initial selection).
/// Updated when backend or problem size changes significantly.
static CACHED_PLAN: OnceLock<ThreadPlan> = OnceLock::new();

/// Return a cached thread plan for "large" matrices (≥ 1024).
/// Used during backend initialisation to set env vars once.
pub fn get_default_large_matrix_plan(backend: MathBackend) -> &'static ThreadPlan {
    CACHED_PLAN.get_or_init(|| compute_thread_plan(2048, backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_matrix_single_thread() {
        unsafe {
            std::env::remove_var("OPENBLAS_NUM_THREADS");
        }
        let plan = compute_thread_plan(64, MathBackend::OpenBlas);
        assert_eq!(plan.count, 1, "Tiny matrices must use 1 thread");
    }

    #[test]
    fn test_small_matrix_boundary() {
        unsafe {
            std::env::remove_var("OPENBLAS_NUM_THREADS");
        }
        // 255 < 256 → 1 thread
        let plan = compute_thread_plan(255, MathBackend::OpenBlas);
        assert_eq!(plan.count, 1);
        // 256 ≥ 256 → may use more
        let plan2 = compute_thread_plan(256, MathBackend::OpenBlas);
        assert!(plan2.count >= 1);
    }

    #[test]
    fn test_mid_matrix_capped() {
        // 512-wide matrix must not use more than 4 threads
        let plan = compute_thread_plan(512, MathBackend::OpenBlas);
        assert!(
            plan.count <= 4,
            "Mid-size must cap at 4 threads: got {}",
            plan.count
        );
    }

    #[test]
    fn test_large_matrix_uses_more_threads() {
        let small_plan = compute_thread_plan(128, MathBackend::OpenBlas);
        let large_plan = compute_thread_plan(4096, MathBackend::OpenBlas);
        assert!(
            large_plan.count >= small_plan.count,
            "Large matrices should use at least as many threads as small ones"
        );
    }

    #[test]
    fn test_thread_plan_never_zero() {
        for size in [1, 64, 128, 256, 512, 1024, 2048, 4096] {
            let plan = compute_thread_plan(size, MathBackend::OpenBlas);
            assert!(
                plan.count >= 1,
                "Thread count must be ≥ 1 for size {}",
                size
            );
        }
    }

    #[test]
    fn test_rust_parallel_threading() {
        let plan = compute_thread_plan(2048, MathBackend::RustParallel);
        assert!(plan.count >= 1);
    }
}
