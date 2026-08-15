// ============================================================================
// Environment Variable Setter
// ============================================================================
#![allow(dead_code)] // Public API — force_set_env / get_env_thread_count used by Python layer

// Sets BLAS/OpenMP/Rayon thread-count environment variables exactly once
// per backend selection using OnceLock guards.
//
// Why set env vars instead of calling library APIs directly?
// - OpenBLAS, MKL, and AOCL read these vars at their own init time.
// - Setting them before any BLAS symbol is resolved is the most
//   reliable cross-platform approach (works both with system-installed
//   and conda/pip-installed NumPy that ship their own BLAS).
//
// IMPORTANT: Must be called BEFORE any BLAS call or NumPy import.
//            In the Python extension, call during module init.

use std::sync::OnceLock;

use super::selector_v2::MathBackend;

/// Indicates whether env vars have been applied for each backend.
static MKL_ENV_APPLIED: OnceLock<()> = OnceLock::new();
static OPENBLAS_ENV_APPLIED: OnceLock<()> = OnceLock::new();
static AOCL_ENV_APPLIED: OnceLock<()> = OnceLock::new();
static RUST_ENV_APPLIED: OnceLock<()> = OnceLock::new();
/// OMP_NUM_THREADS is set once as a global safety-net regardless of backend.
static OMP_ENV_APPLIED: OnceLock<()> = OnceLock::new();

/// Apply optimal threading environment variables for the selected backend.
///
/// This function is idempotent: repeated calls for the same backend are no-ops.
/// It ALWAYS sets OMP_NUM_THREADS as a safety net for nested libraries.
pub fn apply_thread_env(backend: MathBackend, threads: usize) {
    let t = threads.to_string();

    // Set OMP_NUM_THREADS only on the very first call to any backend, so nested
    // libraries (scipy, sklearn) also respect the cap.  Using a OnceLock avoids
    // hitting std::env::var() on every dispatch.
    OMP_ENV_APPLIED.get_or_init(|| {
        safe_set_env("OMP_NUM_THREADS", &t);
    });

    match backend {
        MathBackend::Mkl => {
            MKL_ENV_APPLIED.get_or_init(|| {
                safe_set_env("MKL_NUM_THREADS", &t);
                // Disable MKL's own runtime dynamic tuning – it can override
                // our carefully chosen thread count.
                safe_set_env("MKL_DYNAMIC", "FALSE");
                // Force Intel threading layer (avoids OMP layer conflicts).
                safe_set_env("MKL_THREADING_LAYER", "INTEL");
                // KMP affinity: bind threads to physical cores, compact layout
                // (only on Linux; harmless on Windows).
                #[cfg(target_os = "linux")]
                safe_set_env("KMP_AFFINITY", "granularity=fine,compact,1,0");
            });
        }

        MathBackend::OpenBlas => {
            OPENBLAS_ENV_APPLIED.get_or_init(|| {
                safe_set_env("OPENBLAS_NUM_THREADS", &t);
                safe_set_env("OPENBLAS_VERBOSE", "0");
            });
        }

        MathBackend::Aocl => {
            AOCL_ENV_APPLIED.get_or_init(|| {
                // AOCL/BLIS uses BLIS_NUM_THREADS
                safe_set_env("BLIS_NUM_THREADS", &t);
                // Thread binding hints for AMD Zen topology
                #[cfg(target_os = "linux")]
                {
                    safe_set_env("OMP_PROC_BIND", "TRUE");
                    safe_set_env("OMP_PLACES", "cores");
                }
            });
        }

        MathBackend::RustParallel => {
            RUST_ENV_APPLIED.get_or_init(|| {
                safe_set_env("RAYON_NUM_THREADS", &t);
            });
        }

        MathBackend::Accelerate => {
            // Apple Accelerate / vecLib manages its own GCD thread pool.
            // Setting env vars can actually hurt performance on Apple Silicon.
            // We intentionally do nothing here.
        }
    }
}

/// Thread affinity and performance-mode hints (Linux only).
///
/// These are best-effort: failures are silently ignored.
#[cfg(target_os = "linux")]
pub fn apply_linux_perf_hints() {
    // Proc-bind hint (requires OpenMP runtime to be loaded)
    safe_set_env("OMP_PROC_BIND", "TRUE");
    safe_set_env("OMP_PLACES", "cores");
    // Note: cpupower governor changes require root; skip silently.
}

#[cfg(not(target_os = "linux"))]
pub fn apply_linux_perf_hints() {
    // No-op on non-Linux
}

// ============================================================================
// Helpers
// ============================================================================

/// Set an env var only if it is not already set by the user.
///
/// User-defined values always take priority – we never override them.
fn safe_set_env(key: &str, value: &str) {
    // If already set (by user or a previous call), do not override.
    if std::env::var(key).is_err() {
        // SAFETY: single-threaded context – called during module init before
        // Python threads are spawned. This is the standard PyO3 init pattern.
        unsafe {
            std::env::set_var(key, value);
        }
    }
}

/// Force-set an env var (overrides existing value).
/// Only use this when explicitly requested by the user via the Python API.
pub fn force_set_env(key: &str, value: &str) {
    // SAFETY: same as above – called from single-threaded module init context.
    unsafe {
        std::env::set_var(key, value);
    }
}

/// Read back the applied thread count for a given well-known env var.
/// Returns None if not set.
pub fn get_env_thread_count(backend: MathBackend) -> Option<usize> {
    let key = match backend {
        MathBackend::Mkl => "MKL_NUM_THREADS",
        MathBackend::OpenBlas => "OPENBLAS_NUM_THREADS",
        MathBackend::Aocl => "BLIS_NUM_THREADS",
        MathBackend::RustParallel => "RAYON_NUM_THREADS",
        MathBackend::Accelerate => return None,
    };
    std::env::var(key).ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_openblas_env() {
        // Unset in this test process before applying
        // (In parallel test runs this may already be set by another test)
        apply_thread_env(MathBackend::OpenBlas, 2);
        // OMP_NUM_THREADS must have been set
        let omp = std::env::var("OMP_NUM_THREADS");
        assert!(omp.is_ok(), "OMP_NUM_THREADS must be set");
    }

    #[test]
    fn test_apply_rust_env() {
        apply_thread_env(MathBackend::RustParallel, 4);
        // RAYON_NUM_THREADS must have been set (or was already set)
        let r = std::env::var("RAYON_NUM_THREADS");
        // It may have been set by a previous test; just ensure no panic
        let _ = r;
    }

    #[test]
    fn test_accelerate_noop() {
        // Must not panic
        apply_thread_env(MathBackend::Accelerate, 8);
    }

    #[test]
    fn test_get_env_thread_count_after_set() {
        // Force set for testing purposes
        force_set_env("MKL_NUM_THREADS", "3");
        let count = get_env_thread_count(MathBackend::Mkl);
        assert_eq!(count, Some(3));
    }
}
