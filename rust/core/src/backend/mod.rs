// ============================================================================
// Backend Module — Dispatcher Registry
// ============================================================================
//
// Central module for all backend infrastructure:
//   - BackendPolicy: user-settable dispatch strategy
//   - Dispatch recording (for diagnostics / profiling)
//   - Sub-modules: capabilities, cpu, vendor, thread_policy,
//                  env_setter, selector_v2, registry, scoring, traits

pub mod capabilities;
pub mod cpu;
pub mod device;
pub mod env_setter;
pub mod optimizer;
pub mod registry;
pub mod scoring;
pub mod selector_v2;
pub mod thread_policy;
pub mod traits;
pub mod vendor;

pub use capabilities::get_capabilities;
pub use selector_v2::{get_math_backend, MathBackend};
pub use traits::ComputeBackend;

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

// ============================================================================
// Backend Policy (user-visible, via Python set_backend_policy)
// ============================================================================

/// Strategy for selecting the execution backend.
///
/// The policy controls which backend is attempted for matmul.
/// `Default` uses the smart auto-detection logic in `selector_v2`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum BackendPolicy {
    /// Auto: runtime selection (MKL → AOCL → Accelerate → OpenBLAS → Rust)
    Default = 0,
    /// Force OpenBLAS (uses cblas_sgemm)
    Openblas = 1,
    /// Generic BLAS (alias for Openblas)
    Blas = 2,
    /// CUDA GPU backend
    Cuda = 3,
    /// Metal GPU backend (macOS)
    Metal = 4,
    /// Intel MKL (requires mkl feature or MKLROOT)
    Mkl = 5,
    /// AMD AOCL / BLIS
    Aocl = 6,
    /// Apple Accelerate (vecLib)
    Accelerate = 7,
    /// Pure-Rust parallel fallback (rayon + matrixmultiply)
    RustParallel = 8,
}

impl BackendPolicy {
    /// Convert to the underlying `MathBackend` for matmul dispatch.
    #[allow(dead_code)]
    pub fn to_math_backend(self) -> Option<MathBackend> {
        match self {
            BackendPolicy::Mkl => Some(MathBackend::Mkl),
            BackendPolicy::Aocl => Some(MathBackend::Aocl),
            BackendPolicy::Accelerate => Some(MathBackend::Accelerate),
            BackendPolicy::Openblas | BackendPolicy::Blas => Some(MathBackend::OpenBlas),
            BackendPolicy::RustParallel => Some(MathBackend::RustParallel),
            _ => None, // Default / Cuda / Metal handled by outer dispatcher
        }
    }
}

// ============================================================================
// Dispatch Information
// ============================================================================

/// Information about a dispatch decision (stored for the Python profiler).
/// Stored as plain atomics — no allocation, no mutex, no syscall on hot path.
#[derive(Debug, Clone)]
pub struct DispatchInfo {
    pub backend_id: u8,
    pub operation: String,
    pub dimensions: (usize, usize, usize), // M, N, K
    pub policy: BackendPolicy,
    pub timestamp: std::time::Instant,
}

/// Global backend policy state (user can override via Python)
static CURRENT_POLICY: AtomicU8 = AtomicU8::new(BackendPolicy::Default as u8);

/// Simple atomic tracking (fast path)
static LAST_DISPATCH: AtomicU8 = AtomicU8::new(0);

// Detailed dispatch info stored as lock-free atomics (zero allocation).
// Reconstructed into DispatchInfo only when explain_last_dispatch() is called.
static LAST_DISPATCH_BACKEND: AtomicU8 = AtomicU8::new(0);
static LAST_DISPATCH_POLICY: AtomicU8 = AtomicU8::new(0);
static LAST_DISPATCH_M: AtomicU64 = AtomicU64::new(0);
static LAST_DISPATCH_N: AtomicU64 = AtomicU64::new(0);
static LAST_DISPATCH_K: AtomicU64 = AtomicU64::new(0);

// ============================================================================
// Policy Management
// ============================================================================

/// Get the current global backend selection policy.
pub fn get_policy() -> BackendPolicy {
    match CURRENT_POLICY.load(Ordering::Relaxed) {
        0 => BackendPolicy::Default,
        1 => BackendPolicy::Openblas,
        2 => BackendPolicy::Blas,
        3 => BackendPolicy::Cuda,
        4 => BackendPolicy::Metal,
        5 => BackendPolicy::Mkl,
        6 => BackendPolicy::Aocl,
        7 => BackendPolicy::Accelerate,
        8 => BackendPolicy::RustParallel,
        _ => BackendPolicy::Default,
    }
}

/// Change the global backend selection policy.
pub fn set_policy(policy: BackendPolicy) {
    CURRENT_POLICY.store(policy as u8, Ordering::Relaxed);
}

// ============================================================================
// Dispatch Recording
// ============================================================================

/// Record which backend was used (called from matmul implementations).
pub fn record_dispatch(backend_id: u8) {
    LAST_DISPATCH.store(backend_id, Ordering::Relaxed);
}

/// Record detailed dispatch metrics (lock-free, zero allocation).
///
/// Stores backend_id, policy, dimensions as plain atomics.
/// Reconstructed only when `explain_last_dispatch()` is called (cold path).
#[inline(always)]
pub fn record_detailed_dispatch(
    backend_id: u8,
    _operation: &str, // kept for API compat; stored implicitly in backend_id
    m: usize,
    n: usize,
    k: usize,
    policy: BackendPolicy,
) {
    LAST_DISPATCH_BACKEND.store(backend_id, Ordering::Relaxed);
    LAST_DISPATCH_POLICY.store(policy as u8, Ordering::Relaxed);
    LAST_DISPATCH_M.store(m as u64, Ordering::Relaxed);
    LAST_DISPATCH_N.store(n as u64, Ordering::Relaxed);
    LAST_DISPATCH_K.store(k as u64, Ordering::Relaxed);
    // Timestamp: only pay for syscall when diagnostics are needed
    // (read lazily in explain_last_dispatch, not here)
}

/// Reconstruct a DispatchInfo from the atomic store (called on the cold/diagnostic path).
pub fn last_dispatch_info() -> DispatchInfo {
    let backend_id = LAST_DISPATCH_BACKEND.load(Ordering::Relaxed);
    let policy_raw = LAST_DISPATCH_POLICY.load(Ordering::Relaxed);
    let m = LAST_DISPATCH_M.load(Ordering::Relaxed) as usize;
    let n = LAST_DISPATCH_N.load(Ordering::Relaxed) as usize;
    let k = LAST_DISPATCH_K.load(Ordering::Relaxed) as usize;
    let policy = match policy_raw {
        1 => BackendPolicy::Openblas,
        2 => BackendPolicy::Blas,
        3 => BackendPolicy::Cuda,
        4 => BackendPolicy::Metal,
        5 => BackendPolicy::Mkl,
        6 => BackendPolicy::Aocl,
        7 => BackendPolicy::Accelerate,
        8 => BackendPolicy::RustParallel,
        _ => BackendPolicy::Default,
    };
    DispatchInfo {
        backend_id,
        operation: "matmul".to_string(),
        dimensions: (m, n, k),
        policy,
        timestamp: std::time::Instant::now(), // only paid on cold diagnostic path
    }
}

/// Backend ID → human-readable name.
#[allow(clippy::needless_return)] // returns inside #[cfg(...)] blocks are required for multi-arch
fn backend_id_name(id: u8) -> &'static str {
    match id {
        0 => {
            #[cfg(target_arch = "aarch64")]
            return "Corepy NEON";
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return "Corepy AVX-512";
                }
                return "Corepy AVX2";
            }
            #[cfg(not(any(
                target_arch = "aarch64",
                target_arch = "x86",
                target_arch = "x86_64"
            )))]
            return "Corepy Scalar";
        }
        1 => "OpenBLAS",
        2 => "BLAS",
        3 => "CUDA",
        4 => "Metal",
        5 => "MKL",
        6 => "AOCL",
        7 => "Accelerate",
        8 => "RustParallel",
        _ => "Unknown",
    }
}

/// Get description of last backend used.
pub fn get_last_dispatch() -> String {
    let info = last_dispatch_info();
    // If dimensions are all 0, no dispatch has happened yet — use simple fallback
    let (m, n, k) = info.dimensions;
    if m == 0 && n == 0 && k == 0 {
        let id = LAST_DISPATCH.load(Ordering::Relaxed);
        return format!("{} backend", backend_id_name(id));
    }
    let elapsed = info.timestamp.elapsed();
    format!(
        "{} → {} (size={}x{}x{}, policy={:?}, {}µs ago)",
        info.operation,
        backend_id_name(info.backend_id),
        m,
        n,
        k,
        info.policy,
        elapsed.as_micros()
    )
}
