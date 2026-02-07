pub mod capabilities;
pub mod cpu;
pub mod traits;

pub use capabilities::get_capabilities;
pub use traits::ComputeBackend;

use lazy_static::lazy_static;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

/// Strategy for selecting the execution backend
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum BackendPolicy {
    Default = 0,  // Runtime decision based on heuristics (Auto)
    Openblas = 1, // OpenBLAS backend
    Blas = 2,     // Generic BLAS backend
    Cuda = 3,     // CUDA backend (future)
}

/// Information about a dispatch decision
#[derive(Debug, Clone)]
pub struct DispatchInfo {
    pub backend_id: u8,
    pub operation: String,
    pub dimensions: (usize, usize, usize), // M, N, K
    pub policy: BackendPolicy,
    pub timestamp: std::time::Instant,
}

lazy_static! {
    /// Global backend policy state
    static ref CURRENT_POLICY: AtomicU8 = AtomicU8::new(BackendPolicy::Default as u8);

    /// Storage for the last backend ID used (simple tracking)
    static ref LAST_DISPATCH: AtomicU8 = AtomicU8::new(0);

    /// Storage for detailed dispatch info (rich tracking)
    static ref LAST_DISPATCH_DETAILED: Mutex<Option<DispatchInfo>> = Mutex::new(None);
}

/// Get the current global backend selection policy
pub fn get_policy() -> BackendPolicy {
    match CURRENT_POLICY.load(Ordering::Relaxed) {
        0 => BackendPolicy::Default,
        1 => BackendPolicy::Openblas,
        2 => BackendPolicy::Blas,
        3 => BackendPolicy::Cuda,
        _ => BackendPolicy::Default,
    }
}

/// Change the global backend selection policy
pub fn set_policy(policy: BackendPolicy) {
    CURRENT_POLICY.store(policy as u8, Ordering::Relaxed);
}

/// Record which backend was used (called from matmul implementations)
pub fn record_dispatch(backend_id: u8) {
    LAST_DISPATCH.store(backend_id, Ordering::Relaxed);
}

/// Record detailed dispatch metrics
pub fn record_detailed_dispatch(
    backend_id: u8,
    operation: &str,
    m: usize,
    n: usize,
    k: usize,
    policy: BackendPolicy,
) {
    let info = DispatchInfo {
        backend_id,
        operation: operation.to_string(),
        dimensions: (m, n, k),
        policy,
        timestamp: std::time::Instant::now(),
    };

    if let Ok(mut guard) = LAST_DISPATCH_DETAILED.lock() {
        *guard = Some(info);
    }
}

/// Get description of last backend used (Simple string)
pub fn get_last_dispatch() -> String {
    // Check detailed info first
    if let Ok(guard) = LAST_DISPATCH_DETAILED.lock() {
        if let Some(info) = &*guard {
            let (m, n, k) = info.dimensions;
            let elapsed = info.timestamp.elapsed();
            return format!(
                "{} → {} (size={}x{}x{}, policy={:?}, {}µs ago)",
                info.operation,
                match info.backend_id {
                    0 => {
                        // Architecture-aware SIMD backend name with runtime detection
                        #[cfg(target_arch = "aarch64")]
                        { "Corepy NEON" }
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        {
                            if is_x86_feature_detected!("avx512f") {
                                "Corepy AVX-512"
                            } else {
                                "Corepy AVX2"
                            }
                        }
                        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
                        { "Corepy Scalar" }
                    },
                    1 => "OpenBLAS",
                    2 => "BLAS",
                    3 => "CUDA",
                    4 => "Metal",
                    _ => "Unknown",
                },
                m,
                n,
                k,
                info.policy,
                elapsed.as_micros()
            );
        }
    }

    // Fallback to simple atomic tracking
    match LAST_DISPATCH.load(Ordering::Relaxed) {
        0 => "Default CPU backend".to_string(),
        1 => "OpenBLAS backend".to_string(),
        2 => "BLAS backend".to_string(),
        3 => "CUDA backend".to_string(),
        id => format!("Unknown backend ({})", id),
    }
}
