// ============================================================================
// Backend Selector v2 — Vendor-Aware Math Library Selection
// ============================================================================
//
// Selection priority (checked in order at first call, result cached):
//
//   COREPY_BACKEND env var  → always respected if set
//   Intel CPU               → Intel MKL  (if libmkl_rt found)
//   AMD CPU                 → AMD AOCL   (if libblis-mt found)
//   Apple Silicon (aarch64) → Accelerate (built-in on macOS)
//   Any CPU                 → OpenBLAS   (if libopenblas found)
//   Fallback                → Pure-Rust  (always available)
//
// The result is cached in a OnceLock — detection runs exactly once
// per process, then every matmul call pays just an atomic load.
#![allow(dead_code)] // Public API — env_name / try_override_backend used by Python layer

use std::path::Path;
use std::sync::OnceLock;

use super::vendor::{detect_vendor, CpuVendor};

/// Available math backends in priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBackend {
    /// Intel MKL (libmkl_rt)
    Mkl,
    /// AMD AOCL / BLIS (libblis-mt)
    Aocl,
    /// Apple Accelerate framework (vecLib BLAS)
    Accelerate,
    /// Reference OpenBLAS (libopenblas)
    OpenBlas,
    /// Pure-Rust parallel GEMM via rayon + matrixmultiply
    RustParallel,
}

impl std::fmt::Display for MathBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathBackend::Mkl => write!(f, "MKL"),
            MathBackend::Aocl => write!(f, "AOCL"),
            MathBackend::Accelerate => write!(f, "Accelerate"),
            MathBackend::OpenBlas => write!(f, "OpenBLAS"),
            MathBackend::RustParallel => write!(f, "RustParallel"),
        }
    }
}

impl MathBackend {
    /// Human-readable short name used in COREPY_BACKEND env var.
    pub fn env_name(&self) -> &'static str {
        match self {
            MathBackend::Mkl => "mkl",
            MathBackend::Aocl => "aocl",
            MathBackend::Accelerate => "accelerate",
            MathBackend::OpenBlas => "openblas",
            MathBackend::RustParallel => "rust",
        }
    }
}

/// Cached selected backend (detected once per process).
static SELECTED_BACKEND: OnceLock<MathBackend> = OnceLock::new();

/// Return the selected math backend (cached after first call).
pub fn get_math_backend() -> MathBackend {
    *SELECTED_BACKEND.get_or_init(select_math_backend_impl)
}

/// Manually override the backend (only works before first call to `get_math_backend`).
/// Returns `true` if the override was accepted, `false` if already initialised.
pub fn try_override_backend(backend: MathBackend) -> bool {
    SELECTED_BACKEND.set(backend).is_ok()
}

// ============================================================================
// Core selection logic
// ============================================================================

fn select_math_backend_impl() -> MathBackend {
    // ── Priority 0: COREPY_BACKEND env var override ──────────────────────────
    if let Ok(val) = std::env::var("COREPY_BACKEND") {
        let parsed = parse_backend_name(val.trim());
        if let Some(b) = parsed {
            eprintln!("[corepy] COREPY_BACKEND override → {}", b);
            return b;
        }
        eprintln!(
            "[corepy] Unknown COREPY_BACKEND='{}'; using auto-detection.",
            val.trim()
        );
    }

    let vendor = detect_vendor();
    eprintln!("[corepy] CPU vendor detected: {}", vendor);

    // ── Priority 1: Intel CPU → MKL ──────────────────────────────────────────
    if vendor == CpuVendor::Intel {
        if let Some(path) = find_mkl() {
            eprintln!("[corepy] Intel CPU + MKL found at {} → using MKL", path);
            return MathBackend::Mkl;
        }
        eprintln!("[corepy] Intel CPU detected but MKL not found.");
        eprintln!("[corepy] Falling back to OpenBLAS.");
        return MathBackend::OpenBlas;
    }

    // ── Priority 2: AMD CPU → AOCL ───────────────────────────────────────────
    if vendor == CpuVendor::Amd {
        if let Some(path) = find_aocl() {
            eprintln!("[corepy] AMD CPU + AOCL found at {} → using AOCL", path);
            return MathBackend::Aocl;
        }
        eprintln!("[corepy] AMD CPU detected but AOCL not found.");
        eprintln!("[corepy] Falling back to OpenBLAS.");
        return MathBackend::OpenBlas;
    }

    // ── Priority 3: Apple Silicon → Accelerate ───────────────────────────────
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if vendor == CpuVendor::AppleSilicon {
        eprintln!("[corepy] Apple Silicon → using Accelerate (AMX)");
        return MathBackend::Accelerate;
    }

    // ── Priority 4: Default Fallback to OpenBLAS ─────────────────────────────
    eprintln!("[corepy] Pre-installed vendor library not found/matched.");
    eprintln!("[corepy] Falling back to OpenBLAS.");
    if let Some(path) = find_openblas() {
        eprintln!("[corepy] OpenBLAS found at {} → using OpenBLAS", path);
        return MathBackend::OpenBlas;
    }

    // ── Priority 5: Pure-Rust fallback ──────────────────────────────────────
    eprintln!("[corepy] No BLAS library found — using pure-Rust fallback.");
    eprintln!("         Install OpenBLAS for better performance:");
    eprintln!("         Ubuntu: sudo apt install libopenblas-dev");
    MathBackend::RustParallel
}

fn parse_backend_name(s: &str) -> Option<MathBackend> {
    match s.to_lowercase().as_str() {
        "mkl" | "intel" | "intel-mkl" => Some(MathBackend::Mkl),
        "aocl" | "amd" | "blis" => Some(MathBackend::Aocl),
        "accelerate" | "apple" | "veclib" => Some(MathBackend::Accelerate),
        "openblas" | "blas" | "open-blas" => Some(MathBackend::OpenBlas),
        "rust" | "rustparallel" | "pure-rust" => Some(MathBackend::RustParallel),
        _ => None,
    }
}

// ============================================================================
// Library path probing — returns first found path for logging
// ============================================================================

/// Check each path; return Some(path) for the first that exists.
fn first_existing(paths: &[&str]) -> Option<String> {
    paths
        .iter()
        .find(|&&p| Path::new(p).exists())
        .map(|&p| p.to_string())
}

fn find_mkl() -> Option<String> {
    // 1. MKLROOT env var (set by `source setvars.sh`)
    if let Ok(root) = std::env::var("MKLROOT") {
        for suffix in &[
            "lib/intel64/libmkl_rt.so",
            "lib/libmkl_rt.so",
            "lib/libmkl_rt.dylib",
        ] {
            let path = format!("{}/{}", root, suffix);
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
    }

    // 2. LD_LIBRARY_PATH entries
    if let Ok(ldpath) = std::env::var("LD_LIBRARY_PATH") {
        for dir in ldpath.split(':') {
            let p = format!("{}/libmkl_rt.so", dir);
            if Path::new(&p).exists() {
                return Some(p);
            }
        }
    }

    // 3. VIRTUAL_ENV / CONDA_PREFIX (pip install mkl / conda install mkl)
    for env_var in &["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Ok(prefix) = std::env::var(env_var) {
            for suffix in &[
                // mkl-devel puts .so directly in lib/
                "lib/libmkl_rt.so",
                "lib/libmkl_rt.so.2",
                "lib/libmkl_rt.so.1",
                // mkl runtime package puts it in lib/
                "lib/libmkl_core.so",
                // site-packages layout for older pip mkl
                "lib/python3.12/site-packages/mkl/lib/libmkl_rt.so",
                "lib/python3.11/site-packages/mkl/lib/libmkl_rt.so",
                "lib/python3.10/site-packages/mkl/lib/libmkl_rt.so",
            ] {
                let p = format!("{}/{}", prefix, suffix);
                if Path::new(&p).exists() {
                    return Some(p);
                }
            }
        }
    }

    // 4. Standard system / oneAPI locations
    first_existing(&[
        "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so",
        "/opt/intel/mkl/lib/intel64/libmkl_rt.so",
        "/usr/lib/x86_64-linux-gnu/libmkl_rt.so",
        "/usr/lib/x86_64-linux-gnu/libmkl_rt.so.2",
        "/usr/local/lib/libmkl_rt.so",
        // macOS
        "/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.dylib",
        "/usr/local/lib/libmkl_rt.dylib",
    ])
}

fn find_aocl() -> Option<String> {
    if let Ok(root) = std::env::var("AOCL_ROOT") {
        let p = format!("{}/lib/libblis-mt.so", root);
        if Path::new(&p).exists() {
            return Some(p);
        }
    }
    first_existing(&[
        "/opt/aocl/lib/libblis-mt.so",
        "/opt/amd/aocl/aocl-linux-gcc/lib/libblis-mt.so",
        "/usr/lib/x86_64-linux-gnu/libblis-mt.so",
        "/usr/local/lib/libblis-mt.so",
        "/usr/lib/libblis.so",
    ])
}

fn find_openblas() -> Option<String> {
    // OPENBLAS_PATH hint from user
    if let Ok(p) = std::env::var("OPENBLAS_PATH") {
        if Path::new(&p).exists() {
            return Some(p);
        }
    }
    first_existing(&[
        // Linux multiarch
        "/usr/lib/x86_64-linux-gnu/libopenblas.so.0",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so",
        "/usr/lib/aarch64-linux-gnu/libopenblas.so.0",
        "/usr/lib/aarch64-linux-gnu/libopenblas.so",
        // Generic Linux
        "/usr/lib/libopenblas.so",
        "/usr/local/lib/libopenblas.so",
        // macOS Homebrew
        "/opt/homebrew/lib/libopenblas.dylib",
        "/usr/local/lib/libopenblas.dylib",
        // conda-forge
        "/opt/conda/lib/libopenblas.so",
    ])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection_returns_valid_backend() {
        let b = get_math_backend();
        println!("Selected backend: {}", b);
        assert!(matches!(
            b,
            MathBackend::Mkl
                | MathBackend::Aocl
                | MathBackend::Accelerate
                | MathBackend::OpenBlas
                | MathBackend::RustParallel
        ));
    }

    #[test]
    fn test_parse_backend_name() {
        assert_eq!(parse_backend_name("mkl"), Some(MathBackend::Mkl));
        assert_eq!(parse_backend_name("intel"), Some(MathBackend::Mkl));
        assert_eq!(parse_backend_name("amd"), Some(MathBackend::Aocl));
        assert_eq!(parse_backend_name("aocl"), Some(MathBackend::Aocl));
        assert_eq!(parse_backend_name("apple"), Some(MathBackend::Accelerate));
        assert_eq!(parse_backend_name("openblas"), Some(MathBackend::OpenBlas));
        assert_eq!(parse_backend_name("rust"), Some(MathBackend::RustParallel));
        assert_eq!(parse_backend_name("unknown-xyz"), None);
    }

    #[test]
    fn test_env_override_rust() {
        let parsed = parse_backend_name("rust");
        assert_eq!(parsed, Some(MathBackend::RustParallel));
    }

    #[test]
    fn test_env_name_round_trips() {
        let backends = [
            MathBackend::Mkl,
            MathBackend::Aocl,
            MathBackend::Accelerate,
            MathBackend::OpenBlas,
            MathBackend::RustParallel,
        ];
        for b in &backends {
            assert!(!b.env_name().is_empty());
        }
    }
}
