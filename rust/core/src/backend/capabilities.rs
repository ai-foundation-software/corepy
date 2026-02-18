// ============================================================================
// Backend Capabilities Detection
// ============================================================================
// Runtime detection of CPU and GPU features for optimal backend selection.

use lazy_static::lazy_static;

/// CPU feature capabilities detected at runtime
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    pub arch: CpuArch,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
    pub has_neon: bool,
    pub core_count: usize,
}

/// CPU architecture enum
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // Variants are conditionally compiled per architecture
pub enum CpuArch {
    X86,
    X86_64,
    Aarch64,
    Other,
}

/// GPU backend availability
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub metal_available: bool,
    pub cuda_available: bool,
    pub rocm_available: bool,
}

/// Combined system capabilities
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub cpu: CpuCapabilities,
    pub gpu: GpuCapabilities,
}

lazy_static! {
    static ref CAPABILITIES: SystemCapabilities = detect_capabilities_impl();
}

/// Get cached system capabilities (detected once at first access)
pub fn get_capabilities() -> &'static SystemCapabilities {
    &CAPABILITIES
}

/// Internal implementation of capability detection
fn detect_capabilities_impl() -> SystemCapabilities {
    SystemCapabilities {
        cpu: detect_cpu_capabilities(),
        gpu: detect_gpu_capabilities(),
    }
}

/// Detect CPU features at runtime
fn detect_cpu_capabilities() -> CpuCapabilities {
    let arch = detect_arch();

    CpuCapabilities {
        arch,
        has_avx2: detect_avx2(),
        has_avx512f: detect_avx512f(),
        has_fma: detect_fma(),
        has_neon: detect_neon(),
        core_count: num_cpus::get(),
    }
}

/// Detect GPU backend availability
fn detect_gpu_capabilities() -> GpuCapabilities {
    GpuCapabilities {
        metal_available: detect_metal(),
        cuda_available: false, // Future: detect CUDA runtime
        rocm_available: false, // Future: detect ROCm runtime
    }
}

// ============================================================================
// Architecture Detection
// ============================================================================

fn detect_arch() -> CpuArch {
    #[cfg(target_arch = "x86")]
    {
        CpuArch::X86
    }

    #[cfg(target_arch = "x86_64")]
    {
        CpuArch::X86_64
    }

    #[cfg(target_arch = "aarch64")]
    {
        CpuArch::Aarch64
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        CpuArch::Other
    }
}

// ============================================================================
// x86/x86_64 Feature Detection
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx2() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx512f() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_fma() -> bool {
    is_x86_feature_detected!("fma")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_fma() -> bool {
    false
}

// ============================================================================
// ARM64 Feature Detection
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn detect_neon() -> bool {
    // NEON is mandatory on aarch64
    true
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_neon() -> bool {
    false
}

// ============================================================================
// GPU Backend Detection
// ============================================================================

#[cfg(target_os = "macos")]
fn detect_metal() -> bool {
    // Use our existing Metal detection
    crate::ops::metal::is_available()
}

#[cfg(not(target_os = "macos"))]
fn detect_metal() -> bool {
    false
}

// ============================================================================
// Public API Helpers
// ============================================================================

impl SystemCapabilities {
    /// Get a human-readable description of the best available SIMD backend
    pub fn best_simd_backend(&self) -> &'static str {
        if self.cpu.has_avx512f {
            "AVX-512"
        } else if self.cpu.has_avx2 && self.cpu.has_fma {
            "AVX2+FMA"
        } else if self.cpu.has_neon {
            "NEON"
        } else {
            "Scalar"
        }
    }

    /// Get a human-readable description of the best available GPU backend
    pub fn best_gpu_backend(&self) -> Option<&'static str> {
        if self.gpu.metal_available {
            Some("Metal")
        } else if self.gpu.cuda_available {
            Some("CUDA")
        } else if self.gpu.rocm_available {
            Some("ROCm")
        } else {
            None
        }
    }

    /// Generate a summary string for display
    pub fn summary(&self) -> String {
        let gpu_str = self.best_gpu_backend().unwrap_or("None");
        format!(
            "CPU: {:?} ({} cores, SIMD: {}), GPU: {}",
            self.cpu.arch,
            self.cpu.core_count,
            self.best_simd_backend(),
            gpu_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_detection() {
        let caps = get_capabilities();

        // Basic sanity checks
        assert!(caps.cpu.core_count > 0);

        // Architecture should be detected
        #[cfg(target_arch = "x86_64")]
        assert_eq!(caps.cpu.arch, CpuArch::X86_64);

        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(caps.cpu.arch, CpuArch::Aarch64);
            assert!(caps.cpu.has_neon); // NEON is mandatory on aarch64
        }
    }

    #[test]
    fn test_summary() {
        let caps = get_capabilities();
        let summary = caps.summary();

        // Should contain meaningful info
        assert!(summary.contains("CPU:"));
        assert!(summary.contains("cores"));
        assert!(summary.contains("GPU:"));
    }
}
