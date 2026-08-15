// ============================================================================
// Backend Capabilities Detection
// ============================================================================
// Runtime detection of CPU and GPU features for optimal backend selection.

/// CPU feature capabilities detected at runtime
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CpuCapabilities {
    pub arch: CpuArch,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_amx: bool,
    pub has_fma: bool,
    pub has_neon: bool,
    pub has_sve: bool,
    pub has_mma: bool,
    pub core_count: usize,
    pub physical_core_count: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
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

use std::sync::OnceLock;

/// Get cached system capabilities (detected once at first access)
pub fn get_capabilities() -> &'static SystemCapabilities {
    static CAPABILITIES: OnceLock<SystemCapabilities> = OnceLock::new();
    CAPABILITIES.get_or_init(detect_capabilities_impl)
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

    let (l1_cache_size, l2_cache_size, l3_cache_size) = detect_cache_sizes();

    CpuCapabilities {
        arch,
        has_avx2: detect_avx2(),
        has_avx512f: detect_avx512f(),
        has_amx: false, // Wait, std::arch doesn't expose AMX easily yet, need CPUID or feature
        has_fma: detect_fma(),
        has_neon: detect_neon(),
        has_sve: detect_sve(),
        has_mma: false, // ARM MMA not natively exposed in std::arch yet
        core_count: num_cpus::get(),
        physical_core_count: num_cpus::get_physical(),
        l1_cache_size,
        l2_cache_size,
        l3_cache_size,
    }
}

fn detect_cache_sizes() -> (usize, usize, usize) {
    #[cfg(target_os = "linux")]
    {
        let l1 = read_linux_cache_size(0, 1).unwrap_or(32 * 1024);
        let l2 = read_linux_cache_size(0, 2).unwrap_or(512 * 1024);
        let l3 = read_linux_cache_size(0, 3).unwrap_or(16 * 1024 * 1024);
        (l1, l2, l3)
    }
    #[cfg(target_os = "macos")]
    {
        let l1 = read_mac_sysctl("hw.l1dcachesize").unwrap_or(32 * 1024);
        let l2 = read_mac_sysctl("hw.l2cachesize").unwrap_or(512 * 1024);
        let l3 = read_mac_sysctl("hw.l3cachesize").unwrap_or(16 * 1024 * 1024);
        (l1, l2, l3)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        (32 * 1024, 512 * 1024, 16 * 1024 * 1024)
    }
}

#[cfg(target_os = "linux")]
fn read_linux_cache_size(cpu: usize, level: usize) -> Option<usize> {
    use std::fs;
    for idx in 0..4 {
        let level_path = format!(
            "/sys/devices/system/cpu/cpu{}/cache/index{}/level",
            cpu, idx
        );
        if let Ok(lvl_str) = fs::read_to_string(&level_path) {
            if lvl_str.trim().parse::<usize>().unwrap_or(0) == level {
                let size_path =
                    format!("/sys/devices/system/cpu/cpu{}/cache/index{}/size", cpu, idx);
                if let Ok(size_str) = fs::read_to_string(&size_path) {
                    let size_str = size_str.trim();
                    let mut multiplier = 1;
                    let num_str = if let Some(stripped) = size_str.strip_suffix('K') {
                        multiplier = 1024;
                        stripped
                    } else if let Some(stripped) = size_str.strip_suffix('M') {
                        multiplier = 1024 * 1024;
                        stripped
                    } else {
                        size_str
                    };
                    if let Ok(num) = num_str.parse::<usize>() {
                        return Some(num * multiplier);
                    }
                }
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn read_mac_sysctl(name: &str) -> Option<usize> {
    use std::process::Command;
    if let Ok(output) = Command::new("sysctl").arg("-n").arg(name).output() {
        if let Ok(val_str) = std::str::from_utf8(&output.stdout) {
            if let Ok(val) = val_str.trim().parse::<usize>() {
                return Some(val);
            }
        }
    }
    None
}

/// Detect GPU backend availability
fn detect_gpu_capabilities() -> GpuCapabilities {
    GpuCapabilities {
        metal_available: detect_metal(),
        cuda_available: detect_cuda(),
        rocm_available: detect_rocm(),
    }
}

fn detect_rocm() -> bool {
    std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists()
        || std::env::var("ROCM_PATH").is_ok()
}

fn detect_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::ops::cuda::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
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
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_neon() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn detect_sve() -> bool {
    // SVE detection using std::arch macro if available, or fallback
    #[cfg(target_feature = "sve")]
    return true;
    #[cfg(not(target_feature = "sve"))]
    return false;
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_sve() -> bool {
    false
}

// ============================================================================
// GPU Backend Detection
// ============================================================================

#[cfg(target_os = "macos")]
fn detect_metal() -> bool {
    #[cfg(feature = "metal")]
    {
        crate::ops::metal::is_available()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
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
        } else if self.cpu.has_sve {
            "SVE"
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
            "CPU: {:?} ({} cores, {:?} L2 Cache, SIMD: {}), GPU: {}",
            self.cpu.arch,
            self.cpu.core_count,
            self.cpu.l2_cache_size,
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
        assert!(caps.cpu.core_count > 0);
    }

    #[test]
    fn test_summary() {
        let caps = get_capabilities();
        let summary = caps.summary();
        assert!(summary.contains("CPU:"));
    }
}
