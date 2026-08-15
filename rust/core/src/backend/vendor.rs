// ============================================================================
// CPU Vendor Detection
// ============================================================================
#![allow(dead_code)] // Public API — used cross-platform; some paths only active on specific targets

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use raw_cpuid::CpuId;
use std::sync::OnceLock;

/// CPU vendor classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    /// GenuineIntel – prefer MKL
    Intel,
    /// AuthenticAMD – prefer AOCL / OpenBLAS-Zen
    Amd,
    /// Apple ARM (M1/M2/M3/M4) – prefer Accelerate
    AppleSilicon,
    /// Unknown / other
    Unknown,
}

impl std::fmt::Display for CpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuVendor::Intel => write!(f, "Intel"),
            CpuVendor::Amd => write!(f, "AMD"),
            CpuVendor::AppleSilicon => write!(f, "Apple Silicon"),
            CpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Cached vendor and HT info (detected once)
#[derive(Debug, Clone)]
pub struct VendorInfo {
    pub vendor: CpuVendor,
    /// True if logical cores > physical cores (SMT/Hyperthreading active)
    pub has_hyperthreading: bool,
    /// Human-readable brand string (e.g. "Intel(R) Core(TM) i7-12700K")
    pub brand: String,
}

static VENDOR_INFO: OnceLock<VendorInfo> = OnceLock::new();

/// Return cached vendor info (detected once at first call).
pub fn get_vendor_info() -> &'static VendorInfo {
    VENDOR_INFO.get_or_init(detect_vendor_impl)
}

/// Convenience – just the vendor enum.
pub fn detect_vendor() -> CpuVendor {
    get_vendor_info().vendor
}

/// True if SMT/HT is active (logical > physical threads).
pub fn has_hyperthreading() -> bool {
    get_vendor_info().has_hyperthreading
}

// ============================================================================
// Internal detection logic
// ============================================================================

fn detect_vendor_impl() -> VendorInfo {
    // Apple Silicon: aarch64 on macOS → always Apple
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let logical = num_cpus::get();
        let physical = num_cpus::get_physical();
        return VendorInfo {
            vendor: CpuVendor::AppleSilicon,
            has_hyperthreading: logical > physical,
            brand: "Apple Silicon".to_string(),
        };
    }

    // x86/x86_64: use CPUID
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let cpuid = CpuId::new();
        let vendor = cpuid
            .get_vendor_info()
            .map(|v| v.as_str().to_owned())
            .unwrap_or_default();

        let brand = cpuid
            .get_processor_brand_string()
            .map(|b| b.as_str().trim().to_owned())
            .unwrap_or_else(|| vendor.clone());

        let cpu_vendor = match vendor.as_str() {
            "GenuineIntel" => CpuVendor::Intel,
            "AuthenticAMD" => CpuVendor::Amd,
            _ => CpuVendor::Unknown,
        };

        let logical = num_cpus::get();
        let physical = num_cpus::get_physical();
        let has_hyperthreading = logical > physical;

        VendorInfo {
            vendor: cpu_vendor,
            has_hyperthreading,
            brand,
        }
    }

    // Generic fallback (aarch64 non-Apple, RISC-V, etc.)
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        all(target_os = "macos", target_arch = "aarch64")
    )))]
    {
        let logical = num_cpus::get();
        let physical = num_cpus::get_physical();
        VendorInfo {
            vendor: CpuVendor::Unknown,
            has_hyperthreading: logical > physical,
            brand: "Unknown CPU".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_detect_not_panics() {
        // Must not panic on any target
        let info = get_vendor_info();
        println!(
            "Vendor: {} | HT: {} | Brand: {}",
            info.vendor, info.has_hyperthreading, info.brand
        );
        // On real hardware the vendor should be something specific
        // (can be Unknown in VMs with masked CPUID)
        assert!(matches!(
            info.vendor,
            CpuVendor::Intel | CpuVendor::Amd | CpuVendor::AppleSilicon | CpuVendor::Unknown
        ));
    }

    #[test]
    fn test_hyperthreading_detection_consistent() {
        // Calling twice must return the same result (cached)
        let a = has_hyperthreading();
        let b = has_hyperthreading();
        assert_eq!(a, b);
    }
}
