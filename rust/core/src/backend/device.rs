// ============================================================================
// Hardware Device Detection Layer
// ============================================================================
// Dynamic hardware detection & abstraction for robust runtime dispatch.

use sysinfo::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Apple,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub cpu_vendor: CpuVendor,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub has_neon: bool,
    pub gpu_available: bool,
    pub gpu_name: Option<String>,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        detect_device()
    }
}

pub fn detect_device() -> DeviceInfo {
    let cpu_vendor = detect_cpu_vendor();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let (has_avx2, has_avx512, has_fma) = (
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("fma"),
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let (has_avx2, has_avx512, has_fma) = (false, false, false);

    #[cfg(target_arch = "aarch64")]
    let has_neon = true;
    #[cfg(not(target_arch = "aarch64"))]
    let has_neon = false;

    // Detect GPU
    let (gpu_available, device_type, gpu_name) = detect_gpu();

    DeviceInfo {
        device_type,
        cpu_vendor,
        has_avx2,
        has_avx512,
        has_fma,
        has_neon,
        gpu_available,
        gpu_name,
    }
}

fn detect_cpu_vendor() -> CpuVendor {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let cpuid = raw_cpuid::CpuId::new();
        if let Some(vendor) = cpuid.get_vendor_info() {
            let vendor_str = vendor.as_str();
            if vendor_str == "GenuineIntel" {
                return CpuVendor::Intel;
            } else if vendor_str == "AuthenticAMD" {
                return CpuVendor::Amd;
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return CpuVendor::Apple;

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        let mut sys = System::new();
        sys.refresh_cpu_all();
        if let Some(cpu) = sys.cpus().first() {
            let vendor_id = cpu.vendor_id().to_lowercase();
            if vendor_id.contains("intel") {
                return CpuVendor::Intel;
            } else if vendor_id.contains("amd") {
                return CpuVendor::Amd;
            } else if vendor_id.contains("apple") {
                return CpuVendor::Apple;
            }
        }

        CpuVendor::Unknown
    }
}

fn detect_gpu() -> (bool, DeviceType, Option<String>) {
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        use libloading::{Library, Symbol};
        use std::os::raw::{c_int, c_uint};

        type CuInitFn = unsafe extern "system" fn(c_uint) -> c_int;
        type CuDeviceGetCountFn = unsafe extern "system" fn(*mut c_int) -> c_int;

        let mut found_gpu = false;
        if let Ok(lib) = unsafe { Library::new("libcuda.so.1") } {
            unsafe {
                if let Ok(cu_init) = lib.get::<Symbol<CuInitFn>>(b"cuInit\0") {
                    if cu_init(0) == 0 {
                        // CUDA_SUCCESS
                        if let Ok(cu_device_get_count) =
                            lib.get::<Symbol<CuDeviceGetCountFn>>(b"cuDeviceGetCount\0")
                        {
                            let mut count = 0;
                            if cu_device_get_count(&mut count) == 0 && count > 0 {
                                found_gpu = true;
                            }
                        }
                    }
                }
            }
        }

        if found_gpu {
            return (
                true,
                DeviceType::Cuda,
                Some("NVIDIA GPU (Dynamically Loaded)".to_string()),
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Some(device) = metal::Device::system_default() {
            return (true, DeviceType::Metal, Some(device.name().to_string()));
        }
    }

    (false, DeviceType::Cpu, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = detect_device();
        println!("Detected Device: {:#?}", device);
        assert!(
            device.device_type == DeviceType::Cpu
                || device.device_type == DeviceType::Cuda
                || device.device_type == DeviceType::Metal
        );
    }
}
