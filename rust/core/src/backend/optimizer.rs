// ============================================================================
// Workload-Based Optimizer Layer
// ============================================================================
// Makes intelligent backend dispatch decisions based on runtime devices and matrix sizes.

use super::device::{detect_device, CpuVendor, DeviceInfo, DeviceType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    Mkl,
    OpenBlas,
    Faer,
    Simd,
    Naive,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub small_threshold: usize,
    pub gpu_threshold: usize,
    pub mkl_enabled: bool,
    pub openblas_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            small_threshold: 128,
            gpu_threshold: 512,
            #[cfg(feature = "mkl")]
            mkl_enabled: true,
            #[cfg(not(feature = "mkl"))]
            mkl_enabled: false,

            #[cfg(feature = "openblas")]
            openblas_enabled: true,
            #[cfg(not(feature = "openblas"))]
            openblas_enabled: false,
        }
    }
}

pub fn smart_optimize(matrix_size: usize, config: &RuntimeConfig, device: &DeviceInfo) -> Backend {
    // 1. GPU Check
    if device.gpu_available && matrix_size > config.gpu_threshold {
        return match device.device_type {
            DeviceType::Cuda => Backend::Cuda,
            DeviceType::Metal => Backend::Metal,
            _ => Backend::Faer, // Should not happen if gpu_available is true
        };
    }

    // 2. Small Matrix (CPU SIMD / Naive)
    if matrix_size <= config.small_threshold {
        if device.has_avx2 || device.has_neon {
            return Backend::Simd;
        } else {
            return Backend::Naive;
        }
    }

    // 3. Medium & Large Matrix CPU Logic
    if device.cpu_vendor == CpuVendor::Intel && config.mkl_enabled {
        return Backend::Mkl;
    }

    if device.cpu_vendor == CpuVendor::Amd && config.openblas_enabled {
        return Backend::OpenBlas;
    }

    // Default Fallback
    Backend::Faer
}

pub fn analyse_workload(matrix_size: usize, config: &RuntimeConfig) -> String {
    let device = detect_device();
    let chosen_backend = smart_optimize(matrix_size, config, &device);

    let performance_class = match (chosen_backend, matrix_size) {
        (Backend::Cuda | Backend::Metal | Backend::Mkl, size) if size > config.gpu_threshold => {
            "HPC"
        }
        (Backend::OpenBlas | Backend::Faer, _) => "Medium",
        (Backend::Simd, _) => "Medium (Small Kernel)",
        (Backend::Naive, _) => "Low",
        _ => "Medium",
    };

    let fallback_chain = "Cuda/Metal → MKL → OpenBLAS → Faer → SIMD → Naive";

    format!(
        "Hardware Summary:
CPU: {:?}
AVX2: {}
AVX512: {}
GPU: {} ({})

Workload: {}x{} matmul

Decision:
→ {:?} selected
Reason: {}
Expected Performance Class: {}
Fallback Chain: {}",
        device.cpu_vendor,
        if device.has_avx2 { "Yes" } else { "No" },
        if device.has_avx512 { "Yes" } else { "No" },
        if device.gpu_available {
            "Available"
        } else {
            "None"
        },
        device.gpu_name.as_deref().unwrap_or("N/A"),
        matrix_size,
        matrix_size,
        chosen_backend,
        get_reason_for_decision(chosen_backend, matrix_size, config, &device),
        performance_class,
        fallback_chain
    )
}

fn get_reason_for_decision(
    backend: Backend,
    size: usize,
    config: &RuntimeConfig,
    _device: &DeviceInfo,
) -> String {
    match backend {
        Backend::Cuda | Backend::Metal => format!(
            "GPU available + large matrix ({} > {})",
            size, config.gpu_threshold
        ),
        Backend::Simd => format!(
            "Small matrix ({} <= {}) with SIMD features available",
            size, config.small_threshold
        ),
        Backend::Naive => format!(
            "Small matrix ({} <= {}) but no SIMD features detected",
            size, config.small_threshold
        ),
        Backend::Mkl => "Optimized MKL available for Intel CPU".to_string(),
        Backend::OpenBlas => "Optimized OpenBLAS available for AMD CPU".to_string(),
        Backend::Faer => "Pure Rust fallback (default safe)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_optimize_small_cpu() {
        let mut device = detect_device();
        device.gpu_available = false;
        device.has_avx2 = true;

        let config = RuntimeConfig::default();
        let backend = smart_optimize(64, &config, &device);

        assert_eq!(backend, Backend::Simd);
    }

    #[test]
    fn test_smart_optimize_large_gpu() {
        let mut device = detect_device();
        device.gpu_available = true;
        device.device_type = DeviceType::Cuda;

        let config = RuntimeConfig::default();
        let backend = smart_optimize(1024, &config, &device);

        assert_eq!(backend, Backend::Cuda);
    }
}
