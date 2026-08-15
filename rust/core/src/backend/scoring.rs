use crate::backend::capabilities::get_capabilities;
use pyo3::prelude::*;

/// Available backend flavors to dispatch to.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFlavor {
    CustomAVX2,
    CustomNeon,
    Faer,
    MKL,
    OpenBLAS,
    Metal,
    CUDA,
    GenericCPU,
}

/// Input traits for a specific array operation to predict best backend.
pub struct OperationProfile {
    pub flops: usize,
    pub memory_bytes: usize,
    #[allow(dead_code)]
    pub is_batched: bool,
}

/// Dynamically score backends and pick the optimal one for this operation profile.
pub fn select_best_backend(op: &OperationProfile) -> BackendFlavor {
    let caps = get_capabilities();

    // 1. GPU Selection (Priority: CUDA > Metal > CPU for large FLOPs)
    if op.flops > 100_000_000 {
        if caps.gpu.cuda_available {
            return BackendFlavor::CUDA;
        }
        if caps.gpu.metal_available {
            return BackendFlavor::Metal;
        }
    }

    let memory_bytes = if op.memory_bytes == 0 {
        1
    } else {
        op.memory_bytes
    };
    let intensity = (op.flops as f32) / (memory_bytes as f32);

    if intensity < 0.5 {
        // Memory bound
        if memory_bytes < caps.cpu.l2_cache_size {
            if caps.cpu.has_avx2 || caps.cpu.has_avx512f {
                return BackendFlavor::CustomAVX2;
            }
            if caps.cpu.has_neon {
                return BackendFlavor::CustomNeon;
            }
            return BackendFlavor::GenericCPU;
        }
        if memory_bytes < caps.cpu.l3_cache_size {
            return BackendFlavor::Faer;
        }
        BackendFlavor::OpenBLAS // default large memory bound to standard BLAS
    } else {
        // Compute bound
        if op.flops < 100_000 {
            if caps.cpu.has_avx2 || caps.cpu.has_avx512f {
                return BackendFlavor::CustomAVX2;
            }
            if caps.cpu.has_neon {
                return BackendFlavor::CustomNeon;
            }
            return BackendFlavor::GenericCPU;
        }
        if caps.cpu.has_avx512f {
            return BackendFlavor::MKL; // or OpenBLAS depending on compilation target
        }
        BackendFlavor::Faer
    }
}

/// PyO3 exposed scoring API. Takes properties of an operation and returns recommended backend string.
#[pyfunction]
pub fn recommend_backend(flops: usize, memory_bytes: usize, is_batched: bool) -> PyResult<String> {
    let profile = OperationProfile {
        flops,
        memory_bytes,
        is_batched,
    };

    let best = select_best_backend(&profile);

    let backend_str = match best {
        BackendFlavor::CustomAVX2 => "CustomAVX2",
        BackendFlavor::CustomNeon => "CustomNeon",
        BackendFlavor::Faer => "Faer",
        BackendFlavor::MKL => "MKL",
        BackendFlavor::OpenBLAS => "OpenBLAS",
        BackendFlavor::Metal => "Metal",
        BackendFlavor::CUDA => "CUDA",
        BackendFlavor::GenericCPU => "GenericCPU",
    };

    Ok(backend_str.to_string())
}
