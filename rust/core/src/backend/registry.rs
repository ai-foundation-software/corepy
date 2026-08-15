// ============================================================================
// Backend Registry: Global Singleton for Trait-Based Dispatch
// ============================================================================
//
// Provides a global CpuBackend instance and dispatch functions that route
// through the ComputeBackend trait. This makes the backend hot-swappable
// and decouples FFI functions from concrete implementations.

use std::sync::OnceLock;

use super::cpu::CpuBackend;
use super::scoring::{select_best_backend, OperationProfile};
use super::traits::ComputeBackend;
use super::{get_policy, BackendPolicy};
#[cfg(feature = "cuda")]
use crate::ops::cuda::CudaBackend;
#[cfg(feature = "metal")]
use crate::ops::metal::MetalBackend;

/// Global backend instances
static CPU_BACKEND: OnceLock<CpuBackend> = OnceLock::new();
#[cfg(feature = "cuda")]
static CUDA_BACKEND: OnceLock<Option<CudaBackend>> = OnceLock::new();
#[cfg(feature = "metal")]
static METAL_BACKEND: OnceLock<Option<MetalBackend>> = OnceLock::new();

fn get_cpu_backend() -> &'static CpuBackend {
    CPU_BACKEND.get_or_init(CpuBackend::new)
}

#[cfg(feature = "cuda")]
fn get_cuda_backend() -> Option<&'static CudaBackend> {
    CUDA_BACKEND.get_or_init(CudaBackend::new).as_ref()
}

#[cfg(feature = "metal")]
fn get_metal_backend() -> Option<&'static MetalBackend> {
    METAL_BACKEND.get_or_init(MetalBackend::new).as_ref()
}

/// Helper to get the best backend based on policy and operation profile
fn get_best_backend(op: &OperationProfile) -> &'static dyn ComputeBackend {
    let policy = get_policy();

    match policy {
        BackendPolicy::Cuda =>
        {
            #[cfg(feature = "cuda")]
            if let Some(cuda) = get_cuda_backend() {
                return cuda;
            }
        }
        BackendPolicy::Metal =>
        {
            #[cfg(feature = "metal")]
            if let Some(metal) = get_metal_backend() {
                return metal;
            }
        }
        BackendPolicy::Openblas | BackendPolicy::Blas => {
            // CPU backend handles internal BLAS switching
            return get_cpu_backend();
        }
        // MKL, AOCL, Accelerate, RustParallel: all CPU-side; matmul dispatch
        // layer in ops/matmul.rs selects the actual implementation.
        BackendPolicy::Mkl
        | BackendPolicy::Aocl
        | BackendPolicy::Accelerate
        | BackendPolicy::RustParallel => {
            return get_cpu_backend();
        }
        BackendPolicy::Default => {
            let flavor = select_best_backend(op);
            match flavor {
                crate::backend::scoring::BackendFlavor::CUDA => {
                    #[cfg(feature = "cuda")]
                    if let Some(cuda) = get_cuda_backend() {
                        return cuda;
                    }
                }
                crate::backend::scoring::BackendFlavor::Metal => {
                    #[cfg(feature = "metal")]
                    if let Some(metal) = get_metal_backend() {
                        return metal;
                    }
                }
                _ => {}
            }
        }
    }

    get_cpu_backend()
}

// ============================================================================
// Dispatch Functions — Route through ComputeBackend trait
// ============================================================================

/// Dispatch element-wise add through the active backend.
///
/// # Safety
/// Pointers must be valid, aligned, and point to `count` f32 elements.
pub unsafe fn dispatch_add_f32(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) -> Result<(), String> {
    let op = OperationProfile {
        flops: count,                // Estimate 1 FLOP per element for addition
        memory_bytes: count * 4 * 3, // A, B, and Result
        is_batched: false,
    };

    get_best_backend(&op)
        .add_f32(a, b, result, count)
        .map_err(|e| e.to_string())
}

/// Dispatch element-wise sub through the active backend.
///
/// # Safety
/// Pointers must be valid, aligned, and point to `count` f32 elements.
pub unsafe fn dispatch_sub_f32(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) -> Result<(), String> {
    let op = OperationProfile {
        flops: count,
        memory_bytes: count * 4 * 3,
        is_batched: false,
    };
    get_best_backend(&op)
        .sub_f32(a, b, result, count)
        .map_err(|e| e.to_string())
}

/// Dispatch element-wise mul through the active backend.
///
/// # Safety
/// Pointers must be valid, aligned, and point to `count` f32 elements.
pub unsafe fn dispatch_mul_f32(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) -> Result<(), String> {
    let op = OperationProfile {
        flops: count,
        memory_bytes: count * 4 * 3,
        is_batched: false,
    };
    get_best_backend(&op)
        .mul_f32(a, b, result, count)
        .map_err(|e| e.to_string())
}

/// Dispatch element-wise div through the active backend.
///
/// # Safety
/// Pointers must be valid, aligned, and point to `count` f32 elements.
pub unsafe fn dispatch_div_f32(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) -> Result<(), String> {
    let op = OperationProfile {
        flops: count,
        memory_bytes: count * 4 * 3,
        is_batched: false,
    };
    get_best_backend(&op)
        .div_f32(a, b, result, count)
        .map_err(|e| e.to_string())
}

/// Dispatch sum reduction through the active backend.
///
/// # Safety
/// Pointer must be valid and point to `count` f32 elements.
pub unsafe fn dispatch_sum_f32(data: *const f32, count: usize) -> Result<f32, String> {
    let op = OperationProfile {
        flops: count,
        memory_bytes: count * 4,
        is_batched: false,
    };
    get_best_backend(&op)
        .sum_f32(data, count)
        .map_err(|e| e.to_string())
}

/// Dispatch mean reduction through the active backend.
///
/// # Safety
/// Pointer must be valid and point to `count` f32 elements.
pub unsafe fn dispatch_mean_f32(data: *const f32, count: usize) -> Result<f32, String> {
    let op = OperationProfile {
        flops: count,
        memory_bytes: count * 4,
        is_batched: false,
    };
    get_best_backend(&op)
        .mean_f32(data, count)
        .map_err(|e| e.to_string())
}

/// Dispatch matmul through the active backend.
///
/// # Safety
/// Pointers must be valid and correctly sized for (m,k) x (k,n) -> (m,n).
#[allow(dead_code)]
pub unsafe fn dispatch_matmul_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let flops = 2 * m * n * k;
    let mem = (m * k + k * n + m * n) * 4;
    let op = OperationProfile {
        flops,
        memory_bytes: mem,
        is_batched: false,
    };

    get_best_backend(&op)
        .matmul_f32(a, b, c, m, k, n)
        .map_err(|e| e.to_string())
}

/// Dispatch dot product through the active backend.
///
/// # Safety
/// Pointers must be valid and point to `count` f32 elements.
pub unsafe fn dispatch_dot_product_f32(
    a: *const f32,
    b: *const f32,
    count: usize,
) -> Result<f32, String> {
    // Dot product = element-wise multiply then sum
    // Use the backend's native dot product if available,
    // otherwise fall back to mul + sum
    let a_slice = std::slice::from_raw_parts(a, count);
    let b_slice = std::slice::from_raw_parts(b, count);
    let result: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
    Ok(result)
}

// ============================================================================
// Boolean / i32 Reduction Dispatch (specialized, not trait-based)
// ============================================================================

/// Dispatch all() boolean reduction.
///
/// # Safety
/// Pointer must be valid and point to `count` u8 elements.
pub unsafe fn dispatch_all_bool(data: *const u8, count: usize) -> Result<bool, String> {
    use crate::ops::reduce::all_bool_cpu_dispatch;
    Ok(all_bool_cpu_dispatch(data, count))
}

/// Dispatch any() boolean reduction.
///
/// # Safety
/// Pointer must be valid and point to `count` u8 elements.
pub unsafe fn dispatch_any_bool(data: *const u8, count: usize) -> Result<bool, String> {
    use crate::ops::reduce::any_bool_cpu_dispatch;
    Ok(any_bool_cpu_dispatch(data, count))
}

/// Dispatch sum() for i32 arrays.
///
/// # Safety
/// Pointer must be valid and point to `count` i32 elements.
pub unsafe fn dispatch_sum_i32(data: *const i32, count: usize) -> Result<i32, String> {
    use crate::ops::reduce::sum_i32_cpu_dispatch;
    Ok(sum_i32_cpu_dispatch(data, count))
}

// ============================================================================
// Strided Reduction Dispatch
// ============================================================================

/// Dispatch strided sum for non-contiguous f32 arrays.
///
/// # Safety
/// data_ptr must be valid for the entire strided iteration range.
pub unsafe fn dispatch_sum_f32_strided(
    data_ptr: *const f32,
    shape: &[i64],
    strides: &[i64],
) -> Result<f32, String> {
    use crate::ops::reduce::sum_f32_strided_dispatch;
    Ok(sum_f32_strided_dispatch(data_ptr, shape, strides))
}

/// Dispatch strided mean for non-contiguous f32 arrays.
///
/// # Safety
/// data_ptr must be valid for the entire strided iteration range.
pub unsafe fn dispatch_mean_f32_strided(
    data_ptr: *const f32,
    shape: &[i64],
    strides: &[i64],
) -> Result<f32, String> {
    use crate::ops::reduce::mean_f32_strided_dispatch;
    Ok(mean_f32_strided_dispatch(data_ptr, shape, strides))
}

/// Dispatch strided max for non-contiguous f32 arrays.
///
/// # Safety
/// data_ptr must be valid for the entire strided iteration range.
pub unsafe fn dispatch_max_f32_strided(
    data_ptr: *const f32,
    shape: &[i64],
    strides: &[i64],
) -> Result<f32, String> {
    use crate::ops::reduce::max_f32_strided_dispatch;
    Ok(max_f32_strided_dispatch(data_ptr, shape, strides))
}

/// Dispatch strided min for non-contiguous f32 arrays.
///
/// # Safety
/// data_ptr must be valid for the entire strided iteration range.
pub unsafe fn dispatch_min_f32_strided(
    data_ptr: *const f32,
    shape: &[i64],
    strides: &[i64],
) -> Result<f32, String> {
    use crate::ops::reduce::min_f32_strided_dispatch;
    Ok(min_f32_strided_dispatch(data_ptr, shape, strides))
}

/// Get the name of the active CPU backend.
#[allow(dead_code)]
pub fn active_backend_name() -> &'static str {
    get_cpu_backend().name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_add() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut result = [0.0f32; 4];

        unsafe {
            dispatch_add_f32(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), 4).unwrap();
        }

        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_dispatch_sum() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = unsafe { dispatch_sum_f32(data.as_ptr(), 4).unwrap() };
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_backend_name() {
        let name = active_backend_name();
        assert!(
            name.starts_with("CPU"),
            "Expected CPU backend, got: {}",
            name
        );
    }
}
