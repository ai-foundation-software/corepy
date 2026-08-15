#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

use pyo3::prelude::*;

// Global profiler instance for this module (and the process)
struct LazyProfiler;
impl std::ops::Deref for LazyProfiler {
    type Target = crate::profiler::Profiler;
    fn deref(&self) -> &Self::Target {
        static GLOBAL_PROFILER_INSTANCE: std::sync::OnceLock<crate::profiler::Profiler> =
            std::sync::OnceLock::new();
        GLOBAL_PROFILER_INSTANCE.get_or_init(crate::profiler::Profiler::new)
    }
}
static GLOBAL_PROFILER: LazyProfiler = LazyProfiler;

/// Export all FFI functions to Python
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Reduction operations (contiguous)
    m.add_function(wrap_pyfunction!(array_all, m)?)?;
    m.add_function(wrap_pyfunction!(array_any, m)?)?;
    m.add_function(wrap_pyfunction!(array_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_sum_i32, m)?)?;
    m.add_function(wrap_pyfunction!(array_mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_matmul_2d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_dot_product_f32, m)?)?;

    // Strided reduction operations (zero-copy for non-contiguous arrays)
    m.add_function(wrap_pyfunction!(array_sum_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(array_mean_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(array_max_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(array_min_f32_strided, m)?)?;

    // Backend control
    m.add_function(wrap_pyfunction!(set_backend_policy, m)?)?;
    m.add_function(wrap_pyfunction!(get_backend_policy, m)?)?;
    m.add_function(wrap_pyfunction!(explain_last_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(get_math_backend_info, m)?)?;

    // Element-wise operations
    m.add_function(wrap_pyfunction!(array_add_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_sub_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_mul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(array_div_f32, m)?)?;

    // Profiling functions
    m.add_function(wrap_pyfunction!(enable_profiling, m)?)?;
    m.add_function(wrap_pyfunction!(disable_profiling, m)?)?;
    m.add_function(wrap_pyfunction!(clear_profile, m)?)?;
    m.add_function(wrap_pyfunction!(get_profile_report, m)?)?;
    m.add_function(wrap_pyfunction!(set_profile_context, m)?)?;

    // Metal GPU operations (always register, returns error on non-macOS)
    m.add_function(wrap_pyfunction!(metal_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(metal_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_add_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_sub_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_mul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_div_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_transpose_f32, m)?)?;
    m.add_function(wrap_pyfunction!(metal_broadcast_op, m)?)?;

    // CUDA GPU operations
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_add_f32, m)?)?;

    // System capabilities
    m.add_function(wrap_pyfunction!(get_system_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(get_capabilities_summary, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_backend, m)?)?;
    m.add_function(wrap_pyfunction!(analyse_workload, m)?)?;

    // Demo functions (backward compatibility)
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    // Random Generation
    m.add_function(wrap_pyfunction!(random_uniform_f32, m)?)?;
    m.add_function(wrap_pyfunction!(random_normal_f32, m)?)?;

    // DataFrame Functions
    m.add_function(wrap_pyfunction!(crate::dataframe::csv::read_csv, m)?)?;

    // Architectural Demo Workload
    m.add_function(wrap_pyfunction!(process_workload, m)?)?;

    Ok(())
}

// ============================================================================
// Profiling Control
// ============================================================================

#[pyfunction]
fn enable_profiling() -> PyResult<()> {
    GLOBAL_PROFILER.enable();
    Ok(())
}

#[pyfunction]
fn disable_profiling() -> PyResult<()> {
    GLOBAL_PROFILER.disable();
    Ok(())
}

#[pyfunction]
fn clear_profile() -> PyResult<()> {
    GLOBAL_PROFILER.clear();
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (context=None))]
fn get_profile_report(context: Option<String>) -> PyResult<String> {
    GLOBAL_PROFILER
        .export_json(context.as_deref())
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (context=None))]
fn set_profile_context(context: Option<String>) -> PyResult<()> {
    crate::profiler::set_context(context);
    Ok(())
}

// ============================================================================
// System Capabilities
// ============================================================================

/// Get system capabilities as a Python dict
#[pyfunction]
fn get_system_capabilities(py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use crate::backend::get_capabilities;
    use pyo3::types::PyDict;

    let caps = get_capabilities();

    let cpu_dict = PyDict::new(py);
    cpu_dict.set_item("arch", format!("{:?}", caps.cpu.arch))?;
    cpu_dict.set_item("cores", caps.cpu.core_count)?;
    cpu_dict.set_item("physical_cores", caps.cpu.physical_core_count)?;
    cpu_dict.set_item("has_avx2", caps.cpu.has_avx2)?;
    cpu_dict.set_item("has_avx512", caps.cpu.has_avx512f)?;
    cpu_dict.set_item("has_fma", caps.cpu.has_fma)?;
    cpu_dict.set_item("has_neon", caps.cpu.has_neon)?;
    cpu_dict.set_item("l1_cache", caps.cpu.l1_cache_size)?;
    cpu_dict.set_item("l2_cache", caps.cpu.l2_cache_size)?;
    cpu_dict.set_item("l3_cache", caps.cpu.l3_cache_size)?;
    cpu_dict.set_item("best_simd", caps.best_simd_backend())?;

    let gpu_dict = PyDict::new(py);
    gpu_dict.set_item("metal_available", caps.gpu.metal_available)?;
    gpu_dict.set_item("cuda_available", caps.gpu.cuda_available)?;
    gpu_dict.set_item("rocm_available", caps.gpu.rocm_available)?;
    if let Some(best) = caps.best_gpu_backend() {
        gpu_dict.set_item("best_gpu", best)?;
    }

    let result = PyDict::new(py);
    result.set_item("cpu", cpu_dict)?;
    result.set_item("gpu", gpu_dict)?;

    Ok(result.into())
}

/// Get a human-readable summary of system capabilities
#[pyfunction]
fn get_capabilities_summary() -> String {
    crate::backend::get_capabilities().summary()
}

/// Recommend ideal backend using Hardware Scoring
#[pyfunction]
fn recommend_backend(flops: usize, memory_bytes: usize, is_batched: bool) -> PyResult<String> {
    crate::backend::scoring::recommend_backend(flops, memory_bytes, is_batched)
}

/// Run the optimizer and return the analysis report
#[pyfunction]
fn analyse_workload(matrix_size: usize, small_threshold: usize, gpu_threshold: usize) -> String {
    let config = crate::backend::optimizer::RuntimeConfig {
        small_threshold,
        gpu_threshold,
        mkl_enabled: cfg!(feature = "mkl"),
        openblas_enabled: cfg!(feature = "openblas"),
    };
    crate::backend::optimizer::analyse_workload(matrix_size, &config)
}

// ============================================================================
// Reduction Operations
// ============================================================================

#[pyfunction]
fn array_all(data_ptr: usize, count: usize) -> PyResult<bool> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_all",
        ));
    }

    if count == 0 {
        return Ok(true);
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "all".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_all_bool(data_ptr as *const u8, count)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_any(data_ptr: usize, count: usize) -> PyResult<bool> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_any",
        ));
    }

    if count == 0 {
        return Ok(false);
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "any".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_any_bool(data_ptr as *const u8, count)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_sum_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_sum_f32",
        ));
    }

    if count == 0 {
        return Ok(0.0);
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "sum".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_sum_f32(data_ptr as *const f32, count)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_sum_i32(data_ptr: usize, count: usize) -> PyResult<i32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_sum_i32",
        ));
    }

    if count == 0 {
        return Ok(0);
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "sum".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_sum_i32(data_ptr as *const i32, count)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_mean_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_mean_f32",
        ));
    }

    if count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute mean of empty array",
        ));
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "mean".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_mean_f32(data_ptr as *const f32, count)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

// ============================================================================
// Strided Reduction Operations (Zero-Copy)
// ============================================================================

#[pyfunction]
fn array_sum_f32_strided(data_ptr: usize, shape: Vec<i64>, strides: Vec<i64>) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_sum_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Ok(0.0);
    }

    let count: i64 = shape.iter().product();

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "sum_strided".to_string(),
        "CPU".to_string(),
        count as usize,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_sum_f32_strided(data_ptr as *const f32, &shape, &strides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_mean_f32_strided(data_ptr: usize, shape: Vec<i64>, strides: Vec<i64>) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_mean_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute mean of empty array",
        ));
    }

    let count: i64 = shape.iter().product();

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "mean_strided".to_string(),
        "CPU".to_string(),
        count as usize,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_mean_f32_strided(
            data_ptr as *const f32,
            &shape,
            &strides,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_max_f32_strided(data_ptr: usize, shape: Vec<i64>, strides: Vec<i64>) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_max_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute max of empty array",
        ));
    }

    let count: i64 = shape.iter().product();

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "max_strided".to_string(),
        "CPU".to_string(),
        count as usize,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_max_f32_strided(data_ptr as *const f32, &shape, &strides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_min_f32_strided(data_ptr: usize, shape: Vec<i64>, strides: Vec<i64>) -> PyResult<f32> {
    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_min_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute min of empty array",
        ));
    }

    let count: i64 = shape.iter().product();

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "min_strided".to_string(),
        "CPU".to_string(),
        count as usize,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_min_f32_strided(data_ptr as *const f32, &shape, &strides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_dot_product_f32(a_ptr: usize, b_ptr: usize, count: usize) -> PyResult<f32> {
    if a_ptr == 0 || b_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_dot_product_f32",
        ));
    }

    if count == 0 {
        return Ok(0.0);
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "dot_product".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe {
        crate::backend::registry::dispatch_dot_product_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            count,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    };

    Ok(result)
}

#[pyfunction]
fn array_matmul_2d_f32(
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<()> {
    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_matmul_2d_f32",
        ));
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "matmul_2d".to_string(),
        "CPU".to_string(),
        m * k * n, // FLOPs approximation
    );

    unsafe {
        crate::backend::registry::dispatch_matmul_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            m,
            k,
            n,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    }

    Ok(())
}

#[pyfunction]
fn array_matmul_f32(a_ptr: usize, b_ptr: usize, count: usize) -> PyResult<f32> {
    // Legacy/Existing wrapper that calls the same kernel
    array_dot_product_f32(a_ptr, b_ptr, count)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

#[pyfunction]
fn array_add_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_add_f32",
        ));
    }

    if count == 0 {
        return Ok(());
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "add".to_string(),
        "CPU".to_string(),
        count,
    );

    unsafe {
        crate::backend::registry::dispatch_add_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    }

    Ok(())
}

#[pyfunction]
fn array_sub_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_sub_f32",
        ));
    }

    if count == 0 {
        return Ok(());
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "sub".to_string(),
        "CPU".to_string(),
        count,
    );

    unsafe {
        crate::backend::registry::dispatch_sub_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    }

    Ok(())
}

#[pyfunction]
fn array_mul_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_mul_f32",
        ));
    }

    if count == 0 {
        return Ok(());
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "mul".to_string(),
        "CPU".to_string(),
        count,
    );

    unsafe {
        crate::backend::registry::dispatch_mul_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    }

    Ok(())
}

#[pyfunction]
fn array_div_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to array_div_f32",
        ));
    }

    if count == 0 {
        return Ok(());
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "div".to_string(),
        "CPU".to_string(),
        count,
    );

    unsafe {
        crate::backend::registry::dispatch_div_f32(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    }

    Ok(())
}

// ============================================================================
// Backend Control
// ============================================================================

#[pyfunction]
fn set_backend_policy(policy: u8) -> PyResult<()> {
    use crate::backend::{set_policy, BackendPolicy};
    let p = match policy {
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
    set_policy(p);
    Ok(())
}

#[pyfunction]
fn get_backend_policy() -> PyResult<u8> {
    use crate::backend::get_policy;
    Ok(get_policy() as u8)
}

#[pyfunction]
fn explain_last_dispatch() -> String {
    crate::backend::get_last_dispatch()
}

/// Return information about the currently selected math backend.
///
/// Returns a dict with keys:
///   - backend (str): "MKL" | "AOCL" | "Accelerate" | "OpenBLAS" | "RustParallel"
///   - vendor  (str): "Intel" | "AMD" | "Apple Silicon" | "Unknown"
///   - threads (int): thread count for a 2048x2048 matmul
///   - hyperthreading (bool): whether HT/SMT is active
///   - brand   (str): CPU brand string
#[pyfunction]
fn get_math_backend_info(py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use crate::backend::{
        get_math_backend, thread_policy::compute_thread_plan, vendor::get_vendor_info,
    };
    use pyo3::types::PyDict;

    let backend = get_math_backend();
    let vendor_info = get_vendor_info();
    let plan = compute_thread_plan(2048, backend);

    let d = PyDict::new(py);
    d.set_item("backend", backend.to_string())?;
    d.set_item("vendor", vendor_info.vendor.to_string())?;
    d.set_item("threads", plan.count)?;
    d.set_item("hyperthreading", vendor_info.has_hyperthreading)?;
    d.set_item("brand", vendor_info.brand.clone())?;
    d.set_item("physical_cores", num_cpus::get_physical())?;
    d.set_item("logical_cores", num_cpus::get())?;
    Ok(d.into())
}

// ============================================================================
// Demo Functions (Backward Compatibility)
// ============================================================================

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn process_workload(data: Vec<f64>) -> PyResult<Vec<f64>> {
    // Passes the Python List[float] directly into the Rust vector and triggers the pipeline
    Ok(crate::ops::workload::process_workload(data))
}

// ============================================================================
// Random Generation
// ============================================================================

#[pyfunction]
#[pyo3(signature = (shape, seed, algo=crate::ops::random::RngAlgorithm::PCG64))]
fn random_uniform_f32(
    shape: Vec<usize>,
    seed: u64,
    algo: crate::ops::random::RngAlgorithm,
) -> PyResult<crate::array::core_array::CoreArray> {
    crate::ops::random::uniform_f32(shape, seed, algo)
}

#[pyfunction]
#[pyo3(signature = (shape, seed, algo=crate::ops::random::RngAlgorithm::PCG64))]
fn random_normal_f32(
    shape: Vec<usize>,
    seed: u64,
    algo: crate::ops::random::RngAlgorithm,
) -> PyResult<crate::array::core_array::CoreArray> {
    crate::ops::random::normal_f32(shape, seed, algo)
}

// ============================================================================
// Metal GPU Operations
// ============================================================================

#[pyfunction]
fn metal_is_available() -> PyResult<bool> {
    #[cfg(feature = "metal")]
    {
        Ok(crate::ops::metal::is_available())
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(false)
    }
}

#[pyfunction]
fn metal_sum_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::sum_f32_metal_dispatch;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_sum_f32",
            ));
        }

        if count == 0 {
            return Ok(0.0);
        }

        // PROFILING
        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "sum".to_string(),
            "Metal".to_string(),
            count,
        );

        let result = unsafe { sum_f32_metal_dispatch(data_ptr as *const f32, count) };
        Ok(result)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_mean_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::mean_f32_metal_dispatch;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_mean_f32",
            ));
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot compute mean of empty array",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "mean".to_string(),
            "Metal".to_string(),
            count,
        );

        let result = unsafe { mean_f32_metal_dispatch(data_ptr as *const f32, count) };
        Ok(result)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_matmul_f32(
    a_ptr: usize,
    b_ptr: usize,
    c_ptr: usize,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::matmul_f32_metal_dispatch;

        if a_ptr == 0 || b_ptr == 0 || c_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_matmul_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "matmul".to_string(),
            "Metal".to_string(),
            m * k * n,
        );

        unsafe {
            matmul_f32_metal_dispatch(
                a_ptr as *const f32,
                b_ptr as *const f32,
                c_ptr as *mut f32,
                m,
                k,
                n,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (a_ptr, b_ptr, c_ptr, m, k, n);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

// ============================================================================
// Metal Reduction Operations (max, min)
// ============================================================================

#[pyfunction]
fn metal_max_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::max_f32_metal_dispatch;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_max_f32",
            ));
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot compute max of empty array",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "max".to_string(),
            "Metal".to_string(),
            count,
        );

        let result = unsafe { max_f32_metal_dispatch(data_ptr as *const f32, count) };
        Ok(result)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_min_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::min_f32_metal_dispatch;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_min_f32",
            ));
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot compute min of empty array",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "min".to_string(),
            "Metal".to_string(),
            count,
        );

        let result = unsafe { min_f32_metal_dispatch(data_ptr as *const f32, count) };
        Ok(result)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

// ============================================================================
// Metal Element-wise Operations (add, sub, mul, div)
// ============================================================================

#[pyfunction]
fn metal_add_f32(a_ptr: usize, b_ptr: usize, result_ptr: usize, count: usize) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::add_f32_metal_dispatch;

        if a_ptr == 0 || b_ptr == 0 || result_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_add_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "add".to_string(),
            "Metal".to_string(),
            count,
        );

        unsafe {
            add_f32_metal_dispatch(
                a_ptr as *const f32,
                b_ptr as *const f32,
                result_ptr as *mut f32,
                count,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (a_ptr, b_ptr, result_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_sub_f32(a_ptr: usize, b_ptr: usize, result_ptr: usize, count: usize) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::sub_f32_metal_dispatch;

        if a_ptr == 0 || b_ptr == 0 || result_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_sub_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "sub".to_string(),
            "Metal".to_string(),
            count,
        );

        unsafe {
            sub_f32_metal_dispatch(
                a_ptr as *const f32,
                b_ptr as *const f32,
                result_ptr as *mut f32,
                count,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (a_ptr, b_ptr, result_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_mul_f32(a_ptr: usize, b_ptr: usize, result_ptr: usize, count: usize) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::mul_f32_metal_dispatch;

        if a_ptr == 0 || b_ptr == 0 || result_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_mul_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "mul".to_string(),
            "Metal".to_string(),
            count,
        );

        unsafe {
            mul_f32_metal_dispatch(
                a_ptr as *const f32,
                b_ptr as *const f32,
                result_ptr as *mut f32,
                count,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (a_ptr, b_ptr, result_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_div_f32(a_ptr: usize, b_ptr: usize, result_ptr: usize, count: usize) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::div_f32_metal_dispatch;

        if a_ptr == 0 || b_ptr == 0 || result_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_div_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "div".to_string(),
            "Metal".to_string(),
            count,
        );

        unsafe {
            div_f32_metal_dispatch(
                a_ptr as *const f32,
                b_ptr as *const f32,
                result_ptr as *mut f32,
                count,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (a_ptr, b_ptr, result_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}
// ============================================================================
// Metal Transpose
// ============================================================================

#[pyfunction]
fn metal_transpose_f32(in_ptr: usize, out_ptr: usize, m: usize, n: usize) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::transpose_f32_metal_dispatch;

        if in_ptr == 0 || out_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_transpose_f32",
            ));
        }

        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "transpose".to_string(),
            "Metal".to_string(),
            m * n,
        );

        unsafe {
            transpose_f32_metal_dispatch(in_ptr as *const f32, out_ptr as *mut f32, m, n);
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (in_ptr, out_ptr, m, n);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}

#[pyfunction]
fn metal_broadcast_op(
    op: i32,
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    shape: Vec<i32>,
    strides_a: Vec<i32>,
    strides_b: Vec<i32>,
    size: usize,
    size_a: usize,
    size_b: usize,
) -> PyResult<()> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use crate::ops::metal::broadcast_op;

        if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_broadcast_op",
            ));
        }

        let rank = shape.len();
        if strides_a.len() != rank || strides_b.len() != rank {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Strides rank mismatch in metal_broadcast_op",
            ));
        }

        unsafe {
            broadcast_op(
                op,
                a_ptr as *const f32,
                b_ptr as *const f32,
                out_ptr as *mut f32,
                shape.as_ptr(),
                strides_a.as_ptr(),
                strides_b.as_ptr(),
                rank as i32,
                size as i32,
                size_a as i32,
                size_b as i32,
            );
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (
            op, a_ptr, b_ptr, out_ptr, shape, strides_a, strides_b, size, size_a, size_b,
        );
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is not available in this build or on this platform",
        ))
    }
}
// ============================================================================
// CUDA GPU Operations
// ============================================================================

#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    #[cfg(feature = "cuda")]
    {
        Ok(crate::ops::cuda::is_available())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(false)
    }
}

#[pyfunction]
fn cuda_add_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    {
        use crate::ops::cuda::cuda_add_f32;

        if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to cuda_add_f32",
            ));
        }

        if count == 0 {
            return Ok(());
        }

        // PROFILING
        let _scope = crate::profiler::ProfileScope::new(
            GLOBAL_PROFILER.clone(),
            "add".to_string(),
            "CUDA".to_string(),
            count,
        );

        let success = unsafe {
            cuda_add_f32(
                a_ptr as *const f32,
                b_ptr as *const f32,
                out_ptr as *mut f32,
                count,
            )
        };

        if !success {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CUDA kernel execution failed",
            ));
        }

        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a_ptr, b_ptr, out_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA is not enabled in this build",
        ))
    }
}
