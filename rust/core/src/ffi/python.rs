// ============================================================================
// FFI: Python ↔ Rust Bridge
// ============================================================================
// This module handles PyO3 integration and exports Rust functions to Python.

use pyo3::prelude::*;

// Global profiler instance for this module (and the process)
lazy_static::lazy_static! {
    static ref GLOBAL_PROFILER: crate::profiler::Profiler = crate::profiler::Profiler::new();
}

/// Export all FFI functions to Python
pub fn register_functions(m: &PyModule) -> PyResult<()> {
    // Reduction operations (contiguous)
    m.add_function(wrap_pyfunction!(tensor_all, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_any, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_sum_i32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_matmul_2d_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_dot_product_f32, m)?)?;

    // Strided reduction operations (zero-copy for non-contiguous arrays)
    m.add_function(wrap_pyfunction!(tensor_sum_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_mean_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_max_f32_strided, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_min_f32_strided, m)?)?;

    // Backend control
    m.add_function(wrap_pyfunction!(set_backend_policy, m)?)?;
    m.add_function(wrap_pyfunction!(get_backend_policy, m)?)?;
    m.add_function(wrap_pyfunction!(explain_last_dispatch, m)?)?;

    // Element-wise operations
    m.add_function(wrap_pyfunction!(tensor_add_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_sub_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_mul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_div_f32, m)?)?;

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
    m.add_function(wrap_pyfunction!(metal_matmul_f32, m)?)?;

    // Demo functions (backward compatibility)
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

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
fn get_profile_report(context: Option<String>) -> PyResult<String> {
    GLOBAL_PROFILER
        .export_json(context.as_deref())
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
fn set_profile_context(context: Option<String>) -> PyResult<()> {
    crate::profiler::set_context(context);
    Ok(())
}

// ============================================================================
// Reduction Operations
// ============================================================================

#[pyfunction]
fn tensor_all(data_ptr: usize, count: usize) -> PyResult<bool> {
    use crate::ops::reduce::all_bool_cpu_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_all",
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

    let result = unsafe { all_bool_cpu_dispatch(data_ptr as *const u8, count) };

    Ok(result)
}

#[pyfunction]
fn tensor_any(data_ptr: usize, count: usize) -> PyResult<bool> {
    use crate::ops::reduce::any_bool_cpu_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_any",
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

    let result = unsafe { any_bool_cpu_dispatch(data_ptr as *const u8, count) };

    Ok(result)
}

#[pyfunction]
fn tensor_sum_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    use crate::ops::reduce::sum_f32_cpu_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_sum_f32",
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

    let result = unsafe { sum_f32_cpu_dispatch(data_ptr as *const f32, count) };

    Ok(result)
}

#[pyfunction]
fn tensor_sum_i32(data_ptr: usize, count: usize) -> PyResult<i32> {
    use crate::ops::reduce::sum_i32_cpu_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_sum_i32",
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

    let result = unsafe { sum_i32_cpu_dispatch(data_ptr as *const i32, count) };

    Ok(result)
}

#[pyfunction]
fn tensor_mean_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    use crate::ops::reduce::mean_f32_cpu_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_mean_f32",
        ));
    }

    if count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute mean of empty tensor",
        ));
    }

    // PROFILING
    let _scope = crate::profiler::ProfileScope::new(
        GLOBAL_PROFILER.clone(),
        "mean".to_string(),
        "CPU".to_string(),
        count,
    );

    let result = unsafe { mean_f32_cpu_dispatch(data_ptr as *const f32, count) };

    Ok(result)
}

// ============================================================================
// Strided Reduction Operations (Zero-Copy)
// ============================================================================

#[pyfunction]
fn tensor_sum_f32_strided(
    data_ptr: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> PyResult<f32> {
    use crate::ops::reduce::sum_f32_strided_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_sum_f32_strided",
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

    let result = unsafe { sum_f32_strided_dispatch(data_ptr as *const f32, &shape, &strides) };

    Ok(result)
}

#[pyfunction]
fn tensor_mean_f32_strided(
    data_ptr: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> PyResult<f32> {
    use crate::ops::reduce::mean_f32_strided_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_mean_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute mean of empty tensor",
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

    let result = unsafe { mean_f32_strided_dispatch(data_ptr as *const f32, &shape, &strides) };

    Ok(result)
}

#[pyfunction]
fn tensor_max_f32_strided(
    data_ptr: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> PyResult<f32> {
    use crate::ops::reduce::max_f32_strided_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_max_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute max of empty tensor",
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

    let result = unsafe { max_f32_strided_dispatch(data_ptr as *const f32, &shape, &strides) };

    Ok(result)
}

#[pyfunction]
fn tensor_min_f32_strided(
    data_ptr: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> PyResult<f32> {
    use crate::ops::reduce::min_f32_strided_dispatch;

    if data_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_min_f32_strided",
        ));
    }

    if shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute min of empty tensor",
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

    let result = unsafe { min_f32_strided_dispatch(data_ptr as *const f32, &shape, &strides) };

    Ok(result)
}

#[pyfunction]
fn tensor_dot_product_f32(a_ptr: usize, b_ptr: usize, count: usize) -> PyResult<f32> {
    use crate::ops::matmul::dot_product_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_dot_product_f32",
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

    let result =
        unsafe { dot_product_f32_cpu_dispatch(a_ptr as *const f32, b_ptr as *const f32, count) };

    Ok(result)
}

#[pyfunction]
fn tensor_matmul_2d_f32(
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<()> {
    use crate::ops::matmul::matmul_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_matmul_2d_f32",
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
        matmul_f32_cpu_dispatch(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            m,
            k,
            n,
        );
    }

    Ok(())
}

#[pyfunction]
fn tensor_matmul_f32(a_ptr: usize, b_ptr: usize, count: usize) -> PyResult<f32> {
    // Legacy/Existing wrapper that calls the same kernel
    tensor_dot_product_f32(a_ptr, b_ptr, count)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

#[pyfunction]
fn tensor_add_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    use crate::ops::elementwise::add_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_add_f32",
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
        add_f32_cpu_dispatch(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        );
    }

    Ok(())
}

#[pyfunction]
fn tensor_sub_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    use crate::ops::elementwise::sub_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_sub_f32",
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
        sub_f32_cpu_dispatch(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        );
    }

    Ok(())
}

#[pyfunction]
fn tensor_mul_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    use crate::ops::elementwise::mul_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_mul_f32",
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
        mul_f32_cpu_dispatch(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        );
    }

    Ok(())
}

#[pyfunction]
fn tensor_div_f32(a_ptr: usize, b_ptr: usize, out_ptr: usize, count: usize) -> PyResult<()> {
    use crate::ops::elementwise::div_f32_cpu_dispatch;

    if a_ptr == 0 || b_ptr == 0 || out_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Null pointer passed to tensor_div_f32",
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
        div_f32_cpu_dispatch(
            a_ptr as *const f32,
            b_ptr as *const f32,
            out_ptr as *mut f32,
            count,
        );
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

// ============================================================================
// Demo Functions (Backward Compatibility)
// ============================================================================

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// ============================================================================
// Metal GPU Operations
// ============================================================================

#[pyfunction]
fn metal_is_available() -> PyResult<bool> {
    Ok(crate::ops::metal::is_available())
}

#[pyfunction]
fn metal_sum_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(target_os = "macos")]
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

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is only available on macOS",
        ))
    }
}

#[pyfunction]
fn metal_mean_f32(data_ptr: usize, count: usize) -> PyResult<f32> {
    #[cfg(target_os = "macos")]
    {
        use crate::ops::metal::mean_f32_metal_dispatch;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null pointer passed to metal_mean_f32",
            ));
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot compute mean of empty tensor",
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

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (data_ptr, count);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is only available on macOS",
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
    #[cfg(target_os = "macos")]
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

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a_ptr, b_ptr, c_ptr, m, k, n);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is only available on macOS",
        ))
    }
}
