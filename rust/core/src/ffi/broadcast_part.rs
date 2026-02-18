
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
    #[cfg(target_os = "macos")]
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
                size_b as i32
            );
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (op, a_ptr, b_ptr, out_ptr, shape, strides_a, strides_b, size, size_a, size_b);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Metal is only available on macOS",
        ))
    }
}
