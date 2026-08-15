// ============================================================================
// Linear Algebra: Decompositions
// ============================================================================
// Wrapper over `faer` crate for advanced linear algebra operations.

use pyo3::prelude::*;

use crate::array::core_array::CoreArray;
#[cfg(feature = "faer")]
use crate::array::dtype::DType;
#[cfg(feature = "faer")]
use faer::linalg::solvers::DenseSolveCore;

#[cfg(feature = "faer")]
pub fn solve_cholesky(a: &CoreArray) -> PyResult<CoreArray> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cholesky requires a square 2D matrix",
        ));
    }

    let n = a.shape()[0];
    let mut out = CoreArray::zeros(vec![n, n], Some(DType::Float32))?;

    // Copy data to `faer::Mat` because `faer` uses its own column-major layout standard usually.
    let slice_a = unsafe { a.as_f32_slice() };

    let mut mat_a = faer::Mat::<f32>::from_fn(n, n, |i, j| slice_a[i * n + j]);

    let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f32>(
        n,
        faer::Par::Seq,
        Default::default(),
    );
    let mut mem = faer::dyn_stack::MemBuffer::new(req);
    let stack = faer::dyn_stack::MemStack::new(&mut mem);

    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        mat_a.as_mut(),
        Default::default(),
        faer::Par::Seq,
        stack,
        Default::default(),
    )
    .map_err(|_| pyo3::exceptions::PyValueError::new_err("Matrix is not positive definite"))?;

    // write back to row-major `out`, extracting lower triangle
    let slice_out = unsafe { out.as_f32_slice_mut() };
    for i in 0..n {
        for j in 0..=i {
            slice_out[i * n + j] = mat_a[(i, j)];
        }
    }

    Ok(out)
}

#[cfg(feature = "faer")]
pub fn linalg_inv(a: &CoreArray) -> PyResult<CoreArray> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Inverse requires a square 2D matrix",
        ));
    }
    let n = a.shape()[0];
    let slice_a = unsafe { a.as_f32_slice() };
    let mat_a = faer::Mat::<f32>::from_fn(n, n, |i, j| slice_a[i * n + j]);

    // Use PartialPivLu for competitive performance and stability
    let lu = mat_a.partial_piv_lu();
    let inv_mat = lu.inverse();

    let mut out_data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out_data[i * n + j] = inv_mat[(i, j)];
        }
    }
    CoreArray::new(out_data, vec![n, n])
}

#[cfg(feature = "faer")]
pub fn linalg_det(a: &CoreArray) -> PyResult<f32> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Determinant requires a square 2D matrix",
        ));
    }
    let n = a.shape()[0];
    let slice_a = unsafe { a.as_f32_slice() };
    let mat_a = faer::Mat::<f32>::from_fn(n, n, |i, j| slice_a[i * n + j]);

    Ok(mat_a.determinant())
}

#[cfg(feature = "faer")]
pub fn linalg_norm(a: &CoreArray) -> PyResult<f32> {
    let slice_a = unsafe { a.as_f32_slice() };
    // Frobenius norm: sqrt(sum of squares)
    let sum_sq: f32 = slice_a.iter().map(|&x| x * x).sum();
    Ok(sum_sq.sqrt())
}

#[cfg(not(feature = "faer"))]
pub fn solve_cholesky(_a: &CoreArray) -> PyResult<CoreArray> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "faer feature is disabled",
    ))
}

#[cfg(not(feature = "faer"))]
pub fn linalg_inv(_a: &CoreArray) -> PyResult<CoreArray> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "faer feature is disabled",
    ))
}

#[cfg(not(feature = "faer"))]
pub fn linalg_det(_a: &CoreArray) -> PyResult<f32> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "faer feature is disabled",
    ))
}

#[cfg(not(feature = "faer"))]
pub fn linalg_norm(_a: &CoreArray) -> PyResult<f32> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "faer feature is disabled",
    ))
}
