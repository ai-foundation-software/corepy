// ============================================================================
// Operations: Element-wise Kernels
// ============================================================================
// This module handles element-wise operations (add, sub, mul, div, etc.)
//
// RESPONSIBILITIES:
// - Validate operation parameters
// - Dispatch to appropriate C++ kernel or Rust fallback
// - Handle different data types and backends

use rayon::prelude::*;

// Rust implementations
#[inline]
unsafe fn add_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    let a_slice = std::slice::from_raw_parts(a, count);
    let b_slice = std::slice::from_raw_parts(b, count);
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    out_slice
        .par_iter_mut()
        .zip(a_slice.par_iter())
        .zip(b_slice.par_iter())
        .for_each(|((o, &x), &y)| *o = x + y);
}

#[inline]
unsafe fn sub_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    let a_slice = std::slice::from_raw_parts(a, count);
    let b_slice = std::slice::from_raw_parts(b, count);
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    out_slice
        .par_iter_mut()
        .zip(a_slice.par_iter())
        .zip(b_slice.par_iter())
        .for_each(|((o, &x), &y)| *o = x - y);
}

#[inline]
unsafe fn mul_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    let a_slice = std::slice::from_raw_parts(a, count);
    let b_slice = std::slice::from_raw_parts(b, count);
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    out_slice
        .par_iter_mut()
        .zip(a_slice.par_iter())
        .zip(b_slice.par_iter())
        .for_each(|((o, &x), &y)| *o = x * y);
}

#[inline]
unsafe fn div_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    let a_slice = std::slice::from_raw_parts(a, count);
    let b_slice = std::slice::from_raw_parts(b, count);
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    out_slice
        .par_iter_mut()
        .zip(a_slice.par_iter())
        .zip(b_slice.par_iter())
        .for_each(|((o, &x), &y)| *o = x / y);
}

/// Dispatch add operation to CPU kernel
pub unsafe fn add_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    add_f32_cpu(a, b, out, count);
}

/// Dispatch subtract operation to CPU kernel
pub unsafe fn sub_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    sub_f32_cpu(a, b, out, count);
}

/// Dispatch multiply operation to CPU kernel
pub unsafe fn mul_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    mul_f32_cpu(a, b, out, count);
}

/// Dispatch divide operation to CPU kernel
pub unsafe fn div_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    div_f32_cpu(a, b, out, count);
}
