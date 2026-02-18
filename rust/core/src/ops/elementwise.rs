// ============================================================================
// Operations: Element-wise Kernels
// ============================================================================
// This module handles element-wise operations (add, sub, mul, div, etc.)
//
// RESPONSIBILITIES:
// - Validate operation parameters
// - Dispatch to appropriate C++ kernel or Rust fallback
// - Handle different data types and backends

// FFI declarations for C++ kernels (optional)
#[cfg(feature = "cpp_kernels")]
extern "C" {
    pub fn add_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);
    pub fn sub_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);
    pub fn mul_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);
    pub fn div_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);
}

// Rust fallback implementations
#[cfg(not(feature = "cpp_kernels"))]
#[inline]
unsafe fn add_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    for i in 0..count {
        *out.add(i) = *a.add(i) + *b.add(i);
    }
}

#[cfg(not(feature = "cpp_kernels"))]
#[inline]
unsafe fn sub_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    for i in 0..count {
        *out.add(i) = *a.add(i) - *b.add(i);
    }
}

#[cfg(not(feature = "cpp_kernels"))]
#[inline]
unsafe fn mul_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    for i in 0..count {
        *out.add(i) = *a.add(i) * *b.add(i);
    }
}

#[cfg(not(feature = "cpp_kernels"))]
#[inline]
unsafe fn div_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    for i in 0..count {
        *out.add(i) = *a.add(i) / *b.add(i);
    }
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
