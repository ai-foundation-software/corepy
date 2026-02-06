// ============================================================================
// Operations: Element-wise Kernels
// ============================================================================
// This module handles element-wise operations (add, sub, mul, div, etc.)
//
// RESPONSIBILITIES:
// - Validate operation parameters
// - Dispatch to appropriate C++ kernel
// - Handle different data types and backends

// FFI declarations for C++ kernels
extern "C" {
    // Float32 element-wise operations
    /// Element-wise addition: out[i] = a[i] + b[i]
    /// C++ signature: void add_f32_cpu(const float* a, const float* b, float* out, size_t count)
    pub fn add_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Element-wise subtraction: out[i] = a[i] - b[i]
    pub fn sub_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Element-wise multiplication: out[i] = a[i] * b[i]
    pub fn mul_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Element-wise division: out[i] = a[i] / b[i]
    pub fn div_f32_cpu(a: *const f32, b: *const f32, out: *mut f32, count: usize);
}

/// Dispatch add operation to CPU kernel
///
/// # Safety
/// Caller must ensure:
/// - a, b are valid for `count` elements
/// - out is valid for `count` elements and non-overlapping with inputs
/// - All pointers' lifetimes exceed this function call
pub unsafe fn add_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    add_f32_cpu(a, b, out, count);
}

/// Dispatch subtract operation to CPU kernel
///
/// # Safety
/// Caller must ensure:
/// - a, b are valid for `count` elements
/// - out is valid for `count` elements and non-overlapping with inputs
/// - All pointers' lifetimes exceed this function call
pub unsafe fn sub_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    sub_f32_cpu(a, b, out, count);
}

/// Dispatch multiply operation to CPU kernel
///
/// # Safety
/// See `sub_f32_cpu_dispatch`
pub unsafe fn mul_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    mul_f32_cpu(a, b, out, count);
}

/// Dispatch divide operation to CPU kernel
///
/// # Safety
/// See `sub_f32_cpu_dispatch`
pub unsafe fn div_f32_cpu_dispatch(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    div_f32_cpu(a, b, out, count);
}
