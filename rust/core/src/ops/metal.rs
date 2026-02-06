// ============================================================================
// Operations: Metal GPU Backend
// ============================================================================
// This module provides Metal GPU acceleration for macOS/Apple Silicon.
// All functions are conditionally compiled only on macOS.

/// Metal FFI declarations - only available on macOS
#[cfg(target_os = "macos")]
mod ffi {
    extern "C" {
        pub fn metal_is_available() -> bool;
        pub fn metal_init();
        pub fn metal_cleanup();

        // Reduction operations
        pub fn metal_sum_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_mean_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_max_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_min_f32(data: *const f32, size: i32) -> f32;

        // Element-wise operations
        pub fn metal_add(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_sub(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_mul(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_div(a: *const f32, b: *const f32, result: *mut f32, size: i32);

        // Matrix multiplication
        pub fn metal_matmul_f32(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            k: i32,
            n: i32,
        );
    }
}

// ============================================================================
// Public API (macOS only)
// ============================================================================

/// Check if Metal is available on this system
#[cfg(target_os = "macos")]
pub fn is_available() -> bool {
    unsafe { ffi::metal_is_available() }
}

#[cfg(not(target_os = "macos"))]
pub fn is_available() -> bool {
    false
}

/// Initialize Metal (call once before using Metal functions)
#[cfg(target_os = "macos")]
pub fn init() {
    unsafe { ffi::metal_init() }
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub fn init() {}

/// Cleanup Metal resources
#[cfg(target_os = "macos")]
pub fn cleanup() {
    unsafe { ffi::metal_cleanup() }
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub fn cleanup() {}

// ============================================================================
// Reduction Operations
// ============================================================================

#[cfg(target_os = "macos")]
pub unsafe fn sum_f32_metal_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    ffi::metal_sum_f32(data_ptr, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn mean_f32_metal_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    ffi::metal_mean_f32(data_ptr, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn max_f32_metal_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    ffi::metal_max_f32(data_ptr, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn min_f32_metal_dispatch(data_ptr: *const f32, count: usize) -> f32 {
    ffi::metal_min_f32(data_ptr, count as i32)
}

// ============================================================================
// Element-wise Operations
// ============================================================================

#[cfg(target_os = "macos")]
pub unsafe fn add_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) {
    ffi::metal_add(a, b, result, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn sub_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) {
    ffi::metal_sub(a, b, result, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn mul_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) {
    ffi::metal_mul(a, b, result, count as i32)
}

#[cfg(target_os = "macos")]
pub unsafe fn div_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    count: usize,
) {
    ffi::metal_div(a, b, result, count as i32)
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

#[cfg(target_os = "macos")]
pub unsafe fn matmul_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    ffi::metal_matmul_f32(a, b, c, m as i32, k as i32, n as i32)
}
