// ============================================================================
// Operations: Metal GPU Backend
// ============================================================================
// This module provides Metal GPU acceleration for macOS/Apple Silicon.
// All functions are conditionally compiled only on macOS.
//
// Note: Some functions are reserved for future API exposure and may appear unused.
#![allow(dead_code, clippy::too_many_arguments)]

/// Metal FFI declarations - only available with C++ kernels on macOS
#[cfg(all(feature = "cpp_kernels", target_os = "macos"))]
mod ffi {
    extern "C" {
        pub fn metal_is_available() -> bool;
        pub fn metal_init();
        pub fn metal_cleanup();
        pub fn metal_sum_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_mean_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_max_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_min_f32(data: *const f32, size: i32) -> f32;
        pub fn metal_add(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_sub(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_mul(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_div(a: *const f32, b: *const f32, result: *mut f32, size: i32);
        pub fn metal_matmul_f32(a: *const f32, b: *const f32, c: *mut f32, m: i32, k: i32, n: i32);
        pub fn metal_transpose_f32(in_ptr: *const f32, out_ptr: *mut f32, m: i32, n: i32);
        pub fn metal_broadcast_op(
            op: i32,
            a: *const f32,
            b: *const f32,
            result: *mut f32,
            shape: *const i32,
            stridesA: *const i32,
            stridesB: *const i32,
            rank: i32,
            size: i32,
            sizeA: i32,
            sizeB: i32,
        );
    }
}

/// Rust fallbacks when C++ kernels not available OR not on macOS
#[cfg(any(not(feature = "cpp_kernels"), not(target_os = "macos")))]
mod ffi {
    pub unsafe fn metal_is_available() -> bool {
        false
    }
    pub unsafe fn metal_init() {}
    pub unsafe fn metal_cleanup() {}

    pub unsafe fn metal_sum_f32(data: *const f32, size: i32) -> f32 {
        (0..size as usize).map(|i| *data.add(i)).sum()
    }

    pub unsafe fn metal_mean_f32(data: *const f32, size: i32) -> f32 {
        if size == 0 {
            return 0.0;
        }
        metal_sum_f32(data, size) / (size as f32)
    }

    pub unsafe fn metal_max_f32(data: *const f32, size: i32) -> f32 {
        (0..size as usize)
            .map(|i| *data.add(i))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    pub unsafe fn metal_min_f32(data: *const f32, size: i32) -> f32 {
        (0..size as usize)
            .map(|i| *data.add(i))
            .fold(f32::INFINITY, f32::min)
    }

    pub unsafe fn metal_add(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
        for i in 0..size as usize {
            *result.add(i) = *a.add(i) + *b.add(i);
        }
    }

    pub unsafe fn metal_sub(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
        for i in 0..size as usize {
            *result.add(i) = *a.add(i) - *b.add(i);
        }
    }

    pub unsafe fn metal_mul(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
        for i in 0..size as usize {
            *result.add(i) = *a.add(i) * *b.add(i);
        }
    }

    pub unsafe fn metal_div(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
        for i in 0..size as usize {
            *result.add(i) = *a.add(i) / *b.add(i);
        }
    }

    pub unsafe fn metal_matmul_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        k: i32,
        n: i32,
    ) {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0;
                for p in 0..k as usize {
                    sum += *a.add(i * k as usize + p) * *b.add(p * n as usize + j);
                }
                *c.add(i * n as usize + j) = sum;
            }
        }
    }

    pub unsafe fn metal_transpose_f32(in_ptr: *const f32, out_ptr: *mut f32, m: i32, n: i32) {
        for i in 0..m as usize {
            for j in 0..n as usize {
                *out_ptr.add(j * m as usize + i) = *in_ptr.add(i * n as usize + j);
            }
        }
    }

    pub unsafe fn metal_broadcast_op(
        _op: i32,
        _a: *const f32,
        _b: *const f32,
        _result: *mut f32,
        _shape: *const i32,
        _strides_a: *const i32,
        _strides_b: *const i32,
        _rank: i32,
        _size: i32,
        _size_a: i32,
        _size_b: i32,
    ) {
        // Fallback or no-op
    }
}

// ============================================================================
// Public API
// ============================================================================

pub fn is_available() -> bool {
    unsafe { ffi::metal_is_available() }
}

pub fn init() {
    unsafe { ffi::metal_init() }
}

pub fn cleanup() {
    unsafe { ffi::metal_cleanup() }
}

pub unsafe fn sum_f32(data: *const f32, size: i32) -> f32 {
    ffi::metal_sum_f32(data, size)
}

pub unsafe fn mean_f32(data: *const f32, size: i32) -> f32 {
    ffi::metal_mean_f32(data, size)
}

pub unsafe fn max_f32(data: *const f32, size: i32) -> f32 {
    ffi::metal_max_f32(data, size)
}

pub unsafe fn min_f32(data: *const f32, size: i32) -> f32 {
    ffi::metal_min_f32(data, size)
}

pub unsafe fn add(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
    ffi::metal_add(a, b, result, size)
}

pub unsafe fn sub(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
    ffi::metal_sub(a, b, result, size)
}

pub unsafe fn mul(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
    ffi::metal_mul(a, b, result, size)
}

pub unsafe fn div(a: *const f32, b: *const f32, result: *mut f32, size: i32) {
    ffi::metal_div(a, b, result, size)
}

pub unsafe fn matmul_f32(a: *const f32, b: *const f32, c: *mut f32, m: i32, k: i32, n: i32) {
    ffi::metal_matmul_f32(a, b, c, m, k, n)
}

pub unsafe fn transpose_f32(in_ptr: *const f32, out_ptr: *mut f32, m: i32, n: i32) {
    ffi::metal_transpose_f32(in_ptr, out_ptr, m, n)
}

pub unsafe fn broadcast_op(
    op: i32,
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    shape: *const i32,
    strides_a: *const i32,
    strides_b: *const i32,
    rank: i32,
    size: i32,
    size_a: i32,
    size_b: i32,
) {
    ffi::metal_broadcast_op(
        op, a, b, result, shape, strides_a, strides_b, rank, size, size_a, size_b,
    )
}

// ============================================================================
// Dispatch Functions (for Python FFI compatibility)
// ============================================================================

pub unsafe fn sum_f32_metal_dispatch(data: *const f32, count: usize) -> f32 {
    sum_f32(data, count as i32)
}

pub unsafe fn mean_f32_metal_dispatch(data: *const f32, count: usize) -> f32 {
    mean_f32(data, count as i32)
}

pub unsafe fn max_f32_metal_dispatch(data: *const f32, count: usize) -> f32 {
    max_f32(data, count as i32)
}

pub unsafe fn min_f32_metal_dispatch(data: *const f32, count: usize) -> f32 {
    min_f32(data, count as i32)
}

pub unsafe fn add_f32_metal_dispatch(a: *const f32, b: *const f32, result: *mut f32, size: usize) {
    add(a, b, result, size as i32)
}

pub unsafe fn sub_f32_metal_dispatch(a: *const f32, b: *const f32, result: *mut f32, size: usize) {
    sub(a, b, result, size as i32)
}

pub unsafe fn mul_f32_metal_dispatch(a: *const f32, b: *const f32, result: *mut f32, size: usize) {
    mul(a, b, result, size as i32)
}

pub unsafe fn div_f32_metal_dispatch(a: *const f32, b: *const f32, result: *mut f32, size: usize) {
    div(a, b, result, size as i32)
}

pub unsafe fn matmul_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    matmul_f32(a, b, c, m as i32, k as i32, n as i32)
}

pub unsafe fn transpose_f32_metal_dispatch(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    m: usize,
    n: usize,
) {
    transpose_f32(in_ptr, out_ptr, m as i32, n as i32)
}
