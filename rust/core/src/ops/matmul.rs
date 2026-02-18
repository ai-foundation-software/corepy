// ============================================================================
// Operations: Matrix Multiplication
// ============================================================================

// FFI declaration for C++ kernel
// FFI declaration for C++ kernel
extern "C" {
    /// Set number of threads for the backend
    #[allow(dead_code)]
    pub fn corepy_set_num_threads(num_threads: i32);
}

/// Safety wrapper for pointers to be Send/Sync for Rayon
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    fn ptr(&self) -> *const T {
        self.0
    }
}

struct SendPtrMut<T>(*mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}
impl<T> SendPtrMut<T> {
    #[inline]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

/// Scalar fallback for dot product
#[allow(dead_code)]
unsafe fn dot_product_f32_scalar(a: *const f32, b: *const f32, count: usize) -> f32 {
    let mut sum = 0.0;
    for i in 0..count {
        sum += *a.add(i) * *b.add(i);
    }
    sum
}

/// AVX2 implementation of dot product
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_f32_avx2(a: *const f32, b: *const f32, count: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        i += 8;
    }

    // Horizontal sum of the vector
    // Extract upper 128 bits
    let sum_high = _mm256_extractf128_ps(sum_vec, 1);
    let sum_low = _mm256_castps256_ps128(sum_vec);
    let sum128 = _mm_add_ps(sum_low, sum_high);

    // Horizontal sum of 4 floats
    let shuf = _mm_movehdup_ps(sum128); // (2, 3, 2, 3)
    let sums = _mm_add_ps(sum128, shuf); // (1+2, 3+3, ...)
    let shuf = _mm_movehl_ps(shuf, sums); // (3+3, ...)
    let sums = _mm_add_ss(sums, shuf); // (1+2 + 3+3)

    let mut sum = _mm_cvtss_f32(sums);

    // Handle remaining elements scalar
    while i < count {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }

    sum
}

/// NEON implementation of dot product (for ARM64/Apple Silicon)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_f32_neon(a: *const f32, b: *const f32, count: usize) -> f32 {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 elements at a time (NEON register is 128-bit = 4x f32)
    // Loop unrolled 4x (16 elements per iter) for pipeline efficiency
    while i + 16 <= count {
        let a_ptr = a.add(i);
        let b_ptr = b.add(i);

        let va0 = vld1q_f32(a_ptr);
        let vb0 = vld1q_f32(b_ptr);
        sum_vec = vfmaq_f32(sum_vec, va0, vb0);

        let va1 = vld1q_f32(a_ptr.add(4));
        let vb1 = vld1q_f32(b_ptr.add(4));
        sum_vec = vfmaq_f32(sum_vec, va1, vb1);

        let va2 = vld1q_f32(a_ptr.add(8));
        let vb2 = vld1q_f32(b_ptr.add(8));
        sum_vec = vfmaq_f32(sum_vec, va2, vb2);

        let va3 = vld1q_f32(a_ptr.add(12));
        let vb3 = vld1q_f32(b_ptr.add(12));
        sum_vec = vfmaq_f32(sum_vec, va3, vb3);

        i += 16;
    }

    // Handle remaining chunks of 4
    while i + 4 <= count {
        let va = vld1q_f32(a.add(i));
        let vb = vld1q_f32(b.add(i));
        sum_vec = vfmaq_f32(sum_vec, va, vb);
        i += 4;
    }

    // Horizontal sum across vector
    let mut sum = vaddvq_f32(sum_vec);

    // Handle scalar remainder
    while i < count {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }

    sum
}

/// Dispatch dot product operation to CPU kernel
pub unsafe fn dot_product_f32_cpu_dispatch(a: *const f32, b: *const f32, count: usize) -> f32 {
    use crate::scheduler::arena::with_arena;
    with_arena(|_arena| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                dot_product_f32_avx2(a, b, count)
            } else {
                dot_product_f32_scalar(a, b, count)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 always has NEON
            dot_product_f32_neon(a, b, count)
        }

        // Scalar fallback for other architectures
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            dot_product_f32_scalar(a, b, count)
        }
    })
}

// ============================================================================
// Native Matrix Multiplication Kernels
// ============================================================================

/// Scalar MatMul fallback
#[allow(dead_code)]
#[inline(always)]
unsafe fn kernel_matmul_f32_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += *a.add(i * k + l) * *b.add(l * n + j);
            }
            *c.add(i * n + j) = sum;
        }
    }
}

/// AVX2 MatMul with 4x8 Register Tiling (High Performance IJK)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn kernel_matmul_f32_avx2(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j + 8 <= n {
            // Accumulators in registers
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();

            for l in 0..k {
                // Load B vec (8 elements) once
                let vb = _mm256_loadu_ps(b.add(l * n + j));

                // Broadcast A elements for each row
                s0 = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(i * k + l)), vb, s0);
                s1 = _mm256_fmadd_ps(_mm256_set1_ps(*a.add((i + 1) * k + l)), vb, s1);
                s2 = _mm256_fmadd_ps(_mm256_set1_ps(*a.add((i + 2) * k + l)), vb, s2);
                s3 = _mm256_fmadd_ps(_mm256_set1_ps(*a.add((i + 3) * k + l)), vb, s3);
            }

            // Store results (overwrite C)
            _mm256_storeu_ps(c.add(i * n + j), s0);
            _mm256_storeu_ps(c.add((i + 1) * n + j), s1);
            _mm256_storeu_ps(c.add((i + 2) * n + j), s2);
            _mm256_storeu_ps(c.add((i + 3) * n + j), s3);

            j += 8;
        }

        // Handle remaining columns for these 4 rows
        while j < n {
            for row_offset in 0..4 {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += *a.add((i + row_offset) * k + l) * *b.add(l * n + j);
                }
                *c.add((i + row_offset) * n + j) = sum;
            }
            j += 1;
        }
        i += 4;
    }

    // Remaining rows
    while i < m {
        let mut j = 0;
        while j + 8 <= n {
            let mut sum_vec = _mm256_setzero_ps();
            for l in 0..k {
                sum_vec = _mm256_fmadd_ps(
                    _mm256_set1_ps(*a.add(i * k + l)),
                    _mm256_loadu_ps(b.add(l * n + j)),
                    sum_vec,
                );
            }
            _mm256_storeu_ps(c.add(i * n + j), sum_vec);
            j += 8;
        }
        while j < n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += *a.add(i * k + l) * *b.add(l * n + j);
            }
            *c.add(i * n + j) = sum;
            j += 1;
        }
        i += 1;
    }
}

/// NEON MatMul with 4x4 Register Tiling
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kernel_matmul_f32_neon(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j + 4 <= n {
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);

            for l in 0..k {
                let vb = vld1q_f32(b.add(l * n + j));
                s0 = vfmaq_f32(s0, vdupq_n_f32(*a.add(i * k + l)), vb);
                s1 = vfmaq_f32(s1, vdupq_n_f32(*a.add((i + 1) * k + l)), vb);
                s2 = vfmaq_f32(s2, vdupq_n_f32(*a.add((i + 2) * k + l)), vb);
                s3 = vfmaq_f32(s3, vdupq_n_f32(*a.add((i + 3) * k + l)), vb);
            }

            vst1q_f32(c.add(i * n + j), s0);
            vst1q_f32(c.add((i + 1) * n + j), s1);
            vst1q_f32(c.add((i + 2) * n + j), s2);
            vst1q_f32(c.add((i + 3) * n + j), s3);
            j += 4;
        }
        while j < n {
            for row_offset in 0..4 {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += *a.add((i + row_offset) * k + l) * *b.add(l * n + j);
                }
                *c.add((i + row_offset) * n + j) = sum;
            }
            j += 1;
        }
        i += 4;
    }

    while i < m {
        let mut j = 0;
        while j + 4 <= n {
            let mut sum_vec = vdupq_n_f32(0.0);
            for l in 0..k {
                sum_vec = vfmaq_f32(
                    sum_vec,
                    vdupq_n_f32(*a.add(i * k + l)),
                    vld1q_f32(b.add(l * n + j)),
                );
            }
            vst1q_f32(c.add(i * n + j), sum_vec);
            j += 4;
        }
        while j < n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += *a.add(i * k + l) * *b.add(l * n + j);
            }
            *c.add(i * n + j) = sum;
            j += 1;
        }
        i += 1;
    }
}

/// OpenBLAS MatMul using cblas-sys
unsafe fn kernel_matmul_f32_openblas(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use cblas_sys::{cblas_sgemm, CblasNoTrans, CblasRowMajor};

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m as i32,
        n as i32,
        k as i32,
        1.0,
        a,
        k as i32,
        b,
        n as i32,
        0.0,
        c,
        n as i32,
    );
}

/// Internal CPU kernel selector
#[inline(always)]
unsafe fn matmul_f32_cpu_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // Note: On macOS with Apple Silicon, we use Accelerate framework which
    // automatically leverages AMX (Apple Matrix eXtension) coprocessor.
    // AMX is significantly faster than NEON, so we use it more aggressively.

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // AVX-512 path: wider SIMD is competitive with OpenBLAS for larger sizes
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            // AVX-512 can handle larger matrices before OpenBLAS becomes faster
            if m * n * k <= 256 * 256 * 256 {
                // Note: Using AVX2 kernel as we don't have dedicated AVX-512 kernel yet
                // The AVX2 kernel benefits from wider registers via compiler auto-vectorization
                return kernel_matmul_f32_avx2(a, b, c, m, k, n);
            }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // For small/medium sizes, use our native kernels to avoid FFI overhead.
            // Up to 128x128x128 (2M ops) native is competitive and avoids FFI sync.
            if m * n * k <= 128 * 128 * 128 {
                return kernel_matmul_f32_avx2(a, b, c, m, k, n);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // On macOS with Accelerate (AMX), prefer BLAS even for smaller matrices
        // because AMX is ~2x faster than NEON for matrix operations
        #[cfg(use_accelerate)]
        {
            // Use NEON only for very small matrices where overhead dominates
            // AMX is beneficial starting from ~32x32 matrices
            if m * n * k <= 32 * 32 * 32 {
                return kernel_matmul_f32_neon(a, b, c, m, k, n);
            }
        }

        // On non-macOS ARM64 (Linux, etc) or when Accelerate is not available,
        // use NEON for small/medium sizes
        #[cfg(not(use_accelerate))]
        {
            if m * n * k <= 128 * 128 * 128 {
                return kernel_matmul_f32_neon(a, b, c, m, k, n);
            }
        }
    }

    // Fall back to BLAS (OpenBLAS or Accelerate depending on platform)
    kernel_matmul_f32_openblas(a, b, c, m, k, n);
}

/// Dispatch 2D matrix multiplication to CPU kernel with Parallel Thresholding
pub unsafe fn matmul_f32_cpu_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use crate::backend::{get_policy, record_detailed_dispatch, record_dispatch};
    use crate::scheduler::arena::with_arena;

    let policy = get_policy();
    record_dispatch(0); // Corepy ID
    record_detailed_dispatch(0, "matmul", m, n, k, policy);

    // Parallel Thresholding
    let total_ops = m * n * k;

    // If it's a large matrix, use OpenBLAS directly.
    // 256x256x256 = 16.7M ops.
    if total_ops >= 256 * 256 * 256 {
        kernel_matmul_f32_openblas(a, b, c, m, k, n);
        return;
    }

    // For medium/small matrices, use native SIMD kernels.
    // Parallel Threshold: ~2M ops.
    let parallel_threshold = 2_000_000;

    if total_ops < parallel_threshold {
        matmul_f32_cpu_kernel(a, b, c, m, k, n);
        return;
    }

    use rayon::prelude::*;
    let a_wrap = SendPtr(a);
    let b_wrap = SendPtr(b);
    let c_wrap = SendPtrMut(c);

    with_arena(|_arena| {
        let num_threads = num_cpus::get();
        let rows_per_thread = m.div_ceil(num_threads);
        let rows_per_thread = if rows_per_thread == 0 {
            1
        } else {
            rows_per_thread
        };

        (0..m)
            .into_par_iter()
            .chunks(rows_per_thread)
            .for_each(move |row_indices| {
                let start_row = row_indices[0];
                let num_rows = row_indices.len();

                let a_ptr = a_wrap.ptr().add(start_row * k);
                let b_ptr = b_wrap.ptr();
                let c_ptr = c_wrap.ptr().add(start_row * n);

                unsafe {
                    matmul_f32_cpu_kernel(a_ptr, b_ptr, c_ptr, num_rows, k, n);
                }
            });
    });
}
