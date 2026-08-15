// ============================================================================
// Operations: Matrix Multiplication
// ============================================================================

// FFI declaration for C++ kernel

/// Safety wrapper for pointers to be Send/Sync for Rayon
#[allow(dead_code)]
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    #[allow(dead_code)]
    fn ptr(&self) -> *const T {
        self.0
    }
}

#[allow(dead_code)]
struct SendPtrMut<T>(*mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}
impl<T> SendPtrMut<T> {
    #[inline]
    #[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Unblocked 6x16 AVX2 micro-kernel for medium matrices
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn kernel_matmul_f32_avx2_simple(
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
    while i + 6 <= m {
        let mut j = 0;
        while j + 16 <= n {
            let mut s00 = _mm256_setzero_ps();
            let mut s01 = _mm256_setzero_ps();
            let mut s10 = _mm256_setzero_ps();
            let mut s11 = _mm256_setzero_ps();
            let mut s20 = _mm256_setzero_ps();
            let mut s21 = _mm256_setzero_ps();
            let mut s30 = _mm256_setzero_ps();
            let mut s31 = _mm256_setzero_ps();
            let mut s40 = _mm256_setzero_ps();
            let mut s41 = _mm256_setzero_ps();
            let mut s50 = _mm256_setzero_ps();
            let mut s51 = _mm256_setzero_ps();

            for l in 0..k {
                let vb0 = _mm256_loadu_ps(b.add(l * n + j));
                let vb1 = _mm256_loadu_ps(b.add(l * n + j + 8));
                let va0 = _mm256_set1_ps(*a.add(i * k + l));
                let va1 = _mm256_set1_ps(*a.add((i + 1) * k + l));
                let va2 = _mm256_set1_ps(*a.add((i + 2) * k + l));
                let va3 = _mm256_set1_ps(*a.add((i + 3) * k + l));
                let va4 = _mm256_set1_ps(*a.add((i + 4) * k + l));
                let va5 = _mm256_set1_ps(*a.add((i + 5) * k + l));

                s00 = _mm256_fmadd_ps(va0, vb0, s00);
                s01 = _mm256_fmadd_ps(va0, vb1, s01);
                s10 = _mm256_fmadd_ps(va1, vb0, s10);
                s11 = _mm256_fmadd_ps(va1, vb1, s11);
                s20 = _mm256_fmadd_ps(va2, vb0, s20);
                s21 = _mm256_fmadd_ps(va2, vb1, s21);
                s30 = _mm256_fmadd_ps(va3, vb0, s30);
                s31 = _mm256_fmadd_ps(va3, vb1, s31);
                s40 = _mm256_fmadd_ps(va4, vb0, s40);
                s41 = _mm256_fmadd_ps(va4, vb1, s41);
                s50 = _mm256_fmadd_ps(va5, vb0, s50);
                s51 = _mm256_fmadd_ps(va5, vb1, s51);
            }

            _mm256_storeu_ps(c.add(i * n + j), s00);
            _mm256_storeu_ps(c.add(i * n + j + 8), s01);
            _mm256_storeu_ps(c.add((i + 1) * n + j), s10);
            _mm256_storeu_ps(c.add((i + 1) * n + j + 8), s11);
            _mm256_storeu_ps(c.add((i + 2) * n + j), s20);
            _mm256_storeu_ps(c.add((i + 2) * n + j + 8), s21);
            _mm256_storeu_ps(c.add((i + 3) * n + j), s30);
            _mm256_storeu_ps(c.add((i + 3) * n + j + 8), s31);
            _mm256_storeu_ps(c.add((i + 4) * n + j), s40);
            _mm256_storeu_ps(c.add((i + 4) * n + j + 8), s41);
            _mm256_storeu_ps(c.add((i + 5) * n + j), s50);
            _mm256_storeu_ps(c.add((i + 5) * n + j + 8), s51);
            j += 16;
        }
        while j < n {
            for r in 0..6 {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += *a.add((i + r) * k + l) * *b.add(l * n + j);
                }
                *c.add((i + r) * n + j) = sum;
            }
            j += 1;
        }
        i += 6;
    }
    while i < m {
        let mut j = 0;
        while j + 8 <= n {
            let mut s = _mm256_setzero_ps();
            for l in 0..k {
                s = _mm256_fmadd_ps(
                    _mm256_set1_ps(*a.add(i * k + l)),
                    _mm256_loadu_ps(b.add(l * n + j)),
                    s,
                );
            }
            _mm256_storeu_ps(c.add(i * n + j), s);
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

/// AVX2 MatMul with Cache Blocking and 6x16 Register Tiling
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(dead_code)]
unsafe fn kernel_matmul_f32_avx2_blocked(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k_dim: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let l2_size = crate::backend::capabilities::get_capabilities()
        .cpu
        .l2_cache_size;
    let tile_size = ((l2_size as f64) / (3.0 * 4.0)).sqrt() as usize; // 4 is sizeof(T)

    // Register tiling factors: AVX2 uses 6x16
    let mc_block = std::cmp::max(72, (tile_size / 6) * 6);
    let kc_block = std::cmp::max(256, tile_size);

    for ic in (0..m).step_by(mc_block) {
        let mc = std::cmp::min(m - ic, mc_block);
        for lc in (0..k_dim).step_by(kc_block) {
            let kc = std::cmp::min(k_dim - lc, kc_block);

            let mut i = 0;
            while i + 6 <= mc {
                let mut j = 0;
                while j + 16 <= n {
                    let mut s00 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i) * n + j))
                    };
                    let mut s01 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i) * n + j + 8))
                    };
                    let mut s10 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 1) * n + j))
                    };
                    let mut s11 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 1) * n + j + 8))
                    };
                    let mut s20 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 2) * n + j))
                    };
                    let mut s21 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 2) * n + j + 8))
                    };
                    let mut s30 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 3) * n + j))
                    };
                    let mut s31 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 3) * n + j + 8))
                    };
                    let mut s40 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 4) * n + j))
                    };
                    let mut s41 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 4) * n + j + 8))
                    };
                    let mut s50 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 5) * n + j))
                    };
                    let mut s51 = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i + 5) * n + j + 8))
                    };

                    for l in 0..kc {
                        let vb0 = _mm256_loadu_ps(b.add((lc + l) * n + j));
                        let vb1 = _mm256_loadu_ps(b.add((lc + l) * n + j + 8));
                        let va0 = _mm256_set1_ps(*a.add((ic + i) * k_dim + (lc + l)));
                        let va1 = _mm256_set1_ps(*a.add((ic + i + 1) * k_dim + (lc + l)));
                        let va2 = _mm256_set1_ps(*a.add((ic + i + 2) * k_dim + (lc + l)));
                        let va3 = _mm256_set1_ps(*a.add((ic + i + 3) * k_dim + (lc + l)));
                        let va4 = _mm256_set1_ps(*a.add((ic + i + 4) * k_dim + (lc + l)));
                        let va5 = _mm256_set1_ps(*a.add((ic + i + 5) * k_dim + (lc + l)));
                        s00 = _mm256_fmadd_ps(va0, vb0, s00);
                        s01 = _mm256_fmadd_ps(va0, vb1, s01);
                        s10 = _mm256_fmadd_ps(va1, vb0, s10);
                        s11 = _mm256_fmadd_ps(va1, vb1, s11);
                        s20 = _mm256_fmadd_ps(va2, vb0, s20);
                        s21 = _mm256_fmadd_ps(va2, vb1, s21);
                        s30 = _mm256_fmadd_ps(va3, vb0, s30);
                        s31 = _mm256_fmadd_ps(va3, vb1, s31);
                        s40 = _mm256_fmadd_ps(va4, vb0, s40);
                        s41 = _mm256_fmadd_ps(va4, vb1, s41);
                        s50 = _mm256_fmadd_ps(va5, vb0, s50);
                        s51 = _mm256_fmadd_ps(va5, vb1, s51);
                    }

                    _mm256_storeu_ps(c.add((ic + i) * n + j), s00);
                    _mm256_storeu_ps(c.add((ic + i) * n + j + 8), s01);
                    _mm256_storeu_ps(c.add((ic + i + 1) * n + j), s10);
                    _mm256_storeu_ps(c.add((ic + i + 1) * n + j + 8), s11);
                    _mm256_storeu_ps(c.add((ic + i + 2) * n + j), s20);
                    _mm256_storeu_ps(c.add((ic + i + 2) * n + j + 8), s21);
                    _mm256_storeu_ps(c.add((ic + i + 3) * n + j), s30);
                    _mm256_storeu_ps(c.add((ic + i + 3) * n + j + 8), s31);
                    _mm256_storeu_ps(c.add((ic + i + 4) * n + j), s40);
                    _mm256_storeu_ps(c.add((ic + i + 4) * n + j + 8), s41);
                    _mm256_storeu_ps(c.add((ic + i + 5) * n + j), s50);
                    _mm256_storeu_ps(c.add((ic + i + 5) * n + j + 8), s51);
                    j += 16;
                }
                while j < n {
                    for r in 0..6 {
                        let mut sum = if lc == 0 {
                            0.0
                        } else {
                            *c.add((ic + i + r) * n + j)
                        };
                        for l in 0..kc {
                            sum +=
                                *a.add((ic + i + r) * k_dim + (lc + l)) * *b.add((lc + l) * n + j);
                        }
                        *c.add((ic + i + r) * n + j) = sum;
                    }
                    j += 1;
                }
                i += 6;
            }
            while i < mc {
                let mut j = 0;
                while j + 8 <= n {
                    let mut sum_vec = if lc == 0 {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c.add((ic + i) * n + j))
                    };
                    for l in 0..kc {
                        sum_vec = _mm256_fmadd_ps(
                            _mm256_set1_ps(*a.add((ic + i) * k_dim + (lc + l))),
                            _mm256_loadu_ps(b.add((lc + l) * n + j)),
                            sum_vec,
                        );
                    }
                    _mm256_storeu_ps(c.add((ic + i) * n + j), sum_vec);
                    j += 8;
                }
                while j < n {
                    let mut sum = if lc == 0 {
                        0.0
                    } else {
                        *c.add((ic + i) * n + j)
                    };
                    for l in 0..kc {
                        sum += *a.add((ic + i) * k_dim + (lc + l)) * *b.add((lc + l) * n + j);
                    }
                    *c.add((ic + i) * n + j) = sum;
                    j += 1;
                }
                i += 1;
            }
        }
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
#[cfg(has_blas)]
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

/// Fallback MatMul if no BLAS was compiled into the extensions
#[cfg(not(has_blas))]
unsafe fn kernel_matmul_f32_openblas(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let matrix_size = m.max(n).max(k);
    let plan = crate::backend::thread_policy::compute_thread_plan(
        matrix_size,
        crate::backend::selector_v2::MathBackend::RustParallel,
    );
    crate::ops::matmul_rust_parallel::matmul_f32_rust_parallel(a, b, c, m, k, n, plan.count);
}

/// Internal CPU kernel selector
#[inline(always)]
#[allow(dead_code)]
unsafe fn matmul_f32_cpu_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Use simple kernel for matrices that fit in L2/L3 without blocking
            if m * n * k <= 16_777_216 {
                // 256x256x256
                return kernel_matmul_f32_avx2_simple(a, b, c, m, k, n);
            }
            return kernel_matmul_f32_avx2_blocked(a, b, c, m, k, n);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON for all small/medium sizes on ARM
        if m * n * k <= 1024 * 1024 * 1024 {
            return kernel_matmul_f32_neon(a, b, c, m, k, n);
        }
    }

    // Fall back to BLAS
    kernel_matmul_f32_openblas(a, b, c, m, k, n);
}

/// Dispatch 2D matrix multiplication to CPU kernel with Vendor-Aware Thread Policy
///
/// Dispatch priority:
///   1. SIMD ultra-fast path (tiny matrices, < SERIAL_OPS_THRESHOLD)
///   2. Explicit policy set by user (Mkl / Aocl / RustParallel / Openblas / Blas)
///   3. Auto-selection via selector_v2 (MKL → AOCL → Accelerate → OpenBLAS → Rust)
///      with thread count from thread_policy and env vars from env_setter.
pub unsafe fn matmul_f32_cpu_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use crate::backend::{
        env_setter, get_math_backend, get_policy, record_detailed_dispatch, record_dispatch,
        selector_v2::MathBackend, thread_policy::compute_thread_plan, BackendPolicy,
    };

    let total_ops = m * n * k;
    // Matrix side used for thread policy (largest dimension drives cache pressure)
    let matrix_size = m.max(n).max(k);

    // ── Ultra-fast SIMD path for small matrices ( <= 256^3 ops ) ───────────────
    // Bypasses all policy/env logic to minimise overhead for matrices up to 256x256.
    if total_ops <= 16_777_216 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                record_dispatch(0);
                if total_ops <= 2_097_152 {
                    kernel_matmul_f32_avx2_simple(a, b, c, m, k, n);
                } else {
                    kernel_matmul_f32_avx2_blocked(a, b, c, m, k, n);
                }
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            record_dispatch(0);
            kernel_matmul_f32_neon(a, b, c, m, k, n);
            return;
        }
        // Scalar fallback for other arches
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            record_dispatch(0);
            kernel_matmul_f32_scalar(a, b, c, m, k, n);
            return;
        }
    }

    let policy = get_policy();

    // ── Explicit policy override ─────────────────────────────────────────────
    match policy {
        BackendPolicy::Openblas | BackendPolicy::Blas => {
            let plan = compute_thread_plan(matrix_size, MathBackend::OpenBlas);
            env_setter::apply_thread_env(MathBackend::OpenBlas, plan.count);
            record_dispatch(1);
            record_detailed_dispatch(1, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n);
            return;
        }
        BackendPolicy::RustParallel => {
            let plan = compute_thread_plan(matrix_size, MathBackend::RustParallel);
            env_setter::apply_thread_env(MathBackend::RustParallel, plan.count);
            record_dispatch(8);
            record_detailed_dispatch(8, "matmul", m, n, k, policy);
            crate::ops::matmul_rust_parallel::matmul_f32_rust_parallel(
                a, b, c, m, k, n, plan.count,
            );
            return;
        }
        BackendPolicy::Mkl => {
            // MKL: fall through to BLAS call (linked as libmkl_rt when feature active)
            let plan = compute_thread_plan(matrix_size, MathBackend::Mkl);
            env_setter::apply_thread_env(MathBackend::Mkl, plan.count);
            record_dispatch(5);
            record_detailed_dispatch(5, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n); // same CBLAS ABI
            return;
        }
        BackendPolicy::Aocl => {
            let plan = compute_thread_plan(matrix_size, MathBackend::Aocl);
            env_setter::apply_thread_env(MathBackend::Aocl, plan.count);
            record_dispatch(6);
            record_detailed_dispatch(6, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n); // same CBLAS ABI
            return;
        }
        _ => {} // Default: continue to auto-selection below
    }

    // ── Auto-selection (Default policy) ─────────────────────────────────────
    let math_backend = get_math_backend();
    let plan = compute_thread_plan(matrix_size, math_backend);
    env_setter::apply_thread_env(math_backend, plan.count);

    match math_backend {
        MathBackend::RustParallel => {
            record_dispatch(8);
            record_detailed_dispatch(8, "matmul", m, n, k, policy);
            crate::ops::matmul_rust_parallel::matmul_f32_rust_parallel(
                a, b, c, m, k, n, plan.count,
            );
        }
        // MKL, AOCL, Accelerate, OpenBLAS all use the same cblas_sgemm ABI
        // linked at build time. Thread count controlled by env vars above.
        MathBackend::Mkl => {
            record_dispatch(5);
            record_detailed_dispatch(5, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n);
        }
        MathBackend::Aocl => {
            record_dispatch(6);
            record_detailed_dispatch(6, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n);
        }
        MathBackend::Accelerate => {
            record_dispatch(7);
            record_detailed_dispatch(7, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n);
        }
        MathBackend::OpenBlas => {
            record_dispatch(1);
            record_detailed_dispatch(1, "matmul", m, n, k, policy);
            kernel_matmul_f32_openblas(a, b, c, m, k, n);
        }
    }
}

// ============================================================================
// F64 Matrix Multiplication (Phase 3)
// ============================================================================

/// OpenBLAS/MKL MatMul for f64 using cblas_dgemm
#[cfg(has_blas)]
unsafe fn kernel_matmul_f64_openblas(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) {
    use cblas_sys::{cblas_dgemm, CblasNoTrans, CblasRowMajor};

    cblas_dgemm(
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

/// Fallback f64 MatMul when no BLAS is linked — naive scalar implementation
#[cfg(not(has_blas))]
unsafe fn kernel_matmul_f64_openblas(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += *a.add(i * k + p) * *b.add(p * n + j);
            }
            *c.add(i * n + j) = sum;
        }
    }
}

/// Dispatch 2D f64 matrix multiplication to BLAS kernel
pub unsafe fn matmul_f64_cpu_dispatch(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) {
    kernel_matmul_f64_openblas(a, b, c, m, k, n);
}
