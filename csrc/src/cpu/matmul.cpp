// ============================================================================
// Matrix Multiplication Operations (CPU)
// ============================================================================
// This file implements matrix/vector multiplication operations using SIMD.
//
// OPERATIONS:
// - dot_product_f32_cpu: 1D dot product (vector · vector)
// - Future: matmul_f32_cpu: 2D matrix multiplication
//
// OPTIMIZATION: AVX2 SIMD (8x f32 per instruction)

#include "corepy_kernels.h"
#include <cstddef>

// Only include x86-specific SIMD headers on x86/x64 architectures
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>  // AVX2 intrinsics
#endif

#ifndef COREPY_USE_OPENBLAS

namespace corepy::backend::avx2 {

/// Compute dot product of two f32 arrays
float dot_product_f32(const float* a, const float* b, size_t count) {
    #ifdef __AVX2__
    // AVX2 path: process 8 floats at a time
    
    __m256 sum_vec = _mm256_setzero_ps();  // Initialize accumulator to 0
    size_t avx_count = count / 8;          // Number of full AVX2 vectors
    size_t i = 0;
    
    // Main loop: process 8 elements per iteration
    for (; i < avx_count * 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);       // Load 8 floats from a
        __m256 vb = _mm256_loadu_ps(b + i);       // Load 8 floats from b
        __m256 prod = _mm256_mul_ps(va, vb);      // Multiply element-wise
        sum_vec = _mm256_add_ps(sum_vec, prod);   // Accumulate
    }
    
    // Horizontal sum: reduce 8 partial sums to 1 scalar
    // sum_vec = [s0, s1, s2, s3, s4, s5, s6, s7]
    // We need: s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    
    // Extract low and high 128-bit lanes
    __m128 low = _mm256_castps256_ps128(sum_vec);          // [s0, s1, s2, s3]
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);       // [s4, s5, s6, s7]
    __m128 sum128 = _mm_add_ps(low, high);                 // [s0+s4, s1+s5, s2+s6, s3+s7]
    
    // Horizontal add within 128-bit vector
    __m128 shuf = _mm_movehdup_ps(sum128);                 // [s1+s5, s1+s5, s3+s7, s3+s7]
    __m128 sums = _mm_add_ps(sum128, shuf);                // [s0+s4+s1+s5, *, s2+s6+s3+s7, *]
    shuf = _mm_movehl_ps(shuf, sums);                      // [s2+s6+s3+s7, *, *, *]
    __m128 final_sum = _mm_add_ss(sums, shuf);             // All 8 sums in lowest element
    
    float result = _mm_cvtss_f32(final_sum);
    
    // Scalar remainder loop (handle leftover elements)
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
    
    #else
    // Scalar fallback (no AVX2 support)
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
    #endif
}

void matmul_f32(
    const float* a, const float* b, float* c,
    size_t m, size_t k, size_t n
) {
    // Zero-initialize output matrix
    // Zero-initialize output matrix
    for (size_t i = 0; i < m * n; ++i) c[i] = 0.0f;

    #ifdef __AVX2__
    
    // AVX2 Implementation
    // Unrolled (i, p, j) implementation
    size_t i = 0;
    for (; i + 3 < m; i += 4) {
        for (size_t p = 0; p < k; ++p) {
            __m256 va0 = _mm256_set1_ps(a[(i + 0) * k + p]);
            __m256 va1 = _mm256_set1_ps(a[(i + 1) * k + p]);
            __m256 va2 = _mm256_set1_ps(a[(i + 2) * k + p]);
            __m256 va3 = _mm256_set1_ps(a[(i + 3) * k + p]);

            const float* rb = b + p * n;
            float* rc0 = c + (i + 0) * n;
            float* rc1 = c + (i + 1) * n;
            float* rc2 = c + (i + 2) * n;
            float* rc3 = c + (i + 3) * n;

            size_t j = 0;
            for (; j + 7 < n; j += 8) {
                __m256 vb = _mm256_loadu_ps(rb + j);
                
                #ifdef __FMA__
                _mm256_storeu_ps(rc0 + j, _mm256_fmadd_ps(va0, vb, _mm256_loadu_ps(rc0 + j)));
                _mm256_storeu_ps(rc1 + j, _mm256_fmadd_ps(va1, vb, _mm256_loadu_ps(rc1 + j)));
                _mm256_storeu_ps(rc2 + j, _mm256_fmadd_ps(va2, vb, _mm256_loadu_ps(rc2 + j)));
                _mm256_storeu_ps(rc3 + j, _mm256_fmadd_ps(va3, vb, _mm256_loadu_ps(rc3 + j)));
                #else
                _mm256_storeu_ps(rc0 + j, _mm256_add_ps(_mm256_loadu_ps(rc0 + j), _mm256_mul_ps(va0, vb)));
                _mm256_storeu_ps(rc1 + j, _mm256_add_ps(_mm256_loadu_ps(rc1 + j), _mm256_mul_ps(va1, vb)));
                _mm256_storeu_ps(rc2 + j, _mm256_add_ps(_mm256_loadu_ps(rc2 + j), _mm256_mul_ps(va2, vb)));
                _mm256_storeu_ps(rc3 + j, _mm256_add_ps(_mm256_loadu_ps(rc3 + j), _mm256_mul_ps(va3, vb)));
                #endif
            }
            // Scalar remainder for j
            for (; j < n; ++j) {
                float bv = rb[j];
                // Fix MSVC error: Cannot use va0[0]. Use original scalar value.
                rc0[j] += a[(i + 0) * k + p] * bv;
                rc1[j] += a[(i + 1) * k + p] * bv;
                rc2[j] += a[(i + 2) * k + p] * bv;
                rc3[j] += a[(i + 3) * k + p] * bv;
            }
        }
    }

    // Remainder i
    for (; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            float val_a = a[i * k + p];
            for (size_t j = 0; j < n; ++j) {
                c[i * n + j] += val_a * b[p * n + j];
            }
        }
    }

    #else
    // Scalar Fallback (No AVX2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            float val_a = a[i * k + k_idx];
            for (size_t j = 0; j < n; ++j) {
                c[i * n + j] += val_a * b[k_idx * n + j];
            }
        }
    }
    #endif
}

} // namespace corepy::backend::avx2

extern "C" {

float dot_product_f32_cpu(const float* a, const float* b, size_t count) {
    return corepy::backend::avx2::dot_product_f32(a, b, count);
}

void matmul_f32_cpu(const float* a, const float* b, float* c,
                    size_t m, size_t k, size_t n) {
    corepy::backend::avx2::matmul_f32(a, b, c, m, k, n);
}

void corepy_set_num_threads(int num_threads) {
    // No-op for native kernels (parallelized via Rust/Rayon)
}

bool corepy_is_blas_enabled() {
    return false;
}
}
#endif // !COREPY_USE_OPENBLAS
