// ============================================================================
// Pure-Rust Parallel Matrix Multiplication Fallback
// ============================================================================
#![allow(dead_code)] // Public API — used by Python FFI and future callers

use rayon::ThreadPool;
use std::sync::OnceLock;

// Size (ops = M*N*K) below which threading is not worth the overhead.
const SERIAL_THRESHOLD: usize = 64 * 64 * 64; // 262_144

// Minimum row count per rayon task – avoids over-granulation.
const MIN_ROWS_PER_TASK: usize = 32;

/// Global cached Rayon thread pool for matmul (created once per process).
static MATMUL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Initialise the global matmul thread pool.  Subsequent calls are no-ops.
pub fn init_matmul_pool(threads: usize) {
    MATMUL_POOL.get_or_init(|| build_pool(threads));
}

fn get_pool(threads: usize) -> &'static ThreadPool {
    MATMUL_POOL.get_or_init(|| build_pool(threads))
}

fn build_pool(threads: usize) -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("corepy-mm-{}", i))
        .build()
        .expect("[corepy] Failed to build Rayon ThreadPool for matmul")
}

// ============================================================================
// Public entry points
// ============================================================================

/// Compute C = A @ B  (f32, row-major, no-transpose).
///
/// # Safety
/// * `a`  must be valid for `m * k` f32 reads.
/// * `b`  must be valid for `k * n` f32 reads.
/// * `c`  must be valid for `m * n` f32 writes.
/// * Pointers must not alias.
pub unsafe fn matmul_f32_rust_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    let total_ops = m * n * k;

    if total_ops <= SERIAL_THRESHOLD || threads <= 1 || m < MIN_ROWS_PER_TASK {
        sgemm_serial(a, b, c, m, k, n);
        return;
    }

    let pool = get_pool(threads);
    let rows_per_task = m.div_ceil(threads).max(MIN_ROWS_PER_TASK);

    // Convert to usize for arithmetic inside Send closures.
    let a_addr = a as usize;
    let b_addr = b as usize;
    let c_addr = c as usize;

    pool.scope(|s| {
        let mut row_start = 0_usize;
        while row_start < m {
            let row_end = (row_start + rows_per_task).min(m);
            let block_m = row_end - row_start;
            // Compute addresses as usize (integers are Send)
            let a_block_addr = a_addr + row_start * k * std::mem::size_of::<f32>();
            let c_block_addr = c_addr + row_start * n * std::mem::size_of::<f32>();
            let b_block_addr = b_addr;

            s.spawn(move |_| {
                // Reconstruct pointers inside the task body.
                // SAFETY: addresses computed from valid pointers with correct offsets.
                let a_ptr = a_block_addr as *const f32;
                let b_ptr = b_block_addr as *const f32;
                let c_ptr = c_block_addr as *mut f32;
                unsafe {
                    sgemm_serial(a_ptr, b_ptr, c_ptr, block_m, k, n);
                }
            });

            row_start = row_end;
        }
    });
}

/// Compute C = A @ B  (f64, row-major, no-transpose).
///
/// # Safety
/// Same contract as `matmul_f32_rust_parallel` for f64.
pub unsafe fn matmul_f64_rust_parallel(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    let total_ops = m * n * k;

    if total_ops <= SERIAL_THRESHOLD || threads <= 1 || m < MIN_ROWS_PER_TASK {
        dgemm_serial(a, b, c, m, k, n);
        return;
    }

    let pool = get_pool(threads);
    let rows_per_task = m.div_ceil(threads).max(MIN_ROWS_PER_TASK);

    let a_addr = a as usize;
    let b_addr = b as usize;
    let c_addr = c as usize;

    pool.scope(|s| {
        let mut row_start = 0_usize;
        while row_start < m {
            let row_end = (row_start + rows_per_task).min(m);
            let block_m = row_end - row_start;
            let a_block_addr = a_addr + row_start * k * std::mem::size_of::<f64>();
            let c_block_addr = c_addr + row_start * n * std::mem::size_of::<f64>();
            let b_block_addr = b_addr;

            s.spawn(move |_| {
                let a_ptr = a_block_addr as *const f64;
                let b_ptr = b_block_addr as *const f64;
                let c_ptr = c_block_addr as *mut f64;
                unsafe {
                    dgemm_serial(a_ptr, b_ptr, c_ptr, block_m, k, n);
                }
            });

            row_start = row_end;
        }
    });
}

// ============================================================================
// Serial kernels via matrixmultiply (cache-blocked, AVX2/NEON SIMD)
// ============================================================================

#[inline]
unsafe fn sgemm_serial(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize) {
    // C = 1·A·B + 0·C   (row-major: lda=k, ldb=n, ldc=n)
    matrixmultiply::sgemm(
        m, k, n, 1.0_f32, a, k as isize, 1, b, n as isize, 1, 0.0_f32, c, n as isize, 1,
    );
}

#[inline]
unsafe fn dgemm_serial(a: *const f64, b: *const f64, c: *mut f64, m: usize, k: usize, n: usize) {
    matrixmultiply::dgemm(
        m, k, n, 1.0_f64, a, k as isize, 1, b, n as isize, 1, 0.0_f64, c, n as isize, 1,
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_2x2() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        unsafe {
            matmul_f32_rust_parallel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 1);
        }
        assert!((c[0] - 1.0).abs() < 1e-6, "c[0]={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-6, "c[1]={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-6, "c[2]={}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-6, "c[3]={}", c[3]);
    }

    #[test]
    fn test_3x3_ones() {
        let a = [1.0f32; 9];
        let b = [1.0f32; 9];
        let mut c = [0.0f32; 9];
        unsafe {
            matmul_f32_rust_parallel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 3, 3, 1);
        }
        for &v in &c {
            assert!((v - 3.0).abs() < 1e-5, "Expected 3.0, got {}", v);
        }
    }

    #[test]
    fn test_parallel_matches_serial() {
        let m = 128;
        let a: Vec<f32> = (0..m * m).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..m * m).map(|i| i as f32 * 0.001).collect();
        let mut c_par = vec![0.0f32; m * m];
        let mut c_ser = vec![0.0f32; m * m];
        unsafe {
            matmul_f32_rust_parallel(a.as_ptr(), b.as_ptr(), c_par.as_mut_ptr(), m, m, m, 4);
            matmul_f32_rust_parallel(a.as_ptr(), b.as_ptr(), c_ser.as_mut_ptr(), m, m, m, 1);
        }
        for (i, (p, s)) in c_par.iter().zip(c_ser.iter()).enumerate() {
            assert!((p - s).abs() < 1e-2, "Mismatch @{}: par={} ser={}", i, p, s);
        }
    }

    #[test]
    fn test_pool_init_idempotent() {
        init_matmul_pool(2);
        init_matmul_pool(8); // no-op, should not panic
    }
}
