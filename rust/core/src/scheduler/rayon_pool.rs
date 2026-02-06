// ============================================================================
// Rayon-Based Work-Stealing Scheduler
// ============================================================================
//
// RESPONSIBILITIES:
// - Initialize global thread pool for parallel execution
// - Provide work-stealing task dispatch
// - Integrate with Python's GIL (release during compute)
// - NUMA-aware thread affinity (future)
//
// DESIGN:
// - Lazy initialization via lazy_static
// - Thread count: num_cpus or COREPY_NUM_THREADS env var
// - Each thread has arena allocator via thread_local
// - Panic handler for Rust panics in worker threads

use pyo3::prelude::*;
use rayon;
use std::sync::Once;

#[allow(dead_code)]
static INIT: Once = Once::new();

/// Initialize the global Rayon thread pool
///
/// Called lazily on first use. Thread count determined by:
/// 1. COREPY_NUM_THREADS env var
/// 2. num_cpus::get() (default)
///
/// This sets up the work-stealing scheduler that will be used
/// for all parallel tensor operations.
#[allow(dead_code)]
pub fn init_thread_pool() {
    INIT.call_once(|| {
        let num_threads = std::env::var("COREPY_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(num_cpus::get);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("corepy-worker-{}", idx))
            .panic_handler(|_| {
                eprintln!("Corepy worker thread panicked!");
            })
            .build_global()
            .expect("Failed to initialize Rayon thread pool");

        eprintln!(
            "Corepy: Initialized thread pool with {} workers",
            num_threads
        );
    });
}

/// Execute a parallel operation with GIL released
///
/// This is the core dispatch function for multi-threaded tensor operations.
/// It releases Python's GIL during execution to allow true parallelism.
///
/// # Safety
/// - The closure `f` must not access any Python objects
/// - The closure must be Send + Sync
///
/// # Example
/// ```
/// let result = execute_parallel(py, || {
///     // This runs in parallel with GIL released
///     rayon::join(
///         || compute_chunk_1(),
///         || compute_chunk_2()
///     )
/// });
/// ```
#[allow(dead_code)]
pub fn execute_parallel<F, R>(py: Python, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // Ensure pool is initialized
    init_thread_pool();

    // Release GIL and execute
    py.allow_threads(f)
}

/// Execute parallel iterator operation
///
/// Common pattern for data-parallel operations on tensors.
/// Automatically chunks work across available threads.
///
/// # Example
/// ```
/// execute_parallel_iter(py, data, |chunk| {
///     // Process chunk in parallel
///     process_chunk(chunk)
/// });
/// ```
#[allow(dead_code)]
pub fn execute_parallel_iter<T, F>(py: Python, data: &[T], f: F)
where
    T: Send + Sync,
    F: Fn(&T) + Send + Sync,
{
    init_thread_pool();

    py.allow_threads(|| {
        use rayon::prelude::*;
        data.par_iter().for_each(f);
    });
}

/// Execute parallel map operation
///
/// Maps function over data in parallel, returning collected results.
///
/// # Example
/// ```
/// let results = execute_parallel_map(py, &input_array, |x| x * 2);
/// ```
#[allow(dead_code)]
pub fn execute_parallel_map<T, R, F>(py: Python, data: &[T], f: F) -> Vec<R>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    init_thread_pool();

    py.allow_threads(|| {
        use rayon::prelude::*;
        data.par_iter().map(f).collect()
    })
}

/// Get number of threads in the pool
#[allow(dead_code)]
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Check if currently executing in a Rayon worker thread
#[allow(dead_code)]
pub fn in_worker_thread() -> bool {
    rayon::current_thread_index().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_init() {
        init_thread_pool();
        let count = num_threads();
        assert!(count > 0);
        assert!(count <= num_cpus::get() * 2); // Sanity check
    }

    #[test]
    fn test_parallel_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        init_thread_pool();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        rayon::scope(|s| {
            for _ in 0..100 {
                let counter = counter_clone.clone();
                s.spawn(move |_| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });

        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_in_worker_thread() {
        init_thread_pool();

        // Main thread should not be a worker
        assert!(!in_worker_thread());

        // Inside rayon scope should be a worker
        rayon::scope(|s| {
            s.spawn(|_| {
                assert!(in_worker_thread());
            });
        });
    }
}
