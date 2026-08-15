use rayon::prelude::*;
use std::cmp::max;
use std::sync::Once;

const PARALLEL_THRESHOLD: usize = 10_000;
static INIT_RAYON: Once = Once::new();

/// Initializes the global Rayon thread pool dynamically.
/// Rule: usable_threads = max(1, cpu_threads - 1)
/// Leaves one thread free for the Python GIL and OS scheduling.
pub fn init_thread_pool() {
    INIT_RAYON.call_once(|| {
        let logical_cpus = num_cpus::get();
        let usable_threads = max(1, logical_cpus.saturating_sub(1));

        // Build the global rayon ThreadPool. This won't override it if it's already built.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(usable_threads)
            .build_global();
    });
}

/// A heavy CPU workload implementation demonstrating threshold-based hybrid parallelization.
pub fn process_workload(data: Vec<f64>) -> Vec<f64> {
    init_thread_pool();

    let workload_size = data.len();

    // PARALLEL THRESHOLD RULE
    // Avoids overhead on small arrays by dropping into a fast sequential fallback.
    // Extremely effective when evaluating micro-batches deep inside recursive calls.
    if workload_size < PARALLEL_THRESHOLD {
        // Sequential map
        data.into_iter()
            .map(|x| (x.powf(2.5) * std::f64::consts::PI).sin())
            .collect()
    } else {
        // Parallel map - Rayon dynamically shards and work-steals
        data.into_par_iter()
            .map(|x| (x.powf(2.5) * std::f64::consts::PI).sin())
            .collect()
    }
}
