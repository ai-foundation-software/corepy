// ============================================================================
// Corepy Runtime: Rust Layer (System Brain)
// ============================================================================
//
// This is the Rust runtime layer of Corepy's 3-layer architecture:
//
//   Python (UX) → Rust (Brain) → C++ (Muscle)
//
// RESPONSIBILITIES (see docs/execution_model.md):
// - Array validation (shape/dtype/device compatibility)
// - Memory lifetime management (arena allocators)
// - Work-stealing scheduler (rayon)
// - Backend dispatch (CPU → C++, GPU → CUDA/Metal)
// - FFI safety boundary (Send/Sync enforcement)
// - NEVER in math hot path
//
// MODULES:
// - ffi/: Python ↔ Rust bridge (PyO3)
// - ops/: Operation dispatch to C++ kernels
// - array/: Internal array representation (future)
// - scheduler/: Rayon-based work-stealing (future)
// - backend/: CPU/GPU backend selection (future)

#![allow(clippy::useless_conversion)] // PyO3 macro-generated code triggers this
#![cfg_attr(feature = "nightly", feature(portable_simd))]

use pyo3::prelude::*;

// Module declarations
mod array;
mod backend; // Future: Backend dispatch
pub mod dataframe;
mod ffi;
pub mod linalg;
mod ops;
mod profiler;
mod scheduler; // Future: Rayon scheduler // Future: Shape, dtype, buffer management // Performance profiling system

// ============================================================================
// PyO3 Module Definition
// ============================================================================

/// Python module implemented in Rust
///
/// This exports Rust functions to Python via PyO3.
/// All function signatures use raw pointers for zero-copy performance.
#[pymodule]
#[cfg_attr(feature = "nightly", feature(portable_simd))]
fn _corepy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all FFI functions from ffi/python.rs
    ffi::python::register_functions(m)?;

    // Register PyO3 classes
    m.add_class::<array::CoreArray>()?;
    m.add_class::<array::DType>()?;
    m.add_class::<dataframe::core::DataFrame>()?;
    m.add_class::<dataframe::series::Series>()?;
    m.add_class::<dataframe::groupby::GroupBy>()?;
    m.add_class::<ops::random::RngAlgorithm>()?;

    Ok(())
}
