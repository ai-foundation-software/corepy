// ============================================================================
// Corepy Runtime: Rust Layer (System Brain)
// ============================================================================
//
// This is the Rust runtime layer of Corepy's 3-layer architecture:
//
//   Python (UX) → Rust (Brain) → C++ (Muscle)
//
// RESPONSIBILITIES (see docs/execution_model.md):
// - Tensor validation (shape/dtype/device compatibility)
// - Memory lifetime management (arena allocators)
// - Work-stealing scheduler (rayon)
// - Backend dispatch (CPU → C++, GPU → CUDA/Metal)
// - FFI safety boundary (Send/Sync enforcement)
// - NEVER in math hot path
//
// MODULES:
// - ffi/: Python ↔ Rust bridge (PyO3)
// - ops/: Operation dispatch to C++ kernels
// - tensor/: Internal tensor representation (future)
// - scheduler/: Rayon-based work-stealing (future)
// - backend/: CPU/GPU backend selection (future)

use pyo3::prelude::*;

// Module declarations
mod backend; // Future: Backend dispatch
mod ffi;
mod ops;
mod profiler;
mod scheduler; // Future: Rayon scheduler
mod tensor; // Future: Shape, dtype, buffer management // Performance profiling system

// ============================================================================
// PyO3 Module Definition
// ============================================================================

/// Python module implemented in Rust
///
/// This exports Rust functions to Python via PyO3.
/// All function signatures use raw pointers for zero-copy performance.
#[pymodule]
fn _corepy_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register all FFI functions from ffi/python.rs
    ffi::python::register_functions(m)?;

    Ok(())
}
