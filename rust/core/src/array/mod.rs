// ============================================================================
// Array: Internal Representation
// ============================================================================
// Core array system for corepy. Replaces NumPy as primary backing store.
//
// MODULES:
// - dtype: Element data types (Float32, Float64, Int32, Int64, Bool)
// - core_array: 64-byte aligned array struct with PyO3 bindings

pub mod aligned_buffer;
pub mod core_array;
pub mod dtype;

pub use core_array::CoreArray;
pub use dtype::DType;
