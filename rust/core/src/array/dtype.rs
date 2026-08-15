// ============================================================================
// DType: Array Element Data Types
// ============================================================================

use pyo3::prelude::*;

/// Supported array element data types.
/// Mirrors Python's DataType enum for seamless FFI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(
    module = "corepy._corepy_rust",
    name = "_RustDType",
    eq,
    eq_int,
    from_py_object
)]
pub enum DType {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    Bool = 4,
    String = 5,
}

#[pymethods]
impl DType {
    /// Size of a single element in bytes.
    #[getter]
    pub fn itemsize(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
            DType::String => 0, // Sentinel for non-numeric data
        }
    }

    fn __repr__(&self) -> String {
        format!("DType.{:?}", self)
    }
}
