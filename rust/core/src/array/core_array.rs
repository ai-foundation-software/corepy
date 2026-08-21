// ============================================================================
// CoreArray: Rust-Native Array with 64-byte Aligned Memory
// ============================================================================
//
// This is the primary backing store for corepy.ndarray, replacing NumPy.
// Memory is 64-byte aligned for optimal SIMD (AVX-512 = 64-byte registers).
//
// Ownership model:
// - Rust owns the heap allocation via AlignedBuffer
// - Python holds an Arc<CoreArray> via PyO3
// - to_numpy() creates a zero-copy view (or copy if needed)

use crate::array::aligned_buffer::AlignedBuffer;
use crate::array::dtype::DType;
use crate::linalg::decompose::{linalg_det, linalg_inv, linalg_norm};
use pyo3::ffi::{PyBUF_FORMAT, Py_buffer};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ptr;

macro_rules! check_binop_shapes {
    ($self:expr, $other:expr, $name:expr) => {
        if $self.shape != $other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for {}: {:?} vs {:?}",
                $name, $self.shape, $other.shape
            )));
        }
        if $self.dtype != $other.dtype {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Dtype mismatch for {}: {:?} vs {:?}",
                $name, $self.dtype, $other.dtype
            )));
        }
    };
}

macro_rules! impl_binop {
    ($self:expr, $other:expr, $op:tt, $name:expr, $dispatch:path) => {{
        check_binop_shapes!($self, $other, $name);
        use rayon::prelude::*;
        match $self.dtype {
            DType::Float32 => {
                let a = unsafe { $self.as_f32_slice() };
                let b = unsafe { $other.as_f32_slice() };
                let mut res = vec![0.0f32; $self.element_count()];
                unsafe {
                    $dispatch(a.as_ptr(), b.as_ptr(), res.as_mut_ptr(), $self.element_count())
                        .expect(&format!("{} dispatch failed", $name));
                }
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float32)
            }
            DType::Float64 => {
                let a = unsafe { $self.as_f64_slice() };
                let b = unsafe { $other.as_f64_slice() };
                let res: Vec<f64> = a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x $op y).collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float64)
            }
            DType::Int32 => {
                let a = unsafe { $self.as_i32_slice() };
                let b = unsafe { $other.as_i32_slice() };
                let res: Vec<i32> = a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x $op y).collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Int32)
            }
            DType::Int64 => {
                let a = unsafe { $self.as_i64_slice() };
                let b = unsafe { $other.as_i64_slice() };
                let res: Vec<i64> = a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x $op y).collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Int64)
            }
            DType::Bool => {
                Err(pyo3::exceptions::PyTypeError::new_err(format!("Operation {} not supported for Bool dtype", $name)))
            }
            DType::String => {
                Err(pyo3::exceptions::PyTypeError::new_err(format!("Operation {} not supported for String dtype", $name)))
            }
        }
    }};
}

macro_rules! impl_div_binop {
    ($self:expr, $other:expr, $dispatch:path) => {{
        check_binop_shapes!($self, $other, "div");
        use rayon::prelude::*;
        match $self.dtype {
            DType::Float32 => {
                let a = unsafe { $self.as_f32_slice() };
                let b = unsafe { $other.as_f32_slice() };
                let mut res = vec![0.0f32; $self.element_count()];
                unsafe {
                    $dispatch(
                        a.as_ptr(),
                        b.as_ptr(),
                        res.as_mut_ptr(),
                        $self.element_count(),
                    )
                    .expect("div dispatch failed");
                }
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float32)
            }
            DType::Float64 => {
                let a = unsafe { $self.as_f64_slice() };
                let b = unsafe { $other.as_f64_slice() };
                let res: Vec<f64> = a
                    .par_iter()
                    .zip(b.par_iter())
                    .map(|(&x, &y)| x / y)
                    .collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float64)
            }
            DType::Int32 => {
                let a = unsafe { $self.as_i32_slice() };
                let b = unsafe { $other.as_i32_slice() };
                let res: Vec<f64> = a
                    .par_iter()
                    .zip(b.par_iter())
                    .map(|(&x, &y)| (x as f64) / (y as f64))
                    .collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float64)
            }
            DType::Int64 => {
                let a = unsafe { $self.as_i64_slice() };
                let b = unsafe { $other.as_i64_slice() };
                let res: Vec<f64> = a
                    .par_iter()
                    .zip(b.par_iter())
                    .map(|(&x, &y)| (x as f64) / (y as f64))
                    .collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float64)
            }
            DType::Bool => Err(pyo3::exceptions::PyTypeError::new_err(
                "Operation div not supported for Bool dtype",
            )),
            DType::String => Err(pyo3::exceptions::PyTypeError::new_err(
                "Operation div not supported for String dtype",
            )),
        }
    }};
}

macro_rules! impl_compare_binop {
    ($self:expr, $other:expr, $method:ident, $name:expr) => {{
        check_binop_shapes!($self, $other, $name);
        use rayon::prelude::*;
        match $self.dtype {
            DType::Float32 => {
                let a = unsafe { $self.as_f32_slice() };
                let b = unsafe { $other.as_f32_slice() };
                let res: Vec<f32> = a
                    .par_iter()
                    .zip(b.par_iter())
                    .map(|(&x, &y)| x.$method(y))
                    .collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float32)
            }
            DType::Float64 => {
                let a = unsafe { $self.as_f64_slice() };
                let b = unsafe { $other.as_f64_slice() };
                let res: Vec<f64> = a
                    .par_iter()
                    .zip(b.par_iter())
                    .map(|(&x, &y)| x.$method(y))
                    .collect();
                CoreArray::new_from_vec(res, $self.shape.clone(), DType::Float64)
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Operation {} not implemented completely",
                $name
            ))),
        }
    }};
}

// ============================================================================
// CoreArray
// ============================================================================

/// Rust-native array. Primary backing store for corepy arrays.
///
/// Features:
/// - 64-byte aligned memory (SIMD-optimal)
/// - C-contiguous row-major layout
/// - Zero-copy pointer exposure for FFI
/// - PyO3-exposed for direct Python access
#[pyclass(
    module = "corepy._corepy_rust",
    name = "_RustCoreArray",
    from_py_object
)]
#[derive(Debug)]
pub struct CoreArray {
    buffer: AlignedBuffer,
    pub data_str: Option<Vec<String>>,
    shape: Vec<usize>,
    strides: Vec<usize>, // byte strides
    dtype: DType,
    // FFI-compatible shape and strides for the Buffer Protocol
    shape_ffi: Vec<isize>,
    strides_ffi: Vec<isize>,
}

impl Clone for CoreArray {
    fn clone(&self) -> Self {
        CoreArray {
            buffer: self.buffer.clone(),
            data_str: self.data_str.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            shape_ffi: self.shape_ffi.clone(),
            strides_ffi: self.strides_ffi.clone(),
        }
    }
}

impl CoreArray {
    /// Internal helper to create a CoreArray with pre-calculated strides and FFI metadata.
    fn from_raw(
        buffer: AlignedBuffer,
        data_str: Option<Vec<String>>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let strides = Self::compute_c_strides(&shape, dtype.itemsize());
        let shape_ffi = shape.iter().map(|&x| x as isize).collect();
        let strides_ffi = strides.iter().map(|&x| x as isize).collect();

        CoreArray {
            buffer,
            data_str,
            shape,
            strides,
            dtype,
            shape_ffi,
            strides_ffi,
        }
    }
}

impl CoreArray {
    /// Generic internal constructor for arbitrary Vec<T>.
    pub fn new_from_vec<T>(data: Vec<T>, shape: Vec<usize>, dtype: DType) -> PyResult<Self> {
        let expected_count: usize = shape.iter().product();
        if data.len() != expected_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_count
            )));
        }

        let byte_len = expected_count * std::mem::size_of::<T>();
        let buffer = AlignedBuffer::new(byte_len);

        // Copy data into aligned buffer
        // SAFETY: buffer is freshly allocated with correct size
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.as_mut_ptr(),
                byte_len,
            );
        }

        Ok(Self::from_raw(buffer, None, shape, dtype))
    }
}

impl CoreArray {
    /// Create a new array from a flat list of f32 values and a shape.
    /// Internal Rust constructor.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        Self::new_from_vec(data, shape, DType::Float32)
    }
}

#[allow(clippy::useless_conversion)]
#[pymethods]
impl CoreArray {
    /// PyO3 constructor for Pickle and direct Python use.
    #[new]
    #[pyo3(signature = (data=None, shape=None))]
    pub fn py_new(data: Option<Vec<f32>>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let d = data.unwrap_or_default();
        let s = shape.unwrap_or_else(|| vec![d.len()]);
        Self::new(d, s)
    }

    /// Create a new array from a list of strings and a shape.
    #[staticmethod]
    #[pyo3(signature = (data_str, shape))]
    pub fn from_strings(data_str: Vec<String>, shape: Vec<usize>) -> PyResult<Self> {
        let expected_count: usize = shape.iter().product();
        if data_str.len() != expected_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "String data length {} does not match shape {:?}",
                data_str.len(),
                shape
            )));
        }
        let buffer = AlignedBuffer::new(0);
        Ok(Self::from_raw(buffer, Some(data_str), shape, DType::String))
    }

    /// Raw pointer to the data buffer (for FFI).
    #[getter]
    pub fn data_ptr(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    /// Shape of the array.
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Byte strides of the array.
    #[getter]
    pub fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    /// Data type of the array.
    #[getter]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of dimensions.
    #[getter]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    #[getter]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    #[getter]
    pub fn nbytes(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the array is C-contiguous.
    pub fn is_contiguous(&self) -> bool {
        let expected = Self::compute_c_strides(&self.shape, self.dtype.itemsize());
        self.strides == expected
    }

    /// Return the data as a flat list of values (for Python interop).
    pub fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let count = self.element_count();
        match self.dtype {
            DType::Float32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(self.buffer.as_ptr() as *const f32, count)
                };
                Ok(pyo3::types::PyList::new(py, slice)?)
            }
            DType::Float64 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(self.buffer.as_ptr() as *const f64, count)
                };
                Ok(pyo3::types::PyList::new(py, slice)?)
            }
            DType::Int32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(self.buffer.as_ptr() as *const i32, count)
                };
                Ok(pyo3::types::PyList::new(py, slice)?)
            }
            DType::Int64 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(self.buffer.as_ptr() as *const i64, count)
                };
                Ok(pyo3::types::PyList::new(py, slice)?)
            }
            DType::Bool => {
                let slice = unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), count) };
                let bools: Vec<bool> = slice.iter().map(|&v| v != 0).collect();
                Ok(pyo3::types::PyList::new(py, bools)?)
            }
            DType::String => {
                if let Some(ref data) = self.data_str {
                    Ok(pyo3::types::PyList::new(py, data)?)
                } else {
                    Ok(pyo3::types::PyList::new(py, vec![""; count])?)
                }
            }
        }
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", &self.shape)?;
        dict.set_item("dtype", format!("{:?}", self.dtype))?;

        if let Some(ref data_str) = self.data_str {
            dict.set_item("data_str", data_str)?;
        } else {
            let bytes =
                unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), self.buffer.len()) };
            dict.set_item("buffer", pyo3::types::PyBytes::new(py, bytes))?;
        }

        Ok(dict)
    }

    pub fn __setstate__(&mut self, state: pyo3::Py<pyo3::PyAny>, py: Python<'_>) -> PyResult<()> {
        let dict = state.bind(py).cast::<PyDict>()?;
        self.shape = dict
            .get_item("shape")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing shape"))?
            .extract()?;

        let dtype_str: String = dict
            .get_item("dtype")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing dtype"))?
            .extract()?;

        self.dtype = match dtype_str.as_str() {
            "Float32" => DType::Float32,
            "Float64" => DType::Float64,
            "Int32" => DType::Int32,
            "Int64" => DType::Int64,
            "Bool" => DType::Bool,
            "String" => DType::String,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported dtype: {}",
                    dtype_str
                )))
            }
        };

        if self.dtype == DType::String {
            self.data_str = Some(
                dict.get_item("data_str")?
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing data_str"))?
                    .extract()?,
            );
            self.buffer = AlignedBuffer::new(0);
        } else {
            let bytes: Bound<'_, pyo3::types::PyBytes> = dict
                .get_item("buffer")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing buffer"))?
                .cast_into()?;
            let data = bytes.as_bytes();
            let buffer = AlignedBuffer::new(data.len());
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr(), data.len());
            }
            self.buffer = buffer;
            self.data_str = None;
        }

        self.strides = Self::compute_c_strides(&self.shape, self.dtype.itemsize());
        self.shape_ffi = self.shape.iter().map(|&x| x as isize).collect();
        self.strides_ffi = self.strides.iter().map(|&x| x as isize).collect();

        Ok(())
    }

    /// Implement the Python Buffer Protocol (__getbuffer__) for zero-copy memory views.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }

        let view = &mut *view;
        view.obj = std::ptr::null_mut();
        view.buf = slf.buffer.as_ptr() as *mut std::os::raw::c_void;
        view.len = (slf.element_count() * slf.dtype.itemsize()) as isize;
        view.readonly = 1; // Mark as read-only for safety
        view.itemsize = slf.dtype.itemsize() as isize;

        view.format = std::ptr::null_mut();
        if (flags & PyBUF_FORMAT) == PyBUF_FORMAT {
            view.format = match slf.dtype {
                DType::Float32 => c"f".as_ptr() as *mut _,
                DType::Float64 => c"d".as_ptr() as *mut _,
                DType::Int32 => c"i".as_ptr() as *mut _,
                DType::Int64 => c"q".as_ptr() as *mut _,
                DType::Bool => c"?".as_ptr() as *mut _,
                DType::String => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Buffer Protocol not supported for String arrays",
                    ))
                }
            };
        }

        view.ndim = slf.shape.len() as std::os::raw::c_int;
        view.shape = slf.shape_ffi.as_ptr() as *mut _;
        view.strides = slf.strides_ffi.as_ptr() as *mut _;
        view.suboffsets = std::ptr::null_mut();
        view.internal = std::ptr::null_mut();

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut Py_buffer) {}

    /// Get a single element by its flat index.
    pub fn get_flat(&self, index: usize) -> PyResult<f64> {
        let count = self.element_count();
        if index >= count {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }
        match self.dtype {
            DType::Float32 => {
                let v = unsafe { *(self.buffer.as_ptr() as *const f32).add(index) };
                Ok(v as f64)
            }
            DType::Float64 => {
                let v = unsafe { *(self.buffer.as_ptr() as *const f64).add(index) };
                Ok(v)
            }
            DType::Int32 => {
                let v = unsafe { *(self.buffer.as_ptr() as *const i32).add(index) };
                Ok(v as f64)
            }
            DType::Int64 => {
                let v = unsafe { *(self.buffer.as_ptr() as *const i64).add(index) };
                Ok(v as f64)
            }
            DType::Bool => {
                let v = unsafe { *(self.buffer.as_ptr()).add(index) };
                Ok(if v != 0 { 1.0 } else { 0.0 })
            }
            DType::String => Err(pyo3::exceptions::PyTypeError::new_err(
                "get_flat not supported for String dtype",
            )),
        }
    }

    /// Create a array filled with zeros.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    pub fn zeros(shape: Vec<usize>, dtype: Option<DType>) -> PyResult<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let count: usize = shape.iter().product();
        let byte_len = count * dt.itemsize();
        let buffer = AlignedBuffer::new(byte_len); // alloc_zeroed
        Ok(Self::from_raw(buffer, None, shape, dt))
    }

    /// Create a array filled with ones (f32).
    #[staticmethod]
    #[pyo3(signature = (shape))]
    pub fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let count: usize = shape.iter().product();
        let data = vec![1.0f32; count];
        CoreArray::new(data, shape)
    }

    /// Create a array filled with a constant value.
    #[staticmethod]
    #[pyo3(signature = (shape, value))]
    pub fn full(shape: Vec<usize>, value: f32) -> PyResult<Self> {
        let count: usize = shape.iter().product();
        let data = vec![value; count];
        CoreArray::new(data, shape)
    }

    /// Create a array with evenly spaced values: [start, start+step, ...).
    #[staticmethod]
    #[pyo3(signature = (start, stop, step=None))]
    pub fn arange(start: f32, stop: f32, step: Option<f32>) -> PyResult<Self> {
        let s = step.unwrap_or(1.0);
        if s == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "arange step cannot be zero",
            ));
        }
        let mut data = Vec::new();
        let mut val = start;
        if s > 0.0 {
            while val < stop {
                data.push(val);
                val += s;
            }
        } else {
            while val > stop {
                data.push(val);
                val += s;
            }
        }
        let count = data.len();
        CoreArray::new(data, vec![count])
    }

    /// Concatenate multiple arrays along axis 0.
    #[staticmethod]
    #[pyo3(signature = (arrays))]
    pub fn concatenate(arrays: Vec<pyo3::PyRef<'_, CoreArray>>) -> PyResult<Self> {
        if arrays.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "need at least one array to concatenate",
            ));
        }

        // Validate all have same dtype and compatible shapes
        let dtype = arrays[0].dtype;
        for (i, t) in arrays.iter().enumerate() {
            if t.dtype != dtype {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "All arrays must have the same dtype for concatenation, got {:?} and {:?}",
                    dtype, t.dtype
                )));
            }
            if t.shape.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot concatenate 0-d array (at index {})",
                    i
                )));
            }
        }

        if dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "concatenate() currently only supports Float32",
            ));
        }

        // For 1D: just concatenate data
        if arrays[0].shape.len() == 1 {
            let mut all_data = Vec::new();
            for t in &arrays {
                let slice = unsafe { t.as_f32_slice() };
                all_data.extend_from_slice(slice);
            }
            let count = all_data.len();
            return CoreArray::new(all_data, vec![count]);
        }

        // For ND: validate inner dimensions match, sum first dim
        let inner_shape = &arrays[0].shape[1..];
        let mut total_first_dim = 0usize;
        for t in &arrays {
            if t.shape[1..] != *inner_shape {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "incompatible shapes for concatenation: {:?} vs {:?}",
                    arrays[0].shape, t.shape
                )));
            }
            total_first_dim += t.shape[0];
        }

        let mut all_data = Vec::new();
        for t in &arrays {
            let slice = unsafe { t.as_f32_slice() };
            all_data.extend_from_slice(slice);
        }

        let mut new_shape = vec![total_first_dim];
        new_shape.extend_from_slice(inner_shape);
        CoreArray::new(all_data, new_shape)
    }

    // =========================================================================
    // Element-wise Operations (Phase 3)
    // =========================================================================

    /// Element-wise addition
    pub fn add(&self, other: &CoreArray) -> PyResult<Self> {
        impl_binop!(self, other, +, "add", crate::backend::registry::dispatch_add_f32)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &CoreArray) -> PyResult<Self> {
        impl_binop!(self, other, -, "sub", crate::backend::registry::dispatch_sub_f32)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &CoreArray) -> PyResult<Self> {
        impl_binop!(self, other, *, "mul", crate::backend::registry::dispatch_mul_f32)
    }

    /// Element-wise division (true division)
    pub fn div(&self, other: &CoreArray) -> PyResult<Self> {
        impl_div_binop!(self, other, crate::backend::registry::dispatch_div_f32)
    }

    /// Element-wise maximum
    pub fn maximum(&self, other: &CoreArray) -> PyResult<Self> {
        impl_compare_binop!(self, other, max, "maximum")
    }

    /// Element-wise minimum
    pub fn minimum(&self, other: &CoreArray) -> PyResult<Self> {
        impl_compare_binop!(self, other, min, "minimum")
    }

    /// Sum all elements.
    pub fn sum(&self) -> PyResult<f32> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "sum() requires Float32",
            ));
        }
        let slice = unsafe { self.as_f32_slice() };
        unsafe {
            crate::backend::registry::dispatch_sum_f32(slice.as_ptr(), slice.len())
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        }
    }

    /// Mean of all elements.
    pub fn mean(&self) -> PyResult<f32> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "mean() requires Float32",
            ));
        }
        let slice = unsafe { self.as_f32_slice() };
        let count = slice.len();
        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mean of empty array",
            ));
        }
        unsafe {
            crate::backend::registry::dispatch_mean_f32(slice.as_ptr(), count)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        }
    }

    // ---- Axis-aware reductions (Phase 2) ------------------------------------

    /// Helper: compute output shape and iteration params for axis reduction.
    /// Returns (out_shape, outer_size, axis_size, inner_size).
    fn axis_reduction_params(&self, axis: usize) -> PyResult<(Vec<usize>, usize, usize, usize)> {
        if axis >= self.shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} out of range for {}D array",
                axis,
                self.shape.len()
            )));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let mut out_shape: Vec<usize> = self.shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok((out_shape, outer, axis_size, inner))
    }

    /// Sum along a single axis → reduced array.
    pub fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    out[o * inner + i] += src[o * axis_size * inner + a * inner + i];
                }
            }
        }
        CoreArray::new(out, out_shape)
    }

    /// Mean along a single axis.
    pub fn mean_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    out[o * inner + i] += src[o * axis_size * inner + a * inner + i];
                }
            }
        }
        let denom = axis_size as f32;
        for v in &mut out {
            *v /= denom;
        }
        CoreArray::new(out, out_shape)
    }

    /// Min along a single axis.
    pub fn min_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out = vec![f32::INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let v = src[o * axis_size * inner + a * inner + i];
                    if v < out[o * inner + i] {
                        out[o * inner + i] = v;
                    }
                }
            }
        }
        CoreArray::new(out, out_shape)
    }

    /// Max along a single axis.
    pub fn max_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out = vec![f32::NEG_INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let v = src[o * axis_size * inner + a * inner + i];
                    if v > out[o * inner + i] {
                        out[o * inner + i] = v;
                    }
                }
            }
        }
        CoreArray::new(out, out_shape)
    }

    /// Argmin along a single axis → integer indices.
    pub fn argmin_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out_idx = vec![0usize; outer * inner];
        let mut out_val = vec![f32::INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let v = src[o * axis_size * inner + a * inner + i];
                    if v < out_val[o * inner + i] {
                        out_val[o * inner + i] = v;
                        out_idx[o * inner + i] = a;
                    }
                }
            }
        }
        let data: Vec<f32> = out_idx.iter().map(|&x| x as f32).collect();
        CoreArray::new(data, out_shape)
    }

    /// Argmax along a single axis → integer indices.
    pub fn argmax_axis(&self, axis: usize) -> PyResult<Self> {
        let (out_shape, outer, axis_size, inner) = self.axis_reduction_params(axis)?;
        let src = unsafe { self.as_f32_slice() };
        let mut out_idx = vec![0usize; outer * inner];
        let mut out_val = vec![f32::NEG_INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let v = src[o * axis_size * inner + a * inner + i];
                    if v > out_val[o * inner + i] {
                        out_val[o * inner + i] = v;
                        out_idx[o * inner + i] = a;
                    }
                }
            }
        }
        let data: Vec<f32> = out_idx.iter().map(|&x| x as f32).collect();
        CoreArray::new(data, out_shape)
    }

    // =========================================================================
    // Comparison Operations (Phase 14) — return Bool arrays
    // =========================================================================

    /// Element-wise equality: self == other → Bool CoreArray
    pub fn eq(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a == b, "eq")
    }

    /// Element-wise not-equal: self != other
    pub fn ne(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a != b, "ne")
    }

    /// Element-wise greater-than: self > other
    pub fn gt(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a > b, "gt")
    }

    /// Element-wise less-than: self < other
    pub fn lt(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a < b, "lt")
    }

    /// Element-wise greater-or-equal: self >= other
    pub fn ge(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a >= b, "ge")
    }

    /// Element-wise less-or-equal: self <= other
    pub fn le(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_cmp(other, |a, b| a <= b, "le")
    }

    // =========================================================================
    // Unary Operations (Phase 15)
    // =========================================================================

    /// Element-wise negation: -self
    pub fn neg(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| -a, "neg")
    }

    /// Element-wise absolute value: |self|
    pub fn abs(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.abs(), "abs")
    }

    /// Element-wise square root: sqrt(self)
    pub fn sqrt(&self) -> PyResult<Self> {
        use crate::ops::ufunc::UFunc;
        crate::ops::ufunc::Sqrt.apply(self)
    }

    /// Element-wise exponential: exp(self)
    pub fn exp(&self) -> PyResult<Self> {
        use crate::ops::ufunc::UFunc;
        crate::ops::ufunc::Exp.apply(self)
    }

    /// Element-wise natural logarithm: log(self)
    pub fn log(&self) -> PyResult<Self> {
        use crate::ops::ufunc::UFunc;
        crate::ops::ufunc::Log.apply(self)
    }

    /// Element-wise sine: sin(self)
    pub fn sin(&self) -> PyResult<Self> {
        use crate::ops::ufunc::UFunc;
        crate::ops::ufunc::Sin.apply(self)
    }

    /// Element-wise power: self^exponent (scalar)
    pub fn pow(&self, exponent: f32) -> PyResult<Self> {
        self.elementwise_unary(|a| a.powf(exponent), "pow")
    }

    /// Element-wise power: self^other (array)
    pub fn power(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| a.powf(b), "power")
    }

    /// Element-wise modulo: self % other
    pub fn mod_op(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| a % b, "mod")
    }

    /// Element-wise floor division: self // other
    pub fn floor_div(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| (a / b).floor(), "floor_div")
    }

    // =========================================================================
    // Logical Operations (UFUNC CORE-12)
    // =========================================================================

    /// Element-wise logical AND (0.0 = false, non-zero = true)
    pub fn logical_and(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| {
                if a != 0.0 && b != 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            "logical_and",
        )
    }

    /// Element-wise logical OR
    pub fn logical_or(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| {
                if a != 0.0 || b != 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            "logical_or",
        )
    }

    /// Element-wise logical NOT (unary)
    pub fn logical_not(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| if a == 0.0 { 1.0 } else { 0.0 }, "logical_not")
    }

    /// Element-wise logical XOR
    pub fn logical_xor(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| {
                let ab = a != 0.0;
                let bb = b != 0.0;
                if ab ^ bb {
                    1.0
                } else {
                    0.0
                }
            },
            "logical_xor",
        )
    }

    // =========================================================================
    // Sorting Operations (UFUNC CORE-12)
    // =========================================================================

    /// Sort elements in ascending order, returns new sorted array.
    #[pyo3(signature = (descending=false))]
    pub fn sort(&self, descending: bool) -> PyResult<Self> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "sort() currently only supports Float32",
            ));
        }
        let mut data: Vec<f32> = unsafe { self.as_f32_slice() }.to_vec();
        if data.len() > 100_000 {
            use rayon::prelude::*;
            data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        if descending {
            data.reverse();
        }
        CoreArray::new(data, self.shape.clone())
    }

    /// Return indices that would sort the array.
    #[pyo3(signature = (descending=false))]
    pub fn argsort(&self, descending: bool) -> PyResult<Self> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "argsort() currently only supports Float32",
            ));
        }
        let slice = unsafe { self.as_f32_slice() };
        let mut indices: Vec<usize> = (0..slice.len()).collect();
        indices.sort_by(|&i, &j| {
            slice[i]
                .partial_cmp(&slice[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if descending {
            indices.reverse();
        }
        let data: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
        CoreArray::new(data, self.shape.clone())
    }

    // =========================================================================
    // Searching Operations (UFUNC CORE-12)
    // =========================================================================

    /// Return index of maximum element.
    pub fn argmax(&self) -> PyResult<usize> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "argmax() requires Float32",
            ));
        }
        let slice = unsafe { self.as_f32_slice() };
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "argmax of empty array",
            ));
        }
        let mut max_idx = 0;
        let mut max_val = slice[0];
        for (i, &v) in slice.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Ok(max_idx)
    }

    /// Return index of minimum element.
    pub fn argmin(&self) -> PyResult<usize> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "argmin() requires Float32",
            ));
        }
        let slice = unsafe { self.as_f32_slice() };
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "argmin of empty array",
            ));
        }
        let mut min_idx = 0;
        let mut min_val = slice[0];
        for (i, &v) in slice.iter().enumerate().skip(1) {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        Ok(min_idx)
    }

    // =========================================================================
    // Utility Operations (UFUNC CORE-12)
    // =========================================================================

    /// Element-wise is_even: returns 1.0 where x % 2 == 0, else 0.0
    pub fn is_even(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| if a % 2.0 == 0.0 { 1.0 } else { 0.0 }, "is_even")
    }

    /// Element-wise is_odd: returns 1.0 where x % 2 != 0, else 0.0
    pub fn is_odd(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| if a % 2.0 != 0.0 { 1.0 } else { 0.0 }, "is_odd")
    }

    /// Create evenly spaced values over [start, stop].
    #[staticmethod]
    #[pyo3(signature = (start, stop, num))]
    pub fn linspace(start: f32, stop: f32, num: usize) -> PyResult<Self> {
        if num == 0 {
            return CoreArray::new(vec![], vec![0]);
        }
        if num == 1 {
            return CoreArray::new(vec![start], vec![1]);
        }
        let step = (stop - start) / (num - 1) as f32;
        let data: Vec<f32> = (0..num).map(|i| start + step * i as f32).collect();
        CoreArray::new(data, vec![num])
    }

    // =========================================================================
    // Advanced Indexing (UFUNC CORE-12)
    // =========================================================================

    /// Take elements from array at given indices.
    pub fn take(&self, indices: &CoreArray) -> PyResult<Self> {
        if self.dtype != DType::Float32 || indices.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "take() currently only supports Float32",
            ));
        }
        let src = unsafe { self.as_f32_slice() };
        let idx_slice = unsafe { indices.as_f32_slice() };
        let mut result = Vec::with_capacity(idx_slice.len());
        for &idx_f in idx_slice {
            let idx = idx_f as usize;
            if idx >= src.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {} out of bounds for array with {} elements",
                    idx,
                    src.len()
                )));
            }
            result.push(src[idx]);
        }
        CoreArray::new(result, indices.shape.clone())
    }

    /// Select elements where mask is non-zero.
    pub fn boolean_index(&self, mask: &CoreArray) -> PyResult<Self> {
        if self.dtype != DType::Float32 || mask.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "boolean_index() currently only supports Float32",
            ));
        }
        let src = unsafe { self.as_f32_slice() };
        let mask_slice = unsafe { mask.as_f32_slice() };
        if src.len() != mask_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "boolean_index: array length {} != mask length {}",
                src.len(),
                mask_slice.len()
            )));
        }
        let result: Vec<f32> = src
            .iter()
            .zip(mask_slice.iter())
            .filter(|(_, &m)| m != 0.0)
            .map(|(&v, _)| v)
            .collect();
        let count = result.len();
        CoreArray::new(result, vec![count])
    }

    // =========================================================================
    // Trigonometric Operations (UFUNC CORE-50)
    // =========================================================================

    pub fn cos(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.cos(), "cos")
    }
    pub fn tan(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.tan(), "tan")
    }
    pub fn arcsin(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.asin(), "arcsin")
    }
    pub fn arccos(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.acos(), "arccos")
    }
    pub fn arctan(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.atan(), "arctan")
    }
    pub fn arctan2(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| a.atan2(b), "arctan2")
    }

    // =========================================================================
    // Hyperbolic Operations (UFUNC CORE-50)
    // =========================================================================

    pub fn sinh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.sinh(), "sinh")
    }
    pub fn cosh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.cosh(), "cosh")
    }
    pub fn tanh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.tanh(), "tanh")
    }
    pub fn arcsinh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.asinh(), "arcsinh")
    }
    pub fn arccosh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.acosh(), "arccosh")
    }
    pub fn arctanh(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.atanh(), "arctanh")
    }

    // =========================================================================
    // Exponential / Logarithmic (UFUNC CORE-50)
    // =========================================================================

    pub fn exp2(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.exp2(), "exp2")
    }
    pub fn expm1(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.exp() - 1.0, "expm1")
    }
    pub fn log2(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.log2(), "log2")
    }
    pub fn log10(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.log10(), "log10")
    }
    pub fn log1p(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| (1.0 + a).ln(), "log1p")
    }

    // =========================================================================
    // Rounding Operations (UFUNC CORE-50)
    // =========================================================================

    pub fn floor_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.floor(), "floor")
    }
    pub fn ceil_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.ceil(), "ceil")
    }
    pub fn round_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.round(), "round")
    }
    pub fn trunc_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.trunc(), "trunc")
    }
    pub fn rint(&self) -> PyResult<Self> {
        // Round to nearest even (banker's rounding)
        self.elementwise_unary(
            |a| {
                let rounded = a.round();
                if (a - rounded).abs() == 0.5 {
                    // Round to even
                    if rounded as i32 % 2 != 0 {
                        rounded - a.signum()
                    } else {
                        rounded
                    }
                } else {
                    rounded
                }
            },
            "rint",
        )
    }

    // =========================================================================
    // Sign / Clip (UFUNC CORE-50)
    // =========================================================================

    pub fn sign_op(&self) -> PyResult<Self> {
        self.elementwise_unary(
            |a| {
                if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            "sign",
        )
    }

    #[pyo3(signature = (min_val, max_val))]
    pub fn clip(&self, min_val: f32, max_val: f32) -> PyResult<Self> {
        self.elementwise_unary(|a| a.max(min_val).min(max_val), "clip")
    }

    pub fn copysign(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| a.abs() * b.signum(), "copysign")
    }

    // =========================================================================
    // Special Functions (UFUNC CORE-50)
    // =========================================================================

    pub fn square(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a * a, "square")
    }
    pub fn reciprocal(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| 1.0 / a, "reciprocal")
    }
    pub fn cbrt(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.cbrt(), "cbrt")
    }
    pub fn degrees_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.to_degrees(), "degrees")
    }
    pub fn radians_op(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| a.to_radians(), "radians")
    }
    pub fn hypot(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| a.hypot(b), "hypot")
    }

    // =========================================================================
    // Bitwise Operations (UFUNC CORE-50) — cast f32→i32 internally
    // =========================================================================

    pub fn bitwise_and(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| ((a as i32) & (b as i32)) as f32,
            "bitwise_and",
        )
    }
    pub fn bitwise_or(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(other, |a, b| ((a as i32) | (b as i32)) as f32, "bitwise_or")
    }
    pub fn bitwise_xor(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| ((a as i32) ^ (b as i32)) as f32,
            "bitwise_xor",
        )
    }
    pub fn bitwise_not(&self) -> PyResult<Self> {
        self.elementwise_unary(|a| (!(a as i32)) as f32, "bitwise_not")
    }
    pub fn left_shift(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| ((a as i32) << (b as i32)) as f32,
            "left_shift",
        )
    }
    pub fn right_shift(&self, other: &CoreArray) -> PyResult<Self> {
        self.elementwise_binop(
            other,
            |a, b| ((a as i32) >> (b as i32)) as f32,
            "right_shift",
        )
    }

    // =========================================================================
    // Reduction Operations (UFUNC CORE-50)
    // =========================================================================

    /// Product of all elements.
    pub fn prod(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        Ok(slice.iter().product())
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "std of empty array",
            ));
        }
        let n = slice.len() as f32;
        let mean = slice.iter().sum::<f32>() / n;
        let variance = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        Ok(variance.sqrt())
    }

    /// Population variance.
    pub fn var(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "var of empty array",
            ));
        }
        let n = slice.len() as f32;
        let mean = slice.iter().sum::<f32>() / n;
        Ok(slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n)
    }

    /// Cumulative sum, returns new array.
    pub fn cumsum(&self) -> PyResult<Self> {
        let slice = unsafe { self.as_f32_slice() };
        let mut result = Vec::with_capacity(slice.len());
        let mut acc = 0.0f32;
        for &v in slice {
            acc += v;
            result.push(acc);
        }
        CoreArray::new(result, self.shape.clone())
    }

    /// Cumulative product, returns new array.
    pub fn cumprod(&self) -> PyResult<Self> {
        let slice = unsafe { self.as_f32_slice() };
        let mut result = Vec::with_capacity(slice.len());
        let mut acc = 1.0f32;
        for &v in slice {
            acc *= v;
            result.push(acc);
        }
        CoreArray::new(result, self.shape.clone())
    }

    /// Sum ignoring NaN values.
    pub fn nansum(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        Ok(slice.iter().filter(|x| !x.is_nan()).sum())
    }

    /// Mean ignoring NaN values.
    pub fn nanmean(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        let valid: Vec<f32> = slice.iter().filter(|x| !x.is_nan()).copied().collect();
        if valid.is_empty() {
            return Ok(f32::NAN);
        }
        Ok(valid.iter().sum::<f32>() / valid.len() as f32)
    }

    /// Max ignoring NaN values.
    pub fn nanmax(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        let result = slice
            .iter()
            .filter(|x| !x.is_nan())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(result)
    }

    /// Min ignoring NaN values.
    pub fn nanmin(&self) -> PyResult<f32> {
        let slice = unsafe { self.as_f32_slice() };
        let result = slice
            .iter()
            .filter(|x| !x.is_nan())
            .copied()
            .fold(f32::INFINITY, f32::min);
        Ok(result)
    }

    // =========================================================================
    // Creation from existing (UFUNC CORE-50)
    // =========================================================================

    /// Create zeros with same shape.
    pub fn zeros_like(&self) -> PyResult<Self> {
        CoreArray::zeros(self.shape.clone(), Some(self.dtype))
    }

    /// Create ones with same shape.
    pub fn ones_like(&self) -> PyResult<Self> {
        let count = self.element_count();
        CoreArray::new(vec![1.0f32; count], self.shape.clone())
    }

    /// Create filled with value, same shape.
    pub fn full_like(&self, value: f32) -> PyResult<Self> {
        let count = self.element_count();
        CoreArray::new(vec![value; count], self.shape.clone())
    }

    // =========================================================================
    // Shape Manipulation (Phase 9)
    // =========================================================================

    /// Reshape the array (zero-copy when element count matches).
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_count: usize = new_shape.iter().product();
        if new_count != self.element_count() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot reshape array of {} elements into shape {:?} ({} elements)",
                self.element_count(),
                new_shape,
                new_count
            )));
        }

        // Zero-copy: reuse buffer, just change shape/strides metadata
        let byte_len = self.buffer.len();
        let new_buffer = AlignedBuffer::new(byte_len);
        unsafe {
            ptr::copy_nonoverlapping(self.buffer.as_ptr(), new_buffer.as_mut_ptr(), byte_len);
        }

        let _strides = Self::compute_c_strides(&new_shape, self.dtype.itemsize());
        Ok(Self::from_raw(new_buffer, None, new_shape, self.dtype))
    }

    /// Transpose a 2D array (swap rows and columns).
    pub fn transpose(&self) -> PyResult<Self> {
        if self.shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transpose() requires 2D array, got {}D",
                self.shape.len()
            )));
        }
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "transpose() currently only supports Float32",
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let count = rows * cols;
        let mut out_data = vec![0.0f32; count];

        let src = unsafe { self.as_f32_slice() };
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = src[r * cols + c];
            }
        }

        CoreArray::new(out_data, vec![cols, rows])
    }

    /// Linear algebra: Matrix inverse.
    pub fn linalg_inv(&self) -> PyResult<Self> {
        linalg_inv(self)
    }

    /// Linear algebra: Determinant.
    pub fn linalg_det(&self) -> PyResult<f32> {
        linalg_det(self)
    }

    /// Linear algebra: Norm (Frobenius).
    pub fn linalg_norm(&self) -> PyResult<f32> {
        linalg_norm(self)
    }

    /// Cast to a new dtype (returns a copy).
    pub fn astype(&self, target_dtype: DType) -> PyResult<Self> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }

        let count = self.element_count();

        if target_dtype == DType::String {
            let data_str = self.to_list_vec_str()?;
            return CoreArray::from_strings(data_str, self.shape.clone());
        }

        if self.dtype == DType::String {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Casting from String to numeric is not yet supported",
            ));
        }

        // Numeric to Numeric
        let src_f32 = unsafe { self.as_f32_slice() };
        let byte_len = count * target_dtype.itemsize();
        let buffer = AlignedBuffer::new(byte_len);

        match target_dtype {
            DType::Float32 => {
                let data: Vec<f32> = src_f32.to_vec();
                CoreArray::new(data, self.shape.clone())
            }
            DType::Float64 => {
                let data: Vec<f64> = src_f32.iter().map(|&x| x as f64).collect();
                let ptr = buffer.as_mut_ptr() as *mut f64;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, count);
                }
                Ok(Self::from_raw(
                    buffer,
                    None,
                    self.shape.clone(),
                    target_dtype,
                ))
            }
            DType::Int32 => {
                let data: Vec<i32> = src_f32.iter().map(|&x| x as i32).collect();
                let ptr = buffer.as_mut_ptr() as *mut i32;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, count);
                }
                Ok(Self::from_raw(
                    buffer,
                    None,
                    self.shape.clone(),
                    target_dtype,
                ))
            }
            DType::Int64 => {
                let data: Vec<i64> = src_f32.iter().map(|&x| x as i64).collect();
                let ptr = buffer.as_mut_ptr() as *mut i64;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, count);
                }
                Ok(Self::from_raw(
                    buffer,
                    None,
                    self.shape.clone(),
                    target_dtype,
                ))
            }
            DType::Bool => {
                let data: Vec<u8> = src_f32
                    .iter()
                    .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
                    .collect();
                let ptr = buffer.as_mut_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, count);
                }
                Ok(Self::from_raw(
                    buffer,
                    None,
                    self.shape.clone(),
                    target_dtype,
                ))
            }
            DType::String => unreachable!(),
        }
    }

    /// Internal helper to get values as a Vec<String>
    fn to_list_vec_str(&self) -> PyResult<Vec<String>> {
        let count = self.element_count();
        match self.dtype {
            DType::Float32 => {
                let slice = unsafe { self.as_f32_slice() };
                Ok(slice.iter().map(|s| s.to_string()).collect())
            }
            DType::Float64 => {
                let slice = unsafe { self.as_f64_slice() };
                Ok(slice.iter().map(|s| s.to_string()).collect())
            }
            DType::Int32 => {
                let slice = unsafe { self.as_i32_slice() };
                Ok(slice.iter().map(|s| s.to_string()).collect())
            }
            DType::Int64 => {
                let slice = unsafe { self.as_i64_slice() };
                Ok(slice.iter().map(|s| s.to_string()).collect())
            }
            DType::Bool => {
                let slice = unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), count) };
                Ok(slice
                    .iter()
                    .map(|&v| if v != 0 { "True" } else { "False" }.to_string())
                    .collect())
            }
            DType::String => {
                if let Some(ref data) = self.data_str {
                    Ok(data.clone())
                } else {
                    Ok(vec!["".to_string(); count])
                }
            }
        }
    }

    /// Create an identity matrix of size n×m (default m=n).
    #[staticmethod]
    #[pyo3(signature = (n, m=None))]
    pub fn eye(n: usize, m: Option<usize>) -> PyResult<Self> {
        let cols = m.unwrap_or(n);
        let mut data = vec![0.0f32; n * cols];
        for i in 0..n.min(cols) {
            data[i * cols + i] = 1.0;
        }
        CoreArray::new(data, vec![n, cols])
    }

    /// Flatten to 1D (contiguous copy).
    pub fn ravel(&self) -> PyResult<Self> {
        self.reshape(vec![self.element_count()])
    }

    /// Flatten to 1D (always returns a copy).
    pub fn flatten(&self) -> PyResult<Self> {
        self.reshape(vec![self.element_count()])
    }

    /// Remove axes of length 1.
    #[pyo3(signature = (axis=None))]
    pub fn squeeze(&self, axis: Option<usize>) -> PyResult<Self> {
        let new_shape: Vec<usize> = match axis {
            Some(ax) => {
                if ax >= self.shape.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "axis {} out of range for {}D array",
                        ax,
                        self.shape.len()
                    )));
                }
                if self.shape[ax] != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "cannot squeeze axis {} with size {}",
                        ax, self.shape[ax]
                    )));
                }
                self.shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &s)| s)
                    .collect()
            }
            None => self.shape.iter().copied().filter(|&s| s != 1).collect(),
        };
        // If nothing to squeeze, return clone
        if new_shape.is_empty() {
            return self.reshape(vec![1]);
        }
        self.reshape(new_shape)
    }

    /// Split array into sub-arrays along axis 0.
    #[pyo3(signature = (indices_or_sections))]
    pub fn split(&self, indices_or_sections: Vec<usize>) -> PyResult<Vec<Self>> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "split() currently only supports Float32",
            ));
        }
        let src = unsafe { self.as_f32_slice() };
        let n = self.shape[0];
        let row_size: usize = self.shape[1..].iter().product::<usize>().max(1);

        let mut result = Vec::new();
        let mut prev = 0usize;
        for &idx in &indices_or_sections {
            let idx = idx.min(n);
            let start = prev * row_size;
            let end = idx * row_size;
            let chunk = src[start..end].to_vec();
            let mut new_shape = vec![idx - prev];
            new_shape.extend_from_slice(&self.shape[1..]);
            result.push(CoreArray::new(chunk, new_shape)?);
            prev = idx;
        }
        // Remaining
        if prev < n {
            let start = prev * row_size;
            let chunk = src[start..].to_vec();
            let mut new_shape = vec![n - prev];
            new_shape.extend_from_slice(&self.shape[1..]);
            result.push(CoreArray::new(chunk, new_shape)?);
        }
        Ok(result)
    }

    /// Stack arrays along a new axis (axis=0).
    #[staticmethod]
    #[pyo3(signature = (arrays, axis=None))]
    pub fn stack(arrays: Vec<pyo3::PyRef<'_, CoreArray>>, axis: Option<usize>) -> PyResult<Self> {
        if arrays.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "need at least one array to stack",
            ));
        }
        let ax = axis.unwrap_or(0);
        let base_shape = &arrays[0].shape;
        for arr in &arrays[1..] {
            if arr.shape != *base_shape {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "all arrays must have the same shape: {:?} vs {:?}",
                    base_shape, arr.shape
                )));
            }
        }

        // For axis=0: new shape is [n_arrays, *base_shape]
        if ax == 0 {
            let mut all_data = Vec::new();
            for arr in &arrays {
                let s = unsafe { arr.as_f32_slice() };
                all_data.extend_from_slice(s);
            }
            let mut new_shape = vec![arrays.len()];
            new_shape.extend_from_slice(base_shape);
            return CoreArray::new(all_data, new_shape);
        }

        // For other axes, we need to interleave data
        if ax > base_shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} out of range for stacking {}D arrays",
                ax,
                base_shape.len()
            )));
        }

        // General axis stacking: insert new dim at position ax
        let n_arrays = arrays.len();
        let mut new_shape = base_shape.clone();
        new_shape.insert(ax, n_arrays);
        let total = new_shape.iter().product::<usize>();
        let mut out = vec![0.0f32; total];

        // Compute strides for the output shape
        let ndim = new_shape.len();
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
        }

        // For each array, copy its elements into the right positions
        let elem_count = base_shape.iter().product::<usize>();
        for (arr_idx, arr) in arrays.iter().enumerate() {
            let src = unsafe { arr.as_f32_slice() };
            #[allow(clippy::needless_range_loop)]
            for flat_i in 0..elem_count {
                // Convert flat_i to multi-index in base_shape
                let mut remaining = flat_i;
                let mut out_flat = 0usize;
                for d in 0..base_shape.len() {
                    let dim_idx = if d < base_shape.len() - 1 {
                        let stride: usize = base_shape[d + 1..].iter().product();
                        let idx = remaining / stride;
                        remaining %= stride;
                        idx
                    } else {
                        remaining
                    };
                    // Map to output index: dimensions before ax stay, ax gets arr_idx, after shift by 1
                    let out_d = if d < ax { d } else { d + 1 };
                    out_flat += dim_idx * out_strides[out_d];
                }
                out_flat += arr_idx * out_strides[ax];
                out[out_flat] = src[flat_i];
            }
        }

        CoreArray::new(out, new_shape)
    }

    // =========================================================================
    // Indexing (Phase 10)
    // =========================================================================

    /// Get a single f32 element by flat index.
    pub fn get_scalar(&self, index: usize) -> PyResult<f32> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "get_scalar() only supports Float32",
            ));
        }
        let count = self.element_count();
        if index >= count {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for array with {} elements",
                index, count
            )));
        }
        let slice = unsafe { self.as_f32_slice() };
        Ok(slice[index])
    }

    /// Index along the first axis (like array[i]).
    /// For 1D: returns a scalar wrapper (1-element array).
    /// For ND: returns a sub-array with shape[1:].
    pub fn getitem(&self, index: i64) -> PyResult<Self> {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "getitem() only supports Float32",
            ));
        }
        if self.shape.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "cannot index a scalar array",
            ));
        }

        let dim0 = self.shape[0] as i64;
        let actual_idx = if index < 0 { dim0 + index } else { index };
        if actual_idx < 0 || actual_idx >= dim0 {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for axis 0 with size {}",
                index, dim0
            )));
        }
        let idx = actual_idx as usize;

        let src = unsafe { self.as_f32_slice() };

        if self.shape.len() == 1 {
            // 1D: return scalar as 1-element array
            CoreArray::new(vec![src[idx]], vec![1])
        } else {
            // ND: slice along first axis
            let inner_count: usize = self.shape[1..].iter().product();
            let sub_data = src[idx * inner_count..(idx + 1) * inner_count].to_vec();
            let new_shape = self.shape[1..].to_vec();
            CoreArray::new(sub_data, new_shape)
        }
    }

    // =========================================================================
    // Linear Algebra Extensions
    // =========================================================================

    /// Cholesky decomposition
    pub fn cholesky(&self) -> PyResult<Self> {
        crate::linalg::decompose::solve_cholesky(self)
    }

    /// Matrix multiplication (supports Float32 and Float64)
    pub fn matmul(&self, other: &CoreArray) -> PyResult<Self> {
        if self.dtype != other.dtype {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "matmul() dtype mismatch: {:?} vs {:?}",
                self.dtype, other.dtype
            )));
        }

        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "matmul() currently only supports 2D arrays",
            ));
        }

        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k1 != k2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "matmul() shape mismatch: inner dimensions {} and {} must match",
                k1, k2
            )));
        }

        match self.dtype {
            DType::Float32 => {
                let result = CoreArray::zeros(vec![m, n], Some(DType::Float32))?;
                unsafe {
                    crate::backend::registry::dispatch_matmul_f32(
                        self.buffer.as_ptr() as *const f32,
                        other.buffer.as_ptr() as *const f32,
                        result.buffer.as_ptr() as *mut f32,
                        m,
                        k1,
                        n,
                    )
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                }
                Ok(result)
            }
            DType::Float64 => {
                let result = CoreArray::zeros(vec![m, n], Some(DType::Float64))?;
                unsafe {
                    crate::ops::matmul::matmul_f64_cpu_dispatch(
                        self.buffer.as_ptr() as *const f64,
                        other.buffer.as_ptr() as *const f64,
                        result.buffer.as_ptr() as *mut f64,
                        m,
                        k1,
                        n,
                    );
                }
                Ok(result)
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "matmul() not supported for dtype {:?}",
                self.dtype
            ))),
        }
    }

    fn __repr__(&self) -> String {
        let count = self.element_count();
        let preview = if self.dtype == DType::Float32 && count <= 8 {
            let slice =
                unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const f32, count) };
            format!("{:?}", slice)
        } else {
            format!("[{} elements]", count)
        };
        format!(
            "CoreArray(shape={:?}, dtype={:?}, data={})",
            self.shape, self.dtype, preview
        )
    }
}

// Non-PyO3 methods (Rust-internal API)
impl CoreArray {
    /// Compute C-contiguous (row-major) byte strides for a given shape.
    pub fn compute_c_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![0usize; ndim];
        strides[ndim - 1] = itemsize;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // =========================================================================
    // Raw Pointers / Unsafe Access
    // =========================================================================

    /// Get a slice to the underlying f32 buffer.
    pub unsafe fn as_f32_slice(&self) -> &[f32] {
        std::slice::from_raw_parts(self.buffer.as_ptr() as *const f32, self.element_count())
    }

    pub unsafe fn as_f64_slice(&self) -> &[f64] {
        std::slice::from_raw_parts(self.buffer.as_ptr() as *const f64, self.element_count())
    }

    pub unsafe fn as_i32_slice(&self) -> &[i32] {
        std::slice::from_raw_parts(self.buffer.as_ptr() as *const i32, self.element_count())
    }

    pub unsafe fn as_i64_slice(&self) -> &[i64] {
        std::slice::from_raw_parts(self.buffer.as_ptr() as *const i64, self.element_count())
    }

    pub unsafe fn as_bool_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.buffer.as_ptr(), self.element_count())
    }

    pub unsafe fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut f32, self.element_count())
    }

    /// Generic element-wise binary operation (Float32 only fallback).
    fn elementwise_binop<F>(&self, other: &CoreArray, op: F, name: &str) -> PyResult<Self>
    where
        F: Fn(f32, f32) -> f32 + Sync + Send,
    {
        if self.dtype != DType::Float32 || other.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} currently only supports Float32 (fallback)",
                name
            )));
        }
        if self.shape != other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for {}: {:?} vs {:?}",
                name, self.shape, other.shape
            )));
        }

        let a = unsafe { self.as_f32_slice() };
        let b = unsafe { other.as_f32_slice() };

        use rayon::prelude::*;
        let result_data: Vec<f32> = a
            .par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| op(x, y))
            .collect();

        CoreArray::new(result_data, self.shape.clone())
    }

    /// Generic element-wise comparison operation (returns Float32 where 1.0=true, 0.0=false).
    fn elementwise_cmp<F>(&self, other: &CoreArray, op: F, name: &str) -> PyResult<Self>
    where
        F: Fn(f32, f32) -> bool + Sync + Send,
    {
        if self.dtype != DType::Float32 || other.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} currently only supports Float32",
                name
            )));
        }

        let a = unsafe { self.as_f32_slice() };
        let b = unsafe { other.as_f32_slice() };
        use rayon::prelude::*;

        if other.element_count() == 1 && !b.is_empty() {
            let scalar_b = b[0];
            let result_data: Vec<f32> = a
                .par_iter()
                .map(|&x| if op(x, scalar_b) { 1.0 } else { 0.0 })
                .collect();
            CoreArray::new(result_data, self.shape.clone())
        } else if self.element_count() == 1 && !a.is_empty() {
            let scalar_a = a[0];
            let result_data: Vec<f32> = b
                .par_iter()
                .map(|&y| if op(scalar_a, y) { 1.0 } else { 0.0 })
                .collect();
            CoreArray::new(result_data, other.shape.clone())
        } else {
            if self.shape != other.shape {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Shape mismatch for {}: {:?} vs {:?}",
                    name, self.shape, other.shape
                )));
            }
            let result_data: Vec<f32> = a
                .par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| if op(x, y) { 1.0 } else { 0.0 })
                .collect();
            CoreArray::new(result_data, self.shape.clone())
        }
    }

    /// Generic element-wise unary operation.
    fn elementwise_unary<F>(&self, op: F, name: &str) -> PyResult<Self>
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        if self.dtype != DType::Float32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} currently only supports Float32",
                name
            )));
        }

        let a = unsafe { self.as_f32_slice() };
        use rayon::prelude::*;
        let result_data: Vec<f32> = a.par_iter().map(|&x| op(x)).collect();

        CoreArray::new(result_data, self.shape.clone())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_array() {
        let t = CoreArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.element_count(), 6);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.shape(), vec![2, 3]);
        assert!(t.is_contiguous());
        assert_eq!(t.nbytes(), 24); // 6 * 4
    }

    #[test]
    fn test_alignment() {
        let t = CoreArray::new(vec![0.0; 1024], vec![1024]).unwrap();
        assert_eq!(t.data_ptr() % 64, 0, "Buffer must be 64-byte aligned");
    }

    #[test]
    fn test_zeros() {
        let t = CoreArray::zeros(vec![3, 3], None).unwrap();
        let data = unsafe { t.as_f32_slice() };
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = CoreArray::ones(vec![2, 2]).unwrap();
        let data = unsafe { t.as_f32_slice() };
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_shape_mismatch() {
        let result = CoreArray::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_c_strides() {
        let strides = CoreArray::compute_c_strides(&[2, 3, 4], 4);
        assert_eq!(strides, vec![48, 16, 4]); // 3*4*4, 4*4, 4
    }
}
