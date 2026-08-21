use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::array::core_array::CoreArray;
use crate::array::DType;

#[pyclass(module = "corepy._corepy_rust", name = "_RustSeries", from_py_object)]
#[derive(Clone, Debug)]
pub struct Series {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub data: CoreArray,
    #[pyo3(get, set)]
    pub index: Vec<String>,
}

impl Series {
    /// Internal Rust constructor.
    pub fn new(name: String, data: CoreArray, index: Option<Vec<String>>) -> PyResult<Self> {
        let count = data.element_count();
        let idx = match index {
            Some(idx) => {
                if idx.len() != count {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Index length must match data length",
                    ));
                }
                idx
            }
            None => (0..count).map(|i| i.to_string()).collect(),
        };

        Ok(Series {
            name,
            data,
            index: idx,
        })
    }
}

#[pymethods]
impl Series {
    /// PyO3 constructor for Pickle and direct Python use.
    #[new]
    #[pyo3(signature = (name=None, data=None, index=None))]
    pub fn py_new(
        name: Option<String>,
        data: Option<pyo3::PyRef<'_, CoreArray>>,
        index: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| "unnamed".to_string());

        // If data is missing (e.g. during Pickle reconstruction), create an empty f32 array
        if data.is_none() {
            return Self::new(
                name,
                CoreArray::new_from_vec::<f32>(vec![], vec![0], DType::Float32)?,
                index,
            );
        }

        let data_ref = data.unwrap();
        Self::new(name, data_ref.clone(), index)
    }

    pub fn head(&self, n: Option<isize>) -> PyResult<Series> {
        let count = self.data.element_count();
        let n_raw = n.unwrap_or(5);
        let n = if n_raw < 0 {
            count.saturating_sub(n_raw.unsigned_abs())
        } else {
            (n_raw as usize).min(count)
        };
        let mut new_index = Vec::with_capacity(n);
        new_index.extend_from_slice(&self.index[..n]);

        let dtype = self.data.dtype();
        let new_core_array = if dtype == DType::String {
            if let Some(ref data) = self.data.data_str {
                let subset = data[..n].to_vec();
                CoreArray::from_strings(subset, vec![n])?
            } else {
                CoreArray::from_strings(vec!["".to_string(); n], vec![n])?
            }
        } else {
            let mut new_data_vec = Vec::with_capacity(n);
            if dtype == DType::Float32 {
                let data_slice =
                    unsafe { std::slice::from_raw_parts(self.data.data_ptr() as *const f32, n) };
                new_data_vec.extend_from_slice(data_slice);
            } else if dtype == DType::Int32 {
                let data_slice =
                    unsafe { std::slice::from_raw_parts(self.data.data_ptr() as *const i32, n) };
                for &val in data_slice {
                    new_data_vec.push(val as f32);
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Unsupported DType for head",
                ));
            }
            CoreArray::new(new_data_vec, vec![n])?
        };

        Ok(Series {
            name: self.name.clone(),
            data: new_core_array,
            index: new_index,
        })
    }

    pub fn tail(&self, n: Option<isize>) -> PyResult<Series> {
        let count = self.data.element_count();
        let n_raw = n.unwrap_or(5);
        let n = if n_raw < 0 {
            count.saturating_sub(n_raw.unsigned_abs())
        } else {
            (n_raw as usize).min(count)
        };
        let start = count - n;
        let mut new_index = Vec::with_capacity(n);
        new_index.extend_from_slice(&self.index[start..]);

        let dtype = self.data.dtype();
        let new_core_array = if dtype == DType::String {
            if let Some(ref data) = self.data.data_str {
                let subset = data[start..].to_vec();
                CoreArray::from_strings(subset, vec![n])?
            } else {
                CoreArray::from_strings(vec!["".to_string(); n], vec![n])?
            }
        } else {
            let mut new_data_vec = Vec::with_capacity(n);
            if dtype == DType::Float32 {
                let data_slice = unsafe {
                    std::slice::from_raw_parts((self.data.data_ptr() as *const f32).add(start), n)
                };
                new_data_vec.extend_from_slice(data_slice);
            } else if dtype == DType::Int32 {
                let data_slice = unsafe {
                    std::slice::from_raw_parts((self.data.data_ptr() as *const i32).add(start), n)
                };
                for &val in data_slice {
                    new_data_vec.push(val as f32);
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Unsupported DType for tail",
                ));
            }
            CoreArray::new(new_data_vec, vec![n])?
        };

        Ok(Series {
            name: self.name.clone(),
            data: new_core_array,
            index: new_index,
        })
    }

    #[getter]
    pub fn values(&self) -> PyResult<CoreArray> {
        Ok(self.data.clone())
    }

    #[getter]
    pub fn dtype(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.data.dtype()))
    }

    pub fn unique(&self) -> PyResult<Series> {
        let dtype = self.data.dtype();
        let mut seen = std::collections::HashSet::new();

        if dtype == DType::String {
            let mut unique_strings = Vec::new();
            if let Some(ref data) = self.data.data_str {
                for s in data {
                    if seen.insert(s.clone()) {
                        unique_strings.push(s.clone());
                    }
                }
            }
            let n = unique_strings.len();
            let new_array = CoreArray::from_strings(unique_strings, vec![n])?;
            let new_index: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            return Ok(Series {
                name: self.name.clone(),
                data: new_array,
                index: new_index,
            });
        }

        if dtype != DType::Float32 && dtype != DType::Int32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported DType for unique",
            ));
        }

        let mut unique_vals = Vec::new();

        if dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    self.data.data_ptr() as *const f32,
                    self.data.element_count(),
                )
            };
            for &val in slice {
                let key = val.to_bits(); // Handle floats
                if seen.insert(key.to_string()) {
                    unique_vals.push(val);
                }
            }
        } else if dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    self.data.data_ptr() as *const i32,
                    self.data.element_count(),
                )
            };
            for &val in slice {
                if seen.insert(val.to_string()) {
                    unique_vals.push(val as f32);
                }
            }
        }

        let n = unique_vals.len();
        let new_array = CoreArray::new(unique_vals, vec![n])?;
        let new_index: Vec<String> = (0..n).map(|i| i.to_string()).collect();

        Ok(Series {
            name: self.name.clone(),
            data: new_array,
            index: new_index,
        })
    }

    pub fn value_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let dtype = self.data.dtype();

        if dtype == DType::String {
            let mut counts: HashMap<String, usize> = HashMap::new();
            if let Some(ref data) = self.data.data_str {
                for s in data {
                    *counts.entry(s.clone()).or_insert(0) += 1;
                }
            }
            for (val, count) in counts {
                dict.set_item(val, count)?;
            }
            return Ok(dict);
        }

        if dtype == DType::Float32 {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    self.data.data_ptr() as *const f32,
                    self.data.element_count(),
                )
            };
            for &val in slice {
                *counts.entry(val.to_bits()).or_insert(0) += 1;
            }
            for (bits, count) in counts {
                let val = f32::from_bits(bits);
                dict.set_item(val, count)?;
            }
        } else if dtype == DType::Int32 {
            let mut counts: HashMap<i32, usize> = HashMap::new();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    self.data.data_ptr() as *const i32,
                    self.data.element_count(),
                )
            };
            for &val in slice {
                *counts.entry(val).or_insert(0) += 1;
            }
            for (val, count) in counts {
                dict.set_item(val, count)?;
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported DType for value_counts",
            ));
        }

        Ok(dict)
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.name)?;
        dict.set_item("index", &self.index)?;
        dict.set_item("dtype", format!("{:?}", self.data.dtype()))?;

        let data_list = self.data.to_list(py)?;
        dict.set_item("data", data_list)?;

        Ok(dict)
    }

    pub fn __setstate__(&mut self, state: pyo3::Py<pyo3::PyAny>, py: Python<'_>) -> PyResult<()> {
        let dict = state.bind(py).cast::<PyDict>()?;

        self.name = dict
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing name"))?
            .extract()?;
        self.index = dict
            .get_item("index")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing index"))?
            .extract()?;

        let dtype_str: String = dict
            .get_item("dtype")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing dtype"))?
            .extract()?;
        let data_list: Bound<'_, pyo3::types::PyList> = dict
            .get_item("data")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing data"))?
            .cast_into()?;

        let count = data_list.len();
        let shape = vec![count];

        if dtype_str.contains("String") {
            let data: Vec<String> = data_list.extract()?;
            self.data = CoreArray::from_strings(data, shape)?;
        } else if dtype_str.contains("Float32") {
            let data: Vec<f32> = data_list.extract()?;
            self.data = CoreArray::new(data, shape)?;
        } else if dtype_str.contains("Int32") {
            let data: Vec<i32> = data_list.extract()?;
            self.data = CoreArray::new_from_vec(data, shape, DType::Int32)?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype for pickle: {}",
                dtype_str
            )));
        }

        Ok(())
    }
}
