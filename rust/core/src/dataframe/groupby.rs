use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use super::core::DataFrame;
use super::series::Series;
use crate::array::core_array::CoreArray;
use crate::array::DType;

#[pyclass(module = "corepy._corepy_rust", name = "_RustGroupBy", from_py_object)]
#[derive(Clone)]
pub struct GroupBy {
    df: DataFrame,
    by: String,

    // For f32 keys (using to_bits), or i32 keys
    groups_f32: Option<HashMap<u32, Vec<usize>>>,
    groups_i32: Option<HashMap<i32, Vec<usize>>>,
    groups_str: Option<HashMap<String, Vec<usize>>>,
}

impl GroupBy {
    pub fn new(df: DataFrame, by: String) -> PyResult<Self> {
        let col = df.get_column(by.clone())?;

        let dtype = col.data.dtype();
        let mut groups_f32 = None;
        let mut groups_i32 = None;
        let mut groups_str = None;

        if dtype == DType::Float32 {
            let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
            let slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    col.data.data_ptr() as *const f32,
                    col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                map.entry(val.to_bits()).or_default().push(i);
            }
            groups_f32 = Some(map);
        } else if dtype == DType::Int32 {
            let mut map: HashMap<i32, Vec<usize>> = HashMap::new();
            let slice: &[i32] = unsafe {
                std::slice::from_raw_parts(
                    col.data.data_ptr() as *const i32,
                    col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                map.entry(val).or_default().push(i);
            }
            groups_i32 = Some(map);
        } else if dtype == DType::String {
            let mut map: HashMap<String, Vec<usize>> = HashMap::new();
            if let Some(ref data) = col.data.data_str {
                for (i, val) in data.iter().enumerate() {
                    let s: String = val.to_string();
                    map.entry(s).or_default().push(i);
                }
            }
            groups_str = Some(map);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported grouping column dtype",
            ));
        }

        Ok(GroupBy {
            df,
            by,
            groups_f32,
            groups_i32,
            groups_str,
        })
    }
}

#[pymethods]
impl GroupBy {
    pub fn sum(&self) -> PyResult<DataFrame> {
        self.aggregate_sum()
    }

    pub fn mean(&self) -> PyResult<DataFrame> {
        self.aggregate_mean()
    }
}

impl GroupBy {
    fn aggregate_sum(&self) -> PyResult<DataFrame> {
        let mut new_df = DataFrame::new();

        // We need to iterate over groups in a consistent order, so we extract and sort keys
        let (keys_f32, keys_i32, keys_str) = if let Some(ref map) = self.groups_f32 {
            let mut keys: Vec<u32> = map.keys().cloned().collect();
            // Sort by f32 value
            keys.sort_by(|a, b| {
                f32::from_bits(*a)
                    .partial_cmp(&f32::from_bits(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            (Some((keys, map)), None, None)
        } else if let Some(ref map) = self.groups_i32 {
            let mut keys: Vec<i32> = map.keys().cloned().collect();
            keys.sort();
            (None, Some((keys, map)), None)
        } else if let Some(ref map) = self.groups_str {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            (None, None, Some((keys, map)))
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("No groups found"));
        };

        let num_groups = keys_f32
            .as_ref()
            .map(|(k, _)| k.len())
            .or_else(|| keys_i32.as_ref().map(|(k, _)| k.len()))
            .unwrap_or_else(|| keys_str.as_ref().unwrap().0.len());
        // ... (continued in next chunk for aggregate_sum)

        for (name, series) in &self.df.columns {
            if name == &self.by {
                // Grouping key column
                if let Some((ref keys, _)) = keys_f32 {
                    let mut new_data_vec = Vec::with_capacity(num_groups);
                    for &k in keys {
                        new_data_vec.push(f32::from_bits(k));
                    }
                    let array = CoreArray::new(new_data_vec, vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                } else if let Some((ref keys, _)) = keys_i32 {
                    let mut new_data_vec = Vec::with_capacity(num_groups);
                    for &k in keys {
                        new_data_vec.push(k as f32); // Fallback float storage
                    }
                    let array = CoreArray::new(new_data_vec, vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                } else if let Some((ref keys, _)) = keys_str {
                    let array = CoreArray::from_strings(keys.clone(), vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                }
                continue;
            }

            // Numeric aggregation column
            let mut new_data_vec = Vec::with_capacity(num_groups);
            let dtype = series.data.dtype();

            let aggregate = |indices: &Vec<usize>| -> f32 {
                let mut sum = 0.0;
                if dtype == DType::Float32 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            series.data.data_ptr() as *const f32,
                            series.data.element_count(),
                        )
                    };
                    for &idx in indices {
                        sum += slice[idx];
                    }
                } else if dtype == DType::Int32 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            series.data.data_ptr() as *const i32,
                            series.data.element_count(),
                        )
                    };
                    for &idx in indices {
                        sum += slice[idx] as f32;
                    }
                }
                sum
            };

            if let Some((ref keys, map)) = keys_f32 {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            } else if let Some((ref keys, map)) = keys_i32 {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            } else if let Some((ref keys, map)) = keys_str {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            }

            let array = CoreArray::new(new_data_vec, vec![num_groups])?;
            let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
            new_df.columns.insert(
                name.clone(),
                Series {
                    name: name.clone(),
                    data: array,
                    index,
                },
            );
        }

        new_df.row_count = num_groups;
        Ok(new_df)
    }

    fn aggregate_mean(&self) -> PyResult<DataFrame> {
        let mut new_df = DataFrame::new();

        let (keys_f32, keys_i32, keys_str) = if let Some(ref map) = self.groups_f32 {
            let mut keys: Vec<u32> = map.keys().cloned().collect();
            keys.sort_by(|a, b| {
                f32::from_bits(*a)
                    .partial_cmp(&f32::from_bits(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            (Some((keys, map)), None, None)
        } else if let Some(ref map) = self.groups_i32 {
            let mut keys: Vec<i32> = map.keys().cloned().collect();
            keys.sort();
            (None, Some((keys, map)), None)
        } else if let Some(ref map) = self.groups_str {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            (None, None, Some((keys, map)))
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("No groups found"));
        };

        let num_groups = keys_f32
            .as_ref()
            .map(|(k, _)| k.len())
            .or_else(|| keys_i32.as_ref().map(|(k, _)| k.len()))
            .unwrap_or_else(|| keys_str.as_ref().unwrap().0.len());

        for (name, series) in &self.df.columns {
            if name == &self.by {
                if let Some((ref keys, _)) = keys_f32 {
                    let mut new_data_vec = Vec::with_capacity(num_groups);
                    for &k in keys {
                        new_data_vec.push(f32::from_bits(k));
                    }
                    let array = CoreArray::new(new_data_vec, vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                } else if let Some((ref keys, _)) = keys_i32 {
                    let mut new_data_vec = Vec::with_capacity(num_groups);
                    for &k in keys {
                        new_data_vec.push(k as f32);
                    }
                    let array = CoreArray::new(new_data_vec, vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                } else if let Some((ref keys, _)) = keys_str {
                    let array = CoreArray::from_strings(keys.clone(), vec![num_groups])?;
                    let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
                    new_df.columns.insert(
                        name.clone(),
                        Series {
                            name: name.clone(),
                            data: array,
                            index,
                        },
                    );
                }
                continue;
            }

            let mut new_data_vec = Vec::with_capacity(num_groups);
            let dtype = series.data.dtype();

            let aggregate = |indices: &Vec<usize>| -> f32 {
                let mut sum = 0.0;
                let count = indices.len() as f32;
                if count == 0.0 {
                    return f32::NAN;
                }

                if dtype == DType::Float32 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            series.data.data_ptr() as *const f32,
                            series.data.element_count(),
                        )
                    };
                    for &idx in indices {
                        sum += slice[idx];
                    }
                } else if dtype == DType::Int32 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            series.data.data_ptr() as *const i32,
                            series.data.element_count(),
                        )
                    };
                    for &idx in indices {
                        sum += slice[idx] as f32;
                    }
                }
                sum / count
            };

            if let Some((ref keys, map)) = keys_f32 {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            } else if let Some((ref keys, map)) = keys_i32 {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            } else if let Some((ref keys, map)) = keys_str {
                new_data_vec = keys
                    .par_iter()
                    .map(|k| aggregate(map.get(k).unwrap()))
                    .collect();
            }

            let array = CoreArray::new(new_data_vec, vec![num_groups])?;
            let index: Vec<String> = (0..num_groups).map(|i| i.to_string()).collect();
            new_df.columns.insert(
                name.clone(),
                Series {
                    name: name.clone(),
                    data: array,
                    index,
                },
            );
        }

        new_df.row_count = num_groups;
        Ok(new_df)
    }
}
