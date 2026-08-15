use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use super::groupby::GroupBy;
use super::series::Series;
use crate::array::core_array::CoreArray;
use crate::array::DType;

#[pyclass(
    module = "corepy._corepy_rust",
    name = "_RustDataFrame",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct DataFrame {
    #[pyo3(get, set)]
    pub columns: HashMap<String, Series>,
    #[pyo3(get, set)]
    pub row_count: usize,
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl DataFrame {
    #[new]
    pub fn new() -> Self {
        DataFrame {
            columns: HashMap::new(),
            row_count: 0,
        }
    }

    pub fn insert(&mut self, name: String, series: pyo3::PyRef<'_, Series>) -> PyResult<()> {
        let series_len = series.data.element_count();
        if self.columns.is_empty() {
            self.row_count = series_len;
        } else if series_len != self.row_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Column length mismatch",
            ));
        }
        self.columns.insert(name, series.clone());
        Ok(())
    }

    pub fn get_column(&self, name: String) -> PyResult<Series> {
        self.columns.get(&name).cloned().ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", name))
        })
    }

    pub fn drop(&mut self, columns: Vec<String>) -> PyResult<()> {
        for col in columns {
            self.columns.remove(&col);
        }
        Ok(())
    }

    pub fn rename(&mut self, columns: HashMap<String, String>) -> PyResult<()> {
        let mut new_columns = HashMap::new();
        for old_name in columns.keys() {
            if !self.columns.contains_key(old_name) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Column {} not found",
                    old_name
                )));
            }
        }
        for (old_name, new_name) in columns {
            if let Some(mut series) = self.columns.remove(&old_name) {
                series.name = new_name.clone();
                new_columns.insert(new_name, series);
            }
        }
        self.columns.extend(new_columns);
        Ok(())
    }

    pub fn iloc(&self, start: Option<usize>, end: Option<usize>) -> PyResult<DataFrame> {
        let n = self.row_count;
        let s = start.unwrap_or(0).min(n);
        let e = end.unwrap_or(n).min(n);
        let s = s.min(e);
        let size = e - s;

        let mut new_df = DataFrame::new();
        for (name, series) in &self.columns {
            let mut new_data_vec = Vec::with_capacity(size);
            let s_dtype = series.data.dtype();

            if s_dtype == DType::Float32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts((series.data.data_ptr() as *const f32).add(s), size)
                };
                new_data_vec.extend_from_slice(s_slice);
            } else if s_dtype == DType::Int32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts((series.data.data_ptr() as *const i32).add(s), size)
                };
                for &val in s_slice {
                    new_data_vec.push(val as f32);
                }
            }

            let new_array = if s_dtype == DType::String {
                if let Some(ref data) = series.data.data_str {
                    let subset = data[s..e].to_vec();
                    CoreArray::from_strings(subset, vec![size])?
                } else {
                    CoreArray::from_strings(vec!["".to_string(); size], vec![size])?
                }
            } else {
                CoreArray::new(new_data_vec, vec![size])?
            };

            let mut new_index = Vec::with_capacity(size);
            new_index.extend_from_slice(&series.index[s..e]);
            let new_series = Series {
                name: series.name.clone(),
                data: new_array,
                index: new_index,
            };
            new_df.columns.insert(name.clone(), new_series);
        }
        new_df.row_count = size;
        Ok(new_df)
    }

    pub fn filter_eq(&self, col_name: String, value: Bound<'_, PyAny>) -> PyResult<DataFrame> {
        let col = self.columns.get(&col_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", col_name))
        })?;

        let mut indices_to_keep = Vec::new();
        let dtype = col.data.dtype();
        match dtype {
            DType::Float32 => {
                let val: f32 = value.extract()?;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        col.data.data_ptr() as *const f32,
                        col.data.element_count(),
                    )
                };
                for (i, &v) in slice.iter().enumerate() {
                    if v == val {
                        indices_to_keep.push(i);
                    }
                }
            }
            DType::Float64 => {
                let val: f64 = value.extract()?;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        col.data.data_ptr() as *const f64,
                        col.data.element_count(),
                    )
                };
                for (i, &v) in slice.iter().enumerate() {
                    if v == val {
                        indices_to_keep.push(i);
                    }
                }
            }
            DType::Int32 => {
                let val: i32 = value.extract()?;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        col.data.data_ptr() as *const i32,
                        col.data.element_count(),
                    )
                };
                for (i, &v) in slice.iter().enumerate() {
                    if v == val {
                        indices_to_keep.push(i);
                    }
                }
            }
            DType::Int64 => {
                let val: i64 = value.extract()?;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        col.data.data_ptr() as *const i64,
                        col.data.element_count(),
                    )
                };
                for (i, &v) in slice.iter().enumerate() {
                    if v == val {
                        indices_to_keep.push(i);
                    }
                }
            }
            DType::Bool => {
                let val: bool = value.extract()?;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        col.data.data_ptr() as *const u8,
                        col.data.element_count(),
                    )
                };
                let byte_val = if val { 1 } else { 0 };
                for (i, &v) in slice.iter().enumerate() {
                    if v == byte_val {
                        indices_to_keep.push(i);
                    }
                }
            }
            DType::String => {
                let val: String = value.extract()?;
                if let Some(ref data) = col.data.data_str {
                    for (i, v) in data.iter().enumerate() {
                        if v == &val {
                            indices_to_keep.push(i);
                        }
                    }
                }
            }
        }

        let mut new_df = DataFrame::new();
        for (name, series) in &self.columns {
            let n = indices_to_keep.len();
            let mut new_data_vec = Vec::with_capacity(n);
            let s_dtype = series.data.dtype();

            if s_dtype == DType::Float32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const f32,
                        series.data.element_count(),
                    )
                };
                for &i in &indices_to_keep {
                    new_data_vec.push(s_slice[i]);
                }
            } else if s_dtype == DType::Int32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const i32,
                        series.data.element_count(),
                    )
                };
                for &i in &indices_to_keep {
                    new_data_vec.push(s_slice[i] as f32);
                }
            }

            let new_array = if s_dtype == DType::String {
                if let Some(ref data) = series.data.data_str {
                    let subset: Vec<String> =
                        indices_to_keep.iter().map(|&i| data[i].clone()).collect();
                    CoreArray::from_strings(subset, vec![n])?
                } else {
                    CoreArray::from_strings(vec!["".to_string(); n], vec![n])?
                }
            } else {
                CoreArray::new(new_data_vec, vec![n])?
            };
            let new_index: Vec<String> = indices_to_keep
                .iter()
                .map(|&i| series.index[i].clone())
                .collect();
            let new_series = Series {
                name: series.name.clone(),
                data: new_array,
                index: new_index,
            };
            new_df.columns.insert(name.clone(), new_series);
        }
        new_df.row_count = indices_to_keep.len();
        Ok(new_df)
    }

    pub fn sort_values(&self, col_name: String, ascending: bool) -> PyResult<DataFrame> {
        let col = self.columns.get(&col_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", col_name))
        })?;

        let mut indices: Vec<usize> = (0..self.row_count).collect();

        let dtype = col.data.dtype();
        if dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    col.data.data_ptr() as *const f32,
                    col.data.element_count(),
                )
            };
            indices.sort_by(|&a, &b| {
                slice[a]
                    .partial_cmp(&slice[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    col.data.data_ptr() as *const i32,
                    col.data.element_count(),
                )
            };
            indices.sort_by(|&a, &b| slice[a].cmp(&slice[b]));
        } else if dtype == DType::String {
            if let Some(ref data) = col.data.data_str {
                indices.sort_by(|&a, &b| data[a].cmp(&data[b]));
            }
        }

        if !ascending {
            indices.reverse();
        }

        let mut new_df = DataFrame::new();
        for (name, series) in &self.columns {
            let n = indices.len();
            let mut new_data_vec = Vec::with_capacity(n);
            let s_dtype = series.data.dtype();

            if s_dtype == DType::Float32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const f32,
                        series.data.element_count(),
                    )
                };
                for &i in &indices {
                    new_data_vec.push(s_slice[i]);
                }
            } else if s_dtype == DType::Int32 {
                let s_slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const i32,
                        series.data.element_count(),
                    )
                };
                for &i in &indices {
                    new_data_vec.push(s_slice[i] as f32);
                }
            }

            let new_array = if s_dtype == DType::String {
                if let Some(ref data) = series.data.data_str {
                    let subset: Vec<String> = indices.iter().map(|&i| data[i].clone()).collect();
                    CoreArray::from_strings(subset, vec![n])?
                } else {
                    CoreArray::from_strings(vec!["".to_string(); n], vec![n])?
                }
            } else {
                CoreArray::new(new_data_vec, vec![n])?
            };
            let new_index: Vec<String> = indices.iter().map(|&i| series.index[i].clone()).collect();
            let new_series = Series {
                name: series.name.clone(),
                data: new_array,
                index: new_index,
            };
            new_df.columns.insert(name.clone(), new_series);
        }
        new_df.row_count = indices.len();
        Ok(new_df)
    }

    pub fn join(
        &self,
        other: &DataFrame,
        left_on: String,
        right_on: String,
        how: String,
    ) -> PyResult<DataFrame> {
        let left_col = self.columns.get(&left_on).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "Column {} not found in left DataFrame",
                left_on
            ))
        })?;
        let right_col = other.columns.get(&right_on).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "Column {} not found in right DataFrame",
                right_on
            ))
        })?;

        let left_dtype = left_col.data.dtype();
        let right_dtype = right_col.data.dtype();

        if left_dtype != right_dtype {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Join columns must have the same dtype",
            ));
        }

        if how != "inner" && how != "left" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only 'inner' and 'left' joins are supported",
            ));
        }

        // Build hash map from right dataframe
        let mut right_map_f32: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut right_map_i32: HashMap<i32, Vec<usize>> = HashMap::new();

        if right_dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    right_col.data.data_ptr() as *const f32,
                    right_col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                right_map_f32.entry(val.to_bits()).or_default().push(i);
            }
        } else if right_dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    right_col.data.data_ptr() as *const i32,
                    right_col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                right_map_i32.entry(val).or_default().push(i);
            }
        }

        // Probe with left dataframe
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        if left_dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    left_col.data.data_ptr() as *const f32,
                    left_col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                if let Some(r_idxs) = right_map_f32.get(&val.to_bits()) {
                    for &r_idx in r_idxs {
                        left_indices.push(i);
                        right_indices.push(Some(r_idx));
                    }
                } else if how == "left" {
                    left_indices.push(i);
                    right_indices.push(None);
                }
            }
        } else if left_dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    left_col.data.data_ptr() as *const i32,
                    left_col.data.element_count(),
                )
            };
            for (i, &val) in slice.iter().enumerate() {
                if let Some(r_idxs) = right_map_i32.get(&val) {
                    for &r_idx in r_idxs {
                        left_indices.push(i);
                        right_indices.push(Some(r_idx));
                    }
                } else if how == "left" {
                    left_indices.push(i);
                    right_indices.push(None);
                }
            }
        }

        // Construct new dataframe
        let mut new_df = DataFrame::new();
        let n = left_indices.len();

        // 1. Add columns from left
        for (name, series) in &self.columns {
            let dtype = series.data.dtype();
            let mut new_data_vec = Vec::with_capacity(n);

            if dtype == DType::Float32 {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const f32,
                        series.data.element_count(),
                    )
                };
                for &l_idx in &left_indices {
                    new_data_vec.push(slice[l_idx]);
                }
            } else if dtype == DType::Int32 {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const i32,
                        series.data.element_count(),
                    )
                };
                for &l_idx in &left_indices {
                    new_data_vec.push(slice[l_idx] as f32);
                }
            }

            let array = CoreArray::new(new_data_vec, vec![n])?;
            let index: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            new_df.columns.insert(
                name.clone(),
                Series {
                    name: name.clone(),
                    data: array,
                    index,
                },
            );
        }

        // 2. Add columns from right (excluding the join key, which is already in left)
        for (name, series) in &other.columns {
            if name == &right_on {
                continue;
            } // Skip join key

            // For conflicting column names, append "_y" (simple collision handling)
            let mut col_name = name.clone();
            if self.columns.contains_key(name) {
                col_name.push_str("_y");
            }

            let dtype = series.data.dtype();
            let mut new_data_vec = Vec::with_capacity(n);

            if dtype == DType::Float32 {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const f32,
                        series.data.element_count(),
                    )
                };
                for &r_idx_opt in &right_indices {
                    if let Some(r_idx) = r_idx_opt {
                        new_data_vec.push(slice[r_idx]);
                    } else {
                        new_data_vec.push(f32::NAN); // NaN for missing
                    }
                }
            } else if dtype == DType::Int32 {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        series.data.data_ptr() as *const i32,
                        series.data.element_count(),
                    )
                };
                for &r_idx_opt in &right_indices {
                    if let Some(r_idx) = r_idx_opt {
                        new_data_vec.push(slice[r_idx] as f32);
                    } else {
                        new_data_vec.push(f32::NAN); // Fallback to NaN
                    }
                }
            }

            let array = CoreArray::new(new_data_vec, vec![n])?;
            let index: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            new_df.columns.insert(
                col_name,
                Series {
                    name: name.clone(),
                    data: array,
                    index,
                },
            );
        }

        new_df.row_count = n;
        Ok(new_df)
    }

    pub fn groupby(&self, by: String) -> PyResult<GroupBy> {
        GroupBy::new(self.clone(), by)
    }

    pub fn pivot(&self, index: String, columns: String, values: String) -> PyResult<DataFrame> {
        // pivot turns unique values in `columns` into actual columns, and uses `index` as row identifier.
        // `values` are populated into the new columns.

        let idx_col = self.columns.get(&index).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", index))
        })?;
        let col_col = self.columns.get(&columns).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", columns))
        })?;
        let val_col = self.columns.get(&values).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Column {} not found", values))
        })?;

        // 1. Find all unique values in the `columns` column to become new column headers
        let mut unique_col_vals_f32: Vec<u32> = Vec::new(); // Store bits for exact matching
        let mut unique_col_vals_i32: Vec<i32> = Vec::new();
        let mut seen_col_f32 = std::collections::HashSet::new();
        let mut seen_col_i32 = std::collections::HashSet::new();

        let col_dtype = col_col.data.dtype();
        if col_dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    col_col.data.data_ptr() as *const f32,
                    col_col.data.element_count(),
                )
            };
            for &val in slice {
                let bits = val.to_bits();
                if seen_col_f32.insert(bits) {
                    unique_col_vals_f32.push(bits);
                }
            }
            // Sort to make column order deterministic
            unique_col_vals_f32.sort_by(|a, b| {
                f32::from_bits(*a)
                    .partial_cmp(&f32::from_bits(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if col_dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    col_col.data.data_ptr() as *const i32,
                    col_col.data.element_count(),
                )
            };
            for &val in slice {
                if seen_col_i32.insert(val) {
                    unique_col_vals_i32.push(val);
                }
            }
            unique_col_vals_i32.sort();
        }

        // 2. Find all unique values in the `index` column to become the new rows
        let mut unique_idx_vals_f32: Vec<u32> = Vec::new();
        let mut unique_idx_vals_i32: Vec<i32> = Vec::new();
        let mut seen_idx_f32 = std::collections::HashSet::new();
        let mut seen_idx_i32 = std::collections::HashSet::new();

        let idx_dtype = idx_col.data.dtype();
        if idx_dtype == DType::Float32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    idx_col.data.data_ptr() as *const f32,
                    idx_col.data.element_count(),
                )
            };
            for &val in slice {
                let bits = val.to_bits();
                if seen_idx_f32.insert(bits) {
                    unique_idx_vals_f32.push(bits);
                }
            }
            unique_idx_vals_f32.sort_by(|a, b| {
                f32::from_bits(*a)
                    .partial_cmp(&f32::from_bits(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if idx_dtype == DType::Int32 {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    idx_col.data.data_ptr() as *const i32,
                    idx_col.data.element_count(),
                )
            };
            for &val in slice {
                if seen_idx_i32.insert(val) {
                    unique_idx_vals_i32.push(val);
                }
            }
            unique_idx_vals_i32.sort();
        }

        let num_rows = if idx_dtype == DType::Float32 {
            unique_idx_vals_f32.len()
        } else {
            unique_idx_vals_i32.len()
        };
        let num_cols = if col_dtype == DType::Float32 {
            unique_col_vals_f32.len()
        } else {
            unique_col_vals_i32.len()
        };

        // We will store everything internally as f32 for the pivoted values
        let mut pivoted_data = vec![vec![f32::NAN; num_rows]; num_cols];

        // 3. Populate the cells
        let idx_slice_f32 = if idx_dtype == DType::Float32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    idx_col.data.data_ptr() as *const f32,
                    idx_col.data.element_count(),
                )
            })
        } else {
            None
        };
        let idx_slice_i32 = if idx_dtype == DType::Int32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    idx_col.data.data_ptr() as *const i32,
                    idx_col.data.element_count(),
                )
            })
        } else {
            None
        };

        let col_slice_f32 = if col_dtype == DType::Float32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    col_col.data.data_ptr() as *const f32,
                    col_col.data.element_count(),
                )
            })
        } else {
            None
        };
        let col_slice_i32 = if col_dtype == DType::Int32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    col_col.data.data_ptr() as *const i32,
                    col_col.data.element_count(),
                )
            })
        } else {
            None
        };

        let val_slice_f32 = if val_col.data.dtype() == DType::Float32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    val_col.data.data_ptr() as *const f32,
                    val_col.data.element_count(),
                )
            })
        } else {
            None
        };
        let val_slice_i32 = if val_col.data.dtype() == DType::Int32 {
            Some(unsafe {
                std::slice::from_raw_parts(
                    val_col.data.data_ptr() as *const i32,
                    val_col.data.element_count(),
                )
            })
        } else {
            None
        };

        for i in 0..self.row_count {
            // Find row index
            let row_idx = if let Some(s) = idx_slice_f32 {
                unique_idx_vals_f32.binary_search(&s[i].to_bits()).unwrap()
            } else if let Some(s) = idx_slice_i32 {
                unique_idx_vals_i32.binary_search(&s[i]).unwrap()
            } else {
                unreachable!()
            };

            // Find col index
            let col_idx = if let Some(s) = col_slice_f32 {
                unique_col_vals_f32.binary_search(&s[i].to_bits()).unwrap()
            } else if let Some(s) = col_slice_i32 {
                unique_col_vals_i32.binary_search(&s[i]).unwrap()
            } else {
                unreachable!()
            };

            // Get Value
            let val = if let Some(s) = val_slice_f32 {
                s[i]
            } else if let Some(s) = val_slice_i32 {
                s[i] as f32
            } else {
                unreachable!()
            };

            pivoted_data[col_idx][row_idx] = val;
        }

        // 4. Construct Final DataFrame
        let mut new_df = DataFrame::new();

        // 4a. Add the index column
        let mut new_idx_data = Vec::with_capacity(num_rows);
        if idx_dtype == DType::Float32 {
            for &bits in &unique_idx_vals_f32 {
                new_idx_data.push(f32::from_bits(bits));
            }
        } else {
            for &val in &unique_idx_vals_i32 {
                new_idx_data.push(val as f32);
            }
        }
        let array = CoreArray::new(new_idx_data, vec![num_rows])?;
        let str_index: Vec<String> = (0..num_rows).map(|i| i.to_string()).collect();
        new_df.columns.insert(
            index.clone(),
            Series {
                name: index.clone(),
                data: array,
                index: str_index.clone(),
            },
        );

        // 4b. Add the value columns
        for c in 0..num_cols {
            let col_name = if col_dtype == DType::Float32 {
                format!("{}", f32::from_bits(unique_col_vals_f32[c]))
            } else {
                format!("{}", unique_col_vals_i32[c])
            };

            let data = std::mem::take(&mut pivoted_data[c]);
            let array = CoreArray::new(data, vec![num_rows])?;
            new_df.columns.insert(
                col_name.clone(),
                Series {
                    name: col_name,
                    data: array,
                    index: str_index.clone(),
                },
            );
        }

        new_df.row_count = num_rows;
        Ok(new_df)
    }

    fn __repr__(&self) -> String {
        format!(
            "DataFrame(rows={}, cols={})",
            self.row_count,
            self.columns.len()
        )
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let columns_dict = PyDict::new(py);
        for (name, series) in &self.columns {
            columns_dict.set_item(name, series.clone())?;
        }
        dict.set_item("columns", columns_dict)?;
        dict.set_item("row_count", self.row_count)?;
        Ok(dict)
    }

    pub fn __setstate__(&mut self, state: pyo3::Py<pyo3::PyAny>, py: Python<'_>) -> PyResult<()> {
        let dict = state.bind(py).cast::<PyDict>()?;
        self.row_count = dict
            .get_item("row_count")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing row_count"))?
            .extract()?;
        for (name, series) in dict
            .get_item("columns")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing columns"))?
            .cast::<PyDict>()?
            .into_iter()
        {
            let name_str: String = name.extract()?;
            let series_obj: Series = series.extract()?;
            self.columns.insert(name_str, series_obj);
        }
        Ok(())
    }
}
