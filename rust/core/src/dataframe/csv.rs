use csv::ReaderBuilder;
use pyo3::prelude::*;
use std::fs::File;

use super::core::DataFrame;
use super::series::Series;
use crate::array::core_array::CoreArray;

#[pyfunction]
pub fn read_csv(path: String) -> PyResult<DataFrame> {
    let file = File::open(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e)))?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = match rdr.headers() {
        Ok(h) => h.clone(),
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to read headers: {}",
                e
            )))
        }
    };

    let num_cols = headers.len();
    let mut float_columns: Vec<Vec<f32>> = vec![Vec::new(); num_cols];

    for result in rdr.records() {
        let record = result.map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to read record: {}", e))
        })?;
        for (i, field) in record.iter().enumerate() {
            // For now, parse everything as f32 since CoreArray primarily supports f32.
            // If it fails, fallback to 0.0 (or NaN).
            let val = field.parse::<f32>().unwrap_or(f32::NAN);
            float_columns[i].push(val);
        }
    }

    let mut df = DataFrame::new();
    let row_count = if num_cols > 0 {
        float_columns[0].len()
    } else {
        0
    };

    for i in 0..num_cols {
        let name = headers[i].to_string();
        let data = std::mem::take(&mut float_columns[i]);
        let array = CoreArray::new(data, vec![row_count])?;
        let index: Vec<String> = (0..row_count).map(|idx| idx.to_string()).collect();
        let series = Series {
            name: name.clone(),
            data: array,
            index,
        };
        df.columns.insert(name, series);
    }

    df.row_count = row_count;

    Ok(df)
}
