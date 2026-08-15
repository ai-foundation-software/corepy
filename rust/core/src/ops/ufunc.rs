// ============================================================================
// Operations: Universal Functions (UFuncs)
// ============================================================================
// Trait-based definitions for element-wise mathematical operations with
// Rayon parallelism.

use crate::array::core_array::CoreArray;
use crate::scheduler::arena::with_arena;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Trait definition for universal functions
pub trait UFunc {
    fn apply(&self, input: &CoreArray) -> PyResult<CoreArray>;
    // Future: fn apply_inplace(&self, input: &mut CoreArray) -> PyResult<()>;
}

// ----------------------------------------------------------------------------
// Mathematical Functions
// ----------------------------------------------------------------------------

pub struct Exp;
impl UFunc for Exp {
    fn apply(&self, input: &CoreArray) -> PyResult<CoreArray> {
        let size = input.element_count();
        let out = CoreArray::zeros(input.shape(), Some(input.dtype()))?;

        with_arena(|_arena| {
            let in_data =
                unsafe { std::slice::from_raw_parts(input.data_ptr() as *const f32, size) };
            let out_data =
                unsafe { std::slice::from_raw_parts_mut(out.data_ptr() as *mut f32, size) };

            // Rayon parallel map
            in_data
                .par_iter()
                .zip(out_data.par_iter_mut())
                .for_each(|(&i, o)| {
                    *o = i.exp();
                });
        });

        Ok(out)
    }
}

pub struct Log;
impl UFunc for Log {
    fn apply(&self, input: &CoreArray) -> PyResult<CoreArray> {
        let size = input.element_count();
        let out = CoreArray::zeros(input.shape(), Some(input.dtype()))?;

        with_arena(|_arena| {
            let in_data =
                unsafe { std::slice::from_raw_parts(input.data_ptr() as *const f32, size) };
            let out_data =
                unsafe { std::slice::from_raw_parts_mut(out.data_ptr() as *mut f32, size) };

            in_data
                .par_iter()
                .zip(out_data.par_iter_mut())
                .for_each(|(&i, o)| {
                    *o = i.ln();
                });
        });

        Ok(out)
    }
}

pub struct Sin;
impl UFunc for Sin {
    fn apply(&self, input: &CoreArray) -> PyResult<CoreArray> {
        let size = input.element_count();
        let out = CoreArray::zeros(input.shape(), Some(input.dtype()))?;

        with_arena(|_arena| {
            let in_data =
                unsafe { std::slice::from_raw_parts(input.data_ptr() as *const f32, size) };
            let out_data =
                unsafe { std::slice::from_raw_parts_mut(out.data_ptr() as *mut f32, size) };

            in_data
                .par_iter()
                .zip(out_data.par_iter_mut())
                .for_each(|(&i, o)| {
                    *o = i.sin();
                });
        });

        Ok(out)
    }
}

pub struct Sqrt;
impl UFunc for Sqrt {
    fn apply(&self, input: &CoreArray) -> PyResult<CoreArray> {
        let size = input.element_count();
        let out = CoreArray::zeros(input.shape(), Some(input.dtype()))?;

        with_arena(|_arena| {
            let in_data =
                unsafe { std::slice::from_raw_parts(input.data_ptr() as *const f32, size) };
            let out_data =
                unsafe { std::slice::from_raw_parts_mut(out.data_ptr() as *mut f32, size) };

            in_data
                .par_iter()
                .zip(out_data.par_iter_mut())
                .for_each(|(&i, o)| {
                    *o = i.sqrt();
                });
        });

        Ok(out)
    }
}
