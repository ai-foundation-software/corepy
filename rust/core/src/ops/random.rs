use pyo3::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::array::core_array::CoreArray;
use crate::array::dtype::DType;

/// Generator algorithm choices
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RngAlgorithm {
    PCG64 = 0,
    Xoshiro256PP = 1,
}

#[pymethods]
impl RngAlgorithm {
    fn __repr__(&self) -> String {
        match self {
            RngAlgorithm::PCG64 => "RngAlgorithm.PCG64".to_string(),
            RngAlgorithm::Xoshiro256PP => "RngAlgorithm.Xoshiro256PP".to_string(),
        }
    }
}

pub fn uniform_f32(shape: Vec<usize>, seed: u64, algo: RngAlgorithm) -> PyResult<CoreArray> {
    let size: usize = shape.iter().product();
    if size == 0 {
        return CoreArray::zeros(shape, Some(DType::Float32));
    }

    let mut out = CoreArray::zeros(shape.clone(), Some(DType::Float32))?;
    let data = unsafe { out.as_f32_slice_mut() };

    let threads = rayon::current_num_threads();
    let chunk_size = size.div_ceil(threads);

    data.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let chunk_seed = seed.wrapping_add(i as u64);
            match algo {
                RngAlgorithm::PCG64 => {
                    let mut rng = Pcg64::seed_from_u64(chunk_seed);
                    for val in chunk.iter_mut() {
                        *val = rng.random::<f32>();
                    }
                }
                RngAlgorithm::Xoshiro256PP => {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);
                    for val in chunk.iter_mut() {
                        *val = rng.random::<f32>();
                    }
                }
            }
        });

    Ok(out)
}

pub fn normal_f32(shape: Vec<usize>, seed: u64, algo: RngAlgorithm) -> PyResult<CoreArray> {
    let size: usize = shape.iter().product();
    if size == 0 {
        return CoreArray::zeros(shape, Some(DType::Float32));
    }

    let mut out = CoreArray::zeros(shape.clone(), Some(DType::Float32))?;
    let data = unsafe { out.as_f32_slice_mut() };

    let threads = rayon::current_num_threads();
    let chunk_size = size.div_ceil(threads);

    data.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let chunk_seed = seed.wrapping_add(i as u64);
            match algo {
                RngAlgorithm::PCG64 => {
                    let mut rng = Pcg64::seed_from_u64(chunk_seed);
                    for val in chunk.iter_mut() {
                        *val = StandardNormal.sample(&mut rng);
                    }
                }
                RngAlgorithm::Xoshiro256PP => {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);
                    for val in chunk.iter_mut() {
                        *val = StandardNormal.sample(&mut rng);
                    }
                }
            }
        });

    Ok(out)
}
