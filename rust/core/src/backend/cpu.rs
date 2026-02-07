// ============================================================================
// CPU Backend Implementation
// ============================================================================
// Implements ComputeBackend trait for CPU (SIMD) execution.
#![allow(dead_code)] // Reserved public API for future unified backend dispatch

use std::any::Any;

use super::capabilities::get_capabilities;
use super::traits::{BackendCapabilities, BackendError, BackendResult, ComputeBackend, DataType};

/// CPU backend using SIMD-accelerated kernels
pub struct CpuBackend {
    /// Backend ID for dispatch tracking
    id: u8,
}

impl CpuBackend {
    /// Create a new CPU backend instance
    pub fn new() -> Self {
        CpuBackend { id: 0 }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        let caps = get_capabilities();
        if caps.cpu.has_avx512f {
            "CPU (AVX-512)"
        } else if caps.cpu.has_avx2 {
            "CPU (AVX2)"
        } else if caps.cpu.has_neon {
            "CPU (NEON)"
        } else {
            "CPU (Scalar)"
        }
    }

    fn backend_id(&self) -> u8 {
        self.id
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        let caps = get_capabilities();
        let mut properties = std::collections::HashMap::new();

        // Report SIMD capabilities
        if caps.cpu.has_avx512f {
            properties.insert("simd".to_string(), "AVX-512".to_string());
        } else if caps.cpu.has_avx2 {
            properties.insert("simd".to_string(), "AVX2".to_string());
        } else if caps.cpu.has_neon {
            properties.insert("simd".to_string(), "NEON".to_string());
        } else {
            properties.insert("simd".to_string(), "Scalar".to_string());
        }

        properties.insert("cores".to_string(), caps.cpu.core_count.to_string());
        properties.insert("fma".to_string(), caps.cpu.has_fma.to_string());

        BackendCapabilities {
            supported_dtypes: vec![
                DataType::F32,
                DataType::F64,
                DataType::I32,
                DataType::I64,
                DataType::Bool,
            ],
            max_tensor_bytes: None, // Limited by system memory
            supports_async: false,
            supports_inplace: true,
            properties,
        }
    }

    unsafe fn matmul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<()> {
        if a.is_null() || b.is_null() || c.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }

        // Delegate to existing matmul implementation
        crate::ops::matmul::matmul_f32_cpu_dispatch(a, b, c, m, k, n);
        Ok(())
    }

    unsafe fn sum_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        if data.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }
        if count == 0 {
            return Err(BackendError::InvalidInput("Empty array".to_string()));
        }

        Ok(crate::ops::reduce::sum_f32_cpu_dispatch(data, count))
    }

    unsafe fn max_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        if data.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }
        if count == 0 {
            return Err(BackendError::InvalidInput("Empty array".to_string()));
        }

        // Simple implementation for now (can be optimized with SIMD later)
        let slice = std::slice::from_raw_parts(data, count);
        Ok(slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
    }

    unsafe fn min_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        if data.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }
        if count == 0 {
            return Err(BackendError::InvalidInput("Empty array".to_string()));
        }

        let slice = std::slice::from_raw_parts(data, count);
        Ok(slice.iter().fold(f32::INFINITY, |a, &b| a.min(b)))
    }

    unsafe fn add_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if a.is_null() || b.is_null() || result.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }

        crate::ops::elementwise::add_f32_cpu_dispatch(a, b, result, count);
        Ok(())
    }

    unsafe fn sub_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if a.is_null() || b.is_null() || result.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }

        crate::ops::elementwise::sub_f32_cpu_dispatch(a, b, result, count);
        Ok(())
    }

    unsafe fn mul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if a.is_null() || b.is_null() || result.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }

        crate::ops::elementwise::mul_f32_cpu_dispatch(a, b, result, count);
        Ok(())
    }

    unsafe fn div_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if a.is_null() || b.is_null() || result.is_null() {
            return Err(BackendError::InvalidInput("Null pointer".to_string()));
        }

        crate::ops::elementwise::div_f32_cpu_dispatch(a, b, result, count);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_available() {
        let backend = CpuBackend::new();
        assert!(backend.is_available());
    }

    #[test]
    fn test_cpu_backend_capabilities() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.supported_dtypes.contains(&DataType::F32));
        assert!(caps.supports_inplace);
    }
}
