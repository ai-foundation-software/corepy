// ============================================================================
// Compute Backend Trait
// ============================================================================
// Abstract interface for compute backends (CPU, GPU, etc.)
// Enables extensible backend system without modifying core dispatch logic.
#![allow(dead_code)] // Reserved public API for future backend implementations

use std::any::Any;

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Errors that can occur during backend operations
#[derive(Debug, Clone)]
pub enum BackendError {
    /// Backend is not available on this platform
    NotAvailable(String),
    /// Operation not supported by this backend
    UnsupportedOperation(String),
    /// Invalid input parameters
    InvalidInput(String),
    /// Runtime execution error
    ExecutionError(String),
    /// Out of memory
    OutOfMemory,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotAvailable(msg) => write!(f, "Backend not available: {}", msg),
            BackendError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            BackendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            BackendError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            BackendError::OutOfMemory => write!(f, "Out of memory"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Supported data types for backend operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// Backend capability flags
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// Supported data types
    pub supported_dtypes: Vec<DataType>,
    /// Maximum tensor size in bytes
    pub max_tensor_bytes: Option<usize>,
    /// Whether async execution is supported
    pub supports_async: bool,
    /// Whether in-place operations are supported
    pub supports_inplace: bool,
    /// Backend-specific properties
    pub properties: std::collections::HashMap<String, String>,
}

/// Abstract compute backend trait
/// 
/// Implementations must be thread-safe (Send + Sync).
/// All methods that perform computation should be safe to call
/// from multiple threads concurrently.
pub trait ComputeBackend: Send + Sync {
    /// Human-readable name of this backend
    fn name(&self) -> &'static str;
    
    /// Unique identifier for this backend type
    fn backend_id(&self) -> u8;
    
    /// Check if this backend is available on the current system
    fn is_available(&self) -> bool;
    
    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
    
    /// Check if a specific data type is supported
    fn supports_dtype(&self, dtype: DataType) -> bool {
        self.capabilities().supported_dtypes.contains(&dtype)
    }
    
    // =========================================================================
    // Matrix Operations
    // =========================================================================
    
    /// Matrix multiplication: C = A @ B
    /// 
    /// # Safety
    /// - Pointers must be valid and aligned
    /// - Output buffer must be pre-allocated with size m * n
    unsafe fn matmul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<()>;
    
    // =========================================================================
    // Reduction Operations
    // =========================================================================
    
    /// Sum reduction
    unsafe fn sum_f32(&self, data: *const f32, count: usize) -> BackendResult<f32>;
    
    /// Mean reduction
    unsafe fn mean_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        let sum = self.sum_f32(data, count)?;
        Ok(sum / count as f32)
    }
    
    /// Max reduction
    unsafe fn max_f32(&self, data: *const f32, count: usize) -> BackendResult<f32>;
    
    /// Min reduction
    unsafe fn min_f32(&self, data: *const f32, count: usize) -> BackendResult<f32>;
    
    // =========================================================================
    // Element-wise Operations
    // =========================================================================
    
    /// Element-wise addition: result = a + b
    unsafe fn add_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()>;
    
    /// Element-wise subtraction: result = a - b
    unsafe fn sub_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()>;
    
    /// Element-wise multiplication: result = a * b
    unsafe fn mul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()>;
    
    /// Element-wise division: result = a / b
    unsafe fn div_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()>;
    
    // =========================================================================
    // Lifecycle
    // =========================================================================
    
    /// Initialize the backend (called once before first use)
    fn init(&self) -> BackendResult<()> {
        Ok(())
    }
    
    /// Cleanup resources (called when backend is no longer needed)
    fn cleanup(&self) -> BackendResult<()> {
        Ok(())
    }
    
    /// Downcast to concrete type for backend-specific operations
    fn as_any(&self) -> &dyn Any;
}

/// Backend registry for managing available backends
pub struct BackendRegistry {
    backends: Vec<Box<dyn ComputeBackend>>,
    default_backend_idx: usize,
}

impl BackendRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        BackendRegistry {
            backends: Vec::new(),
            default_backend_idx: 0,
        }
    }
    
    /// Register a new backend
    pub fn register(&mut self, backend: Box<dyn ComputeBackend>) {
        self.backends.push(backend);
    }
    
    /// Get all available backends
    pub fn available_backends(&self) -> Vec<&dyn ComputeBackend> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .map(|b| b.as_ref())
            .collect()
    }
    
    /// Get backend by name
    pub fn get_by_name(&self, name: &str) -> Option<&dyn ComputeBackend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
    
    /// Get the default backend
    pub fn default_backend(&self) -> Option<&dyn ComputeBackend> {
        self.backends.get(self.default_backend_idx).map(|b| b.as_ref())
    }
    
    /// Set default backend by name
    pub fn set_default(&mut self, name: &str) -> bool {
        if let Some(idx) = self.backends.iter().position(|b| b.name() == name) {
            self.default_backend_idx = idx;
            true
        } else {
            false
        }
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing
    struct MockBackend {
        available: bool,
    }

    impl ComputeBackend for MockBackend {
        fn name(&self) -> &'static str { "mock" }
        fn backend_id(&self) -> u8 { 255 }
        fn is_available(&self) -> bool { self.available }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                supported_dtypes: vec![DataType::F32],
                ..Default::default()
            }
        }
        
        unsafe fn matmul_f32(&self, _: *const f32, _: *const f32, _: *mut f32, _: usize, _: usize, _: usize) -> BackendResult<()> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn sum_f32(&self, _: *const f32, _: usize) -> BackendResult<f32> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn max_f32(&self, _: *const f32, _: usize) -> BackendResult<f32> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn min_f32(&self, _: *const f32, _: usize) -> BackendResult<f32> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn add_f32(&self, _: *const f32, _: *const f32, _: *mut f32, _: usize) -> BackendResult<()> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn sub_f32(&self, _: *const f32, _: *const f32, _: *mut f32, _: usize) -> BackendResult<()> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn mul_f32(&self, _: *const f32, _: *const f32, _: *mut f32, _: usize) -> BackendResult<()> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        unsafe fn div_f32(&self, _: *const f32, _: *const f32, _: *mut f32, _: usize) -> BackendResult<()> {
            Err(BackendError::UnsupportedOperation("mock".into()))
        }
        fn as_any(&self) -> &dyn Any { self }
    }

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();
        registry.register(Box::new(MockBackend { available: true }));
        registry.register(Box::new(MockBackend { available: false }));
        
        assert_eq!(registry.available_backends().len(), 1);
        assert!(registry.get_by_name("mock").is_some());
    }
}
