# GPU Backend Extension Guide

This guide explains how to add a new GPU backend to CorePy (e.g., CUDA, ROCm, Vulkan).

## Architecture Overview

```
rust/core/src/
├── backend/
│   ├── mod.rs           # Backend policy and dispatch
│   └── capabilities.rs  # Runtime feature detection
└── ops/
    ├── metal.rs         # Metal backend (reference implementation)
    └── {cuda,rocm}.rs   # Your new backend

csrc/src/
├── metal/               # Metal implementation (reference)
│   ├── kernels.metal
│   └── metal_backend.mm
└── {cuda,rocm,vulkan}/  # Your new backend
    ├── kernels.{cu,hip,spv}
    └── backend.cpp
```

## Step 1: Create Kernel Directory

```bash
mkdir -p csrc/src/cuda
touch csrc/src/cuda/kernels.cu
touch csrc/src/cuda/cuda_backend.cpp
```

## Step 2: Implement C/C++ Backend

Create `csrc/src/cuda/cuda_backend.cpp`:

```cpp
#include <cuda_runtime.h>

extern "C" {
    bool cuda_is_available() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        return (err == cudaSuccess && deviceCount > 0);
    }
    
    void cuda_init() {
        cudaSetDevice(0);
    }
    
    void cuda_cleanup() {
        cudaDeviceReset();
    }
    
    void cuda_matmul_f32(const float* a, const float* b, float* c,
                         int m, int k, int n) {
        // Launch CUDA kernel
    }
}
```

## Step 3: Create Rust FFI Module

Create `rust/core/src/ops/cuda.rs`:

```rust
// Compile only when CUDA feature is enabled
#![allow(dead_code)]

#[cfg(feature = "cuda")]
mod ffi {
    extern "C" {
        pub fn cuda_is_available() -> bool;
        pub fn cuda_init();
        pub fn cuda_cleanup();
        pub fn cuda_matmul_f32(
            a: *const f32, b: *const f32, c: *mut f32,
            m: i32, k: i32, n: i32
        );
    }
}

#[cfg(feature = "cuda")]
pub fn is_available() -> bool {
    unsafe { ffi::cuda_is_available() }
}

#[cfg(not(feature = "cuda"))]
pub fn is_available() -> bool {
    false
}

// ... dispatch functions
```

## Step 4: Register in Operations Module

Edit `rust/core/src/ops/mod.rs`:

```rust
pub mod elementwise;
pub mod matmul;
pub mod metal;
pub mod reduce;

#[cfg(feature = "cuda")]
pub mod cuda;
```

## Step 5: Update Capabilities Detection

Edit `rust/core/src/backend/capabilities.rs`:

```rust
fn detect_gpu_capabilities() -> GpuCapabilities {
    GpuCapabilities {
        metal_available: detect_metal(),
        cuda_available: detect_cuda(),  // Add this
        rocm_available: false,
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda() -> bool {
    crate::ops::cuda::is_available()
}

#[cfg(not(feature = "cuda"))]
fn detect_cuda() -> bool {
    false
}
```

## Step 6: Add Python FFI

Edit `rust/core/src/ffi/python.rs`:

```rust
// Register functions
m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
m.add_function(wrap_pyfunction!(cuda_matmul_f32, m)?)?;

// Implement with platform guards
#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    #[cfg(feature = "cuda")]
    { Ok(crate::ops::cuda::is_available()) }
    
    #[cfg(not(feature = "cuda"))]
    { Ok(false) }
}
```

## Step 7: Update Build System

Edit `csrc/CMakeLists.txt`:

```cmake
option(ENABLE_CUDA "Enable CUDA backend" OFF)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(cuda_backend STATIC
        src/cuda/cuda_backend.cpp
        src/cuda/kernels.cu
    )
    target_link_libraries(cuda_backend CUDA::cudart)
endif()
```

Edit `rust/core/Cargo.toml`:

```toml
[features]
default = []
cuda = []
```

## Step 8: Add Tests

Create `tests/test_cuda.py`:

```python
import pytest
import sys

try:
    from corepy._corepy_rust import cuda_is_available
    HAS_CUDA = cuda_is_available()
except ImportError:
    HAS_CUDA = False

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_matmul():
    # Test implementation
    pass
```

## Checklist

- [ ] C/C++ backend with `extern "C"` exports
- [ ] Rust FFI module with `#[cfg(feature)]` guards
- [ ] Register in `ops/mod.rs`
- [ ] Update `capabilities.rs`
- [ ] Add Python FFI functions
- [ ] Update CMakeLists.txt
- [ ] Add Cargo feature flag
- [ ] Write tests with proper skip markers
- [ ] Ensure CPU fallback remains intact

## Key Principles

1. **Never break CPU path** - GPU backends are optional enhancements
2. **Use feature flags** - Don't compile GPU code by default
3. **Runtime detection** - Check availability before dispatch
4. **Graceful fallback** - Return errors, don't crash
5. **Consistent API** - Follow existing patterns in `metal.rs`
