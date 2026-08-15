// ============================================================================
// Operations: Metal GPU Backend
// ============================================================================
// This module provides Metal GPU acceleration for macOS/Apple Silicon.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::backend::traits::{
    BackendCapabilities, BackendError, BackendResult, ComputeBackend, DataType,
};
#[cfg(feature = "metal")]
use metal::*;

#[cfg(feature = "metal")]
struct LazyMetalDevice;
#[cfg(feature = "metal")]
impl std::ops::Deref for LazyMetalDevice {
    type Target = Option<Device>;
    fn deref(&self) -> &Self::Target {
        static INSTANCE: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| Device::system_default())
    }
}
#[cfg(feature = "metal")]
static METAL_DEVICE: LazyMetalDevice = LazyMetalDevice;

#[cfg(feature = "metal")]
struct LazyMetalQueue;
#[cfg(feature = "metal")]
impl std::ops::Deref for LazyMetalQueue {
    type Target = Option<CommandQueue>;
    fn deref(&self) -> &Self::Target {
        static INSTANCE: std::sync::OnceLock<Option<CommandQueue>> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| METAL_DEVICE.as_ref().map(|d| d.new_command_queue()))
    }
}
#[cfg(feature = "metal")]
static METAL_QUEUE: LazyMetalQueue = LazyMetalQueue;

#[cfg(feature = "metal")]
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    result[id] = a[id] + b[id];
}

kernel void matmul_f32(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    constant uint3& dims [[ buffer(3) ]],
    uint2 id [[ thread_position_in_grid ]]
) {
    uint row = id.y;
    uint col = id.x;
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        result[row * N + col] = sum;
    }
}

kernel void sub_f32(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    result[id] = a[id] - b[id];
}

kernel void mul_f32(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    result[id] = a[id] * b[id];
}

kernel void div_f32(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    result[id] = a[id] / b[id];
}

kernel void sum_f32(
    device const float* data [[ buffer(0) ]],
    device float* result [[ buffer(1) ]],
    constant uint& count [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < count; ++i) {
            sum += data[i];
        }
        result[0] = sum;
    }
}

kernel void max_f32(
    device const float* data [[ buffer(0) ]],
    device float* result [[ buffer(1) ]],
    constant uint& count [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id == 0) {
        float max_val = count > 0 ? data[0] : 0.0f;
        for (uint i = 1; i < count; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }
        result[0] = max_val;
    }
}

kernel void min_f32(
    device const float* data [[ buffer(0) ]],
    device float* result [[ buffer(1) ]],
    constant uint& count [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id == 0) {
        float min_val = count > 0 ? data[0] : 0.0f;
        for (uint i = 1; i < count; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }
        result[0] = min_val;
    }
}

kernel void transpose_f32(
    device const float* a [[ buffer(0) ]],
    device float* result [[ buffer(1) ]],
    constant uint2& dims [[ buffer(2) ]],
    uint2 id [[ thread_position_in_grid ]]
) {
    uint row = id.y;
    uint col = id.x;
    uint M = dims.x;
    uint N = dims.y;

    if (row < M && col < N) {
        result[col * M + row] = a[row * N + col];
    }
}
"#;

pub struct MetalBackend {
    #[cfg(feature = "metal")]
    _device: Device,
    #[cfg(feature = "metal")]
    _queue: CommandQueue,
}

impl MetalBackend {
    pub fn new() -> Option<Self> {
        #[cfg(feature = "metal")]
        {
            if let (Some(device), Some(queue)) = (METAL_DEVICE.clone(), METAL_QUEUE.clone()) {
                return Some(Self {
                    _device: device,
                    _queue: queue,
                });
            }
        }
        None
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "Metal"
    }
    fn backend_id(&self) -> u8 {
        4
    }
    fn is_available(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            METAL_DEVICE.is_some()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![DataType::F32],
            ..Default::default()
        }
    }

    unsafe fn add_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if unsafe { metal_add(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "Metal add_f32 failed".to_string(),
            ))
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
        if unsafe { metal_matmul(a, b, c, m, k, n) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "Metal matmul_f32 failed".to_string(),
            ))
        }
    }

    unsafe fn sum_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        let mut result = 0.0f32;
        if unsafe { metal_sum(data, &mut result, count) } {
            Ok(result)
        } else {
            Err(BackendError::ExecutionError(
                "Metal sum_f32 failed".to_string(),
            ))
        }
    }
    unsafe fn max_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        let mut result = 0.0f32;
        if unsafe { metal_max(data, &mut result, count) } {
            Ok(result)
        } else {
            Err(BackendError::ExecutionError(
                "Metal max_f32 failed".to_string(),
            ))
        }
    }
    unsafe fn min_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        let mut result = 0.0f32;
        if unsafe { metal_min(data, &mut result, count) } {
            Ok(result)
        } else {
            Err(BackendError::ExecutionError(
                "Metal min_f32 failed".to_string(),
            ))
        }
    }
    unsafe fn sub_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if unsafe { metal_sub(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "Metal sub_f32 failed".to_string(),
            ))
        }
    }
    unsafe fn mul_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if unsafe { metal_mul(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "Metal mul_f32 failed".to_string(),
            ))
        }
    }
    unsafe fn div_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if unsafe { metal_div(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "Metal div_f32 failed".to_string(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(feature = "metal")]
pub unsafe fn metal_add(a: *const f32, b: *const f32, result: *mut f32, size: usize) -> bool {
    if let (Some(device), Some(queue)) = (METAL_DEVICE.as_ref(), METAL_QUEUE.as_ref()) {
        let options = CompileOptions::new();
        let library = match device.new_library_with_source(SHADER_SOURCE, &options) {
            Ok(lib) => lib,
            Err(_) => return false,
        };
        let function = match library.get_function("add_f32", None) {
            Ok(func) => func,
            Err(_) => return false,
        };
        let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
            Ok(p) => p,
            Err(_) => return false,
        };

        let data_size = (size * 4) as u64;
        let buf_a = device.new_buffer_with_data(
            a as *const _,
            data_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b = device.new_buffer_with_data(
            b as *const _,
            data_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_res = device.new_buffer(data_size, MTLResourceOptions::StorageModeShared);

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_res), 0);

        let threads_per_grid = MTLSize::new(size as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(std::cmp::min(size as u64, 256), 1, 1);

        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        std::ptr::copy_nonoverlapping(buf_res.contents() as *const f32, result, size);
        return true;
    }
    false
}

#[cfg(not(feature = "metal"))]
pub unsafe fn metal_add(_a: *const f32, _b: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}

#[cfg(feature = "metal")]
pub unsafe fn metal_matmul(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if let (Some(device), Some(queue)) = (METAL_DEVICE.as_ref(), METAL_QUEUE.as_ref()) {
        let options = CompileOptions::new();
        let library = match device.new_library_with_source(SHADER_SOURCE, &options) {
            Ok(lib) => lib,
            Err(_) => return false,
        };
        let function = match library.get_function("matmul_f32", None) {
            Ok(func) => func,
            Err(_) => return false,
        };
        let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
            Ok(p) => p,
            Err(_) => return false,
        };

        let buf_a = device.new_buffer_with_data(
            a as *const _,
            (m * k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b = device.new_buffer_with_data(
            b as *const _,
            (k * n * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_res = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let dims = [m as u32, k as u32, n as u32];
        let buf_dims = device.new_buffer_with_data(
            dims.as_ptr() as *const _,
            12,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_res), 0);
        encoder.set_buffer(3, Some(&buf_dims), 0);

        let threads_per_grid = MTLSize::new(n as u64, m as u64, 1);
        let threads_per_threadgroup = MTLSize::new(8, 8, 1);

        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        std::ptr::copy_nonoverlapping(buf_res.contents() as *const f32, result, m * n);
        return true;
    }
    false
}

#[cfg(feature = "metal")]
macro_rules! define_metal_binop {
    ($name:ident, $func_name:expr) => {
        pub unsafe fn $name(a: *const f32, b: *const f32, result: *mut f32, size: usize) -> bool {
            if let (Some(device), Some(queue)) = (METAL_DEVICE.as_ref(), METAL_QUEUE.as_ref()) {
                let options = CompileOptions::new();
                let library = match device.new_library_with_source(SHADER_SOURCE, &options) {
                    Ok(lib) => lib,
                    Err(_) => return false,
                };
                let function = match library.get_function($func_name, None) {
                    Ok(func) => func,
                    Err(_) => return false,
                };
                let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
                    Ok(p) => p,
                    Err(_) => return false,
                };

                let data_size = (size * 4) as u64;
                let buf_a = device.new_buffer_with_data(
                    a as *const _,
                    data_size,
                    MTLResourceOptions::StorageModeShared,
                );
                let buf_b = device.new_buffer_with_data(
                    b as *const _,
                    data_size,
                    MTLResourceOptions::StorageModeShared,
                );
                let buf_res = device.new_buffer(data_size, MTLResourceOptions::StorageModeShared);

                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&buf_a), 0);
                encoder.set_buffer(1, Some(&buf_b), 0);
                encoder.set_buffer(2, Some(&buf_res), 0);

                let threads_per_grid = MTLSize::new(size as u64, 1, 1);
                let threads_per_threadgroup = MTLSize::new(std::cmp::min(size as u64, 256), 1, 1);

                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                std::ptr::copy_nonoverlapping(buf_res.contents() as *const f32, result, size);
                return true;
            }
            false
        }
    };
}

#[cfg(feature = "metal")]
define_metal_binop!(metal_sub, "sub_f32");
#[cfg(feature = "metal")]
define_metal_binop!(metal_mul, "mul_f32");
#[cfg(feature = "metal")]
define_metal_binop!(metal_div, "div_f32");

#[cfg(feature = "metal")]
macro_rules! define_metal_reduce {
    ($name:ident, $func_name:expr) => {
        pub unsafe fn $name(data: *const f32, result: *mut f32, size: usize) -> bool {
            if let (Some(device), Some(queue)) = (METAL_DEVICE.as_ref(), METAL_QUEUE.as_ref()) {
                let options = CompileOptions::new();
                let library = match device.new_library_with_source(SHADER_SOURCE, &options) {
                    Ok(lib) => lib,
                    Err(_) => return false,
                };
                let function = match library.get_function($func_name, None) {
                    Ok(func) => func,
                    Err(_) => return false,
                };
                let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
                    Ok(p) => p,
                    Err(_) => return false,
                };

                let data_size = (size * 4) as u64;
                let buf_data = device.new_buffer_with_data(
                    data as *const _,
                    data_size,
                    MTLResourceOptions::StorageModeShared,
                );
                let buf_res = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
                let count_val = size as u32;
                let buf_count = device.new_buffer_with_data(
                    &count_val as *const _ as *const _,
                    4,
                    MTLResourceOptions::StorageModeShared,
                );

                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&buf_data), 0);
                encoder.set_buffer(1, Some(&buf_res), 0);
                encoder.set_buffer(2, Some(&buf_count), 0);

                let threads_per_grid = MTLSize::new(1, 1, 1);
                let threads_per_threadgroup = MTLSize::new(1, 1, 1);

                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                std::ptr::copy_nonoverlapping(buf_res.contents() as *const f32, result, 1);
                return true;
            }
            false
        }
    };
}

#[cfg(feature = "metal")]
define_metal_reduce!(metal_sum, "sum_f32");
#[cfg(feature = "metal")]
define_metal_reduce!(metal_max, "max_f32");
#[cfg(feature = "metal")]
define_metal_reduce!(metal_min, "min_f32");

#[cfg(not(feature = "metal"))]
pub unsafe fn metal_matmul(
    _a: *const f32,
    _b: *const f32,
    _result: *mut f32,
    _m: usize,
    _k: usize,
    _n: usize,
) -> bool {
    false
}

#[cfg(not(feature = "metal"))]
pub unsafe fn metal_sub(_a: *const f32, _b: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}
#[cfg(not(feature = "metal"))]
pub unsafe fn metal_mul(_a: *const f32, _b: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}
#[cfg(not(feature = "metal"))]
pub unsafe fn metal_div(_a: *const f32, _b: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}

#[cfg(not(feature = "metal"))]
pub unsafe fn metal_sum(_data: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}
#[cfg(not(feature = "metal"))]
pub unsafe fn metal_max(_data: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}
#[cfg(not(feature = "metal"))]
pub unsafe fn metal_min(_data: *const f32, _result: *mut f32, _size: usize) -> bool {
    false
}

// FFI compatibility functions
pub fn is_available() -> bool {
    #[cfg(feature = "metal")]
    {
        METAL_DEVICE.is_some()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

pub unsafe fn add_f32_metal_dispatch(a: *const f32, b: *const f32, result: *mut f32, size: usize) {
    unsafe { metal_add(a, b, result, size) };
}

pub unsafe fn matmul_f32_metal_dispatch(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe { metal_matmul(a, b, result, m, k, n) };
}

pub unsafe fn sum_f32_metal_dispatch(_a: *const f32, _size: usize) -> f32 {
    let mut res = 0.0f32;
    unsafe { metal_sum(_a, &mut res, _size) };
    res
}
pub unsafe fn mean_f32_metal_dispatch(_a: *const f32, _size: usize) -> f32 {
    if _size == 0 {
        return 0.0;
    }
    let mut res = 0.0f32;
    unsafe { metal_sum(_a, &mut res, _size) };
    res / (_size as f32)
}
pub unsafe fn max_f32_metal_dispatch(_a: *const f32, _size: usize) -> f32 {
    let mut res = 0.0f32;
    unsafe { metal_max(_a, &mut res, _size) };
    res
}
pub unsafe fn min_f32_metal_dispatch(_a: *const f32, _size: usize) -> f32 {
    let mut res = 0.0f32;
    unsafe { metal_min(_a, &mut res, _size) };
    res
}

pub unsafe fn sub_f32_metal_dispatch(_a: *const f32, _b: *const f32, _res: *mut f32, _size: usize) {
    unsafe { metal_sub(_a, _b, _res, _size) };
}
pub unsafe fn mul_f32_metal_dispatch(_a: *const f32, _b: *const f32, _res: *mut f32, _size: usize) {
    unsafe { metal_mul(_a, _b, _res, _size) };
}
pub unsafe fn div_f32_metal_dispatch(_a: *const f32, _b: *const f32, _res: *mut f32, _size: usize) {
    unsafe { metal_div(_a, _b, _res, _size) };
}

pub unsafe fn transpose_f32_metal_dispatch(
    _in_ptr: *const f32,
    _out_ptr: *mut f32,
    _m: usize,
    _n: usize,
) {
    // Basic CPU fallback for transpose until shader exists
    let src = std::slice::from_raw_parts(_in_ptr, _m * _n);
    let dst = std::slice::from_raw_parts_mut(_out_ptr, _m * _n);
    for r in 0.._m {
        for c in 0.._n {
            dst[c * _m + r] = src[r * _n + c];
        }
    }
}

pub unsafe fn broadcast_op(
    _op: i32,
    _a: *const f32,
    _b: *const f32,
    _out: *mut f32,
    _shape: *const i32,
    _strides_a: *const i32,
    _strides_b: *const i32,
    _rank: i32,
    _size: i32,
    _size_a: i32,
    _size_b: i32,
) {
    unimplemented!("Metal broadcast_op not implemented")
}
