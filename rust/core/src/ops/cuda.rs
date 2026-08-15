//! CUDA Backend Implementation using `libloading` (Dynamic Driver Loading)
//!
//! Handles device interaction, memory allocation, and kernel execution for NVIDIA GPUs
//! by loading `libcuda.so.1` dynamically at runtime, avoiding all zero-day NVCC and static
//! linkage compilation issues.

#![allow(dead_code, clippy::too_many_arguments)]

use crate::backend::traits::{
    BackendCapabilities, BackendError, BackendResult, ComputeBackend, DataType,
};
#[cfg(feature = "cuda")]
use libloading::Library;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::os::raw::{c_char, c_int, c_uint};

#[cfg(feature = "cuda")]
type CuInitFn = unsafe extern "system" fn(c_uint) -> c_int;
#[cfg(feature = "cuda")]
type CuDeviceGetFn = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
#[cfg(feature = "cuda")]
type CuCtxCreateFn = unsafe extern "system" fn(*mut *mut c_void, c_uint, c_int) -> c_int;
#[cfg(feature = "cuda")]
type CuCtxDestroyFn = unsafe extern "system" fn(*mut c_void) -> c_int;
#[cfg(feature = "cuda")]
type CuModuleLoadDataFn = unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> c_int;
#[cfg(feature = "cuda")]
type CuModuleGetFunctionFn =
    unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
#[cfg(feature = "cuda")]
type CuMemAllocFn = unsafe extern "system" fn(*mut usize, usize) -> c_int;
#[cfg(feature = "cuda")]
type CuMemFreeFn = unsafe extern "system" fn(usize) -> c_int;
#[cfg(feature = "cuda")]
type CuMemcpyHtoDFn = unsafe extern "system" fn(usize, *const c_void, usize) -> c_int;
#[cfg(feature = "cuda")]
type CuMemcpyDtoHFn = unsafe extern "system" fn(*mut c_void, usize, usize) -> c_int;
#[cfg(feature = "cuda")]
type CuMemsetD8Fn = unsafe extern "system" fn(usize, u8, usize) -> c_int;
#[cfg(feature = "cuda")]
type CuLaunchKernelFn = unsafe extern "system" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
#[cfg(feature = "cuda")]
type CuCtxSynchronizeFn = unsafe extern "system" fn() -> c_int;

#[cfg(feature = "cuda")]
pub struct CudaDriver {
    _lib: Library,
    pub cu_init: CuInitFn,
    pub cu_device_get: CuDeviceGetFn,
    pub cu_ctx_create: CuCtxCreateFn,
    pub cu_ctx_destroy: CuCtxDestroyFn,
    pub cu_module_load_data: CuModuleLoadDataFn,
    pub cu_module_get_function: CuModuleGetFunctionFn,
    pub cu_mem_alloc: CuMemAllocFn,
    pub cu_mem_free: CuMemFreeFn,
    pub cu_memcpy_htod: CuMemcpyHtoDFn,
    pub cu_memcpy_dtoh: CuMemcpyDtoHFn,
    pub cu_memset_d8: CuMemsetD8Fn,
    pub cu_launch_kernel: CuLaunchKernelFn,
    pub cu_ctx_synchronize: CuCtxSynchronizeFn,
}

#[cfg(feature = "cuda")]
impl CudaDriver {
    unsafe fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let lib = Library::new("libcuda.so.1").or_else(|_| Library::new("nvcuda.dll"))?;
        let cu_init = *lib.get(b"cuInit\0")?;
        let cu_device_get = *lib.get(b"cuDeviceGet\0")?;
        let cu_ctx_create = *lib.get(b"cuCtxCreate_v2\0")?;
        let cu_ctx_destroy = *lib.get(b"cuCtxDestroy_v2\0")?;
        let cu_module_load_data = *lib.get(b"cuModuleLoadData\0")?;
        let cu_module_get_function = *lib.get(b"cuModuleGetFunction\0")?;
        let cu_mem_alloc = *lib.get(b"cuMemAlloc_v2\0")?;
        let cu_mem_free = *lib.get(b"cuMemFree_v2\0")?;
        let cu_memcpy_htod = *lib.get(b"cuMemcpyHtoD_v2\0")?;
        let cu_memcpy_dtoh = *lib.get(b"cuMemcpyDtoH_v2\0")?;
        let cu_memset_d8 = *lib.get(b"cuMemsetD8_v2\0")?;
        let cu_launch_kernel = *lib.get(b"cuLaunchKernel\0")?;
        let cu_ctx_synchronize = *lib.get(b"cuCtxSynchronize\0")?;

        Ok(Self {
            _lib: lib,
            cu_init,
            cu_device_get,
            cu_ctx_create,
            cu_ctx_destroy,
            cu_module_load_data,
            cu_module_get_function,
            cu_mem_alloc,
            cu_mem_free,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cu_memset_d8,
            cu_launch_kernel,
            cu_ctx_synchronize,
        })
    }
}

#[cfg(feature = "cuda")]
struct LazyCudaDriver;
#[cfg(feature = "cuda")]
impl std::ops::Deref for LazyCudaDriver {
    type Target = Option<std::sync::Arc<CudaDriver>>;
    fn deref(&self) -> &Self::Target {
        static INSTANCE: std::sync::OnceLock<Option<std::sync::Arc<CudaDriver>>> =
            std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| unsafe { CudaDriver::load().ok().map(std::sync::Arc::new) })
    }
}

#[cfg(feature = "cuda")]
static CUDA_DRIVER: LazyCudaDriver = LazyCudaDriver;

#[cfg(feature = "cuda")]
const ELEMENTWISE_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
) {
    .reg .f32 %f<3>;
    .reg .s32 %r<4>;
    .reg .s64 %rd<10>;

    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r0, %r2, %r3, %tid.x;

    setp.ge.s32 %p1, %r0, %r1;
    @%p1 bra LB0_1;

    cvt.u64.s32 %rd1, %r0;
    mul.wide.u64 %rd2, %rd1, 4;
    
    ld.param.u64 %rd3, [a];
    add.u64 %rd4, %rd3, %rd2;
    ld.global.f32 %f1, [%rd4];

    ld.param.u64 %rd5, [b];
    add.u64 %rd6, %rd5, %rd2;
    ld.global.f32 %f2, [%rd6];

    add.f32 %f1, %f1, %f2;

    ld.param.u64 %rd7, [c];
    add.u64 %rd8, %rd7, %rd2;
    st.global.f32 [%rd8], %f1;

LB0_1:
    ret;
}

.visible .entry sub_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
) {
    .reg .f32 %f<3>;
    .reg .s32 %r<4>;
    .reg .s64 %rd<10>;

    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r0, %r2, %r3, %tid.x;

    setp.ge.s32 %p1, %r0, %r1;
    @%p1 bra LB0_1;

    cvt.u64.s32 %rd1, %r0;
    mul.wide.u64 %rd2, %rd1, 4;

    ld.param.u64 %rd3, [a];
    add.u64 %rd4, %rd3, %rd2;
    ld.global.f32 %f1, [%rd4];

    ld.param.u64 %rd5, [b];
    add.u64 %rd6, %rd5, %rd2;
    ld.global.f32 %f2, [%rd6];

    sub.f32 %f1, %f1, %f2;

    ld.param.u64 %rd7, [c];
    add.u64 %rd8, %rd7, %rd2;
    st.global.f32 [%rd8], %f1;

LB0_1:
    ret;
}

.visible .entry mul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
) {
    .reg .f32 %f<3>;
    .reg .s32 %r<4>;
    .reg .s64 %rd<10>;

    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r0, %r2, %r3, %tid.x;

    setp.ge.s32 %p1, %r0, %r1;
    @%p1 bra LB0_1;

    cvt.u64.s32 %rd1, %r0;
    mul.wide.u64 %rd2, %rd1, 4;

    ld.param.u64 %rd3, [a];
    add.u64 %rd4, %rd3, %rd2;
    ld.global.f32 %f1, [%rd4];

    ld.param.u64 %rd5, [b];
    add.u64 %rd6, %rd5, %rd2;
    ld.global.f32 %f2, [%rd6];

    mul.f32 %f1, %f1, %f2;

    ld.param.u64 %rd7, [c];
    add.u64 %rd8, %rd7, %rd2;
    st.global.f32 [%rd8], %f1;

LB0_1:
    ret;
}

.visible .entry div_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
) {
    .reg .f32 %f<3>;
    .reg .s32 %r<4>;
    .reg .s64 %rd<10>;

    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r0, %r2, %r3, %tid.x;

    setp.ge.s32 %p1, %r0, %r1;
    @%p1 bra LB0_1;

    cvt.u64.s32 %rd1, %r0;
    mul.wide.u64 %rd2, %rd1, 4;

    ld.param.u64 %rd3, [a];
    add.u64 %rd4, %rd3, %rd2;
    ld.global.f32 %f1, [%rd4];

    ld.param.u64 %rd5, [b];
    add.u64 %rd6, %rd5, %rd2;
    ld.global.f32 %f2, [%rd6];

    div.rn.f32 %f1, %f1, %f2;

    ld.param.u64 %rd7, [c];
    add.u64 %rd8, %rd7, %rd2;
    st.global.f32 [%rd8], %f1;

LB0_1:
    ret;
}
"#;

#[cfg(feature = "cuda")]
const MATMUL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry matmul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 M,
    .param .u32 K,
    .param .u32 N
) {
    .reg .f32 %f<6>;
    .reg .s32 %r<10>;
    .reg .s64 %rd<15>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %tid.x; // col
    
    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mad.lo.s32 %r6, %r4, %r5, %tid.y; // row

    ld.param.u32 %r7, [M];
    ld.param.u32 %r8, [N];
    setp.ge.s32 %p1, %r6, %r7;
    setp.ge.s32 %p2, %r3, %r8;
    or.pred %p3, %p1, %p2;
    @%p3 bra LB1_1;

    mov.f32 %f1, 0f00000000;
    ld.param.u32 %r9, [K];
    mov.u32 %r10, 0;

LB1_loop:
    cvt.u64.u32 %rd1, %r6;
    cvt.u64.u32 %rd2, %r9;
    mul.wide.u64 %rd3, %rd1, %rd2;
    cvt.u64.u32 %rd4, %r10;
    add.u64 %rd5, %rd3, %rd4;
    mul.wide.u64 %rd6, %rd5, 4;
    ld.param.u64 %rd7, [a];
    add.u64 %rd8, %rd7, %rd6;
    ld.global.f32 %f2, [%rd8];

    cvt.u64.u32 %rd9, %r10;
    cvt.u64.u32 %rd10, %r8;
    mul.wide.u64 %rd11, %rd9, %rd10;
    cvt.u64.u32 %rd12, %r3;
    add.u64 %rd13, %rd11, %rd12;
    mul.wide.u64 %rd14, %rd13, 4;
    ld.param.u64 %rd15, [b];
    add.u64 %rd16, %rd15, %rd14;
    ld.global.f32 %f3, [%rd16];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.s32 %r10, %r10, 1;
    setp.lt.u32 %p4, %r10, %r9;
    @%p4 bra LB1_loop;

    cvt.u64.u32 %rd1, %r6;
    cvt.u64.u32 %rd2, %r8;
    mul.wide.u64 %rd3, %rd1, %rd2;
    cvt.u64.u32 %rd4, %r3;
    add.u64 %rd5, %rd3, %rd4;
    mul.wide.u64 %rd6, %rd5, 4;
    ld.param.u64 %rd7, [c];
    add.u64 %rd8, %rd7, %rd6;
    st.global.f32 [%rd8], %f1;

LB1_1:
    ret;
}
"#;

pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    _driver: std::sync::Arc<CudaDriver>,
    #[cfg(feature = "cuda")]
    _ctx: *mut c_void,
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            (self._driver.cu_ctx_destroy)(self._ctx);
        }
    }
}

unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    pub fn new() -> Option<Self> {
        #[cfg(feature = "cuda")]
        {
            if let Some(driver) = CUDA_DRIVER.as_ref() {
                unsafe {
                    if (driver.cu_init)(0) == 0 {
                        let mut dev = 0;
                        if (driver.cu_device_get)(&mut dev, 0) == 0 {
                            let mut ctx: *mut c_void = std::ptr::null_mut();
                            if (driver.cu_ctx_create)(&mut ctx, 0, dev) == 0 {
                                return Some(Self {
                                    _driver: driver.clone(),
                                    _ctx: ctx,
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }
    fn backend_id(&self) -> u8 {
        3
    }
    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            CUDA_DRIVER.is_some()
        }
        #[cfg(not(feature = "cuda"))]
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
        if unsafe { cuda_add_f32(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "CUDA add_f32 failed".to_string(),
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
        if unsafe { cuda_matmul(a, b, c, m, k, n) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "CUDA matmul_f32 failed".to_string(),
            ))
        }
    }

    unsafe fn sum_f32(&self, data: *const f32, count: usize) -> BackendResult<f32> {
        Ok(crate::ops::reduce::sum_f32_cpu_dispatch(data, count))
    }
    unsafe fn max_f32(&self, _: *const f32, _: usize) -> BackendResult<f32> {
        Err(BackendError::UnsupportedOperation(
            "CUDA not yet implemented".to_string(),
        ))
    }
    unsafe fn min_f32(&self, _: *const f32, _: usize) -> BackendResult<f32> {
        Err(BackendError::UnsupportedOperation(
            "CUDA not yet implemented".to_string(),
        ))
    }
    unsafe fn sub_f32(
        &self,
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        count: usize,
    ) -> BackendResult<()> {
        if unsafe { cuda_sub_f32(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "CUDA sub_f32 failed".to_string(),
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
        if unsafe { cuda_mul_f32(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "CUDA mul_f32 failed".to_string(),
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
        if unsafe { cuda_div_f32(a, b, result, count) } {
            Ok(())
        } else {
            Err(BackendError::ExecutionError(
                "CUDA div_f32 failed".to_string(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(feature = "cuda")]
macro_rules! define_cuda_binop {
    ($name:ident, $func_name:expr) => {
        pub unsafe fn $name(a: *const f32, b: *const f32, result: *mut f32, size: usize) -> bool {
            if let Some(drv) = CUDA_DRIVER.as_ref() {
                let mut module: *mut c_void = std::ptr::null_mut();
                let ptx = std::ffi::CString::new(ELEMENTWISE_PTX).unwrap();
                if (drv.cu_module_load_data)(&mut module, ptx.as_ptr() as *const c_void) != 0 {
                    return false;
                }

                let mut func: *mut c_void = std::ptr::null_mut();
                let func_name = std::ffi::CString::new($func_name).unwrap();
                if (drv.cu_module_get_function)(&mut func, module, func_name.as_ptr()) != 0 {
                    return false;
                }

                let bytes = size * 4;
                let mut d_a: usize = 0;
                let mut d_b: usize = 0;
                let mut d_c: usize = 0;

                if (drv.cu_mem_alloc)(&mut d_a, bytes) != 0 {
                    return false;
                }
                if (drv.cu_mem_alloc)(&mut d_b, bytes) != 0 {
                    return false;
                }
                if (drv.cu_mem_alloc)(&mut d_c, bytes) != 0 {
                    return false;
                }

                (drv.cu_memcpy_htod)(d_a, a as *const c_void, bytes);
                (drv.cu_memcpy_htod)(d_b, b as *const c_void, bytes);
                (drv.cu_memset_d8)(d_c, 0, bytes);

                let mut args: [*mut c_void; 4] = [
                    &mut d_a as *mut _ as *mut c_void,
                    &mut d_b as *mut _ as *mut c_void,
                    &mut d_c as *mut _ as *mut c_void,
                    &size as *const _ as *mut c_void,
                ];

                let grid = (size as u32 + 255) / 256;
                let block = 256;

                let res = (drv.cu_launch_kernel)(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                (drv.cu_ctx_synchronize)();
                (drv.cu_memcpy_dtoh)(result as *mut c_void, d_c, bytes);

                (drv.cu_mem_free)(d_a);
                (drv.cu_mem_free)(d_b);
                (drv.cu_mem_free)(d_c);

                return res == 0;
            }
            false
        }
    };
}

#[cfg(feature = "cuda")]
define_cuda_binop!(cuda_add_f32, "add_f32");
#[cfg(feature = "cuda")]
define_cuda_binop!(cuda_sub_f32, "sub_f32");
#[cfg(feature = "cuda")]
define_cuda_binop!(cuda_mul_f32, "mul_f32");
#[cfg(feature = "cuda")]
define_cuda_binop!(cuda_div_f32, "div_f32");

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_add_f32(_: *const f32, _: *const f32, _: *mut f32, _: usize) -> bool {
    false
}
#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_sub_f32(_: *const f32, _: *const f32, _: *mut f32, _: usize) -> bool {
    false
}
#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_mul_f32(_: *const f32, _: *const f32, _: *mut f32, _: usize) -> bool {
    false
}
#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_div_f32(_: *const f32, _: *const f32, _: *mut f32, _: usize) -> bool {
    false
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_matmul(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if let Some(drv) = CUDA_DRIVER.as_ref() {
        let mut module: *mut c_void = std::ptr::null_mut();
        let ptx = std::ffi::CString::new(MATMUL_PTX).unwrap();
        if (drv.cu_module_load_data)(&mut module, ptx.as_ptr() as *const c_void) != 0 {
            return false;
        }

        let mut func: *mut c_void = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("matmul_f32").unwrap();
        if (drv.cu_module_get_function)(&mut func, module, func_name.as_ptr()) != 0 {
            return false;
        }

        let a_bytes = m * k * 4;
        let b_bytes = k * n * 4;
        let c_bytes = m * n * 4;

        let mut d_a: usize = 0;
        let mut d_b: usize = 0;
        let mut d_c: usize = 0;
        if (drv.cu_mem_alloc)(&mut d_a, a_bytes) != 0 {
            return false;
        }
        if (drv.cu_mem_alloc)(&mut d_b, b_bytes) != 0 {
            return false;
        }
        if (drv.cu_mem_alloc)(&mut d_c, c_bytes) != 0 {
            return false;
        }

        (drv.cu_memcpy_htod)(d_a, a as *const c_void, a_bytes);
        (drv.cu_memcpy_htod)(d_b, b as *const c_void, b_bytes);
        (drv.cu_memset_d8)(d_c, 0, c_bytes);

        let mut m_u32 = m as u32;
        let mut k_u32 = k as u32;
        let mut n_u32 = n as u32;
        let mut args: [*mut c_void; 6] = [
            &mut d_a as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut m_u32 as *mut _ as *mut c_void,
            &mut k_u32 as *mut _ as *mut c_void,
            &mut n_u32 as *mut _ as *mut c_void,
        ];

        let grid_x = (n as u32 + 15) / 16;
        let grid_y = (m as u32 + 15) / 16;
        let res = (drv.cu_launch_kernel)(
            func,
            grid_x,
            grid_y,
            1,
            16,
            16,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        (drv.cu_ctx_synchronize)();
        (drv.cu_memcpy_dtoh)(result as *mut c_void, d_c, c_bytes);

        (drv.cu_mem_free)(d_a);
        (drv.cu_mem_free)(d_b);
        (drv.cu_mem_free)(d_c);

        return res == 0;
    }
    false
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_matmul(
    _: *const f32,
    _: *const f32,
    _: *mut f32,
    _: usize,
    _: usize,
    _: usize,
) -> bool {
    false
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CUDA_DRIVER.is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
