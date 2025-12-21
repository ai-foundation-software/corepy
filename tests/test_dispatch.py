import pytest
from corepy.tensor import Tensor
from corepy.backend.types import BackendType
from corepy.backend.errors import OperationNotSupportedError
from corepy.backend.dispatch import dispatch_kernel, register_kernel

def test_cpu_add_dispatch():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    
    t3 = t1 + t2
    
    assert t3.backend == BackendType.CPU
    # Our simple implementation returns list
    assert t3._backing_data == [5, 7, 9]

def test_cpu_scalar_add():
    t1 = Tensor([1.0, 2.0])
    t2 = t1 + 10.0
    assert t2._backing_data == [11.0, 12.0]

def test_missing_kernel_error():
    t1 = Tensor([1, 2])
    # "sub" is not implemented yet
    # We need to manually invoke it or mock __sub__ since Tensor doesn't have it yet
    with pytest.raises(OperationNotSupportedError):
        dispatch_kernel("sub", BackendType.CPU, [1], [1])

def test_dispatch_override_gpu(monkeypatch):
    """
    Register a mock GPU kernel and verify dispatch flows there.
    """
    # 1. Register Mock Kernel
    @register_kernel("add", BackendType.GPU)
    def gpu_add_mock(a, b):
        return ["gpu_result"]
        
    # 2. Create GPU Tensor (force backend)
    # Note: Device selection logic might fallback to CPU if no GPU detected, 
    # so we force backend using internal override if needed or patch detection.
    # For this test, we just assume we can set _backend_type or use explicit 'backend="gpu"'
    # which 'select_backend' respects if passed.
    t_gpu = Tensor([1, 2], backend="gpu")
    
    # Verify it thinks it's strictly GPU backend (even if no hardware)
    # Our current select_backend implementation respects explicit 'backend' arg completely.
    assert t_gpu.backend == BackendType.GPU
    
    # 3. Dispatch
    t_res = t_gpu + t_gpu
    assert t_res._backing_data == ["gpu_result"]
    assert t_res.backend == BackendType.GPU
