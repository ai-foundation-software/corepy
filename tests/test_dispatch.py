import pytest

from corepy.backend.dispatch import dispatch_kernel, register_kernel
from corepy.backend.errors import OperationNotSupportedError
from corepy.backend.types import BackendType
from corepy.tensor import Tensor


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
    # Test with a guaranteed non-existent operation
    with pytest.raises(OperationNotSupportedError):
        dispatch_kernel("non_existent_op_12345", BackendType.CPU, [1], [1])


def test_dispatch_override_gpu(monkeypatch):
    """
    Register a mock GPU kernel and verify dispatch flows there.
    """

    # 1. Register Mock Kernel
    @register_kernel("add", BackendType.GPU)
    def gpu_add_mock(a, b):
        return ["gpu_result"]

    # Mock device detection to allow GPU backend selection
    from unittest.mock import MagicMock
    from corepy.backend.device import DeviceInfo
    
    # Needs to patch where it's used or simpler: reset session
    with monkeypatch.context() as m:
        gpu_info = DeviceInfo(cpu_cores=4, gpu_count=1, gpu_names=["MockGPU"])
        m.setattr("corepy.backend.device.detect_devices", lambda: gpu_info)
        m.setattr("corepy.backend.session.detect_devices", lambda: gpu_info)
        
        # Reset session to pick up new device info
        from corepy.backend import session
        old_session_var = session._session
        old_session_instance = session.Session._instance
        session._session = None
        session.Session._instance = None
        
        try:
            # 2. Create GPU Tensor (force backend)
            # Now valid because we fake-detected a GPU
            t_gpu = Tensor([1, 2], backend="gpu")
        
            # Verify it respects the request
            assert t_gpu.backend == BackendType.GPU
        
            # 3. Dispatch
            t_res = t_gpu + t_gpu
            assert t_res._backing_data == ["gpu_result"]
            assert t_res.backend == BackendType.GPU
            
        finally:
            session._session = old_session_var
            session.Session._instance = old_session_instance
