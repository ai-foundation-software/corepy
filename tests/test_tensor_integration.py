from typing import Any
from corepy.tensor import Tensor
from corepy.backend import BackendType
import pytest

def test_tensor_creation_defaults():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.backend == BackendType.CPU
    assert t.shape == (3,)

def test_tensor_auto_gpu_threshold_mock(monkeypatch):
    """
    Test that large tensors default to GPU if GPU is 'detected'.
    """
    # Mock detection to simulate GPU presence
    from corepy.backend.device import DeviceInfo
    mock_info = DeviceInfo(cpu_cores=4, gpu_count=1)
    
    from corepy.backend import session
    
    # Reset singleton manually
    old_session = session._session
    try:
        session._session = None 
        session.Session._instance = None # Also reset class instance tracking if used
        
        # Apply patch to detect_devices BEFORE creating new session
        with monkeypatch.context() as m:
            m.setattr("corepy.backend.device.detect_devices", lambda: mock_info)
            # monkeypatching module-level detect_devices in session.py if it was imported as such
            # checking session.py imports: "from .device import detect_devices, DeviceInfo"
            m.setattr("corepy.backend.session.detect_devices", lambda: mock_info)
            
            # Re-initialize session (will trigger detect_devices)
            s = session.Session()
            session._session = s
            
            # 1. Small Tensor -> CPU
            t_small = Tensor([1.0]*1000)
            assert t_small.backend == BackendType.CPU

            # 2. Large Tensor -> GPU
            # THRESHOLD is 100,000
            t_large = Tensor([1.0]*100_001)
            assert t_large.backend == BackendType.GPU
    finally:
        # Restore session
        session._session = old_session


def test_tensor_explicit_override_api():
    t = Tensor([1,2,3], backend="gpu")
    assert t.backend == BackendType.GPU

def test_tensor_explicit_device_api():
    t = Tensor([1,2,3], device="cuda:0")
    assert t.backend == BackendType.GPU
