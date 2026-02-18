import sys
from unittest.mock import MagicMock, patch

import pytest

import corepy
from corepy import array, ndarray
from corepy.backend.types import BackendType
from corepy.buffer import Device, DeviceType


def test_metal_dispatch_logic_mocked():
    """
    Test that ndarray dispatch logic correctly routes to Metal kernels
    when environment appears to be macOS with Metal available.
    """
    # Mock modules
    with patch.dict(sys.modules, {"corepy._corepy_rust": MagicMock()}):
        mock_rust = sys.modules["corepy._corepy_rust"]

        # Setup mocks
        mock_rust.metal_is_available.return_value = True
        mock_rust.metal_sum_f32.return_value = 123.0

        # Mock BufferView to report Metal device
        with patch.object(ndarray, "_get_buffer_view") as mock_get_view:
            mock_view = MagicMock()
            mock_view.device.is_metal.return_value = True
            mock_view.is_contiguous.return_value = True
            mock_view.data_ptr = 1000
            mock_view.element_count = 10
            mock_get_view.return_value = mock_view

            # Create array
            t = array([1.0] * 10, device="metal")
            # Force backend to GPU to simulate what __init__ does
            t._backend_type = BackendType.GPU

            # Verify sum dispatch
            val = t.sum()

            # Assertions
            mock_rust.metal_sum_f32.assert_called()
            # Result is wrapped in ndarray, unwrap to check
            assert val._get_scalar_value() == 123.0


def test_metal_init_mocked():
    """Test that array(..., device='metal') sets backend to GPU."""
    # Must mock GPU presence for this to succeed now
    from corepy.backend.device import DeviceInfo

    gpu_info = DeviceInfo(cpu_cores=4, gpu_count=1, gpu_names=["MockM1"])

    with patch("corepy.backend.session.detect_devices", return_value=gpu_info):
        with patch("corepy.backend.device.detect_devices", return_value=gpu_info):
            # Reset session
            import corepy.backend.session as session

            old_session_var = session._session
            old_session_instance = session.Session._instance
            session._session = None
            session.Session._instance = None
            try:
                t = array([1, 2, 3], device="metal")
                assert t.backend == BackendType.GPU
            finally:
                session._session = old_session_var
                session.Session._instance = old_session_instance
