import builtins
import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

from corepy.backend.device import (
    CPUDevice,
    Device,
    DeviceInfo,
    GPUDevice,
    _detect_cuda_gpus,
    _detect_metal_gpus,
    detect_devices,
)
from corepy.backend.types import BackendType


class TestDeviceTypes:
    def test_device_info_default(self):
        info = DeviceInfo(cpu_cores=4)
        assert info.cpu_cores == 4
        assert not info.has_gpu
        assert info.gpu_count == 0

    def test_cpu_device(self):
        info = DeviceInfo(cpu_cores=8)
        dev = CPUDevice(info)
        assert dev.type == BackendType.CPU
        assert "8 cores" in dev.name
        assert dev.memory_free > 0

    def test_gpu_device(self):
        dev = GPUDevice(0, "Test GPU", 1024)
        assert dev.type == BackendType.GPU
        assert dev.name == "GPU:0 (Test GPU)"
        assert dev.memory_free == 1024


class TestDetectionLogic:
    @patch("platform.machine")
    def test_cpu_features_x86(self, mock_machine):
        mock_machine.return_value = "x86_64"
        with patch("platform.system", return_value="Linux"):
            with patch("corepy.backend.device._detect_cuda_gpus", return_value=[]):
                info = detect_devices()
                assert info.has_avx2 is True
                assert info.has_neon is False

    @patch("platform.machine")
    def test_cpu_features_arm(self, mock_machine):
        mock_machine.return_value = "aarch64"
        with patch("platform.system", return_value="Linux"):
            with patch("corepy.backend.device._detect_cuda_gpus", return_value=[]):
                info = detect_devices()
                assert info.has_neon is True
                assert info.has_avx2 is False

    @patch("platform.system", return_value="Darwin")
    def test_metal_detection_mock(self, mock_system):
        # Mock Rust extension presence
        # We need to ensure 'from .. import _corepy_rust' works.
        # This implies it looks for corepy._corepy_rust.
        mock_rust = MagicMock()
        mock_rust.metal_is_available.return_value = True

        with patch.dict(
            sys.modules,
            {
                "corepy._corepy_rust": mock_rust,
                "corepy.backend._corepy_rust": mock_rust,
            },
        ):
            # Also ensure that if 'corepy' is imported, it has the attribute
            import corepy

            with patch.object(corepy, "_corepy_rust", mock_rust, create=True):
                with patch("corepy.backend.device._detect_metal_gpus") as mock_detect:
                    mock_detect.return_value = (["Apple M1"], [16 * 1024**3])

                    info = detect_devices()
                    assert info.gpu_count == 1
                    assert info.gpu_names == ["Apple M1"]

    @patch("platform.system", return_value="Linux")
    @patch("corepy.backend.device._detect_cuda_gpus")
    def test_cuda_detection_mock(self, mock_detect, mock_system):
        mock_detect.return_value = [8 * 1024**3, 8 * 1024**3]

        info = detect_devices()
        assert info.gpu_count == 2
        assert len(info.gpu_names) == 2
        assert "CUDA Device 0" in info.gpu_names[0]

    def test_detect_metal_gpus_subprocess(self):
        # Test the subprocess parsing logic
        # Sample JSON output from system_profiler
        sample_json = """
        {
            "SPDisplaysDataType": [
                {
                    "sppci_model": "Apple M1 Pro",
                    "spdisplays_vram": "16 GB"
                }
            ]
        }
        """
        import subprocess

        with patch("subprocess.check_output", return_value=sample_json.encode("utf-8")):
            with patch("sys.platform", "darwin"):
                names, mems = _detect_metal_gpus()
                assert names == ["Apple M1 Pro"]
                assert mems == [16 * 1024**3]

    def test_detect_metal_gpus_failure(self):
        with patch("subprocess.check_output", side_effect=ValueError("fail")):
            with patch("sys.platform", "darwin"):
                names, mems = _detect_metal_gpus()
                assert names == []

    @patch("ctypes.util.find_library")
    def test_detect_cuda_gpus_mock_lib(self, mock_find):
        mock_find.return_value = "/tmp/fake_libcuda.so"

        # Mock ctypes.CDLL to check if it tries to load
        with patch("ctypes.CDLL") as mock_cdll:
            instance = mock_cdll.return_value
            instance.cudaGetDeviceCount.return_value = 0

            # Setup count ref
            with patch("ctypes.c_int") as mock_int:
                mock_int_inst = mock_int.return_value
                mock_int_inst.value = 2

                _detect_cuda_gpus()
                # We can just verify it ran without error
                # Deeper validity checks require mocking byref etc. suited for ctypes
                pass
