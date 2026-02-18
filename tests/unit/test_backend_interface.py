import pytest

from corepy.backend.backend import CPUBackend, GPUBackend
from corepy.backend.types import BackendType, OperationType


class TestBackendInterface:
    def test_cpu_backend(self):
        backend = CPUBackend()
        assert backend.device_type == BackendType.CPU
        assert backend.is_available() is True
        assert backend.supports_operation(OperationType.COMPUTE_VECTOR) is True

    def test_gpu_backend_placeholder(self):
        # Even if not functional, we test the class defines
        backend = GPUBackend()
        assert backend.device_type == BackendType.GPU
        # Currently hardcoded to False in placeholder
        assert backend.is_available() is False

        # Support logic
        assert backend.supports_operation(OperationType.COMPUTE_VECTOR) is True
        assert backend.supports_operation(OperationType.CONTROL) is False
        assert backend.supports_operation(OperationType.SCALAR) is False
