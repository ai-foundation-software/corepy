from enum import IntEnum

from .backend import Backend, CPUBackend, CUDABackend, MetalBackend
from .device import CPUDevice, Device, DeviceInfo, GPUDevice, detect_devices
from .reference import ReferenceBackend
from .selector import select_backend
from .session import Session, get_session
from .types import BackendType, DataType, OperationProperties, OperationType


class BackendPolicy(IntEnum):
    DEFAULT = 0
    OPENBLAS = 1
    BLAS = 2
    CUDA = 3
    METAL = 4


_GLOBAL_POLICY = BackendPolicy.DEFAULT


def set_backend_policy(policy: BackendPolicy):
    """Set the global CPU backend selection policy."""
    global _GLOBAL_POLICY
    _GLOBAL_POLICY = policy
    try:
        from corepy import _corepy_rust

        _corepy_rust.set_backend_policy(int(policy))
    except ImportError:
        # Rust extension not available, running in Python-only mode
        pass


def get_backend_policy() -> BackendPolicy:
    """Get the current global CPU backend selection policy."""
    try:
        from corepy import _corepy_rust

        return BackendPolicy(_corepy_rust.get_backend_policy())
    except ImportError:
        return _GLOBAL_POLICY


def explain_last_dispatch() -> str:
    """Returns a string explaining which backend was used for the last operation."""
    try:
        from corepy import _corepy_rust

        return _corepy_rust.explain_last_dispatch()
    except ImportError:
        return f"Method dispatched via Python Fallback (Backend: {_GLOBAL_POLICY.name})"


def analyse_workload(
    matrix_size: int, small_threshold: int = 64, gpu_threshold: int = 512
) -> str:
    """Analyze the workload and hardware to predict which backend will be used."""
    try:
        from corepy import _corepy_rust

        return _corepy_rust.analyse_workload(
            matrix_size, small_threshold, gpu_threshold
        )
    except ImportError:
        return "Backend analysis unavailable (Rust extension not loaded)."


def get_system_capabilities():
    """Get system hardware capabilities (CPU features, GPU availability)."""
    try:
        from corepy import _corepy_rust

        return _corepy_rust.get_system_capabilities()
    except ImportError:
        return {
            "cpu": {"arch": "unknown", "cores": 1},
            "gpu": {"metal_available": False, "cuda_available": False},
        }


__all__ = [
    "BackendType",
    "OperationType",
    "OperationProperties",
    "DataType",
    "DeviceInfo",
    "BackendPolicy",
    "set_backend_policy",
    "get_backend_policy",
    "set_backend_policy",
    "get_backend_policy",
    "get_system_capabilities",
    "explain_last_dispatch",
    "analyse_workload",
    "BackendError",
    "DeviceNotFoundError",
    "OutOfMemoryError",
    "Device",
    "CPUDevice",
    "GPUDevice",
    "detect_devices",
    "Backend",
    "CPUBackend",
    "CUDABackend",
    "MetalBackend",
    "ReferenceBackend",
    "select_backend",
    "get_session",
    "Session",
]
