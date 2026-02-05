from enum import IntEnum

from .backend import Backend, CPUBackend, GPUBackend
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


__all__ = [
    "BackendType",
    "OperationType",
    "OperationProperties",
    "DataType",
    "DeviceInfo",
    "BackendPolicy",
    "set_backend_policy",
    "get_backend_policy",
    "explain_last_dispatch",
    "BackendError",
    "DeviceNotFoundError",
    "OutOfMemoryError",
    "Device",
    "CPUDevice",
    "GPUDevice",
    "detect_devices",
    "Backend",
    "CPUBackend",
    "GPUBackend",
    "ReferenceBackend",
    "select_backend",
    "get_session",
    "Session",
]
