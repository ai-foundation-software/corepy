from typing import Optional, Dict
from .types import BackendType
from .device import detect_devices, DeviceInfo
from .backend import Backend, CPUBackend, GPUBackend

class Session:
    """
    Manages the global state of the Corepy runtime, including detected devices
    and active backends.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Session, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._device_info: DeviceInfo = detect_devices()
        self._backends: Dict[BackendType, Backend] = {}
        
        # Initialize default backends
        self._backends[BackendType.CPU] = CPUBackend()
        if self._device_info.gpu_count > 0:
            self._backends[BackendType.GPU] = GPUBackend()
            
        self._initialized = True

    @property
    def device_info(self) -> DeviceInfo:
        return self._device_info

    def get_backend(self, backend_type: BackendType) -> Backend:
        if backend_type not in self._backends:
            # Try to lazy load or raise error
            raise ValueError(f"Backend {backend_type} not available or not initialized.")
        return self._backends[backend_type]

# Global session instance
_session = Session()

def get_session() -> Session:
    return _session
