from abc import ABC, abstractmethod
from typing import Optional, List, Any
import platform
import os
from dataclasses import dataclass, field
from .types import BackendType

@dataclass
class DeviceInfo:
    """
    Aggregated information about the execution environment's hardware.
    """
    cpu_cores: int
    memory_limit_bytes: Optional[int] = None
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_bytes: List[int] = field(default_factory=list)
    platform_system: str = platform.system()
    forced_backend: Optional[BackendType] = None

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

class Device(ABC):
    """
    Abstract base class for a hardware device.
    """
    @property
    @abstractmethod
    def type(self) -> BackendType:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def memory_free(self) -> int:
        """Approximate free memory in bytes."""
        pass

class CPUDevice(Device):
    def __init__(self, info: DeviceInfo):
        self._info = info
    
    @property
    def type(self) -> BackendType:
        return BackendType.CPU

    @property
    def name(self) -> str:
        return f"CPU ({self._info.cpu_cores} cores)"

    @property
    def memory_free(self) -> int:
        # Implementation to fetch real memory stats would go here (e.g., using psutil)
        # For now, return a safe large number or implement basic `psutil` check if allowed
        # Fallback to 'unknown'/None if not checking
        return 1024**3 * 16 # Placeholder: 16GB

class GPUDevice(Device):
    def __init__(self, index: int, name: str, memory: int):
        self._index = index
        self._name = name
        self._memory_total = memory
    
    @property
    def type(self) -> BackendType:
        return BackendType.GPU

    @property
    def name(self) -> str:
        return f"GPU:{self._index} ({self._name})"
    
    @property
    def memory_free(self) -> int:
        # Placeholder for actual GPU memory check
        return self._memory_total

def _detect_cuda_gpus() -> List[int]:
    """
    Attempts to detect NVIDIA GPUs via ctypes loading of libcudart/libcuda.
    Returns a list of memory sizes (in bytes) for detected GPUs.
    For this pass, we just return a list of fake memory sizes (e.g. 8GB) 
    if we detect a GPU, since getting exact memory requires complex struct mapping.
    """
    import ctypes.util
    
    # Try locating CUDA runtime
    lib_names = ['cudart', 'cudart.so.11.0', 'cudart.so.12', 'cuda']
    lib_path = None
    for name in lib_names:
        lib_path = ctypes.util.find_library(name)
        if lib_path:
            break
            
    if not lib_path:
        # Fallback for linux if ldconfig not updated but path exists in standard locations
        common_paths = [
            '/usr/local/cuda/lib64/libcudart.so',
            '/usr/lib/x86_64-linux-gnu/libcudart.so'
        ]
        for p in common_paths:
            if os.path.exists(p):
                lib_path = p
                break

    if lib_path:
        try:
            cuda = ctypes.CDLL(lib_path)
            count = ctypes.c_int()
            # cudaGetDeviceCount(int* count)
            if hasattr(cuda, 'cudaGetDeviceCount'):
                ret = cuda.cudaGetDeviceCount(ctypes.byref(count))
                if ret == 0 and count.value > 0:
                    # Detected GPUs!
                    # For now, return a placeholder 8GB for each
                    return [8 * 1024**3] * count.value
        except Exception:
            pass
            
    return []

def detect_devices() -> DeviceInfo:
    """
    Detects available hardware devices on the system.
    """
    info = DeviceInfo(cpu_cores=os.cpu_count() or 1)
    
    # Simple architecture checks
    machine = platform.machine().lower()
    if 'x86_64' in machine or 'amd64' in machine:
        # In a real implementation we would inspect feature flags (/proc/cpuinfo or cpuid)
        info.has_avx2 = True 
    elif 'arm' in machine or 'aarch64' in machine:
        info.has_neon = True

    # GPU Detection
    gpu_mems = _detect_cuda_gpus()
    info.gpu_count = len(gpu_mems)
    info.gpu_memory_bytes = gpu_mems
    if info.gpu_count > 0:
        info.gpu_names = [f"CUDA Device {i}" for i in range(info.gpu_count)]

    return info
