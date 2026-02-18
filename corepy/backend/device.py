import os
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

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
        return 1024**3 * 16  # Placeholder: 16GB


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
    lib_names = ["cudart", "cudart.so.11.0", "cudart.so.12", "cuda"]
    lib_path = None
    for name in lib_names:
        lib_path = ctypes.util.find_library(name)
        if lib_path:
            break

    if not lib_path:
        # Fallback for linux if ldconfig not updated but path exists in standard locations
        common_paths = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
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
            if hasattr(cuda, "cudaGetDeviceCount"):
                ret = cuda.cudaGetDeviceCount(ctypes.byref(count))
                if ret == 0 and count.value > 0:
                    # Detected GPUs!
                    # For now, return a placeholder 8GB for each
                    return [8 * 1024**3] * count.value
        except Exception:
            pass

    return []


def _detect_metal_gpus() -> tuple[List[str], List[int]]:
    """
    Detects Apple Silicon/Metal GPUs using system_profiler.
    Returns (names, memory_bytes).
    """
    import json
    import subprocess
    import sys

    if sys.platform != "darwin":
        return [], []

    try:
        # User system_profiler to get JSON output for graphics/displays
        cmd = ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(output)

        names = []
        mems = []

        items = data.get("SPDisplaysDataType", [])
        for item in items:
            name = item.get("sppci_model", "Unknown GPU")

            # Parse VRAM string (e.g., "16 GB", "1536 MB")
            # Apple Silicon often reports unified memory or specific VRAM allocation
            vram_str = item.get("spdisplays_vram", "0 MB")

            try:
                parts = vram_str.split()
                if len(parts) >= 2:
                    val = float(parts[0])
                    unit = parts[1].upper()
                    if unit == "GB":
                        mem_bytes = int(val * 1024**3)
                    elif unit == "MB":
                        mem_bytes = int(val * 1024**2)
                    elif unit == "KB":
                        mem_bytes = int(val * 1024)
                    else:
                        mem_bytes = int(val)
                else:
                    mem_bytes = 0
            except ValueError:
                mem_bytes = 0

            names.append(name)
            mems.append(mem_bytes)

        return names, mems

    except Exception:
        # Fallback if system_profiler fails or is missing
        return [], []


def detect_devices() -> DeviceInfo:
    """
    Detects available hardware devices on the system.
    """
    info = DeviceInfo(cpu_cores=os.cpu_count() or 1)

    # Simple architecture checks
    machine = platform.machine().lower()
    if "x86_64" in machine or "amd64" in machine:
        info.has_avx2 = True  # optimistically assume AVX2 on modern x86
    elif "arm" in machine or "aarch64" in machine:
        info.has_neon = True

    # GPU Detection
    if platform.system() == "Darwin":
        # Primary Check: Ask Rust runtime if Metal is actually usable
        # This handles cases where system_profiler fails (CI) or Metal is unsupported
        try:
            from .. import _corepy_rust

            if _corepy_rust.metal_is_available():
                gpu_names, gpu_mems = _detect_metal_gpus()
                # If system_profiler failed but Rust says yes, add a generic Metal GPU
                if not gpu_names:
                    gpu_names = ["Metal GPU"]
                    gpu_mems = [0]  # Unknown memory
            else:
                gpu_names, gpu_mems = [], []
        except ImportError:
            # Fallback if Rust not loaded (shouldn't happen in installed pkg)
            gpu_names, gpu_mems = [], []
    else:
        # CUDA Detection
        gpu_mems = _detect_cuda_gpus()
        gpu_names = [f"CUDA Device {i}" for i in range(len(gpu_mems))]

    info.gpu_count = len(gpu_mems)
    info.gpu_memory_bytes = gpu_mems

    # Only overwrite gpu_names if we found something, to handle empty list correctly
    if info.gpu_count > 0:
        info.gpu_names = gpu_names

    return info
