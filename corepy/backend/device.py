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
    cpu_threads: int = 1
    memory_limit_bytes: Optional[int] = None
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False
    l1_cache_size: int = 32768
    l2_cache_size: int = 524288
    l3_cache_size: int = 16777216
    gpu_count: int = 0
    has_cuda: bool = False
    has_metal: bool = False
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_bytes: List[int] = field(default_factory=list)
    gpu_memory_free_bytes: List[int] = field(default_factory=list)
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
        try:
            import psutil

            return psutil.virtual_memory().available
        except ImportError:
            return 1024**3 * 16  # Placeholder: 16GB


class GPUDevice(Device):
    def __init__(
        self,
        index: int,
        name: str,
        memory: int,
        memory_free: int,
        is_metal: bool = False,
    ):
        self._index = index
        self._name = name
        self._memory_total = memory
        self._memory_free = memory_free
        self._is_metal = is_metal

    @property
    def type(self) -> BackendType:
        return BackendType.METAL if self._is_metal else BackendType.CUDA

    @property
    def name(self) -> str:
        return f"GPU:{self._index} ({self._name})"

    @property
    def memory_free(self) -> int:
        return self._memory_free


def _detect_cuda_gpus() -> tuple[List[int], List[int]]:
    """
    Attempts to detect NVIDIA GPUs via nvidia-smi.
    Returns (memory_total_bytes_list, memory_free_bytes_list).
    """
    import shutil
    import subprocess

    if getattr(shutil, "which", lambda x: None)("nvidia-smi"):
        try:
            # Output format: "8192, 7000" (MiB)
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )

            totals = []
            frees = []

            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    # Convert MiB to Bytes
                    totals.append(int(parts[0]) * 1024 * 1024)
                    frees.append(int(parts[1]) * 1024 * 1024)

            if totals:
                return totals, frees
        except Exception:
            pass

    # Fallback to older ctypes method if nvidia-smi isn't available
    import ctypes.util

    lib_names = ["cudart", "cudart.so.11.0", "cudart.so.12", "cuda"]
    lib_path = None
    for name in lib_names:
        lib_path = ctypes.util.find_library(name)
        if lib_path:
            break

    if not lib_path:
        common_paths = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
        ]
        for p in common_paths:
            import os

            if os.path.exists(p):
                lib_path = p
                break

    if lib_path:
        try:
            cuda = ctypes.CDLL(lib_path)
            count = ctypes.c_int()
            if hasattr(cuda, "cudaGetDeviceCount"):
                ret = cuda.cudaGetDeviceCount(ctypes.byref(count))
                if ret == 0 and count.value > 0:
                    return [8 * 1024**3] * count.value, [8 * 1024**3] * count.value
        except Exception:
            pass

    return [], []


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


def _get_cpu_cores_threads():
    """Returns (physical_cores, logical_threads)."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        return physical or logical or 1, logical or 1
    except ImportError:
        import os

        count = os.cpu_count() or 1
        return count, count


def detect_devices() -> DeviceInfo:
    """
    Detects available hardware devices on the system using Rust FFI.
    """
    physical, logical = _get_cpu_cores_threads()
    info = DeviceInfo(cpu_cores=physical, cpu_threads=logical)

    try:
        from .. import _corepy_rust

        caps = _corepy_rust.get_system_capabilities()
        cpu_caps = caps.get("cpu", {})
        gpu_caps = caps.get("gpu", {})

        info.has_avx2 = cpu_caps.get("has_avx2", False)
        info.has_avx512 = cpu_caps.get("has_avx512", False)
        info.has_neon = cpu_caps.get("has_neon", False)

        # Add cache knowledge to the DeviceInfo
        info.l1_cache_size = cpu_caps.get("l1_cache", 32 * 1024)
        info.l2_cache_size = cpu_caps.get("l2_cache", 512 * 1024)
        info.l3_cache_size = cpu_caps.get("l3_cache", 16 * 1024 * 1024)

        if gpu_caps.get("metal_available", False):
            info.has_metal = True
            gpu_names, gpu_mems = _detect_metal_gpus()
            if not gpu_names:
                gpu_names = ["Metal GPU"]
                gpu_mems = [0]
            info.gpu_count = len(gpu_names)
            info.gpu_names = gpu_names
            info.gpu_memory_bytes = gpu_mems
            # For UMA, free memory is system free memory (approx)
            try:
                import psutil

                free_mem = psutil.virtual_memory().available
                info.gpu_memory_free_bytes = [free_mem] * len(gpu_names)
            except ImportError:
                info.gpu_memory_free_bytes = gpu_mems
        elif gpu_caps.get("cuda_available", False) or platform.system() != "Darwin":
            # Fallback to python CUDA detection
            info.has_cuda = True
            gpu_mems, gpu_frees = _detect_cuda_gpus()
            gpu_names = [f"CUDA Device {i}" for i in range(len(gpu_mems))]
            info.gpu_count = len(gpu_names)
            info.gpu_names = gpu_names
            info.gpu_memory_bytes = gpu_mems
            info.gpu_memory_free_bytes = gpu_frees

    except Exception:
        # Fallback if Rust module fails to load
        machine = platform.machine().lower()
        if "x86_64" in machine or "amd64" in machine:
            info.has_avx2 = True
        elif "arm" in machine or "aarch64" in machine:
            info.has_neon = True

        info.l1_cache_size = 32 * 1024
        info.l2_cache_size = 512 * 1024
        info.l3_cache_size = 16 * 1024 * 1024

        if platform.system() == "Darwin":
            info.has_metal = True
            gpu_names, gpu_mems = _detect_metal_gpus()
            info.gpu_count = len(gpu_names)
            info.gpu_names = gpu_names
            info.gpu_memory_bytes = gpu_mems
            try:
                import psutil

                free_mem = psutil.virtual_memory().available
                info.gpu_memory_free_bytes = [free_mem] * len(gpu_names)
            except ImportError:
                info.gpu_memory_free_bytes = gpu_mems
        else:
            info.has_cuda = True
            gpu_mems, gpu_frees = _detect_cuda_gpus()
            gpu_names = [f"CUDA Device {i}" for i in range(len(gpu_mems))]
            info.gpu_count = len(gpu_names)
            info.gpu_names = gpu_names
            info.gpu_memory_bytes = gpu_mems
            info.gpu_memory_free_bytes = gpu_frees

    return info
