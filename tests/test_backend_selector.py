from unittest.mock import patch

import pytest

from corepy.backend.device import DeviceInfo
from corepy.backend.selector import select_backend
from corepy.backend.types import BackendType, OperationProperties, OperationType
from corepy.compute import ComputeConfig, get_config, set_config


@pytest.fixture
def cpu_only_device():
    return DeviceInfo(cpu_cores=4, gpu_count=0)


@pytest.fixture
def gpu_device():
    return DeviceInfo(
        cpu_cores=4,
        gpu_count=1,
        has_cuda=True,
        gpu_names=["TestGPU"],
        gpu_memory_bytes=[8 * 1024**3],
        platform_system="Linux",
    )


@pytest.fixture
def metal_device():
    return DeviceInfo(
        cpu_cores=4,
        gpu_count=1,
        has_metal=True,
        gpu_names=["Apple M2 Max"],
        gpu_memory_bytes=[32 * 1024**3],
        platform_system="Darwin",
    )


def test_select_backend_cpu_default(cpu_only_device):
    op_props = OperationProperties(element_count=1000, shape=(1000,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, cpu_only_device)
    assert backend == BackendType.CPU


def test_select_backend_gpu_vector_threshold(gpu_device):
    # Below threshold (New: 2,000_000)
    op_props_small = OperationProperties(element_count=1_999_999, shape=(1_999_999,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_small, gpu_device)
    assert backend == BackendType.CPU

    # Above threshold
    op_props_large = OperationProperties(element_count=2_000_000, shape=(2_000_000,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_large, gpu_device)
    assert backend == BackendType.CUDA


@patch("corepy.backend.selector._corepy_rust", None)
def test_select_backend_gpu_matrix_threshold(gpu_device):
    # Retrieve the default config and save it
    original_config = get_config()

    # Isolate matrix dimension check by setting a very high FLOP threshold
    test_config = ComputeConfig(
        cuda_min_matrix_dim=4096,
        cuda_min_flops=10**15,  # Very high FLOP boundary
    )
    set_config(test_config)

    # Below threshold (4000 is < 4096)
    op_props_small = OperationProperties(element_count=4000 * 4000, shape=(4000, 4000))
    backend = select_backend(OperationType.COMPUTE_MATRIX, op_props_small, gpu_device)
    assert backend == BackendType.CPU

    # Above threshold
    op_props_large = OperationProperties(element_count=4096 * 4096, shape=(4096, 4096))
    backend = select_backend(OperationType.COMPUTE_MATRIX, op_props_large, gpu_device)
    assert backend == BackendType.CUDA

    set_config(original_config)


def test_select_backend_control_always_cpu(gpu_device):
    op_props = OperationProperties(element_count=10**7, shape=(10**7,))  # Huge
    backend = select_backend(OperationType.CONTROL, op_props, gpu_device)
    assert backend == BackendType.CPU


def test_select_backend_explicit_request():
    # We must mock a GPU device for the request to be honored now
    gpu_info = DeviceInfo(cpu_cores=4, gpu_count=1, platform_system="Linux")
    op_props = OperationProperties(element_count=100, shape=(100,))

    backend = select_backend(
        OperationType.COMPUTE_VECTOR,
        op_props,
        gpu_info,  # Pass fake GPU info
        requested_backend=BackendType.CUDA,
    )
    assert backend == BackendType.CUDA


@patch("os.getenv")
def test_select_backend_env_var_override(mock_getenv, gpu_device):
    mock_getenv.return_value = "cpu"  # Force CPU despite GPU being better
    op_props = OperationProperties(element_count=10**7, shape=(10**7,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, gpu_device)
    assert backend == BackendType.CPU

    mock_getenv.return_value = "gpu"
    # Even small op forced to GPU
    op_props_small = OperationProperties(element_count=100, shape=(100,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_small, gpu_device)
    # the environment variable "gpu" falls back to CUDA automatically on non-Mac platforms
    assert backend == BackendType.CUDA


def test_select_backend_dynamic_compute_config(gpu_device):
    # Retrieve the default config and save it
    original_config = get_config()

    # Normally this is a tiny element count that goes to CPU
    op_props = OperationProperties(element_count=50, shape=(50,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, gpu_device)
    assert backend == BackendType.CPU

    # Modify the config to forcefully trigger GPU routing at a tiny threshold
    new_config = ComputeConfig(cuda_min_vector_size=10, cuda_min_flops=10)
    set_config(new_config)

    # Evaluation should now shift to the CUDA backend due to dynamic thresholds
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, gpu_device)
    assert backend == BackendType.CUDA

    # Restore original configuration
    set_config(original_config)


def test_select_backend_metal_thresholds(metal_device):
    # Retrieve the default config to see the lower limits on metal
    original_config = get_config()

    # Normally this vector size (500,000) does NOT go to CUDA (Threshold: 2,000,000),
    # but since this is a Metal backend, its UMA memory favors offloading smaller arrays!
    op_props_medium = OperationProperties(element_count=500_000, shape=(500_000,))
    backend = select_backend(
        OperationType.COMPUTE_VECTOR, op_props_medium, metal_device
    )
    assert backend == BackendType.METAL

    # For CUDA, this should naturally fail and remain CPU
    cuda_device = DeviceInfo(cpu_cores=4, gpu_count=1, has_cuda=True)
    backend_cuda = select_backend(
        OperationType.COMPUTE_VECTOR, op_props_medium, cuda_device
    )
    assert backend_cuda == BackendType.CPU
