import pytest
from unittest.mock import MagicMock, patch
from corepy.backend.selector import select_backend
from corepy.backend.types import BackendType, OperationType, OperationProperties
from corepy.backend.device import DeviceInfo

@pytest.fixture
def cpu_only_device():
    return DeviceInfo(cpu_cores=4, gpu_count=0)

@pytest.fixture
def gpu_device():
    return DeviceInfo(cpu_cores=4, gpu_count=1, gpu_names=["TestGPU"], gpu_memory_bytes=[8*1024**3])

def test_select_backend_cpu_default(cpu_only_device):
    op_props = OperationProperties(element_count=1000, shape=(1000,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, cpu_only_device)
    assert backend == BackendType.CPU

def test_select_backend_gpu_vector_threshold(gpu_device):
    # Below threshold
    op_props_small = OperationProperties(element_count=100_000, shape=(100_000,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_small, gpu_device)
    assert backend == BackendType.CPU

    # Above threshold
    op_props_large = OperationProperties(element_count=100_001, shape=(100_001,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_large, gpu_device)
    assert backend == BackendType.GPU

def test_select_backend_gpu_matrix_threshold(gpu_device):
    # Below threshold
    op_props_small = OperationProperties(element_count=511*511, shape=(511, 511))
    backend = select_backend(OperationType.COMPUTE_MATRIX, op_props_small, gpu_device)
    assert backend == BackendType.CPU
    
    # Above threshold
    op_props_large = OperationProperties(element_count=512*512, shape=(512, 512))
    backend = select_backend(OperationType.COMPUTE_MATRIX, op_props_large, gpu_device)
    assert backend == BackendType.GPU

def test_select_backend_control_always_cpu(gpu_device):
    op_props = OperationProperties(element_count=10**7, shape=(10**7,)) # Huge
    backend = select_backend(OperationType.CONTROL, op_props, gpu_device)
    assert backend == BackendType.CPU

def test_select_backend_explicit_request(cpu_only_device):
    op_props = OperationProperties(element_count=100, shape=(100,))
    # Arguably if we request GPU on CPU-only device it might fail later, 
    # but selector should respect the request or warn. 
    # In our current impl, selector returns requested_backend.
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, cpu_only_device, requested_backend=BackendType.GPU)
    assert backend == BackendType.GPU

@patch("os.getenv")
def test_select_backend_env_var_override(mock_getenv, gpu_device):
    mock_getenv.return_value = "cpu" # Force CPU despite GPU being better
    op_props = OperationProperties(element_count=10**7, shape=(10**7,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props, gpu_device)
    assert backend == BackendType.CPU

    mock_getenv.return_value = "gpu"
    # Even small op forced to GPU
    op_props_small = OperationProperties(element_count=100, shape=(100,))
    backend = select_backend(OperationType.COMPUTE_VECTOR, op_props_small, gpu_device)
    assert backend == BackendType.GPU
