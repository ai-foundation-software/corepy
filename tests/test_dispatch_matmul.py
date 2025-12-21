from corepy.tensor import Tensor
from corepy.backend.types import BackendType

def test_cpu_matmul_dispatch():
    # Placeholder data (nested lists as 2D array)
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[1, 0], [0, 1]])
    
    t3 = t1.matmul(t2)
    
    assert t3.backend == BackendType.CPU
    # Our placeholder implementation just returns a string, which Tensor wraps in a list
    assert t3._backing_data == ["cpu_matmul_result_placeholder"]
