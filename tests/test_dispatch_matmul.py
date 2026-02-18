from corepy.array import Tensor
from corepy.backend.types import BackendType


def test_cpu_matmul_dispatch():
    # Placeholder data (nested lists as 2D array)
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[1, 0], [0, 1]])

    t3 = t1.matmul(t2)

    # t1 @ t2 (Identity) should match t1
    # Check shape
    assert t3.shape == (2, 2)
    # Check backend
    assert t3.backend == BackendType.CPU

    # Check values (real computation, not placeholder)
    # t3 should be [[1, 2], [3, 4]]
    # Since it's a Tensor wrapping numpy array or list, check data
    import numpy as np

    expected = [[1.0, 2.0], [3.0, 4.0]]
    if isinstance(t3._backing_data, np.ndarray):
        np.testing.assert_array_almost_equal(t3._backing_data, expected)
    else:
        assert t3._backing_data == expected
