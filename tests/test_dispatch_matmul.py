from corepy.array import ndarray
from corepy.backend.types import BackendType


def test_cpu_matmul_dispatch():
    # Placeholder data (nested lists as 2D array)
    t1 = ndarray([[1, 2], [3, 4]])
    t2 = ndarray([[1, 0], [0, 1]])

    t3 = t1.matmul(t2)

    # t1 @ t2 (Identity) should match t1
    # Check shape
    assert t3.shape == (2, 2)
    # Check backend
    assert t3.backend == BackendType.CPU

    # Check values (real computation, not placeholder)
    # t3 should be [[1, 2], [3, 4]]
    # Since it's a ndarray wrapping numpy array or list, check data
    expected = [[1.0, 2.0], [3.0, 4.0]]
    assert t3.to_list() == expected
