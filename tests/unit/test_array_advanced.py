import numpy as np
import pytest

import corepy as cp
from corepy import DataType


class TestArrayAdvanced:
    def test_init_nested_list_shapes(self):
        # 3D list
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        arr = cp.array(data)
        assert arr.shape == (2, 2, 2)
        assert arr.size == 8
        assert np.array(arr.to_list()).shape == (2, 2, 2)

    def test_init_bytearray(self):
        data = bytearray([1, 2, 3, 4])
        arr = cp.array(data, dtype=DataType.INT32)
        # Note: current impl treats bytearray as 1D usually
        assert arr.shape == (4,)

    def test_init_memoryview(self):
        data = memoryview(b"1234")
        arr = cp.array(data)
        assert arr.shape == (4,)

    def test_getitem_list_fallback(self):
        # Force backing data to be list
        arr = cp.array([1, 2, 3])
        # Manually ensure it hasn't converted to numpy yet if possible,
        # or just test slicing works
        # In current impl, it stays as list until compute?
        # Lines 79: self.to_list() = data (list)

        assert arr[0] == 1.0
        assert isinstance(arr[0:2], cp.ndarray)
        assert arr[0:2].shape == (2,)

    def test_getitem_numpy_fallback(self):
        arr = cp.array(np.array([1, 2, 3]))
        assert arr[0] == 1
        assert arr[0:2].shape == (2,)

    def test_binary_ops_scalar_broadcasting(self):
        arr = cp.array([1, 2])
        res = arr + 10
        assert res.to_list() == [11.0, 12.0]

        # Reverse? __radd__ not implemented in snippet?
        # If not implemented, Python tries to call __add__ on scalar, which fails.
        # Check source: __radd__? Snippet didn't show it.
        pass

    def test_binary_ops_cpu_optimization(self):
        # Should hit optimize path if Rust available
        arr1 = cp.array([1.0, 2.0], dtype=DataType.FLOAT32, device="cpu")
        arr2 = cp.array([3.0, 4.0], dtype=DataType.FLOAT32, device="cpu")

        res = arr1 + arr2
        assert res.to_list() == [4.0, 6.0]

    def test_comparison_ops(self):
        arr1 = cp.array([1.0])
        arr2 = cp.array([2.0])
        scalar = 2.0

        assert arr1 < arr2
        assert arr1 < scalar
        assert arr2 <= scalar
        assert arr2 > arr1
        assert arr2 >= arr1

    def test_buffer_view_extraction(self):
        # Test _get_buffer_view for different backing types
        # This is internal but critical for coverage

        # List
        arr = cp.array([1, 2], dtype=DataType.FLOAT32)
        view = arr._get_buffer_view()
        assert view.element_count == 2

        # NumPy
        arr_np = cp.array(np.array([1, 2], dtype=np.float32))
        view_np = arr_np._get_buffer_view()
        assert view_np.element_count == 2
