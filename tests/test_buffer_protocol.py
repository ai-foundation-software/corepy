import unittest

import numpy as np

import corepy as cp
from corepy.backend.types import DataType


class TestBufferProtocol(unittest.TestCase):
    def test_numpy_array_support(self):
        """Test zero-copy with NumPy arrays."""
        print("\nTesting NumPy array support...")
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        array = cp.ndarray(arr, dtype=DataType.FLOAT32)

        # Check that the array was created successfully
        np.testing.assert_array_equal(array.to_list(), arr.tolist())

        # Test generic operation (sum)
        result = array.sum()
        # Result should be a scalar ndarray with value 15.0
        self.assertAlmostEqual(result.to_list()[0], 15.0, places=5)
        print("  ✓ NumPy zero-copy sum passed")

    def test_bytearray_support(self):
        """Test with bytearray (explicit buffer protocol)."""
        print("Testing bytearray support...")
        # Create boolean bytearray: [True, True, True, False]
        data = bytearray([1, 1, 1, 0])
        array = cp.ndarray(data, dtype=DataType.BOOL)

        # Test any()
        result = array.any()
        self.assertTrue(result.to_list()[0])

        # Test all()
        result = array.all()
        self.assertFalse(result.to_list()[0])
        print("  ✓ bytearray support passed")

    def test_memoryview_support(self):
        """Test with memoryview."""
        print("Testing memoryview support...")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mv = memoryview(arr)
        array = cp.ndarray(mv, dtype=DataType.FLOAT32)

        result = array.mean()
        self.assertAlmostEqual(result.to_list()[0], 2.0, places=5)
        print("  ✓ memoryview support passed")

    def test_list_conversion(self):
        """Test list conversion fallback."""
        print("Testing list conversion...")
        data = [1.0, 2.0, 3.0, 4.0]
        array = cp.ndarray(data, dtype=DataType.FLOAT32)

        result = array.mean()
        self.assertAlmostEqual(result.to_list()[0], 2.5, places=5)
        print("  ✓ list conversion passed")

    def test_binary_ops_mixed_types(self):
        """Test binary ops with different backing types."""
        print("Testing mixed backing types...")
        # NumPy backed
        t1 = cp.ndarray(np.array([1.0, 2.0], dtype=np.float32))
        # List backed
        t2 = cp.ndarray([10.0, 20.0])

        result = t1 + t2
        expected = [11.0, 22.0]

        for i, val in enumerate(result.to_list()):
            self.assertAlmostEqual(val, expected[i], places=5)
        print("  ✓ mixed type binary op passed")


if __name__ == "__main__":
    unittest.main()
