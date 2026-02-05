import unittest

import numpy as np

import corepy as cp
from corepy.backend.types import DataType


class TestBufferProtocol(unittest.TestCase):
    def test_numpy_array_support(self):
        """Test zero-copy with NumPy arrays."""
        print("\nTesting NumPy array support...")
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        tensor = cp.Tensor(arr, dtype=DataType.FLOAT32)

        # Accessing private backing data to verify it wasn't copied
        self.assertIs(tensor._backing_data, arr)

        # Test generic operation (sum)
        result = tensor.sum()
        # Result should be a scalar Tensor with value 15.0
        self.assertAlmostEqual(result._backing_data[0], 15.0, places=5)
        print("  ✓ NumPy zero-copy sum passed")

    def test_bytearray_support(self):
        """Test with bytearray (explicit buffer protocol)."""
        print("Testing bytearray support...")
        # Create boolean bytearray: [True, True, True, False]
        data = bytearray([1, 1, 1, 0])
        tensor = cp.Tensor(data, dtype=DataType.BOOL)

        # Test any()
        result = tensor.any()
        self.assertTrue(result._backing_data[0])

        # Test all()
        result = tensor.all()
        self.assertFalse(result._backing_data[0])
        print("  ✓ bytearray support passed")

    def test_memoryview_support(self):
        """Test with memoryview."""
        print("Testing memoryview support...")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mv = memoryview(arr)
        tensor = cp.Tensor(mv, dtype=DataType.FLOAT32)

        result = tensor.mean()
        self.assertAlmostEqual(result._backing_data[0], 2.0, places=5)
        print("  ✓ memoryview support passed")

    def test_list_conversion(self):
        """Test list conversion fallback."""
        print("Testing list conversion...")
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = cp.Tensor(data, dtype=DataType.FLOAT32)

        result = tensor.mean()
        self.assertAlmostEqual(result._backing_data[0], 2.5, places=5)
        print("  ✓ list conversion passed")

    def test_binary_ops_mixed_types(self):
        """Test binary ops with different backing types."""
        print("Testing mixed backing types...")
        # NumPy backed
        t1 = cp.Tensor(np.array([1.0, 2.0], dtype=np.float32))
        # List backed
        t2 = cp.Tensor([10.0, 20.0])

        result = t1 + t2
        expected = [11.0, 22.0]

        for i, val in enumerate(result._backing_data):
            self.assertAlmostEqual(val, expected[i], places=5)
        print("  ✓ mixed type binary op passed")


if __name__ == "__main__":
    unittest.main()
