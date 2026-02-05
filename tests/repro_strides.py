import unittest

import numpy as np

import corepy as cp


class TestSafety(unittest.TestCase):
    def test_non_contiguous_input(self):
        """
        CRITICAL SAFETY TEST:
        Passing a non-contiguous NumPy array (slice) should NOT result in
        reading the wrong memory addresses via the zero-copy path.
        """
        print("\n=== Testing Non-Contiguous Safety ===")

        # 1. Create a contiguous array: [0, 1, 2, 3, 4, 5]
        full_arr = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)

        # 2. PROMPT: Create a non-contiguous slice: [0, 2, 4]
        # Strides will be 8 bytes (skip one float) instead of 4
        sliced_arr = full_arr[::2]

        print(f"Input: {sliced_arr.tolist()}")
        print(f"Input Strides: {sliced_arr.strides}")

        # 3. Create Tensor from slice
        # CURRENT BEHAVIOR:
        # - Extracts ptr to '0'
        # - Passes count=3
        # - C++ reads contiguous bytes: 0, 1, 2
        # - Result is sum(0, 1, 2) = 3.0
        #
        # CORRECT BEHAVIOR:
        # - Result should be sum(0, 2, 4) = 6.0
        t = cp.Tensor(sliced_arr, dtype=cp.DataType.FLOAT32)

        result_tensor = t.sum()
        result_val = result_tensor._backing_data[0]

        print(f"Computed Sum: {result_val}")
        print("Expected Sum: 6.0")

        if abs(result_val - 6.0) > 1e-5:
            print("❌ SAFETY FAILURE: Silent data corruption detected!")
            print("   The system read contiguous memory instead of respecting strides.")
        else:
            print("✅ SAFETY PASS: Correctly handled non-contiguous memory.")

        self.assertAlmostEqual(
            result_val,
            6.0,
            places=5,
            msg=f"Data corruption! Expected 6.0, got {result_val}",
        )


if __name__ == "__main__":
    unittest.main()
