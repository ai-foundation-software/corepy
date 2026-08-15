import numpy as np

import corepy as cp


def test_matmul_2d():
    print("Testing 2D Matmul...")

    # 1. Square matrices
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected = a_np @ b_np

    a_cp = cp.ndarray([[1, 2], [3, 4]])
    b_cp = cp.ndarray([[5, 6], [7, 8]])

    result_cp = a_cp.matmul(b_cp)
    result_np = result_cp.to_list()

    print(f"A:\n{a_np}")
    print(f"B:\n{b_np}")
    print(f"Result Corepy:\n{result_np}")
    print(f"Result NumPy:\n{expected}")

    np.testing.assert_allclose(result_np, expected, rtol=1e-5)
    print("✅ Square matrix test passed!")

    # 2. Rectangular matrices
    a_rect = np.random.rand(3, 5).astype(np.float32)
    b_rect = np.random.rand(5, 2).astype(np.float32)
    expected_rect = a_rect @ b_rect

    a_cp_rect = cp.ndarray(a_rect.tolist())
    b_cp_rect = cp.ndarray(b_rect.tolist())

    result_cp_rect = a_cp_rect.matmul(b_cp_rect)
    result_np_rect = result_cp_rect.to_list()

    np.testing.assert_allclose(result_np_rect, expected_rect, rtol=1e-5)
    print("✅ Rectangular matrix test passed!")


if __name__ == "__main__":
    test_matmul_2d()
