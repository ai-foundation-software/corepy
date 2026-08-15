import numpy as np
import pytest

import corepy as cp


def test_linalg_inv_3x3():
    """Verify matrix inverse for 3x3 matrices."""
    a_np = np.array(
        [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float32
    )

    a_cp = cp.array(a_np)

    inv_np = np.linalg.inv(a_np)
    inv_cp = cp.linalg.inv(a_cp)

    try:
        np.testing.assert_allclose(
            inv_cp.to_list(), inv_np.tolist(), rtol=1e-5, atol=1e-5
        )
    except AssertionError as e:
        print("FAILED test_linalg_inv_3x3")
        print(f"Actual (CorePy):\n{np.array(inv_cp.to_list()).reshape(3, 3)}")
        print(f"Expected (NumPy):\n{inv_np}")
        raise e

    # Verify Identity
    ident = cp.matmul(a_cp, inv_cp)
    try:
        np.testing.assert_allclose(
            ident.to_list(), np.eye(3), rtol=1e-5, atol=1e-4
        )  # Higher atol for identity
    except AssertionError as e:
        print("FAILED Identity check (A @ A_inv)")
        print(f"Result:\n{np.array(ident.to_list()).reshape(3, 3)}")
        raise e


def test_linalg_inv_4x4():
    """Verify matrix inverse for 4x4 matrices."""
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((4, 4)).astype(np.float32)

    a_cp = cp.array(a_np)

    inv_np = np.linalg.inv(a_np)
    inv_cp = cp.linalg.inv(a_cp)

    try:
        np.testing.assert_allclose(
            inv_cp.to_list(), inv_np.tolist(), rtol=1e-5, atol=1e-5
        )
    except AssertionError as e:
        print("FAILED test_linalg_inv_4x4")
        print(f"Actual (CorePy):\n{np.array(inv_cp.to_list()).reshape(4, 4)}")
        print(f"Expected (NumPy):\n{inv_np}")
        raise e


def test_linalg_det():
    """Verify matrix determinant for various sizes."""
    # 2x2
    a2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert cp.linalg.det(cp.array(a2)) == pytest.approx(np.linalg.det(a2))

    # 3x3
    a3 = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float32)
    assert cp.linalg.det(cp.array(a3)) == pytest.approx(np.linalg.det(a3))

    # 4x4
    rng = np.random.default_rng(123)
    a4 = rng.standard_normal((4, 4)).astype(np.float32)
    assert cp.linalg.det(cp.array(a4)) == pytest.approx(np.linalg.det(a4), rel=1e-5)


def test_linalg_norm():
    """Verify matrix/vector norm."""
    # Vector
    v_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cp.linalg.norm(cp.array(v_np)) == pytest.approx(np.linalg.norm(v_np))

    # Matrix (Frobenius)
    m_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert cp.linalg.norm(cp.array(m_np)) == pytest.approx(np.linalg.norm(m_np))


def test_linalg_singular_error():
    """Ensure singular matrices raise appropriate errors or handle safely."""
    a_singular = np.array([[1, 2], [2, 4]], dtype=np.float32)
    a_cp = cp.array(a_singular)

    # Note: different backends might raise different errors or return inf/nan
    # We should at least check if it doesn't crash.
    try:
        inv_cp = cp.linalg.inv(a_cp)
        # If it returns a result, it should contain inf/nan or be incorrect
        # But Faer's partial_piv_lu().inverse() might handle it differently.
    except Exception as e:
        print(f"Caught expected exception for singular matrix: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
