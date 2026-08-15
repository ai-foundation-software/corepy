import sys

import corepy as cp


def test_smoke():
    print("Running CorePy Smoke Tests...")

    # 1. Array Creation
    a = cp.array([1, 2, 3])
    print(f"Array a: {a}, shape: {a.shape}, dtype: {a.dtype}")
    assert a.shape == (3,)

    b = cp.zeros((3, 3))
    print(f"Array b (zeros):\n{b}")
    assert b.shape == (3, 3)

    # 2. Basic Arithmetic
    c = a + 2
    print(f"a + 2 = {c}")
    assert c[0] == 3

    # 3. Broadcasting
    d = b + a
    print(f"b + a (broadcasting):\n{d}")
    assert d.shape == (3, 3)

    # 4. Linear Algebra
    e = cp.eye(3)
    f = cp.matmul(e, b)
    print(f"matmul(eye, zeros):\n{f}")
    assert f.shape == (3, 3)

    # 5. Random
    r = cp.random.rand(2, 2)
    print(f"Random array:\n{r}")
    assert r.shape == (2, 2)

    print("SMOKE TESTS PASSED!")


if __name__ == "__main__":
    try:
        test_smoke()
    except Exception as e:
        print(f"SMOKE TESTS FAILED: {e}")
        sys.exit(1)
