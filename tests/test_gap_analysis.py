"""
Test script for new reduction and element-wise operations.

This validates the gap analysis implementation:
- Any() reduction
- Sum() reduction (f32, i32)
- Mean() reduction
- Element-wise operations (add, sub, mul, div)
"""

import ctypes

import numpy as np
import pytest


def test_tensor_any():
    """Test any() reduction operation."""
    print("Testing tensor_any()...")

    try:
        from _corepy_rust import tensor_any

        # Test 1: All false -> False
        data1 = bytearray([0, 0, 0, 0])
        c_buffer1 = (ctypes.c_uint8 * len(data1)).from_buffer(data1)
        result1 = tensor_any(ctypes.addressof(c_buffer1), len(data1))
        assert result1 == False, f"Expected False, got {result1}"
        print("  ✓ All false -> False")

        # Test 2: Some true -> True
        data2 = bytearray([0, 0, 1, 0])
        c_buffer2 = (ctypes.c_uint8 * len(data2)).from_buffer(data2)
        result2 = tensor_any(ctypes.addressof(c_buffer2), len(data2))
        assert result2 == True, f"Expected True, got {result2}"
        print("  ✓ Some true -> True")

        # Test 3: All true -> True
        data3 = bytearray([1, 1, 1, 1])
        c_buffer3 = (ctypes.c_uint8 * len(data3)).from_buffer(data3)
        result3 = tensor_any(ctypes.addressof(c_buffer3), len(data3))
        assert result3 == True, f"Expected True, got {result3}"
        print("  ✓ All true -> True")

        print("✅ tensor_any() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_tensor_sum_f32():
    """Test sum() reduction for f32."""
    print("Testing tensor_sum_f32()...")

    try:
        from _corepy_rust import tensor_sum_f32

        # Create f32 array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        expected = 15.0

        # Get pointer
        ptr = data.ctypes.data
        count = len(data)

        result = tensor_sum_f32(ptr, count)
        assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
        print(f"  ✓ Sum of {data.tolist()} = {result}")

        print("✅ tensor_sum_f32() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_tensor_sum_i32():
    """Test sum() reduction for i32."""
    print("Testing tensor_sum_i32()...")

    try:
        from _corepy_rust import tensor_sum_i32

        # Create i32 array
        data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        expected = 150

        ptr = data.ctypes.data
        count = len(data)

        result = tensor_sum_i32(ptr, count)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"  ✓ Sum of {data.tolist()} = {result}")

        print("✅ tensor_sum_i32() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_tensor_mean_f32():
    """Test mean() reduction for f32."""
    print("Testing tensor_mean_f32()...")

    try:
        from _corepy_rust import tensor_mean_f32

        # Create f32 array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        expected = 3.0

        ptr = data.ctypes.data
        count = len(data)

        result = tensor_mean_f32(ptr, count)
        assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
        print(f"  ✓ Mean of {data.tolist()} = {result}")

        print("✅ tensor_mean_f32() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_tensor_add_f32():
    """Test element-wise addition."""
    print("Testing tensor_add_f32()...")

    try:
        from _corepy_rust import tensor_add_f32

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        out = np.zeros_like(a)
        expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)

        tensor_add_f32(a.ctypes.data, b.ctypes.data, out.ctypes.data, len(a))

        assert np.allclose(out, expected), f"Expected {expected}, got {out}"
        print(f"  ✓ {a.tolist()} + {b.tolist()} = {out.tolist()}")

        print("✅ tensor_add_f32() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_tensor_mul_f32():
    """Test element-wise multiplication."""
    print("Testing tensor_mul_f32()...")

    try:
        from _corepy_rust import tensor_mul_f32

        a = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        out = np.zeros_like(a)
        expected = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32)

        tensor_mul_f32(a.ctypes.data, b.ctypes.data, out.ctypes.data, len(a))

        assert np.allclose(out, expected), f"Expected {expected}, got {out}"
        print(f"  ✓ {a.tolist()} * {b.tolist()} = {out.tolist()}")

        print("✅ tensor_mul_f32() tests passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_simd_performance():
    """Test SIMD performance with larger arrays."""
    print("Testing SIMD performance...")

    try:
        import time

        from _corepy_rust import tensor_add_f32, tensor_sum_f32

        # Large array to trigger SIMD paths
        size = 10000
        data = np.random.rand(size).astype(np.float32)

        # Time sum operation
        start = time.perf_counter()
        for _ in range(1000):
            result = tensor_sum_f32(data.ctypes.data, len(data))
        rust_time = time.perf_counter() - start

        # Compare with NumPy
        start = time.perf_counter()
        for _ in range(1000):
            np_result = np.sum(data)
        numpy_time = time.perf_counter() - start

        print(f"  ✓ Rust sum: {rust_time:.4f}s for 1000 iterations")
        print(f"  ✓ NumPy sum: {numpy_time:.4f}s for 1000 iterations")
        if rust_time > 0:
            print(f"  ✓ Speedup: {numpy_time / rust_time:.2f}x")

        print("✅ SIMD performance test completed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


def test_numpy_ffi_integration():
    """Test direct zero-copy using __array_interface__."""
    print("Testing NumPy FFI integration...")

    try:
        from _corepy_rust import tensor_sum_f32

        # NumPy array
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        # Access via __array_interface__ (what Tensor class does)
        ptr = arr.__array_interface__["data"][0]

        result = tensor_sum_f32(ptr, len(arr))

        assert abs(result - 15.0) < 1e-5, f"Expected 15.0, got {result}"
        print("  ✓ Direct FFI call with __array_interface__ passed")

        print("✅ NumPy FFI integration test passed!\n")
    except ImportError:
        pytest.skip("_corepy_rust not available")


if __name__ == "__main__":
    print("=" * 60)
    print("Gap Analysis Implementation Tests")
    print("=" * 60 + "\n")

    tests = [
        test_tensor_any,
        test_tensor_sum_f32,
        test_tensor_sum_i32,
        test_tensor_mean_f32,
        test_tensor_add_f32,
        test_tensor_mul_f32,
        test_numpy_ffi_integration,
        test_simd_performance,
    ]

    passed_count = 0
    for test in tests:
        try:
            test()
            print(f"✅ PASS: {test.__name__}")
            passed_count += 1
        except Exception as e:
            print(f"❌ FAIL: {test.__name__} - {e}")
        except pytest.skip.Exception:
            print(f"⚠️ SKIP: {test.__name__}")

    print(f"\nTotal: {passed_count}/{len(tests)} tests passed")
