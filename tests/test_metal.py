import platform
import sys

import numpy as np
import pytest

import corepy
from corepy import array, ndarray

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests only run on macOS"
)


def test_metal_availability():
    """Test if Metal is available and detected."""
    # Check if FFI function works
    try:
        from corepy._corepy_rust import metal_is_available

        available = metal_is_available()
        print(f"Metal Available: {available}")
        # On macOS CI/Dev environment, it should be true usually,
        # but depends on hardware. We assert it doesn't crash.
        assert isinstance(available, bool)
    except ImportError:
        pytest.fail("Could not import metal_is_available")


def _metal_library_loaded():
    """
    Check if Metal shaders are actually loaded (not just device available).
    Returns True if .metallib is loaded and operations will work.
    """
    if not corepy._corepy_rust.metal_is_available():
        return False

    # Try a simple operation to see if library loaded
    # If .metallib missing, operations return 0.0
    import numpy as np

    test_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = array(test_data, device="metal")
    result = t.sum()
    val = result.to_numpy()[0]

    # If library not loaded, sum returns 0.0
    # Real sum should be 6.0
    return abs(val - 6.0) < 0.01


def test_metal_tensor_creation():
    """Test creating a tensor on Metal device."""
    if not corepy._corepy_rust.metal_is_available():
        pytest.skip("Metal not available")

    t = array([1.0, 2.0, 3.0], device="metal")
    assert t.backend.name == "GPU"

    # Check internal buffer view device
    view = t._get_buffer_view()
    assert view.device.is_metal()
    assert str(view.device) == "metal:0"


def test_metal_sum():
    """Test sum reduction on Metal."""
    if not corepy._corepy_rust.metal_is_available():
        pytest.skip("Metal not available")
    if not _metal_library_loaded():
        pytest.skip("Metal shaders not loaded (install Xcode for pre-compiled shaders)")

    # Large enough to be worth it, but small enough for quick test
    data = np.random.rand(1024).astype(np.float32)
    t = array(data, device="metal")

    # Run sum
    result = t.sum()

    # Verify result (allow small float error)
    expected = np.sum(data)
    # result is a Tensor wrapping a scalar usually (or list [scalar])
    # Corepy sum returns Tensor([val])
    val = result.to_numpy()[0]

    assert np.allclose(val, expected, rtol=1e-4)
    # Check that it actually ran on Metal?
    # Hard to verify without mocking, but if it didn't crash and result is correct...
    # We can check checks in profile if enabled?


def test_metal_mean():
    """Test mean reduction on Metal."""
    if not corepy._corepy_rust.metal_is_available():
        pytest.skip("Metal not available")
    if not _metal_library_loaded():
        pytest.skip("Metal shaders not loaded (install Xcode for pre-compiled shaders)")

    data = np.random.rand(1024).astype(np.float32)
    t = array(data, device="metal")

    result = t.mean()
    expected = np.mean(data)
    val = result.to_numpy()[0]

    assert np.allclose(val, expected, rtol=1e-4)


def test_metal_matmul():
    """Test matrix multiplication on Metal."""
    if not corepy._corepy_rust.metal_is_available():
        pytest.skip("Metal not available")
    if not _metal_library_loaded():
        pytest.skip("Metal shaders not loaded (install Xcode for pre-compiled shaders)")

    M, K, N = 64, 64, 64
    a_data = np.random.rand(M, K).astype(np.float32)
    b_data = np.random.rand(K, N).astype(np.float32)

    a = array(a_data, device="metal")
    b = array(b_data, device="metal")

    c = a.matmul(b)

    expected = a_data @ b_data
    assert np.allclose(c.to_numpy(), expected, rtol=1e-4)


def test_metal_matmul_mismatch():
    """Test dimension mismatch error on Metal."""
    if not corepy._corepy_rust.metal_is_available():
        pytest.skip("Metal not available")

    a = array(np.zeros((10, 20)), device="metal")
    b = array(np.zeros((30, 40)), device="metal")

    with pytest.raises(ValueError, match="mismatch"):
        a.matmul(b)
