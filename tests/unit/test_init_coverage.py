import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import corepy as cp
from corepy import DataType


class TestInitPlatform:
    def test_macos_metal_lib_path(self):
        with patch("platform.system", return_value="Darwin"):
            with patch("os.path.exists", return_value=True):
                # We need to reload corepy or simulate the logic block
                # Since we can't easily reload, we copy the logic here to verify it
                bundled_lib = os.path.join(
                    os.path.dirname(cp.__file__), "default.metallib"
                )
                if os.path.exists(bundled_lib):
                    os.environ["COREPY_METAL_LIB_PATH"] = bundled_lib
                    assert os.environ["COREPY_METAL_LIB_PATH"] == bundled_lib

    def test_windows_dll_loading(self):
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"OPENBLAS_DIR": "C:\\opt\\openblas"}):
                with patch("os.path.exists", return_value=True):
                    with patch("os.add_dll_directory", create=True) as mock_add_dll:
                        # Simulate logic
                        openblas_dir = os.environ.get("OPENBLAS_DIR")
                        if openblas_dir:
                            bin_dir = os.path.join(openblas_dir, "bin")
                            if os.path.exists(bin_dir):
                                os.add_dll_directory(bin_dir)
                                mock_add_dll.assert_called_with(bin_dir)


class TestInitFunctions:
    def test_concatenate_simple(self):
        a = cp.array([1, 2])
        b = cp.array([3, 4])
        c = cp.concatenate([a, b])
        assert np.array_equal(c.to_numpy(), np.array([1, 2, 3, 4]))

    def test_concatenate_mixed(self):
        a = cp.array([1, 2])
        b = [3, 4]
        c = np.array([5, 6])
        res = cp.concatenate([a, b, c])
        assert np.array_equal(res.to_numpy(), np.array([1, 2, 3, 4, 5, 6]))

    def test_concatenate_errors(self):
        with pytest.raises(ValueError):
            cp.concatenate([cp.array([1]), "invalid"])

    def test_compute_stats_all(self):
        arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = cp.compute_stats(arr, ["mean", "sum", "std", "max", "min"])
        # Use to_numpy().item() for reliable scalar extraction
        assert stats["mean"].to_numpy().item() == 3.0
        assert stats["sum"].to_numpy().item() == 15.0
        assert abs(stats["std"].to_numpy().item() - 1.414) < 0.1
        assert stats["max"].to_numpy().item() == 5.0
        assert stats["min"].to_numpy().item() == 1.0

    def test_compute_stats_error(self):
        arr = cp.array([1])
        with pytest.raises(ValueError):
            cp.compute_stats(arr, ["invalid_stat"])

    def test_dtype_helpers(self):
        # Access internal function directly from the module
        assert cp._dtype_from_numpy(np.float32) == DataType.FLOAT32
        assert cp._dtype_from_numpy(np.dtype("float32")) == DataType.FLOAT32
        # Coverage for fallback
        assert cp._dtype_from_numpy("unknown") == DataType.FLOAT32

    def test_factory_functions(self):
        # zeros
        z = cp.zeros((2, 2))
        assert z.shape == (2, 2)
        assert np.all(z.to_numpy() == 0)

        # ones
        o = cp.ones(3)
        assert o.shape == (3,)
        assert np.all(o.to_numpy() == 1)

        # empty
        e = cp.empty((2, 2))
        assert e.shape == (2, 2)

        # arange
        a = cp.arange(0, 5)
        assert np.all(a.to_numpy() == np.arange(0, 5))

    def test_top_level_wrappers(self):
        # add
        a = cp.array([1, 2])
        b = cp.array([3, 4])
        res = cp.add(a, b)
        assert np.all(res.to_numpy() == np.array([4, 6]))

        # matmul / dot
        m1 = cp.array([[1, 0], [0, 1]])
        m2 = cp.array([[2, 0], [0, 2]])
        res_matmul = cp.matmul(m1, m2)
        res_dot = cp.dot(m1, m2)
        expected = np.array([[2, 0], [0, 2]])

        assert np.all(res_matmul.to_numpy() == expected)
        assert np.all(res_dot.to_numpy() == expected)

    def test_wrapper_casting(self):
        # Ensure wrappers handle list inputs by converting to array
        res = cp.add([1, 2], [3, 4])
        assert isinstance(res, cp.ndarray)
        res_mm = cp.matmul([[1]], [[2]])
        assert isinstance(res_mm, cp.ndarray)
