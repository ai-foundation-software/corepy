import math
import warnings

import pytest

import corepy as cp


class TestRemediationPlan:
    """Comprehensive test suite for the CorePy Remediation Plan."""

    # -------------------------------------------------------------------------
    # 1. PyO3 Type Unwrapping & Marshalling Defects
    # -------------------------------------------------------------------------
    def test_stack_pure_python_arrays(self):
        a1 = cp.array([1.0, 2.0])
        a2 = cp.array([3.0, 4.0])
        res = cp.stack([a1, a2])
        assert res.shape == (2, 2)
        assert res.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_series_tail_negative_bounds(self):
        s = cp.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        tail_neg = s.tail(-2)
        assert len(tail_neg) == 3
        tail_pos = s.tail(2)
        assert len(tail_pos) == 2

    # -------------------------------------------------------------------------
    # 2. Missing ndarray Object Methods (NumPy Parity)
    # -------------------------------------------------------------------------
    def test_ndarray_instance_methods(self):
        arr = cp.array([[1.0, 2.0], [3.0, 4.0]])
        # squeeze
        arr3d = cp.array([[[1.0, 2.0]]])
        assert arr3d.squeeze().shape == (2,)

        # flatten & ravel
        assert arr.flatten().shape == (4,)
        assert arr.ravel().shape == (4,)
        assert arr.flatten().to_list() == [1.0, 2.0, 3.0, 4.0]

        # var
        assert isinstance(arr.var(), (float, int, cp.ndarray))

        # dot
        dot_res = arr.dot(arr)
        assert dot_res.shape == (2, 2)

        # astype
        arr_casted = arr.astype("float64")
        assert arr_casted.shape == (2, 2)

    # -------------------------------------------------------------------------
    # 3. Series Methods & df.describe() Repair
    # -------------------------------------------------------------------------
    def test_series_methods_and_describe(self):
        s = cp.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="val")
        assert s.mean() == 3.0
        assert s.sum() == 15.0
        assert s.min() == 1.0
        assert s.max() == 5.0
        assert s.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

        # apply & map
        s_applied = s.apply(lambda x: x * 2.0)
        assert s_applied.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]

        s_mapped = s.map({1.0: 10.0, 2.0: 20.0})
        assert s_mapped.tolist() == [10.0, 20.0, 3.0, 4.0, 5.0]

        # DataFrame describe
        df = cp.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        desc = df.describe()
        assert desc is not None
        assert "a" in desc.columns

    # -------------------------------------------------------------------------
    # 4. 2D Scalar Comparisons & Boolean Mask Indexing
    # -------------------------------------------------------------------------
    def test_2d_scalar_comparison_and_boolean_indexing(self):
        arr2d = cp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = arr2d > 2.0
        assert mask.shape == (2, 2)
        assert mask.to_list() == [[0.0, 0.0], [1.0, 1.0]]

        indexed = arr2d[mask]
        assert indexed.to_list() == [3.0, 4.0]

    def test_is_even_odd_mask_indexing(self):
        arr = cp.array([10.0, 15.0, 22.0, 33.0, 40.0, 51.0])
        even_mask = cp.is_even(arr)
        odd_mask = cp.is_odd(arr)

        assert arr[even_mask].to_list() == [10.0, 22.0, 40.0]
        assert arr[odd_mask].to_list() == [15.0, 33.0, 51.0]

    # -------------------------------------------------------------------------
    # 5. NumPy-Compatible Math Domain Error Handling
    # -------------------------------------------------------------------------
    def test_math_domain_errors(self):
        arr = cp.array([2.0])
        with pytest.warns(RuntimeWarning, match="invalid value encountered"):
            res = cp.arctanh(arr)
            assert math.isnan(res.to_list()[0])

        arr_neg = cp.array([-1.0])
        with pytest.warns(RuntimeWarning, match="invalid value encountered"):
            res_log = cp.log(arr_neg)
            assert math.isnan(res_log.to_list()[0])

    # -------------------------------------------------------------------------
    # 6. Hardware / GPU Memory Abstraction
    # -------------------------------------------------------------------------
    def test_gpu_buffer_abstraction(self):
        freed = []
        gpu_buf = cp.GPUBuffer(
            ptr=0x12345,
            size_bytes=64,
            device="metal",
            dealloc_fn=lambda p: freed.append(p),
        )
        assert gpu_buf.ptr == 0x12345
        gpu_buf.free()
        assert freed == [0x12345]
        assert gpu_buf.ptr == 0

    # -------------------------------------------------------------------------
    # 7. Core Preservation Rule
    # -------------------------------------------------------------------------
    def test_linalg_preservation(self):
        mat = cp.array([[2.0, 0.0], [0.0, 2.0]])
        det_val = cp.linalg.det(mat)
        assert det_val == 4.0
