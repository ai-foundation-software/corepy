"""
Tests for UFUNC CORE-50: Reduction, Creation, Special, Stacking.
"""

import math

import pytest

import corepy as cp


class TestReduction:
    def test_prod(self):
        assert cp.prod([2, 3, 4]) == 24.0

    def test_std(self):
        r = cp.std([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(r - 2.0) < 0.1

    def test_var(self):
        r = cp.var([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(r - 4.0) < 0.2

    def test_cumsum(self):
        r = cp.cumsum([1, 2, 3, 4])
        assert r.to_list() == [1.0, 3.0, 6.0, 10.0]

    def test_cumprod(self):
        r = cp.cumprod([1, 2, 3, 4])
        assert r.to_list() == [1.0, 2.0, 6.0, 24.0]

    def test_prod_method(self):
        a = cp.array([2, 3, 4])
        assert a.prod() == 24.0

    def test_std_method(self):
        a = cp.array([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(a.std() - 2.0) < 0.1

    def test_cumsum_method(self):
        a = cp.array([1, 2, 3])
        assert a.cumsum().to_list() == [1.0, 3.0, 6.0]


class TestNanReduction:
    def test_nansum(self):
        r = cp.nansum([1, float("nan"), 3])
        assert abs(r - 4.0) < 1e-5

    def test_nanmean(self):
        r = cp.nanmean([1, float("nan"), 3])
        assert abs(r - 2.0) < 1e-5

    def test_nanmax(self):
        assert cp.nanmax([1, float("nan"), 5]) == 5.0

    def test_nanmin(self):
        assert cp.nanmin([10, float("nan"), 3]) == 3.0


class TestSpecial:
    def test_square(self):
        r = cp.square([2, 3, 4])
        assert r.to_list() == [4.0, 9.0, 16.0]

    def test_reciprocal(self):
        r = cp.reciprocal([2, 4, 5])
        assert abs(r.to_list()[0] - 0.5) < 1e-5
        assert abs(r.to_list()[1] - 0.25) < 1e-5
        assert abs(r.to_list()[2] - 0.2) < 1e-5

    def test_cbrt(self):
        r = cp.cbrt([8, 27])
        assert abs(r.to_list()[0] - 2.0) < 1e-4
        assert abs(r.to_list()[1] - 3.0) < 1e-4

    def test_positive(self):
        r = cp.positive([-1, 2, -3])
        assert r.to_list() == [-1.0, 2.0, -3.0]

    def test_negative(self):
        r = cp.negative([1, -2, 3])
        assert r.to_list() == [-1.0, 2.0, -3.0]

    def test_absolute(self):
        r = cp.absolute([-1, -2, 3])
        assert r.to_list() == [1.0, 2.0, 3.0]

    def test_square_method(self):
        a = cp.array([3, 4])
        assert a.square().to_list() == [9.0, 16.0]


class TestCreation:
    def test_zeros_like(self):
        a = cp.array([1, 2, 3])
        r = cp.zeros_like(a)
        assert r.to_list() == [0.0, 0.0, 0.0]

    def test_ones_like(self):
        a = cp.array([1, 2, 3])
        r = cp.ones_like(a)
        assert r.to_list() == [1.0, 1.0, 1.0]

    def test_eye(self):
        """Test creating an eye matrix."""
        arr = cp.eye(2, 3)
        assert arr.shape == (2, 3)
        expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert arr.to_list() == expected

    def test_identity(self):
        """Test creating an identity matrix."""
        arr = cp.identity(2)
        assert arr.shape == (2, 2)
        assert arr.to_list() == [[1.0, 0.0], [0.0, 1.0]]

    def test_full_like(self):
        a = cp.array([1, 2, 3])
        r = cp.full_like(a, 7.0)
        assert r.to_list() == [7.0, 7.0, 7.0]


class TestStacking:
    def test_hstack(self):
        r = cp.hstack([[1, 2], [3, 4]])
        assert r.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_split(self):
        parts = cp.split([1, 2, 3, 4], 2)
        assert parts[0].to_list() == [1.0, 2.0]
        assert parts[1].to_list() == [3.0, 4.0]

    def test_tile(self):
        r = cp.tile([1, 2], 3)
        assert r.to_list() == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    def test_repeat(self):
        r = cp.repeat([1, 2], 3)
        assert r.to_list() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
