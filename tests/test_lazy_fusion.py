"""
Tests for UFUNC CORE-50: Lazy execution graph and operation fusion.
"""

import pytest

import corepy as cp
from corepy.lazy.array import LazyArray


class TestLazyBasic:
    def test_lazy_add(self):
        a = LazyArray(cp.array([1, 2, 3]))
        b = LazyArray(cp.array([4, 5, 6]))
        c = a + b
        result = c.compute()
        assert result.to_list() == [5.0, 7.0, 9.0]

    def test_lazy_chain(self):
        a = LazyArray(cp.array([1, 2, 3]))
        b = LazyArray(cp.array([4, 5, 6]))
        c = (a + b) * 2
        result = c.compute()
        assert result.to_list() == [10.0, 14.0, 18.0]


class TestLazyCORE50:
    def test_lazy_pow(self):
        a = LazyArray(cp.array([2, 3, 4]))
        c = a**2
        result = c.compute()
        r = result.to_list()
        assert abs(r[0] - 4.0) < 1e-4
        assert abs(r[1] - 9.0) < 1e-4
        assert abs(r[2] - 16.0) < 1e-4

    def test_lazy_mod(self):
        a = LazyArray(cp.array([10, 11, 12]))
        c = a % 3
        result = c.compute()
        assert result.to_list() == [1.0, 2.0, 0.0]

    def test_lazy_floordiv(self):
        a = LazyArray(cp.array([7, 8, 9]))
        c = a // 2
        result = c.compute()
        assert result.to_list() == [3.0, 4.0, 4.0]


class TestLazyReduction:
    def test_lazy_sum(self):
        a = LazyArray(cp.array([1, 2, 3]))
        result = a.sum().compute()
        # sum() may return ndarray or scalar depending on backend
        val = result.to_list()[0] if hasattr(result, "to_list") else float(result)
        assert abs(val - 6.0) < 1e-5

    def test_lazy_mean(self):
        a = LazyArray(cp.array([2, 4, 6]))
        result = a.mean().compute()
        val = result.to_list()[0] if hasattr(result, "to_list") else float(result)
        assert abs(val - 4.0) < 1e-5


class TestLazyFusion:
    def test_fused_chain(self):
        """Multiple elementwise ops should be collected in topological order."""
        a = LazyArray(cp.array([1, 2, 3]))
        b = LazyArray(cp.array([2, 2, 2]))
        # (a + b) * (a - b) = (3,4,5) * (-1,0,1) = (-3,0,5)
        c = (a + b) * (a - b)
        result = c.compute()
        assert result.to_list() == [-3.0, 0.0, 5.0]

    def test_deep_chain(self):
        """6-deep chain: ((a + b) * 2 - 1) + 10 / 2."""
        a = LazyArray(cp.array([1, 2]))
        b = LazyArray(cp.array([3, 4]))
        c = ((a + b) * 2 - 1) + 10
        result = c.compute()
        r = result.to_list()
        # (1+3)*2 -1 +10 = 17, (2+4)*2 -1 +10 = 21
        assert abs(r[0] - 17.0) < 1e-4
        assert abs(r[1] - 21.0) < 1e-4
