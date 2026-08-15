"""
Test suite for UFUNC CORE-12 elementwise operations engine.

Tests cover:
- Arithmetic operations (add, subtract, multiply, divide, power, mod, floor_divide)
- Multi-input chaining (add(a, b, c, d))
- Broadcasting (array + scalar, mismatched shapes)
- Comparison operations (equal, not_equal, greater, less, ge, le)
- Logical operations (and, or, not, xor)
- Operator overloads (__pow__, __mod__, __floordiv__, __neg__, __abs__, right-hand)
- is_even / is_odd detection
- linspace factory
- Edge cases (NaN, Inf, empty, large arrays)
"""

import math

import pytest

import corepy as cp
from corepy.array import ndarray

# ============================================================================
# Arithmetic Operations — Functional API
# ============================================================================


class TestArithmeticFunctional:
    def test_add_basic(self):
        result = cp.add([1, 2, 3], [4, 5, 6])
        assert result.to_list() == [5.0, 7.0, 9.0]

    def test_subtract_basic(self):
        result = cp.subtract([10, 20, 30], [1, 2, 3])
        assert result.to_list() == [9.0, 18.0, 27.0]

    def test_multiply_basic(self):
        result = cp.multiply([2, 3, 4], [5, 6, 7])
        assert result.to_list() == [10.0, 18.0, 28.0]

    def test_divide_basic(self):
        result = cp.divide([10, 20, 30], [2, 4, 5])
        assert result.to_list() == [5.0, 5.0, 6.0]

    def test_power_basic(self):
        result = cp.power([2, 3, 4], [2, 2, 2])
        r = result.to_list()
        assert abs(r[0] - 4.0) < 1e-5
        assert abs(r[1] - 9.0) < 1e-5
        assert abs(r[2] - 16.0) < 1e-5

    def test_mod_basic(self):
        result = cp.mod([10, 11, 12], [3, 3, 3])
        assert result.to_list() == [1.0, 2.0, 0.0]

    def test_floor_divide_basic(self):
        result = cp.floor_divide([7, 8, 9], [2, 3, 4])
        assert result.to_list() == [3.0, 2.0, 2.0]


# ============================================================================
# Multi-Input Chaining
# ============================================================================


class TestMultiInput:
    def test_add_multi(self):
        """add(a, b, c, d) should compute a + b + c + d."""
        result = cp.add([1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4])
        assert result.to_list() == [10.0, 10.0, 10.0]

    def test_multiply_multi(self):
        result = cp.multiply([2, 2], [3, 3], [4, 4])
        assert result.to_list() == [24.0, 24.0]


# ============================================================================
# Broadcasting
# ============================================================================


class TestBroadcasting:
    def test_add_scalar(self):
        """[1, 2, 3] + 5 → [6, 7, 8]."""
        a = cp.array([1, 2, 3])
        result = a + 5
        assert result.to_list() == [6.0, 7.0, 8.0]

    def test_add_scalar_functional(self):
        result = cp.add([1, 2, 3], [5])
        assert result.to_list() == [6.0, 7.0, 8.0]

    def test_multiply_scalar(self):
        a = cp.array([2, 3, 4])
        result = a * 10
        assert result.to_list() == [20.0, 30.0, 40.0]


# ============================================================================
# Comparison Operations
# ============================================================================


class TestComparison:
    def test_equal(self):
        result = cp.equal([1, 2, 3], [1, 0, 3])
        assert result.to_list() == [1.0, 0.0, 1.0]

    def test_not_equal(self):
        result = cp.not_equal([1, 2, 3], [1, 0, 3])
        assert result.to_list() == [0.0, 1.0, 0.0]

    def test_greater(self):
        result = cp.greater([5, 2, 3], [1, 3, 3])
        assert result.to_list() == [1.0, 0.0, 0.0]

    def test_less(self):
        result = cp.less([1, 5, 3], [2, 3, 3])
        assert result.to_list() == [1.0, 0.0, 0.0]

    def test_greater_equal(self):
        result = cp.greater_equal([3, 2, 3], [3, 3, 1])
        assert result.to_list() == [1.0, 0.0, 1.0]

    def test_less_equal(self):
        result = cp.less_equal([3, 2, 3], [3, 3, 1])
        assert result.to_list() == [1.0, 1.0, 0.0]


# ============================================================================
# Logical Operations
# ============================================================================


class TestLogical:
    def test_logical_and(self):
        result = cp.logical_and([1, 0, 1, 0], [1, 1, 0, 0])
        assert result.to_list() == [1.0, 0.0, 0.0, 0.0]

    def test_logical_or(self):
        result = cp.logical_or([1, 0, 1, 0], [1, 1, 0, 0])
        assert result.to_list() == [1.0, 1.0, 1.0, 0.0]

    def test_logical_not(self):
        result = cp.logical_not([1, 0, 1, 0])
        assert result.to_list() == [0.0, 1.0, 0.0, 1.0]

    def test_logical_xor(self):
        result = cp.logical_xor([1, 0, 1, 0], [1, 1, 0, 0])
        assert result.to_list() == [0.0, 1.0, 1.0, 0.0]


# ============================================================================
# Operator Overloads
# ============================================================================


class TestOperatorOverloads:
    def test_pow(self):
        a = cp.array([2, 3, 4])
        result = a**2
        r = result.to_list()
        assert abs(r[0] - 4.0) < 1e-5
        assert abs(r[1] - 9.0) < 1e-5
        assert abs(r[2] - 16.0) < 1e-5

    def test_mod(self):
        a = cp.array([10, 11, 12])
        result = a % 3
        assert result.to_list() == [1.0, 2.0, 0.0]

    def test_floordiv(self):
        a = cp.array([7, 8, 9])
        result = a // 2
        assert result.to_list() == [3.0, 4.0, 4.0]

    def test_neg(self):
        a = cp.array([1, -2, 3])
        result = -a
        assert result.to_list() == [-1.0, 2.0, -3.0]

    def test_abs(self):
        a = cp.array([-1, -2, 3])
        result = abs(a)
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_radd(self):
        """5 + array should work via __radd__."""
        a = cp.array([1, 2, 3])
        result = 5 + a
        assert result.to_list() == [6.0, 7.0, 8.0]

    def test_rmul(self):
        """3 * array should work via __rmul__."""
        a = cp.array([2, 3, 4])
        result = 3 * a
        assert result.to_list() == [6.0, 9.0, 12.0]


# ============================================================================
# is_even / is_odd
# ============================================================================


class TestEvenOdd:
    def test_is_even_functional(self):
        result = cp.is_even([1, 2, 3, 4])
        assert result.to_list() == [0.0, 1.0, 0.0, 1.0]

    def test_is_odd_functional(self):
        result = cp.is_odd([1, 2, 3, 4])
        assert result.to_list() == [1.0, 0.0, 1.0, 0.0]

    def test_is_even_method(self):
        a = cp.array([2, 4, 6, 7])
        result = a.is_even()
        assert result.to_list() == [1.0, 1.0, 1.0, 0.0]

    def test_is_odd_method(self):
        a = cp.array([2, 4, 6, 7])
        result = a.is_odd()
        assert result.to_list() == [0.0, 0.0, 0.0, 1.0]


# ============================================================================
# linspace Factory
# ============================================================================


class TestLinspace:
    def test_linspace_basic(self):
        result = cp.linspace(0, 1, 5)
        r = result.to_list()
        assert len(r) == 5
        assert abs(r[0] - 0.0) < 1e-5
        assert abs(r[4] - 1.0) < 1e-5
        assert abs(r[2] - 0.5) < 1e-5

    def test_linspace_single(self):
        result = cp.linspace(5, 10, 1)
        assert result.to_list() == [5.0]

    def test_linspace_empty(self):
        result = cp.linspace(0, 1, 0)
        assert result.to_list() == []


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_inf_add(self):
        result = cp.add([float("inf")], [1.0])
        assert result.to_list()[0] == float("inf")

    def test_nan_propagation(self):
        result = cp.add([float("nan")], [1.0])
        assert math.isnan(result.to_list()[0])

    def test_large_array(self):
        """Ensure large arrays work without error."""
        size = 100_000
        a = cp.array([1.0] * size)
        b = cp.array([2.0] * size)
        result = a + b
        r = result.to_list()
        assert len(r) == size
        assert r[0] == 3.0
        assert r[-1] == 3.0
