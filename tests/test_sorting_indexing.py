"""
Test suite for UFUNC CORE-12 sorting, searching, and indexing operations.

Tests cover:
- sort() ascending and descending
- argsort() correct indices
- stable_sort()
- argmax() / argmin()
- where_() / where()
- searchsorted()
- take()
- boolean_index()
"""

import pytest

import corepy as cp
from corepy.array import ndarray

# ============================================================================
# Sorting Operations
# ============================================================================


class TestSort:
    def test_sort_ascending(self):
        result = cp.sort([3, 1, 4, 1, 5, 9, 2, 6])
        assert result.to_list() == [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]

    def test_sort_descending(self):
        result = cp.sort([3, 1, 4, 1, 5], descending=True)
        assert result.to_list() == [5.0, 4.0, 3.0, 1.0, 1.0]

    def test_sort_method(self):
        a = cp.array([5, 2, 8, 1])
        result = a.sort()
        assert result.to_list() == [1.0, 2.0, 5.0, 8.0]

    def test_sort_already_sorted(self):
        result = cp.sort([1, 2, 3, 4])
        assert result.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_stable_sort(self):
        result = cp.stable_sort([3, 1, 2])
        assert result.to_list() == [1.0, 2.0, 3.0]


class TestArgsort:
    def test_argsort_basic(self):
        result = cp.argsort([30, 10, 20])
        indices = result.to_list()
        assert indices == [1.0, 2.0, 0.0]

    def test_argsort_descending(self):
        result = cp.argsort([30, 10, 20], descending=True)
        indices = result.to_list()
        assert indices == [0.0, 2.0, 1.0]

    def test_argsort_method(self):
        a = cp.array([5, 1, 3])
        result = a.argsort()
        assert result.to_list() == [1.0, 2.0, 0.0]


# ============================================================================
# Searching Operations
# ============================================================================


class TestArgmaxArgmin:
    def test_argmax(self):
        assert cp.argmax([1, 5, 3, 2]) == 1

    def test_argmin(self):
        assert cp.argmin([4, 1, 3, 2]) == 1

    def test_argmax_method(self):
        a = cp.array([10, 30, 20])
        assert a.argmax() == 1

    def test_argmin_method(self):
        a = cp.array([10, 30, 5])
        assert a.argmin() == 2


class TestWhere:
    def test_where_indices(self):
        """where(condition) returns indices of non-zero elements."""
        result = cp.where([0, 1, 0, 1, 1])
        assert result.to_list() == [1.0, 3.0, 4.0]

    def test_where_select(self):
        """where(cond, x, y) selects from x or y."""
        result = cp.where([1, 0, 1], [10, 20, 30], [100, 200, 300])
        assert result.to_list() == [10.0, 200.0, 30.0]


class TestSearchsorted:
    def test_searchsorted_left(self):
        result = cp.searchsorted([1, 2, 3, 4, 5], 3.5)
        assert result == 3

    def test_searchsorted_right(self):
        result = cp.searchsorted([1, 2, 3, 4, 5], 3, side="right")
        assert result == 3

    def test_searchsorted_array(self):
        result = cp.searchsorted([1, 3, 5, 7], [2, 4, 6])
        assert result.to_list() == [1.0, 2.0, 3.0]


# ============================================================================
# Indexing Operations
# ============================================================================


class TestTake:
    def test_take_basic(self):
        result = cp.take([10, 20, 30, 40, 50], [0, 2, 4])
        assert result.to_list() == [10.0, 30.0, 50.0]

    def test_take_repeated(self):
        result = cp.take([10, 20, 30], [0, 0, 1, 1])
        assert result.to_list() == [10.0, 10.0, 20.0, 20.0]


class TestBooleanIndex:
    def test_boolean_index_basic(self):
        result = cp.boolean_index([10, 20, 30, 40], [1, 0, 1, 0])
        assert result.to_list() == [10.0, 30.0]

    def test_boolean_index_all(self):
        result = cp.boolean_index([10, 20, 30], [1, 1, 1])
        assert result.to_list() == [10.0, 20.0, 30.0]

    def test_boolean_index_none(self):
        result = cp.boolean_index([10, 20, 30], [0, 0, 0])
        assert result.to_list() == []
