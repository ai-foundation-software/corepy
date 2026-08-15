"""
Tests for UFUNC CORE-50: Rounding, Sign, Clip, Bitwise operations.
"""

import pytest

import corepy as cp


class TestRounding:
    def test_floor(self):
        r = cp.floor([1.7, 2.3, -1.2])
        assert r.to_list() == [1.0, 2.0, -2.0]

    def test_ceil(self):
        r = cp.ceil([1.2, 2.7, -1.8])
        assert r.to_list() == [2.0, 3.0, -1.0]

    def test_round(self):
        r = cp.round_([1.5, 2.4, 3.6])
        assert r.to_list() == [2.0, 2.0, 4.0]

    def test_trunc(self):
        r = cp.trunc([1.7, -1.7, 2.3])
        assert r.to_list() == [1.0, -1.0, 2.0]

    def test_floor_method(self):
        a = cp.array([1.7, 2.3])
        r = a.floor()
        assert r.to_list() == [1.0, 2.0]

    def test_ceil_method(self):
        a = cp.array([1.2, 2.7])
        r = a.ceil()
        assert r.to_list() == [2.0, 3.0]


class TestSign:
    def test_sign(self):
        r = cp.sign([-5, 0, 3])
        assert r.to_list() == [-1.0, 0.0, 1.0]


class TestClip:
    def test_clip_functional(self):
        r = cp.clip([1, 5, 10, 15], 3.0, 12.0)
        assert r.to_list() == [3.0, 5.0, 10.0, 12.0]

    def test_clip_method(self):
        a = cp.array([1, 5, 10, 15])
        r = a.clip(3.0, 12.0)
        assert r.to_list() == [3.0, 5.0, 10.0, 12.0]

    def test_clamp_alias(self):
        r = cp.clamp([1, 5, 10], 2.0, 8.0)
        assert r.to_list() == [2.0, 5.0, 8.0]


class TestCopysign:
    def test_copysign(self):
        r = cp.copysign([1, -2, 3], [-1, 1, -1])
        assert r.to_list() == [-1.0, 2.0, -3.0]


class TestBitwise:
    def test_bitwise_and(self):
        r = cp.bitwise_and([7, 12], [3, 10])
        assert r.to_list() == [float(7 & 3), float(12 & 10)]

    def test_bitwise_or(self):
        r = cp.bitwise_or([5, 3], [3, 6])
        assert r.to_list() == [float(5 | 3), float(3 | 6)]

    def test_bitwise_xor(self):
        r = cp.bitwise_xor([5, 3], [3, 6])
        assert r.to_list() == [float(5 ^ 3), float(3 ^ 6)]

    def test_bitwise_not(self):
        r = cp.bitwise_not([0, 1])
        # ~0 = -1, ~1 = -2
        assert r.to_list() == [float(~0), float(~1)]

    def test_left_shift(self):
        r = cp.left_shift([1, 2], [2, 3])
        assert r.to_list() == [float(1 << 2), float(2 << 3)]

    def test_right_shift(self):
        r = cp.right_shift([8, 16], [2, 3])
        assert r.to_list() == [float(8 >> 2), float(16 >> 3)]
