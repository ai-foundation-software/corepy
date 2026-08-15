"""
Tests for UFUNC CORE-50: Trigonometric, Hyperbolic, Exponential, Logarithmic.
"""

import math

import pytest

import corepy as cp


class TestTrig:
    def test_sin(self):
        r = cp.sin([0, math.pi / 2, math.pi])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 1.0) < 1e-5
        assert abs(r.to_list()[2] - 0.0) < 1e-4

    def test_cos(self):
        r = cp.cos([0, math.pi / 2, math.pi])
        assert abs(r.to_list()[0] - 1.0) < 1e-5
        assert abs(r.to_list()[1] - 0.0) < 1e-4
        assert abs(r.to_list()[2] - (-1.0)) < 1e-4

    def test_tan(self):
        r = cp.tan([0, math.pi / 4])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 1.0) < 1e-4

    def test_arcsin(self):
        r = cp.arcsin([0, 0.5, 1.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_arccos(self):
        r = cp.arccos([1.0, 0.5, 0.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_arctan(self):
        r = cp.arctan([0, 1.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - math.pi / 4) < 1e-4

    def test_arctan2(self):
        r = cp.arctan2([1.0, 0.0], [0.0, 1.0])
        assert abs(r.to_list()[0] - math.pi / 2) < 1e-4
        assert abs(r.to_list()[1] - 0.0) < 1e-5


class TestHyperbolic:
    def test_sinh(self):
        r = cp.sinh([0, 1.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_cosh(self):
        r = cp.cosh([0])
        assert abs(r.to_list()[0] - 1.0) < 1e-5

    def test_tanh(self):
        r = cp.tanh([0, 100])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 1.0) < 1e-5

    def test_arcsinh(self):
        r = cp.arcsinh([0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_arccosh(self):
        r = cp.arccosh([1.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_arctanh(self):
        r = cp.arctanh([0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5


class TestAngle:
    def test_degrees(self):
        r = cp.degrees([math.pi, math.pi / 2])
        assert abs(r.to_list()[0] - 180.0) < 0.1
        assert abs(r.to_list()[1] - 90.0) < 0.1

    def test_radians(self):
        r = cp.radians([180, 90])
        assert abs(r.to_list()[0] - math.pi) < 1e-4
        assert abs(r.to_list()[1] - math.pi / 2) < 1e-4

    def test_hypot(self):
        r = cp.hypot([3.0], [4.0])
        assert abs(r.to_list()[0] - 5.0) < 1e-4


class TestExponential:
    def test_exp(self):
        r = cp.exp([0, 1.0])
        assert abs(r.to_list()[0] - 1.0) < 1e-5
        assert abs(r.to_list()[1] - math.e) < 1e-4

    def test_exp2(self):
        r = cp.exp2([0, 3.0])
        assert abs(r.to_list()[0] - 1.0) < 1e-5
        assert abs(r.to_list()[1] - 8.0) < 1e-4

    def test_expm1(self):
        r = cp.expm1([0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_log(self):
        r = cp.log([1.0, math.e])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 1.0) < 1e-4

    def test_log2(self):
        r = cp.log2([1.0, 8.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 3.0) < 1e-4

    def test_log10(self):
        r = cp.log10([1.0, 100.0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 2.0) < 1e-4

    def test_log1p(self):
        r = cp.log1p([0])
        assert abs(r.to_list()[0] - 0.0) < 1e-5

    def test_sqrt(self):
        r = cp.sqrt([4.0, 9.0, 16.0])
        assert abs(r.to_list()[0] - 2.0) < 1e-4
        assert abs(r.to_list()[1] - 3.0) < 1e-4
        assert abs(r.to_list()[2] - 4.0) < 1e-4


class TestMethodProxy:
    def test_sin_method(self):
        a = cp.array([0, math.pi / 2])
        r = a.sin()
        assert abs(r.to_list()[0] - 0.0) < 1e-5
        assert abs(r.to_list()[1] - 1.0) < 1e-5

    def test_exp_method(self):
        a = cp.array([0, 1.0])
        r = a.exp()
        assert abs(r.to_list()[0] - 1.0) < 1e-5

    def test_sqrt_method(self):
        a = cp.array([4.0, 9.0])
        r = a.sqrt()
        assert abs(r.to_list()[0] - 2.0) < 1e-4
