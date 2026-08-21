"""
Corepy Domain Guards & Mathematical Error Handling Module

Provides robust, NumPy-compliant domain checking and exception conversion for
unary and binary mathematical operations (Trigonometric, Hyperbolic, Exponential, Logarithmic).
"""

import math
import warnings
from typing import Any, Callable, Optional


def safe_eval_unary(op_name: str, py_fn: Callable[[float], float], x: float) -> float:
    """
    Safely evaluate a scalar unary mathematical function.

    Converts domain errors, zero division, and overflows into IEEE 754 float
    values (NaN / Inf / -Inf) while issuing NumPy-compatible RuntimeWarnings.

    Args:
        op_name: Operation name (e.g. 'arctanh', 'log', 'sqrt')
        py_fn: Python math function to evaluate
        x: Input scalar float value

    Returns:
        float: Computed result or NaN/Inf
    """
    if math.isnan(x):
        return float("nan")

    try:
        res = py_fn(x)
        if math.isinf(res) and not math.isinf(x):
            warnings.warn(
                f"divide by zero encountered in {op_name}",
                RuntimeWarning,
                stacklevel=3,
            )
        return res
    except ValueError:
        # Math domain error
        warnings.warn(
            f"invalid value encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan")
    except ZeroDivisionError:
        warnings.warn(
            f"divide by zero encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        if op_name.startswith("log"):
            return float("-inf")
        return float("nan") if x < 0 else float("inf")
    except OverflowError:
        warnings.warn(
            f"overflow encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("inf") if x > 0 else float("-inf")


def safe_eval_binary(
    op_name: str, py_fn: Callable[[float, float], float], x: float, y: float
) -> float:
    """
    Safely evaluate a scalar binary mathematical function.

    Args:
        op_name: Operation name (e.g. 'divide', 'arctan2', 'hypot')
        py_fn: Python binary math function to evaluate
        x: First input scalar float
        y: Second input scalar float

    Returns:
        float: Computed result or NaN/Inf
    """
    if math.isnan(x) or math.isnan(y):
        return float("nan")

    try:
        res = py_fn(x, y)
        if math.isinf(res) and not (math.isinf(x) or math.isinf(y)):
            warnings.warn(
                f"divide by zero encountered in {op_name}",
                RuntimeWarning,
                stacklevel=3,
            )
        return res
    except ValueError:
        warnings.warn(
            f"invalid value encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan")
    except ZeroDivisionError:
        warnings.warn(
            f"divide by zero encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("inf") if x > 0 else float("-inf")
    except OverflowError:
        warnings.warn(
            f"overflow encountered in {op_name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("inf")
