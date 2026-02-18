"""
Lazy evaluation module for operation fusion.

Enables building expression trees and fusing operations for improved performance.
"""

from .array import LazyArray
from .context import lazy
from .nodes import BinaryOp, Constant, ExprNode, Reduction, UnaryOp

__all__ = [
    "LazyArray",
    "ExprNode",
    "BinaryOp",
    "UnaryOp",
    "Constant",
    "Reduction",
    "lazy",
]
