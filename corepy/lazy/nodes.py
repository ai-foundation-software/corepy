"""
Expression tree nodes for lazy evaluation.

Each node represents an operation in the expression tree.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .array import LazyArray


class ExprNode:
    """Base class for all expression nodes."""

    def is_elementwise(self) -> bool:
        """Check if this operation is element-wise (fusable)."""
        return False


class Constant(ExprNode):
    """
    Leaf node representing a constant array value.

    This is the base of the expression tree - actual data.
    """

    def __init__(self, value: Any):
        """
        Args:
            value: CorePy ndarray containing the data
        """
        self.value = value

    def __repr__(self):
        return f"Constant(shape={getattr(self.value, 'shape', '?')})"


class BinaryOp(ExprNode):
    """
    Binary operation node (+, -, *, /, etc.).

    Represents operations between two arrays or an array and a scalar.
    """

    def __init__(self, op: str, left: "LazyArray", right: "LazyArray"):
        """
        Args:
            op: Operation name ('add', 'sub', 'mul', 'div')
            left: Left operand
            right: Right operand
        """
        self.op = op
        self.left = left
        self.right = right

    def is_elementwise(self) -> bool:
        """Binary ops are element-wise and can be fused."""
        return self.op in ("add", "sub", "mul", "div")

    def __repr__(self):
        return f"BinaryOp({self.op}, {self.left}, {self.right})"


class UnaryOp(ExprNode):
    """
    Unary operation node (-x, abs(x), etc.).

    Represents operations on a single array.
    """

    def __init__(self, op: str, operand: "LazyArray"):
        """
        Args:
            op: Operation name ('neg', 'abs', 'sqrt', etc.)
            operand: Input operand
        """
        self.op = op
        self.operand = operand

    def is_elementwise(self) -> bool:
        """Most unary ops are element-wise."""
        return self.op in ("neg", "abs", "sqrt", "exp", "log")

    def __repr__(self):
        return f"UnaryOp({self.op}, {self.operand})"


class Reduction(ExprNode):
    """
    Reduction operation node (sum, mean, max, min).

    These operations reduce dimensions and typically can't be fused
    with element-wise operations.
    """

    def __init__(self, op: str, array: "LazyArray"):
        """
        Args:
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            array: Input array
        """
        self.op = op
        self.array = array

    def is_elementwise(self) -> bool:
        """Reductions are not element-wise."""
        return False

    def __repr__(self):
        return f"Reduction({self.op}, {self.array})"
