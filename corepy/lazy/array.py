"""
LazyArray class for deferred computation and operation fusion.

Builds expression trees instead of immediately executing operations.
"""

from typing import Any, Optional, Union

from .nodes import BinaryOp, Constant, ExprNode, Reduction, UnaryOp


class LazyArray:
    """
    Lazy evaluation wrapper that builds expression tree.

    Operations on LazyArray objects build an expression tree instead of
    executing immediately. Call .compute() to materialize the result.

    Example:
        >>> lazy_a = LazyArray(cp.array([1, 2, 3]))
        >>> lazy_b = LazyArray(cp.array([4, 5, 6]))
        >>> lazy_c = (lazy_a + lazy_b) * 2  # Builds tree, doesn't execute
        >>> result = lazy_c.compute()        # Execute fused kernel
    """

    def __init__(self, value: Any = None, expr: Optional[ExprNode] = None):
        """
        Initialize LazyArray from either a concrete value or an expression.

        Args:
            value: Concrete ndarray (if materialized)
            expr: Expression node (if lazy)
        """
        self._value = value
        self._expr = (
            expr
            if expr is not None
            else (Constant(value) if value is not None else None)
        )
        self._is_materialized = value is not None and expr is None

    @staticmethod
    def _wrap(obj: Union["LazyArray", Any]) -> "LazyArray":
        """Wrap a value in LazyArray if it isn't already."""
        if isinstance(obj, LazyArray):
            return obj
        else:
            # Assume it's a concrete ndarray or scalar
            return LazyArray(value=obj)

    # Binary operations
    def __add__(self, other):
        """Addition: lazy_a + lazy_b"""
        return LazyArray(expr=BinaryOp("add", self, self._wrap(other)))

    def __radd__(self, other):
        """Reverse addition: scalar + lazy_a"""
        return LazyArray(expr=BinaryOp("add", self._wrap(other), self))

    def __sub__(self, other):
        """Subtraction: lazy_a - lazy_b"""
        return LazyArray(expr=BinaryOp("sub", self, self._wrap(other)))

    def __rsub__(self, other):
        """Reverse subtraction: scalar - lazy_a"""
        return LazyArray(expr=BinaryOp("sub", self._wrap(other), self))

    def __mul__(self, other):
        """Multiplication: lazy_a * lazy_b"""
        return LazyArray(expr=BinaryOp("mul", self, self._wrap(other)))

    def __rmul__(self, other):
        """Reverse multiplication: scalar * lazy_a"""
        return LazyArray(expr=BinaryOp("mul", self._wrap(other), self))

    def __truediv__(self, other):
        """Division: lazy_a / lazy_b"""
        return LazyArray(expr=BinaryOp("div", self, self._wrap(other)))

    def __rtruediv__(self, other):
        """Reverse division: scalar / lazy_a"""
        return LazyArray(expr=BinaryOp("div", self._wrap(other), self))

    # Unary operations
    def __neg__(self):
        """Negation: -lazy_a"""
        return LazyArray(expr=UnaryOp("neg", self))

    def abs(self):
        """Absolute value: abs(lazy_a)"""
        return LazyArray(expr=UnaryOp("abs", self))

    # Reductions
    def sum(self):
        """Sum reduction: lazy_a.sum()"""
        return LazyArray(expr=Reduction("sum", self))

    def mean(self):
        """Mean reduction: lazy_a.mean()"""
        return LazyArray(expr=Reduction("mean", self))

    def compute(self):
        """
        Force evaluation of the lazy expression tree.

        Compiles the expression into a fused kernel and executes it.

        Returns:
            CorePy ndarray with the computed result
        """
        if self._is_materialized:
            # Already computed
            return self._value

        # Import here to avoid circular dependency
        from .compiler import ExpressionCompiler

        # Compile and execute
        compiler = ExpressionCompiler()
        kernel = compiler.compile(self._expr)
        self._value = kernel.execute()
        self._is_materialized = True

        return self._value

    def __repr__(self):
        if self._is_materialized:
            return (
                f"LazyArray(materialized, shape={getattr(self._value, 'shape', '?')})"
            )
        else:
            return f"LazyArray({self._expr})"
