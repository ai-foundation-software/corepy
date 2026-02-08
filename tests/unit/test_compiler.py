import pytest

from corepy.lazy.array import LazyArray
from corepy.lazy.compiler import ExpressionCompiler, FusedKernel
from corepy.lazy.nodes import BinaryOp, Constant, Reduction, UnaryOp


class MockArray:
    """Simple mock for array operations to avoid dependency on full ndarray."""

    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        return MockArray(self.val + other.val)

    def __sub__(self, other):
        return MockArray(self.val - other.val)

    def __mul__(self, other):
        return MockArray(self.val * other.val)

    def __truediv__(self, other):
        return MockArray(self.val / other.val)

    def __neg__(self):
        return MockArray(-self.val)

    def __abs__(self):
        return MockArray(abs(self.val))

    def sum(self):
        return self.val  # Mock reduction

    def mean(self):
        return self.val  # Mock reduction

    def __eq__(self, other):
        if isinstance(other, MockArray):
            return self.val == other.val
        return self.val == other


def test_compiler_init():
    compiler = ExpressionCompiler()
    assert isinstance(compiler, ExpressionCompiler)


def test_compile_single_node():
    compiler = ExpressionCompiler()
    expr = Constant(MockArray(10))
    kernel = compiler.compile(expr)
    assert isinstance(kernel, FusedKernel)
    assert len(kernel.operations) == 1
    assert kernel.operations[0] is expr


def test_execute_constant():
    # Constant
    node = Constant(MockArray(5))
    kernel = FusedKernel([node])
    res = kernel.execute()
    assert res == MockArray(5)


def test_execute_binary_op():
    # 5 + 3
    left = LazyArray(MockArray(5))
    right = LazyArray(MockArray(3))

    # We need to construct ExprNode tree manually or use LazyArray helper?
    # LazyArray construction wraps in Constant if value provided.

    # left._expr is Constant(5)

    bin_op = BinaryOp("add", left, right)

    kernel = FusedKernel([bin_op])
    res = kernel.execute()
    assert res == MockArray(8)

    # Sub
    res = FusedKernel([BinaryOp("sub", left, right)]).execute()
    assert res == MockArray(2)

    # Mul
    res = FusedKernel([BinaryOp("mul", left, right)]).execute()
    assert res == MockArray(15)

    # Div
    res = FusedKernel([BinaryOp("div", left, right)]).execute()
    assert res == MockArray(5 / 3)


def test_execute_unary_op():
    op = LazyArray(MockArray(5))

    # Neg
    unary = UnaryOp("neg", op)
    res = FusedKernel([unary]).execute()
    assert res == MockArray(-5)

    # Abs
    unary = UnaryOp("abs", LazyArray(MockArray(-5)))
    res = FusedKernel([unary]).execute()
    assert res == MockArray(5)


def test_execute_reduction():
    op = LazyArray(MockArray(10))

    # Sum
    red = Reduction("sum", op)
    res = FusedKernel([red]).execute()
    assert res == 10  # MockArray.sum returns val

    # Mean
    red = Reduction("mean", op)
    res = FusedKernel([red]).execute()
    assert res == 10


def test_nested_expression():
    # (5 + 3) * 2
    a = LazyArray(MockArray(5))
    b = LazyArray(MockArray(3))
    c = LazyArray(MockArray(2))

    # a + b
    sum_expr = BinaryOp("add", a, b)
    # Wrap result in LazyArray for next op
    sum_lazy = LazyArray(expr=sum_expr)

    # sum * c
    mul_expr = BinaryOp("mul", sum_lazy, c)

    kernel = FusedKernel([mul_expr])
    res = kernel.execute()
    # (5+3)*2 = 16
    assert res == MockArray(16)


def test_unknown_ops():
    # Binary
    with pytest.raises(ValueError):
        FusedKernel([BinaryOp("unknown", LazyArray(1), LazyArray(1))]).execute()

    # Unary
    with pytest.raises(ValueError):
        FusedKernel([UnaryOp("unknown", LazyArray(1))]).execute()

    # Reduction
    with pytest.raises(ValueError):
        FusedKernel([Reduction("unknown", LazyArray(1))]).execute()


def test_unknown_node_type():
    class UnknownNode:
        pass

    with pytest.raises(TypeError):
        FusedKernel([UnknownNode()]).execute()  # type: ignore


def test_empty_kernel():
    kernel = FusedKernel([])
    assert kernel.execute() is None
