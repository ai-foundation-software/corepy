from unittest.mock import MagicMock, patch

import pytest

from corepy.lazy.array import LazyArray
from corepy.lazy.nodes import BinaryOp, Constant, Reduction, UnaryOp


class MockKernel:
    def __init__(self, result=None):
        self.result = result

    def execute(self):
        return self.result


class MockCompiler:
    def __init__(self):
        self.compile_calls = []

    def compile(self, expr):
        self.compile_calls.append(expr)
        # Return a mock kernel that returns a dummy result
        return MockKernel(result="mock_result")


def test_lazy_array_init():
    # Test initialization with value
    arr = LazyArray(value=10)
    assert arr._is_materialized
    assert arr._value == 10
    assert isinstance(arr._expr, Constant)
    assert arr._expr.value == 10

    # Test initialization with expression
    expr = Constant(20)
    arr_lazy = LazyArray(expr=expr)
    assert not arr_lazy._is_materialized
    assert arr_lazy._value is None
    assert arr_lazy._expr is expr


def test_lazy_array_wrap():
    # Test wrapping concrete val
    l1 = LazyArray._wrap(5)
    assert isinstance(l1, LazyArray)
    assert l1._value == 5

    # Test wrapping existing LazyArray
    l2 = LazyArray._wrap(l1)
    assert l2 is l1


def test_binary_ops():
    a = LazyArray(1)
    b = LazyArray(2)

    # Add
    c = a + b
    assert isinstance(c, LazyArray)
    assert isinstance(c._expr, BinaryOp)
    assert c._expr.op == "add"
    assert c._expr.left is a
    assert c._expr.right._value == 2  # wrapped

    # Sub
    d = a - b
    assert d._expr.op == "sub"

    # Mul
    e = a * b
    assert e._expr.op == "mul"

    # Div
    f = a / b
    assert f._expr.op == "div"


def test_reverse_binary_ops():
    a = LazyArray(1)

    # Radd
    c = 2 + a
    assert isinstance(c._expr, BinaryOp)
    assert c._expr.op == "add"
    assert c._expr.left._value == 2
    assert c._expr.right is a

    # Rsub
    d = 2 - a
    assert d._expr.op == "sub"
    assert d._expr.left._value == 2

    # Rmul
    e = 2 * a
    assert e._expr.op == "mul"

    # Rdiv
    f = 2 / a
    assert f._expr.op == "div"


def test_unary_ops():
    a = LazyArray(1)

    # Neg
    b = -a
    assert isinstance(b._expr, UnaryOp)
    assert b._expr.op == "neg"
    assert b._expr.operand is a

    # Abs
    c = a.abs()
    assert isinstance(c._expr, UnaryOp)
    assert c._expr.op == "abs"


def test_reductions():
    a = LazyArray(1)

    # Sum
    b = a.sum()
    assert isinstance(b._expr, Reduction)
    assert b._expr.op == "sum"
    assert b._expr.array is a

    # Mean
    c = a.mean()
    assert isinstance(c._expr, Reduction)
    assert c._expr.op == "mean"


def test_compute_materialize():
    # Mock the compiler import within compute
    with patch.dict("sys.modules", {"corepy.lazy.compiler": MagicMock()}):
        # We need to mock the class 'ExpressionCompiler' inside the mocked module
        mock_compiler_cls = MagicMock()
        mock_instance = MockCompiler()
        mock_compiler_cls.return_value = mock_instance

        # Setup sys.modules override
        import sys

        mock_module = MagicMock()
        mock_module.ExpressionCompiler = mock_compiler_cls

        with patch.dict(sys.modules, {"corepy.lazy.compiler": mock_module}):
            a = LazyArray(1)
            b = LazyArray(2)
            c = a + b

            assert not c._is_materialized

            # Compute
            res = c.compute()

            assert res == "mock_result"
            assert c._is_materialized
            assert c._value == "mock_result"

            # Verify compiler was called
            assert len(mock_instance.compile_calls) == 1
            assert mock_instance.compile_calls[0] is c._expr

            # Test re-compute returns cached value
            res2 = c.compute()
            assert res2 == "mock_result"
            # Should not compile again (mock calls count same)
            # Actually our MockCompiler creates new instance each time in the code?
            # Code: compiler = ExpressionCompiler().
            # So mock_compiler_cls is called again.
            # BUT if _is_materialized is True, it returns early.
            # So mock_compiler_cls should NOT be called again.
            assert mock_compiler_cls.call_count == 1


def test_repr():
    a = LazyArray(1)
    # Materialized
    assert "LazyArray(materialized" in repr(a)

    b = LazyArray(expr=Constant(2))
    # Not materialized
    assert "LazyArray(Constant" in repr(b)


def test_node_repr():
    # Constant
    c = Constant(1)
    assert "Constant" in repr(c)

    # Binary
    b = BinaryOp("add", LazyArray(1), LazyArray(2))
    assert "BinaryOp" in repr(b)
    assert "add" in repr(b)

    # Unary
    u = UnaryOp("neg", LazyArray(1))
    assert "UnaryOp" in repr(u)

    # Reduction
    r = Reduction("sum", LazyArray(1))
    assert "Reduction" in repr(r)


def test_node_is_elementwise():
    # Constant (Base ExprNode default is False, wait. Constant inherits ExprNode)
    # is_elementwise is False by default
    assert not Constant(1).is_elementwise()

    # Binary
    assert BinaryOp("add", None, None).is_elementwise()
    assert not BinaryOp("matmul", None, None).is_elementwise()  # Not in list

    # Unary
    assert UnaryOp("neg", None).is_elementwise()
    assert not UnaryOp("unknown", None).is_elementwise()

    # Reduction
    assert not Reduction("sum", None).is_elementwise()
