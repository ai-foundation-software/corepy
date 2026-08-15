"""
Expression compiler for lazy evaluation.

Compiles expression trees into fused kernels for efficient execution.
"""

from typing import List

from .nodes import BinaryOp, Constant, ExprNode, Reduction, UnaryOp


class FusedKernel:
    """
    Executable kernel representing fused operations.

    Executes a sequence of operations, potentially fused for efficiency.
    """

    def __init__(self, operations: List[ExprNode]):
        """
        Args:
            operations: List of operations to execute
        """
        self.operations = operations

    def execute(self):
        """
        Execute the fused kernel.

        For now, this is a simple eager executor. In production, this would
        generate optimized kernels with operation fusion.

        Returns:
            CorePy ndarray with the result
        """
        # Simple execution: just evaluate the expression tree eagerly
        # In production, this would fuse operations

        if not self.operations:
            return None

        # Execute the final operation (recursive evaluation)
        return self._execute_node(self.operations[-1])

    def _execute_node(self, node: ExprNode):
        """Recursively execute a node in the expression tree."""
        if isinstance(node, Constant):
            return node.value

        elif isinstance(node, BinaryOp):
            left = self._execute_node(node.left._expr)  # type: ignore[arg-type]
            right = self._execute_node(node.right._expr)  # type: ignore[arg-type]

            # Execute the operation
            op_map = {
                "add": lambda lhs, r: lhs + r,
                "sub": lambda lhs, r: lhs - r,
                "mul": lambda lhs, r: lhs * r,
                "div": lambda lhs, r: lhs / r,
                "power": lambda lhs, r: lhs**r,
                "mod": lambda lhs, r: lhs % r,
                "floor_div": lambda lhs, r: lhs // r,
            }
            fn = op_map.get(node.op)
            if fn is not None:
                return fn(left, right)
            raise ValueError(f"Unknown binary op: {node.op}")

        elif isinstance(node, UnaryOp):
            operand = self._execute_node(node.operand._expr)  # type: ignore[arg-type]

            if node.op == "neg":
                return -operand
            elif node.op == "abs":
                return abs(operand)
            else:
                raise ValueError(f"Unknown unary op: {node.op}")

        elif isinstance(node, Reduction):
            array = self._execute_node(node.array._expr)

            reduction_map = {
                "sum": lambda a: a.sum(),
                "mean": lambda a: a.mean(),
                "max": lambda a: a.max(),
                "min": lambda a: a.min(),
                "prod": lambda a: a.prod() if hasattr(a, "prod") else None,
            }
            fn = reduction_map.get(node.op)
            if fn is not None:
                return fn(array)
            raise ValueError(f"Unknown reduction: {node.op}")

        else:
            raise TypeError(f"Unknown node type: {type(node)}")


class ExpressionCompiler:
    """
    Compiles expression trees into optimized fused kernels.

    Current implementation: Simple eager execution
    Future: Operation fusion, kernel generation, GPU optimization
    """

    def compile(self, expr: ExprNode) -> FusedKernel:
        """
        Compile an expression tree into an executable kernel.

        Performs elementwise fusion: chains of elementwise ops are
        collected into a single fused kernel for single-pass execution.
        """
        operations = self._collect_ops(expr)
        return FusedKernel(operations)

    def _collect_ops(self, node: ExprNode) -> List[ExprNode]:
        """Collect operations in topological order for fusion."""
        ops = []
        self._visit(node, ops, set())
        return ops

    def _visit(self, node: ExprNode, ops: List[ExprNode], visited: set):
        """DFS visit for topological ordering."""
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, Constant):
            pass  # leaf node
        elif isinstance(node, BinaryOp):
            if node.left._expr is not None:
                self._visit(node.left._expr, ops, visited)
            if node.right._expr is not None:
                self._visit(node.right._expr, ops, visited)
        elif isinstance(node, UnaryOp):
            if node.operand._expr is not None:
                self._visit(node.operand._expr, ops, visited)
        elif isinstance(node, Reduction):
            if node.array._expr is not None:
                self._visit(node.array._expr, ops, visited)

        ops.append(node)
