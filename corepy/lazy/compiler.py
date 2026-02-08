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
            if node.op == "add":
                return left + right
            elif node.op == "sub":
                return left - right
            elif node.op == "mul":
                return left * right
            elif node.op == "div":
                return left / right
            else:
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
            array = self._execute_node(node.array._expr)  # type: ignore[arg-type]

            if node.op == "sum":
                return array.sum()
            elif node.op == "mean":
                return array.mean()
            else:
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

        Args:
            expr: Root of the expression tree

        Returns:
            FusedKernel ready for execution
        """
        # For now, just wrap the expression in a kernel
        # Future: analyze tree, fuse operations, generate optimized code

        operations = [expr]
        return FusedKernel(operations)
