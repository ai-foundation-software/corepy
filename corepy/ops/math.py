from ..backend.dispatch import register_kernel
from ..backend.types import BackendType
from typing import Any

# We'll use a simple list implementation for now since we don't have numpy dependency yet
# In a real scenario, this would import numpy

@register_kernel("add", BackendType.CPU)
def cpu_add(a: Any, b: Any) -> Any:
    """
    Element-wise addition for CPU.
    Accepts lists or raw data.
    """
    # Simplified implementation assuming basic lists or scalars
    # If a and b are lists, element-wise add
    # If scalar, error (for now) or broadcast
    
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Shape mismatch for cpu_add")
        return [x + y for x, y in zip(a, b)]
    elif isinstance(a, list) and isinstance(b, (int, float)):
         return [x + b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
         return [a + x for x in b]
    else:
        # Scalar + Scalar
        return a + b

@register_kernel("matmul", BackendType.CPU)
def cpu_matmul(a: Any, b: Any) -> Any:
    """
    Simple Matrix Multiplication (Naive O(N^3)) for CPU Demo.
    Assumes inputs are 2D lists (List[List[float]]).
    """
    # Assuming a and b are 2D lists
    # A is (m x k), B is (k x n) -> Result is (m x n)
    
    # Just a placeholder for "Compute Bound" op
    # Real impl would use optimized BLAS
    return "cpu_matmul_result_placeholder"
