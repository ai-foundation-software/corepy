
from typing import Any, List, Optional
import math
from corepy.backend.types import BackendType

class ReferenceBackend:
    """
    The 'Gold Standard' reference implementation for Corepy operations.
    
    Rules:
    1. Correctness is the ONLY goal.
    2. Speed does not matter.
    3. Use pure Python or standard library math where possible.
    4. Used to validate C++/Rust optimized kernels.
    """
    
    @staticmethod
    def add(a: Any, b: Any) -> Any:
        # Recursive handle lists/tensors
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), "Shape mismatch in reference add"
            return [ReferenceBackend.add(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list):
            return [ReferenceBackend.add(x, b) for x in a]
        elif isinstance(b, list):
            return [ReferenceBackend.add(a, y) for y in b]
        else:
            return a + b

    @staticmethod
    def sub(a: Any, b: Any) -> Any:
        if isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), "Shape mismatch"
            return [ReferenceBackend.sub(x, y) for x, y in zip(a, b)]
        return a - b

    @staticmethod
    def mul(a: Any, b: Any) -> Any:
        if isinstance(a, list) and isinstance(b, list):
             assert len(a) == len(b), "Shape mismatch"
             return [ReferenceBackend.mul(x, y) for x, y in zip(a, b)]
        return a * b

    @staticmethod
    def div(a: Any, b: Any) -> Any:
        if isinstance(a, list) and isinstance(b, list):
             assert len(a) == len(b), "Shape mismatch"
             return [ReferenceBackend.div(x, y) for x, y in zip(a, b)]
        return a / b

    @staticmethod
    def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        # Naive O(N^3) matrix multiplication
        rows_a = len(a)
        cols_a = len(a[0])
        rows_b = len(b)
        cols_b = len(b[0])
        
        assert cols_a == rows_b, f"Shape mismatch: {cols_a} != {rows_b}"
        
        result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                sum_val = 0.0
                for k in range(cols_a):
                    sum_val += a[i][k] * b[k][j]
                result[i][j] = sum_val
        return result

# Register these if the Dispatcher allows explicit backend selection
# For now, this class stands alone for testing usage.
