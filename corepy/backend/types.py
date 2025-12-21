from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, Any

class BackendType(Enum):
    """
    Supported execution backends.
    """
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    # Future backends can be added here

class OperationType(Enum):
    """
    Classification of operations for backend selection.
    """
    CONTROL = auto()          # Control flow, scalar logic (always CPU)
    MEMORY_BOUND = auto()     # Element-wise, copy, cast (CPU preferred unless huge)
    COMPUTE_VECTOR = auto()   # Vectorized math (potential for GPU if large)
    COMPUTE_MATRIX = auto()   # Matrix/Tensor math (Strong candidate for GPU)
    SCALAR = auto()           # Single value ops (strictly CPU)

@dataclass
class OperationProperties:
    """
    Metadata about an operation to aid backend selection.
    """
    element_count: int
    shape: Tuple[int, ...]
    is_batched: bool = False
    batch_size: int = 1
    is_streaming: bool = False
    dtype_bytes: int = 4  # e.g., float32 = 4 bytes

    @property
    def total_bytes(self) -> int:
        return self.element_count * self.dtype_bytes

class DataType(Enum):
    """
    Data types supported by Corepy.
    """
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    # complex types etc.
