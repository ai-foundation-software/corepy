from typing import Optional, Union, Sequence, Any, Tuple
import logging
from .backend.types import BackendType, OperationType, OperationProperties, DataType
from .backend.selector import select_backend
from .backend.session import get_session
from .backend.errors import BackendError

logger = logging.getLogger("corepy.tensor")

class Tensor:
    """
    A multi-dimensional array object that automatically selects the best
    execution backend (CPU/GPU) based on data size and operation complexity.
    """
    def __init__(
        self, 
        data: Union[Sequence[Any], 'Tensor'], 
        dtype: DataType = DataType.FLOAT32,
        backend: Optional[Union[str, BackendType]] = None,
        device: Optional[str] = None 
    ):
        """
        Initialize a Tensor.

        Args:
            data: Input data (list, tuple, or another Tensor).
            dtype: Data type (default: float32).
            backend: Explicitly requested backend ('cpu', 'gpu').
            device: Explicit device string (e.g. 'cuda:0', 'cpu').
                    If provided, overrides 'backend'.
        """
        self._dtype = dtype
        
        # Determine shape and element count (simplified for this implementation)
        # In a real impl, we'd recursively check list depth/lengths
        if isinstance(data, (list, tuple)):
            self._shape = (len(data),) # Simplified 1D assumption for now
            self._element_count = len(data)
            self._backing_data = data # Placeholder for real storage buffer
        elif hasattr(data, 'shape'): # numpy compatibility
             self._shape = data.shape
             self._element_count = data.size
             self._backing_data = data
        else:
            # scalar or error
            self._shape = (1,)
            self._element_count = 1
            self._backing_data = [data]

        # Resolve requested backend/device
        requested_backend = None
        if device:
            if "cuda" in device or "gpu" in device:
                requested_backend = BackendType.GPU
            elif "cpu" in device:
                requested_backend = BackendType.CPU
        elif backend:
            if isinstance(backend, str):
                requested_backend = BackendType(backend.lower())
            else:
                requested_backend = backend

        # Select Backend
        # We classify Creation as MEMORY_BOUND or SCALAR usually, 
        # but the meaningful decision happens for subsequent ops.
        # However, we must decide where to allocations *now*. 
        # Let's assume creation is MEMORY_BOUND.
        op_props = OperationProperties(
            element_count=self._element_count,
            shape=self._shape,
            # Approximating bytes: len * 4 bytes for float32
            dtype_bytes=4 
        )
        
        # We treat 'allocation' as a memory operation.
        # However, for 'Correctness-First', we usually default to CPU for storage 
        # unless explicitly told otherwise or if we are consuming GPU data.
        # BUT, the goal is "Tensor(data) # auto". 
        # So we should check if this data is "large enough" to justify GPU storage?
        # Usually, just storing is not compute. So auto-placement should default CPU 
        # unless immediate heavy compute is expected? 
        # Actually, "Tensor(data)" usually implies "Ready for compute".
        # Let's use COMPUTE_VECTOR as a proxy for "Will I use this for compute?" 
        # to see if it qualifies for GPU memory residence.
        # This is a heuristic. Stronger approach: Default CPU, move on demand.
        # But per requirements: "Core Principles: Small data -> CPU always wins".
        
        session = get_session()
        self._backend_type = select_backend(
            OperationType.COMPUTE_VECTOR, # Probe: "If I treated this as a compute vector, where would it go?"
            op_props,
            session.device_info,
            requested_backend=requested_backend
        )

        logger.debug(f"Tensor created on {self._backend_type}. Shape={self._shape}")

    @property
    def backend(self) -> BackendType:
        return self._backend_type

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def to(self, device: str) -> 'Tensor':
        """
        Explicitly move tensor to a device.
        Arguments:
            device: 'cpu' or 'gpu'
        """
        # Create a new Tensor with explicit backend
        # In real impl, we would copy data buffer
        return Tensor(self._backing_data, dtype=self._dtype, device=device)

    def __repr__(self):
        return f"Tensor({self._backing_data}, backend='{self._backend_type.value}')"

    def __add__(self, other: Any) -> 'Tensor':
        """
        Element-wise addition.
        """
        # 1. Resolve Other
        if isinstance(other, Tensor):
             # Check compatibility: Same backend?
             # For now, strict requirement: Must indicate same backend or be scalar
             if other.backend != self.backend:
                 raise BackendError(f"Backend mismatch: {self.backend} vs {other.backend}")
             other_data = other._backing_data
        else:
             # Scalar or raw list
             other_data = other
        
        # 2. Dispatch
        # Ensure kernels are loaded! (Usually done at init time or lazy load)
        # For this PoC, we might need to strictly import corepy.ops.math in __init__.py 
        # or inside here (lazy).
        from .backend.dispatch import dispatch_kernel
        # Verify imports of kernels happens somewhere
        
        result_data = dispatch_kernel("add", self.backend, self._backing_data, other_data)
        
        # 3. Return new Tensor on same backend (usually)
        # In real engine, result placement depends on Op rules. 
        # Add usually stays on same device.
        return Tensor(result_data, dtype=self._dtype, backend=self.backend)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication.
        """
        if not isinstance(other, Tensor):
             raise ValueError("matmul requires a Tensor input")
        
        if other.backend != self.backend:
             raise BackendError(f"Backend mismatch: {self.backend} vs {other.backend}")
        
        from .backend.dispatch import dispatch_kernel
        
        result_data = dispatch_kernel("matmul", self.backend, self._backing_data, other._backing_data)
        
        return Tensor(result_data, dtype=self._dtype, backend=self.backend)
