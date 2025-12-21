from abc import ABC, abstractmethod
from typing import Any, Optional
from .types import BackendType, OperationType, OperationProperties
from .device import Device

class Backend(ABC):
    """
    Abstract base class for an execution backend.
    Responsible for memory allocation and operation execution.
    """
    
    @property
    @abstractmethod
    def device_type(self) -> BackendType:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def supports_operation(self, op_type: OperationType) -> bool:
        pass

class CPUBackend(Backend):
    def __init__(self):
        pass

    @property
    def device_type(self) -> BackendType:
        return BackendType.CPU

    def is_available(self) -> bool:
        return True

    def supports_operation(self, op_type: OperationType) -> bool:
        # CPU supports everything, though some things might be slow
        return True

# Placeholder for GPUBackend - would be loaded if available
class GPUBackend(Backend):
    def __init__(self):
        pass

    @property
    def device_type(self) -> BackendType:
        return BackendType.GPU
    
    def is_available(self) -> bool:
        # This would check actual driver availability
        return False

    def supports_operation(self, op_type: OperationType) -> bool:
        # GPU typically doesn't support complex control flow or arbitrary scalar logic well 
        # (at least not via this dispatch mechanism)
        if op_type in (OperationType.CONTROL, OperationType.SCALAR):
            return False
        return True
