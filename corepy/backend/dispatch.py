from typing import Dict, Any, Callable, Tuple, Optional
from .types import BackendType
from .errors import OperationNotSupportedError
import logging

logger = logging.getLogger("corepy.backend.dispatch")

class Dispatcher:
    """
    Registry for backend-specific kernel implementations.
    Maps (operation_name, backend_type) -> kernel_function.
    """
    _registry: Dict[Tuple[str, BackendType], Callable] = {}

    @classmethod
    def register(cls, op_name: str, backend: BackendType):
        """
        Decorator to register a kernel function for a specific operation and backend.
        
        Usage:
            @Dispatcher.register("add", BackendType.CPU)
            def cpu_add(a, b): ...
        """
        def decorator(func: Callable):
            key = (op_name, backend)
            if key in cls._registry:
                logger.warning(f"Overwriting kernel for {key}")
            cls._registry[key] = func
            return func
        return decorator

    @classmethod
    def get_kernel(cls, op_name: str, backend: BackendType) -> Callable:
        """
        Retrieves the kernel for the given operation and backend.
        Raises OperationNotSupportedError if not found.
        """
        key = (op_name, backend)
        kernel = cls._registry.get(key)
        if not kernel:
            raise OperationNotSupportedError(f"No kernel registered for '{op_name}' on {backend.value}")
        return kernel

    @classmethod
    def dispatch(cls, op_name: str, backend: BackendType, *args, **kwargs) -> Any:
        """
        Finds and executes the appropriate kernel.
        """
        kernel = cls.get_kernel(op_name, backend)
        return kernel(*args, **kwargs)

# Alias for easy access
register_kernel = Dispatcher.register
dispatch_kernel = Dispatcher.dispatch
