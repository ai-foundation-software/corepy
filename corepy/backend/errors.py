class CorepyError(Exception):
    """Base class for all Corepy exceptions."""
    pass

class BackendError(CorepyError):
    """Base class for backend-related errors."""
    pass

class DeviceNotFoundError(BackendError):
    """Raised when a requested device is not found or available."""
    pass

class OutOfMemoryError(BackendError):
    """Raised when a backend runs out of memory."""
    pass

class OperationNotSupportedError(BackendError):
    """Raised when an operation is not supported on the selected backend."""
    pass
