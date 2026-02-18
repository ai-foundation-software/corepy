"""
Context manager for lazy evaluation mode.

Enables transparent lazy evaluation with existing cp.array API.
"""

import threading
from contextlib import contextmanager

# Thread-local storage for lazy mode
_lazy_mode = threading.local()


def is_lazy_mode() -> bool:
    """Check if currently in lazy mode."""
    return getattr(_lazy_mode, "enabled", False)


def _enable_lazy_mode():
    """Enable lazy mode (internal)."""
    _lazy_mode.enabled = True


def _disable_lazy_mode():
    """Disable lazy mode (internal)."""
    _lazy_mode.enabled = False


@contextmanager
def lazy():
    """
    Context manager to enable lazy evaluation.

    Within this context, operations on cp.array objects build expression
    trees instead of executing immediately. Call `.compute()` to materialize.

    **User API remains unchanged** - use normal cp.array and operations.

    Example:
        >>> import corepy as cp
        >>>
        >>> # Normal mode (eager)
        >>> a = cp.array([1, 2, 3])
        >>> result = a + a  # Executes immediately
        >>>
        >>> # Lazy mode (same API!)
        >>> with cp.lazy():
        ...     a = cp.array([1, 2, 3])
        ...     b = cp.array([4, 5, 6])
        ...     c = (a + b) * 2  # Builds expression tree
        ...     result = c.compute()  # Execute fused kernel
    """
    _enable_lazy_mode()
    try:
        yield
    finally:
        _disable_lazy_mode()
