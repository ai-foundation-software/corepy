"""
corepy.matmul — High-performance matrix multiplication with automatic backend selection.

Accepts any object implementing the Python buffer protocol (PEP 3118):
  - corepy arrays (preferred)
  - Python array.array
  - Python array.array
  - Any other buffer-compatible object

Usage:
    import corepy

    a = corepy.randn(1024, 1024)  # corepy native array
    b = corepy.randn(1024, 1024)
    c = corepy.matmul(a, b)       # auto-selects best backend

    # Force a specific backend
    c = corepy.matmul(a, b, policy="rust")
    c = corepy.matmul(a, b, policy="mkl")

    # Inspect active backend
    import corepy.matmul as cm
    print(cm.backend_info())  # {'backend':'MKL', 'vendor':'Intel', ...}
"""

from __future__ import annotations

import array
import ctypes
import struct
from enum import IntEnum
from typing import Any, Optional

# ─── Import native module ─────────────────────────────────────────────────────
try:
    from . import _corepy_rust as _cr  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Error importing _corepy_rust: {e}")
    _cr = None  # type: ignore
    _RUST_AVAILABLE = False


# ============================================================================
# Policy enum (mirrors BackendPolicy in Rust backend/mod.rs)
# ============================================================================


class BackendPolicy(IntEnum):
    """Maps to Rust BackendPolicy enum values used by set_backend_policy()."""

    AUTO = 0  # Auto-detect (selector_v2)
    OPENBLAS = 1  # Force OpenBLAS
    BLAS = 2  # Generic BLAS
    CUDA = 3  # CUDA GPU
    METAL = 4  # Metal GPU (macOS)
    MKL = 5  # Intel MKL
    AOCL = 6  # AMD AOCL / BLIS
    ACCELERATE = 7  # Apple Accelerate
    RUST = 8  # Pure-Rust parallel (rayon + matrixmultiply)


_POLICY_ALIASES: dict[str, BackendPolicy] = {
    "auto": BackendPolicy.AUTO,
    "default": BackendPolicy.AUTO,
    "openblas": BackendPolicy.OPENBLAS,
    "blas": BackendPolicy.BLAS,
    "cuda": BackendPolicy.CUDA,
    "metal": BackendPolicy.METAL,
    "mkl": BackendPolicy.MKL,
    "aocl": BackendPolicy.AOCL,
    "accelerate": BackendPolicy.ACCELERATE,
    "rust": BackendPolicy.RUST,
    "rustparallel": BackendPolicy.RUST,
}


def _parse_policy(policy: str | BackendPolicy | int) -> BackendPolicy:
    if isinstance(policy, (BackendPolicy, int)):
        return BackendPolicy(int(policy))
    alias = policy.lower().replace("-", "").replace("_", "")
    if alias not in _POLICY_ALIASES:
        raise ValueError(
            f"Unknown backend policy '{policy}'. "
            f"Valid options: {sorted(_POLICY_ALIASES.keys())}"
        )
    return _POLICY_ALIASES[alias]


# ============================================================================
# Buffer-protocol array handling
# ============================================================================


def _get_buffer_info(obj: Any) -> tuple[int, tuple[int, ...], str]:
    """
    Return (data_ptr, shape, format) for any buffer-protocol object.

    Supports:
      - Objects with .ctypes.data + .shape
      - memoryview
      - array.array
      - CorePy native arrays (via __buffer__ / memoryview)
    """
    # Fast path: buffer-protocol arrays with .ctypes.data and .shape
    if hasattr(obj, "ctypes") and hasattr(obj, "shape") and hasattr(obj, "dtype"):
        data_ptr = obj.ctypes.data
        shape = tuple(obj.shape)
        fmt = getattr(obj.dtype, "char", "f")  # 'f' = float32
        return data_ptr, shape, fmt

    # memoryview path (covers array.array, bytes, bytearray, and any __buffer__ object)
    mv = memoryview(obj)
    if mv.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {mv.ndim}-D buffer")
    data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    return data_ptr, tuple(mv.shape), mv.format


_CP_MODULE = None


def _get_cp_module():
    global _CP_MODULE
    if _CP_MODULE is None:
        try:
            import corepy as _cp

            _CP_MODULE = _cp
        except Exception:
            _CP_MODULE = False
    return _CP_MODULE if _CP_MODULE is not False else None


def _ensure_float32_c_contiguous(obj: Any) -> tuple[int, tuple[int, ...]]:
    """
    Validate `obj` is a 2-D float32 C-contiguous buffer.
    Returns (data_ptr, (rows, cols)).

    Works with any buffer-protocol object.
    """
    # Ultra-fast path for NumPy / CorePy arrays
    try:
        data_ptr = obj.ctypes.data
        shape = obj.shape
        if len(shape) != 2:
            raise ValueError(f"corepy.matmul requires 2-D arrays; got shape {shape}")
        dt = getattr(obj, "dtype", None)
        if dt is not None and getattr(dt, "name", None) != "float32" and str(dt) != "float32":
            raise ValueError(
                f"corepy.matmul requires float32 arrays; got dtype='{dt}'.\n"
                f"Convert with: arr = arr.astype('float32')"
            )
        flags = getattr(obj, "flags", None)
        if flags is not None:
            if hasattr(flags, "c_contiguous") and not flags.c_contiguous:
                raise ValueError(
                    "Array must be C-contiguous. Use arr.copy(order='C') to convert."
                )
            elif hasattr(flags, "get") and not flags.get("C_CONTIGUOUS", True):
                raise ValueError(
                    "Array must be C-contiguous. Use arr.copy(order='C') to convert."
                )

        return data_ptr, shape if type(shape) is tuple else tuple(shape)
    except (AttributeError, TypeError):
        pass

    # memoryview path
    mv = memoryview(obj)
    if mv.ndim != 2:
        raise ValueError(f"corepy.matmul requires 2-D arrays; got {mv.ndim}-D buffer")
    if mv.format != "f":  # 'f' = float32 in buffer protocol
        raise ValueError(
            f"corepy.matmul requires float32 (format='f'); got format='{mv.format}'."
        )
    if not mv.c_contiguous:
        raise ValueError("Buffer must be C-contiguous.")
    data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    return data_ptr, tuple(mv.shape)


def _alloc_float32_2d(rows: int, cols: int) -> tuple[int, Any]:
    """
    Allocate a C-contiguous float32 2-D output buffer.
    Returns (data_ptr, backing_array).
    """
    buf = array.array("f", bytes(rows * cols * 4))
    data_ptr = buf.buffer_info()[0]
    return data_ptr, buf


# ============================================================================
# Public API
# ============================================================================


def matmul(
    a: Any,
    b: Any,
    policy: str | BackendPolicy | int = "auto",
    out: Optional[Any] = None,
) -> Any:
    """Compute matrix product C = A @ B using the optimal CPU backend.

    Parameters
    ----------
    a, b : array-like (buffer protocol)
        2-D float32 arrays in row-major (C) order.
        Accepts corepy arrays, or any PEP 3118 buffer.
    policy : str | BackendPolicy | int, optional
        'auto' (default), 'mkl', 'aocl', 'openblas', 'rust', ...
    out : array-like, optional
        Pre-allocated float32 C-contiguous output buffer.

    Returns
    -------
    Corepy array (or same type as input if it implements __array__).
    Falls back to array.array if no native output type available.

    Raises
    ------
    ValueError
        If shapes are incompatible or dtype is not float32.
    RuntimeError
        If the Rust native module is not built.
    """
    if not _RUST_AVAILABLE:
        raise RuntimeError(
            "CorePy Rust extension not available. "
            "Build with: .venv/bin/maturin build --release && "
            "uv pip install --no-deps dist/corepy_ai-*.whl"
        )

    # ── Validate inputs via buffer protocol ──────────────────────────────────
    a_ptr, (m, k1) = _ensure_float32_c_contiguous(a)
    b_ptr, (k2, n) = _ensure_float32_c_contiguous(b)

    if k1 != k2:
        raise ValueError(f"Incompatible shapes for matmul: ({m},{k1}) @ ({k2},{n})")

    # ── Output allocation ────────────────────────────────────────────────────
    if out is None:
        _cp = _get_cp_module()
        if _cp is not None:
            out = _cp.empty((m, n), dtype=_cp.float32)
            out_ptr = out.ctypes.data
        else:
            out_ptr, out = _alloc_float32_2d(m, n)
    else:
        out_ptr, (om, on) = _ensure_float32_c_contiguous(out)
        if (om, on) != (m, n):
            raise ValueError(f"out has shape ({om},{on}), expected ({m},{n})")

    # ── Policy ───────────────────────────────────────────────────────────────
    bp = _parse_policy(policy)
    if bp != BackendPolicy.AUTO:
        _cr.set_backend_policy(int(bp))

    # ── Dispatch via FFI ─────────────────────────────────────────────────────
    _cr.array_matmul_2d_f32(a_ptr, b_ptr, out_ptr, m, k1, n)

    # Restore auto policy
    if bp != BackendPolicy.AUTO:
        _cr.set_backend_policy(int(BackendPolicy.AUTO))

    return out


def get_last_dispatch() -> str:
    """Return a human-readable description of the last backend used."""
    if not _RUST_AVAILABLE:
        return "RustNotBuilt"
    return _cr.explain_last_dispatch()


def backend_info() -> dict:
    """Return CPU and backend details.

    Keys: 'backend', 'vendor', 'threads', 'hyperthreading',
          'brand', 'physical_cores', 'logical_cores'.
    """
    if not _RUST_AVAILABLE:
        return {
            "backend": "RustNotBuilt",
            "vendor": "Unknown",
            "threads": 1,
            "hyperthreading": False,
            "brand": "N/A",
            "physical_cores": 1,
            "logical_cores": 1,
        }
    return _cr.get_math_backend_info()


def set_policy(policy: str | BackendPolicy | int) -> None:
    """Set the global backend policy for all subsequent matmul calls."""
    if not _RUST_AVAILABLE:
        return
    bp = _parse_policy(policy)
    _cr.set_backend_policy(int(bp))


def reset_policy() -> None:
    """Reset to auto-detect policy."""
    set_policy("auto")
