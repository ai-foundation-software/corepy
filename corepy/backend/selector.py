import os
import logging
from typing import Optional
from .types import BackendType, OperationType, OperationProperties
from .device import DeviceInfo, Device
from .backend import Backend, CPUBackend, GPUBackend
from .errors import DeviceNotFoundError

# Configure logging
logger = logging.getLogger("corepy.backend.selector")

# Thresholds (CONSTANTS)
THRESHOLD_VECTOR_ELEMENTS = 100_000
THRESHOLD_MATRIX_rows = 512
THRESHOLD_MATRIX_COLS = 512
THRESHOLD_BATCH_SIZE = 32

def _get_forced_backend() -> Optional[BackendType]:
    """Check environment variable for forced backend."""
    env_backend = os.getenv("COREPY_BACKEND", "").lower()
    if env_backend == "cpu":
        return BackendType.CPU
    if env_backend == "gpu":
        return BackendType.GPU
    # Add TPU or others if needed
    return None

def select_backend(
    op_type: OperationType,
    op_props: OperationProperties,
    device_info: DeviceInfo,
    requested_backend: Optional[BackendType] = None
) -> BackendType:
    """
    Determines the best backend for an operation based on correctness, 
    availability, and performance cost models.

    Args:
        op_type: Type of operation (CONTROL, COMPUTE_VECTOR, etc.)
        op_props: Properties of the data (size, shape, batching)
        device_info: Available hardware info
        requested_backend: User-requested backend (overrides everything if safe/available)
    
    Returns:
        BackendType: The selected backend
    """
    
    # 1. User Override (API argument)
    if requested_backend:
        logger.debug(f"User requested backend: {requested_backend}")
        # In a real system, we'd verify availability here too. 
        # For now, we trust the user but could add validity checks.
        return requested_backend

    # 2. Environment Variable Override
    env_forced = _get_forced_backend()
    if env_forced:
        logger.debug(f"Environment forced backend: {env_forced}")
        if env_forced == BackendType.GPU and not device_info.has_gpu:
             # Fallback if forced GPU but no GPU found? 
             # Or raise error? Requirement says "safe fallback", but "forced" suggests user intent.
             # "Always provide safe fallbacks" implies we should warn and fallback.
             logger.warning("COREPY_BACKEND=gpu set but no GPU detected. Falling back to CPU.")
             return BackendType.CPU
        return env_forced
    
    # 3. Correctness & Suitability Checks (The "Core Principles")
    
    # Principle: Small data -> CPU always wins
    # Principle: Control/Scalar -> CPU always wins
    if op_type in (OperationType.CONTROL, OperationType.SCALAR, OperationType.MEMORY_BOUND):
        logger.debug(f"Operation {op_type} is best suited for CPU.")
        return BackendType.CPU

    # Principle: Streaming without batching -> CPU
    if op_props.is_streaming and not op_props.is_batched:
        logger.debug("Streaming operation without batching -> forcing CPU for correctness.")
        return BackendType.CPU

    # 4. GPU Candidate Evaluation
    if device_info.gpu_count > 0:
        # Check thresholds
        use_gpu = False
        
        if op_type == OperationType.COMPUTE_VECTOR:
            if op_props.element_count > THRESHOLD_VECTOR_ELEMENTS:
                use_gpu = True
                logger.debug(f"Vector size {op_props.element_count} > {THRESHOLD_VECTOR_ELEMENTS}. GPU Candidate.")
            else:
                logger.debug(f"Vector size {op_props.element_count} <= {THRESHOLD_VECTOR_ELEMENTS}. Keeping CPU.")

        elif op_type == OperationType.COMPUTE_MATRIX:
            rows, cols = op_props.shape[-2:] if len(op_props.shape) >= 2 else (0,0)
            if rows >= THRESHOLD_MATRIX_rows and cols >= THRESHOLD_MATRIX_COLS:
                use_gpu = True
                logger.debug(f"Matrix shape {rows}x{cols} >= {THRESHOLD_MATRIX_rows}x{THRESHOLD_MATRIX_COLS}. GPU Candidate.")
            else:
                return BackendType.CPU # Explicitly return to avoid falling through

        # Check Batching
        if op_props.is_batched and op_props.batch_size >= THRESHOLD_BATCH_SIZE:
             use_gpu = True
             logger.debug(f"Batch size {op_props.batch_size} >= {THRESHOLD_BATCH_SIZE}. GPU Candidate.")

        if use_gpu:
            return BackendType.GPU

    # 5. Default
    return BackendType.CPU
