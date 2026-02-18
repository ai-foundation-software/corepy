import logging
import os
from typing import Optional

from .device import DeviceInfo
from .types import BackendType, OperationProperties, OperationType

# Configure logging
logger = logging.getLogger("corepy.backend.selector")

# Thresholds (CONSTANTS)
# Thresholds (CONSTANTS)
THRESHOLD_VECTOR_ELEMENTS = 1_000_000  # Increased to favor CPU for small/medium ops
THRESHOLD_MATRIX_rows = (
    2048  # Increased based on benchmarks (Metal overhead high for < 2048)
)
THRESHOLD_MATRIX_COLS = 2048
THRESHOLD_BATCH_SIZE = 64


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
    requested_backend: Optional[BackendType] = None,
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

        # Verify availability
        if requested_backend == BackendType.GPU and not device_info.has_gpu:
            logger.warning(
                f"Requested backend {requested_backend} but no GPU/Accelerator detected. Falling back to CPU."
            )
            return BackendType.CPU

        return requested_backend

    # 2. Environment Variable Override
    env_forced = _get_forced_backend()
    if env_forced:
        logger.debug(f"Environment forced backend: {env_forced}")
        if env_forced == BackendType.GPU and not device_info.has_gpu:
            # Fallback if forced GPU but no GPU found?
            # Or raise error? Requirement says "safe fallback", but "forced" suggests user intent.
            # "Always provide safe fallbacks" implies we should warn and fallback.
            logger.warning(
                "COREPY_BACKEND=gpu set but no GPU detected. Falling back to CPU."
            )
            return BackendType.CPU
        return env_forced

    # 3. Correctness & Suitability Checks (The "Core Principles")

    # Principle: Small data -> CPU always wins
    # Principle: Control/Scalar -> CPU always wins
    if op_type in (
        OperationType.CONTROL,
        OperationType.SCALAR,
        # OperationType.MEMORY_BOUND,  # Allow large allocations to go to GPU if thresholds met
    ):
        logger.debug(f"Operation {op_type} is best suited for CPU.")
        return BackendType.CPU

    # Principle: Streaming without batching -> CPU
    if op_props.is_streaming and not op_props.is_batched:
        logger.debug(
            "Streaming operation without batching -> forcing CPU for correctness."
        )
        return BackendType.CPU

    # 4. GPU Candidate Evaluation
    if device_info.gpu_count > 0:
        # Check thresholds
        use_gpu = False

        if op_type == OperationType.COMPUTE_VECTOR:
            if op_props.element_count > THRESHOLD_VECTOR_ELEMENTS:
                use_gpu = True
                logger.debug(
                    f"Vector size {op_props.element_count} > {THRESHOLD_VECTOR_ELEMENTS}. GPU Candidate."
                )
            else:
                logger.debug(
                    f"Vector size {op_props.element_count} <= {THRESHOLD_VECTOR_ELEMENTS}. Keeping CPU."
                )

        elif op_type == OperationType.COMPUTE_MATRIX:
            rows, cols = op_props.shape[-2:] if len(op_props.shape) >= 2 else (0, 0)
            if rows >= THRESHOLD_MATRIX_rows and cols >= THRESHOLD_MATRIX_COLS:
                use_gpu = True
                logger.debug(
                    f"Matrix shape {rows}x{cols} >= {THRESHOLD_MATRIX_rows}x{THRESHOLD_MATRIX_COLS}. GPU Candidate."
                )
            else:
                return BackendType.CPU  # Explicitly return to avoid falling through

        # Check Batching
        if op_props.is_batched and op_props.batch_size >= THRESHOLD_BATCH_SIZE:
            use_gpu = True
            logger.debug(
                f"Batch size {op_props.batch_size} >= {THRESHOLD_BATCH_SIZE}. GPU Candidate."
            )

        if op_type == OperationType.MEMORY_BOUND:
            if op_props.element_count > THRESHOLD_VECTOR_ELEMENTS:
                use_gpu = True
                logger.debug(
                    f"Memory bound size {op_props.element_count} > Threshold. GPU Candidate."
                )

        if use_gpu:
            return BackendType.GPU

    # 5. Default
    return BackendType.CPU
