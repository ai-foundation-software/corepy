import logging
import os
from typing import Optional

from ..compute import get_config
from .device import DeviceInfo
from .types import BackendType, OperationProperties, OperationType

# Configure logging
logger = logging.getLogger("corepy.backend.selector")

try:
    from .. import _corepy_rust
except ImportError:
    _corepy_rust = None


def _get_forced_backend(device_info: DeviceInfo) -> Optional[BackendType]:
    """Check environment variable for forced backend."""
    env_backend = os.getenv("COREPY_BACKEND", "").lower()
    if env_backend == "cpu":
        return BackendType.CPU
    if env_backend == "cuda":
        return BackendType.CUDA
    if env_backend == "metal":
        return BackendType.METAL
    if env_backend == "gpu":
        return (
            BackendType.METAL
            if device_info.platform_system == "Darwin"
            else BackendType.CUDA
        )
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
        if (
            requested_backend in (BackendType.CUDA, BackendType.METAL)
            and not device_info.has_gpu
        ):
            logger.warning(
                f"Requested backend {requested_backend} but no GPU/Accelerator detected. Falling back to CPU."
            )
            return BackendType.CPU

        return requested_backend

    # 2. Environment Variable Override
    env_forced = _get_forced_backend(device_info)
    if env_forced:
        logger.debug(f"Environment forced backend: {env_forced}")
        if (
            env_forced in (BackendType.CUDA, BackendType.METAL)
            and not device_info.has_gpu
        ):
            logger.warning(
                "COREPY_BACKEND requests GPU execution but no GPU detected. Falling back to CPU."
            )
            return BackendType.CPU
        return env_forced

    # 3. Correctness & Suitability Checks
    # Principle: Small data -> CPU always wins
    # Principle: Control/Scalar -> CPU always wins
    if op_type in (
        OperationType.CONTROL,
        OperationType.SCALAR,
    ):
        logger.debug(f"Operation {op_type} is best suited for CPU.")
        return BackendType.CPU

    if op_props.is_streaming and not op_props.is_batched:
        logger.debug(
            "Streaming operation without batching -> forcing CPU for correctness."
        )
        return BackendType.CPU

    # 4. Smart Scoring Evaluation
    cfg = get_config()
    if _corepy_rust is not None:
        # Approximate flops. 1 element ~ 2 operations typically
        flops = op_props.element_count * 2
        # E.g. matrix mul O(N^3) -> 2 * m * n * k
        if op_type == OperationType.COMPUTE_MATRIX and len(op_props.shape) >= 2:
            m, n = op_props.shape[-2:]
            flops = 2 * m * n * n  # very rough approx

        memory_bytes = op_props.element_count * op_props.dtype_bytes

        # Consult Rust AI Brain for the optimal compiled backend
        recommended = _corepy_rust.recommend_backend(
            flops, memory_bytes, op_props.is_batched
        )

        logger.debug(
            f"Rust Brain Evaluation - Flops: {flops}, Mem: {memory_bytes}, Recommendation: {recommended}"
        )

        if recommended == "Metal" and device_info.has_metal:
            return BackendType.METAL
        if recommended == "CUDA" and device_info.has_cuda:
            return BackendType.CUDA
        if recommended in ("Metal", "CUDA") and device_info.has_gpu:
            return BackendType.METAL if device_info.has_metal else BackendType.CUDA

    else:
        # Fallback if rust is not compiled
        flops = op_props.element_count * 2

    # 5. Legacy/Dynamic Threshold GPU Candidate Evaluation
    if device_info.has_gpu:
        use_accel = False
        accel_type = BackendType.METAL if device_info.has_metal else BackendType.CUDA

        min_flops = cfg.metal_min_flops if device_info.has_metal else cfg.cuda_min_flops
        min_vector = (
            cfg.metal_min_vector_size
            if device_info.has_metal
            else cfg.cuda_min_vector_size
        )
        min_matrix = (
            cfg.metal_min_matrix_dim
            if device_info.has_metal
            else cfg.cuda_min_matrix_dim
        )

        if flops >= min_flops:
            use_accel = True

        if op_type == OperationType.COMPUTE_VECTOR:
            if op_props.element_count >= min_vector:
                use_accel = True

        elif op_type == OperationType.COMPUTE_MATRIX:
            rows, cols = op_props.shape[-2:] if len(op_props.shape) >= 2 else (0, 0)
            if rows >= min_matrix and cols >= min_matrix:
                use_accel = True

        if op_props.is_batched and op_props.batch_size >= cfg.cpu_min_batch_size:
            use_accel = True

        if op_type == OperationType.MEMORY_BOUND:
            if op_props.element_count >= min_vector:
                use_accel = True

        if use_accel:
            return accel_type

    # 6. Default
    return BackendType.CPU
