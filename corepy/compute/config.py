from dataclasses import dataclass
from typing import Optional


@dataclass
class ComputeConfig:
    """
    Configuration and threshold defaults for dynamic CorePy compute dispatch operations.
    These can be tuned globally to alter backend selection heuristics at runtime.
    """

    # ---------------------------------------------------------
    # GPU / CUDA Settings
    # ---------------------------------------------------------

    # Minimum number of FLOPs to justify transferring data to CUDA overriding CPU.
    # Increased to 100M to favor CPU for medium-sized operations.
    cuda_min_flops: int = 100_000_000

    # Minimum array size (number of elements) for vector op GPU dispatching
    cuda_min_vector_size: int = 2_000_000

    # Threshold size for both dimensions for matrix multiplication (N >= x)
    cuda_min_matrix_dim: int = 4096
    # ---------------------------------------------------------
    # System & Memory Architecture (Metal)
    # ---------------------------------------------------------
    # Apple Silicon uses Unified Memory, sidestepping costly PCIe transfers.
    # Therefore, Metal backends should trigger at smaller thresholds than CUDA.

    metal_min_flops: int = 25_000_000
    metal_min_vector_size: int = 500_000
    metal_min_matrix_dim: int = 1024

    # ---------------------------------------------------------
    # Parallel CPU Settings
    # ---------------------------------------------------------

    # Enable multi-threaded CPU processing (Rayon) when True
    cpu_parallel_enabled: bool = True

    # Batch size minimum boundary before dispatch considers multi-threaded backend approaches.
    # Lowered to 32 to trigger optimized CPU parallel paths earlier.
    cpu_min_batch_size: int = 32


# Global shared compute configuration
config = ComputeConfig()


def get_config() -> ComputeConfig:
    """Retrieve the global compute configuration."""
    return config


def set_config(new_config: ComputeConfig):
    """Override the global compute configuration with a custom set of thresholds."""
    global config
    config = new_config
