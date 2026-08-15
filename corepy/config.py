"""
CorePy Configuration Settings

Global configuration for performance tuning and optimization parameters.
"""

# ============================================================================
# Performance Thresholds
# ============================================================================

# Small array fast path: Use NumPy directly for arrays smaller than this
# to avoid FFI overhead. For arrays >= threshold, use Rust/Accelerate path.
SMALL_ARRAY_THRESHOLD = 2000

# Parallelization threshold for CPU operations
# Arrays smaller than this use single-threaded execution
CPU_PARALLEL_THRESHOLD = 10_000

# GPU usage threshold: Only use GPU for arrays larger than this
# Below this threshold, CPU is typically faster due to transfer overhead
GPU_THRESHOLD = 100_000

# ============================================================================
# Memory Management
# ============================================================================

# Buffer pool configuration
BUFFER_POOL_ENABLED = True
BUFFER_POOL_MAX_SIZE = 100  # Maximum buffers per size class
BUFFER_POOL_SIZE_CLASSES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

# ============================================================================
# Backend Selection
# ============================================================================

# Default backend priority (higher = preferred)
BACKEND_PRIORITY = {
    "metal": 3,  # Highest for GPU operations
    "cpu": 2,  # Accelerate/NEON
}

# Auto-select backend based on array size
AUTO_BACKEND_SELECTION = True

# ============================================================================
# Debugging & Profiling
# ============================================================================

# Enable detailed performance logging
ENABLE_PROFILING = False

# Log all backend dispatch decisions
LOG_BACKEND_DISPATCH = False

# Validate numerical accuracy (slower but safer)
VALIDATE_NUMERICS = False
