import numpy as np

import corepy
from corepy.backend import BackendType
from corepy.backend.device import detect_devices
from corepy.backend.selector import select_backend
from corepy.backend.types import OperationProperties, OperationType


def check_routing(elements, op_type=OperationType.COMPUTE_VECTOR):
    device_info = detect_devices()
    op_props = OperationProperties(element_count=elements, shape=(elements,))
    backend = select_backend(op_type, op_props, device_info)
    print(f"Elements: {elements:12,} | Selected Backend: {backend}")
    return backend


print("=== CorePy Backend Routing Verification ===")
print(
    "Default Thresholds: CUDA 100M FLOPs / 2M Elements, Metal 25M FLOPs / 500k Elements\n"
)

# 1. Very small (CPU)
check_routing(10_000)

# 2. Medium (previously GPU boundary, should now be CPU)
# 1,500,000 elements * 2 flops/element = 3,000,000 flops
# This should definitely be CPU now
check_routing(1_500_000)

# 3. Large (should be GPU if available, else CPU)
check_routing(100_000_000)

print("\n=== CPU Batching Threshold Verification ===")
print("New Threshold: 32 (Lowered from 64 to trigger parallel CPU paths earlier)")
# Note: Verification of actual parallel execution is harder without deep profiling,
# but we can verify the backend choice logic.
