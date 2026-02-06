"""
Tutorial 04: Metal GPU Acceleration
================================

This tutorial demonstrates how to use the Metal GPU backend on macOS devices (Apple Silicon).
Corepy v0.2.3 introduces native Metal support for high-performance tensor operations.

Prerequisites:
- macOS 12.0+
- Apple Silicon (M1/M2/M3) recommended

concepts:
- Device selection
- GPU vs CPU performance comparison
- Automatic fallback
"""

import time
import corepy as cp
import numpy as np

def benchmark_op(name, tensor_a, tensor_b, op_func, iterations=10):
    # Warmup
    op_func(tensor_a, tensor_b)
    
    start = time.perf_counter()
    for _ in range(iterations):
        res = op_func(tensor_a, tensor_b)
        # Force synchronization if needed (currently synchronous)
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations
    print(f"  {name}: {avg_time*1000:.4f} ms")

def main():
    print("=== Corepy Metal GPU Tutorial ===")
    
    # 1. Check Availability
    # Ideally we'd have cp.is_metal_available(), but for now we try/except or check device
    print("\n1. Initializing Tensors...")
    
    N = 2048
    print(f"Problem size: {N}x{N} matrix ({N*N*4 / 1024**2:.2f} MB)")
    
    # Create data in NumPy (CPU)
    data = np.random.randn(N, N).astype(np.float32)
    
    # 2. CPU Execution
    print("\n--- CPU Backend ---")
    t_cpu_a = cp.Tensor(data) # Default is CPU
    t_cpu_b = cp.Tensor(data)
    
    benchmark_op("Matmul (CPU)", t_cpu_a, t_cpu_b, lambda x, y: x @ y, iterations=5)
    
    # 3. Metal GPU Execution
    print("\n--- Metal Backend ---")
    try:
        # Create tensors directly on Metal device
        t_metal_a = cp.Tensor(data, device="metal")
        t_metal_b = cp.Tensor(data, device="metal")
        
        # Verify device
        # Note: If Metal is unavailable, it might have fallen back to CPU (check valid warning)
        print(f"Tensor device: {t_metal_a._backend_type}") # 2 for Metal usually
        
        benchmark_op("Matmul (Metal)", t_metal_a, t_metal_b, lambda x, y: x @ y, iterations=20)
        
        # 4. Mixed Operations
        print("\n--- Mixed Operations ---")
        # Adding Metal tensor to CPU tensor -> Result is usually on Metal (or CPU depending on promotion rules)
        # Currently Corepy might enforce same-device policies, so let's try strict
        
        res = t_metal_a + t_metal_b
        print(f"Result shape: {res.shape}")
        
    except Exception as e:
        print(f"\n⚠️ Metal backend unavailable or failed: {e}")
        print("Note: This tutorial requires a macOS device with Metal support.")

if __name__ == "__main__":
    main()
