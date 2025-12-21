import time
import statistics
import logging
from corepy.backend.selector import select_backend
from corepy.backend.types import BackendType, OperationType, OperationProperties
from corepy.backend.device import DeviceInfo

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def mock_gpu_transfer_cost_ms(size_bytes):
    # Model: Latency (20us) + Bandwidth (16GB/s approx -> 16 bytes/ns -> 16e9 bytes/s)
    # 1 GB = 1e9 bytes. 
    # Time = size / BW
    latency_s = 20e-6
    bandwidth_Bps = 16 * 1e9
    return (latency_s + size_bytes / bandwidth_Bps) * 1000

def mock_cpu_compute_ms(ops):
    # Model: 100 GFLOPS
    flops = 100 * 1e9
    return (ops / flops) * 1000

def mock_gpu_compute_ms(ops):
    # Model: 10 TFLOPS
    flops = 10 * 1e12
    return (ops / flops) * 1000 + 0.005 # kernel launch overhead 5us

def benchmark_thresholds():
    """
    Simulate cost models to validate thresholds.
    """
    print(f"{'Size (Elem)':<12} {'Bytes':<12} {'CPU (ms)':<10} {'GPU (ms)':<10} {'Winner':<8}")
    print("-" * 60)
    
    sizes = [1000, 10_000, 100_000, 1_000_000, 10_000_000]
    dtype_size = 4
    
    for n in sizes:
        bytes_size = n * dtype_size
        ops = n # Simple element-wise op
        
        cpu_t = mock_cpu_compute_ms(ops)
        
        # GPU = Transfer -> Launch -> Compute -> TransferBack
        gpu_t = (mock_gpu_transfer_cost_ms(bytes_size) * 2) + mock_gpu_compute_ms(ops)
        
        winner = "CPU" if cpu_t < gpu_t else "GPU"
        print(f"{n:<12} {bytes_size:<12} {cpu_t:<10.5f} {gpu_t:<10.5f} {winner:<8}")

        # Validation against our hardcoded threshold (100k)
        if n == 100_000:
            if winner == "GPU" or abs(cpu_t - gpu_t) < 0.01:
                print(f"  -> Validation: 100k is a reasonable transition point for these params.")
            else:
                print(f"  -> Validation: Warning, 100k might be sub-optimal based on this synthetic model.")

if __name__ == "__main__":
    benchmark_thresholds()
