
import time
import gc
import statistics
from typing import Callable, List, Dict, Any, Optional
import platform

class BenchmarkRunner:
    """
     rigorous benchmark runner for Corepy v1.
    
    Principals:
    1. Correctness > Speed (but we measure speed anyway).
    2. CPU Monotonic Clock.
    3. Explicit Warmup.
    4. Forced GC before runs.
    """

    def __init__(self, warmup_iters: int = 3, measure_iters: int = 20):
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results: Dict[str, Dict[str, float]] = {}

    def _now(self) -> int:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    def benchmark(self, name: str, func: Callable[[], Any], setup: Optional[Callable[[], Any]] = None) -> None:
        """
        Run a benchmark for a specific function.
        """
        print(f"Benchmarking: {name}")
        
        # Warmup
        for _ in range(self.warmup_iters):
             if setup: setup()
             func()
        
        latencies_ns: List[int] = []

        # Measurement Loop
        for _ in range(self.measure_iters):
            if setup: setup()
            
            # Force GC to reduce noise, though it might penalize realistic throughput.
            # For latency stability testing, this is acceptable in v1.
            gc.collect()
            
            start = self._now()
            func()
            end = self._now()
            
            latencies_ns.append(end - start)

        # Analysis
        latencies_ms = [t / 1e6 for t in latencies_ns]
        p50 = statistics.median(latencies_ms)
        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        mean = statistics.mean(latencies_ms)
        stdev = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

        print(f"  P50:  {p50:.4f} ms")
        print(f"  P99:  {p99:.4f} ms")
        print(f"  Mean: {mean:.4f} ms")
        print(f"  Std:  {stdev:.4f} ms")

        self.results[name] = {
            "p50_ms": p50,
            "p99_ms": p99,
            "mean_ms": mean,
            "std_ms": stdev
        }

    def save_report(self, path: str = "benchmark_report.json"):
        import json
        import os
        
        report = {
            "system": {
                "os": platform.system(),
                "node": platform.node(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            "config": {
                "warmup": self.warmup_iters,
                "iterations": self.measure_iters
            },
            "results": self.results
        }
        
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {path}")

if __name__ == "__main__":
    # Example Usage
    runner = BenchmarkRunner()
    
    def dummy_workload():
        x = [i**2 for i in range(10000)]
        
    runner.benchmark("list_comprehension_10k", dummy_workload)
    runner.save_report()
