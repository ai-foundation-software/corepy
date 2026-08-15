import time

import corepy as cp


def main():
    print("=" * 60)
    print("CorePy Hardware Discovery & Intelligence")
    print("=" * 60)
    print()

    # 1. Inspect System Topology
    print("1. System Topology Insights")
    print("-" * 30)

    # CorePy natively interfaces via Rust & psutil to map physical
    # versus logical hardware endpoints safely.
    device_info = cp.backend.session.get_session().device_info

    print(f"Platform OS: {device_info.platform_system}")
    print(f"Physical CPU Cores: {device_info.cpu_cores}")
    print(f"Logical OS Threads: {device_info.cpu_threads}")
    print()
    print("CPU Advanced Capabilities:")
    print(f" - AVX2 extensions:  {'Yes' if device_info.has_avx2 else 'No'}")
    print(f" - NEON extensions:  {'Yes' if device_info.has_neon else 'No'}")
    print(f" - L1 Cache:         {device_info.l1_cache_size // 1024} KB")
    print(f" - L3/Smart Cache:   {device_info.l3_cache_size // 1024 // 1024} MB")

    # 2. Extract Memory Constraints
    print()
    print("2. Memory & Capacity Constraints")
    print("-" * 30)

    # Evaluate global memory
    cpu_backend = cp.backend.session.get_session().get_backend(
        cp.backend.BackendType.CPU
    )
    print("CPU Sandbox:")
    print(
        f" - Available Free RAM: {cpu_backend.device_type.name} -> {round(device_info.memory_limit_bytes or 0 / 1024**3, 2)} GB (Fallback tracking)"
    )

    # 3. Dynamic GPU Routing
    print()
    print("3. Discrete & Unified Graphics Profiling")
    print("-" * 30)

    if device_info.has_gpu:
        print(f"Detected {device_info.gpu_count} Graphical Accelerators.")

        # Determine specific architectures
        if device_info.has_metal:
            print(" - Apple Silicon / Metal UMA Architecture Enabled")
        if device_info.has_cuda:
            print(" - NVIDIA CUDA Architecture Enabled")

        print("\nAttached Accelerator Roster:")
        for idx in range(device_info.gpu_count):
            name = device_info.gpu_names[idx]
            vram_total = device_info.gpu_memory_bytes[idx] / 1024**3
            vram_free = device_info.gpu_memory_free_bytes[idx] / 1024**3
            print(f"  [{idx}] {name}")
            print(f"      Total VRAM: {vram_total:.2f} GB")
            print(f"      Free VRAM:  {vram_free:.2f} GB")
    else:
        print("No dedicated accelerators resolved on this environment.")

    print()
    print("4. Context-Aware Backend Auto-Routing")
    print("-" * 30)

    gpu_type = (
        "CUDA"
        if device_info.has_cuda
        else ("Metal" if device_info.has_metal else "None")
    )

    print("By default, CorePy operates matrices dynamically.")
    print("If a large enough heuristic is spotted, execution pivots to the GPU.")

    arr_small = cp.zeros((10, 10))
    print(f"[10x10] Operation   -> Backend mapped to {arr_small.backend.name}")

    # Try large arrays to trigger GPU routing
    if device_info.has_gpu:
        try:
            arr_large = cp.zeros((2000, 2000))
            print(
                f"[2000x2000] Operation -> Backend implicitly raised to {arr_large.backend.name}"
            )

            # Show manual enforcement
            if device_info.has_metal:
                arr_manual = cp.zeros((100, 100), backend="metal")
                print(
                    f"Manual Override     -> Backend rigidly pinned to {arr_manual.backend.name}"
                )
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
