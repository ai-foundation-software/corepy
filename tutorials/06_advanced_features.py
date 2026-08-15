"""
Tutorial 06: Advanced Engine Features
=====================================
This tutorial demonstrates the new advanced features embedded in the Corepy
scientific engine, including hardware detection, tabular DataFrame processing,
and high-performance random generation.
"""

import corepy as cp


def demonstrate_hardware_capabilities():
    print("--- 1. Hardware & Cache Detection ---")
    # Corepy dynamically reads L1, L2, and L3 cache sizes from the OS
    # to automatically block and tile matrix operations.
    caps = cp.get_system_capabilities()
    cpu_caps = caps.get("cpu", {})

    print(f"Architecture: {cpu_caps.get('arch')}")
    print(f"L1 Cache: {cpu_caps.get('l1_cache', 'Unknown')} Bytes")
    print(f"L2 Cache: {cpu_caps.get('l2_cache', 'Unknown')} Bytes")
    print(f"L3 Cache: {cpu_caps.get('l3_cache', 'Unknown')} Bytes")

    print("Supported Features:")
    for feature in ["has_avx2", "has_avx512", "has_neon", "has_fma"]:
        has_feature = cpu_caps.get(feature, False)
        print(
            f"  - {feature.replace('has_', '').upper()}: {'Yes' if has_feature else 'No'}"
        )
    print()


def demonstrate_dataframe():
    print("--- 2. Tabular DataFrame Engine ---")
    # Corepy includes a Pandas-like DataFrame processed strictly in Rust.
    df = cp.DataFrame()

    # Add columns (zero-copy memory sharing where possible)
    df.add_int_column("id", [101, 102, 103, 104, 105])
    df.add_float_column("latency_ms", [12.5, 45.0, 8.2, 102.1, 19.4])
    df.add_int_column("status_code", [200, 400, 200, 500, 200])

    print(f"{df}")

    # Fast filtering entirely in Rust
    print("\nFiltering id == 103:")
    high_latency = df.filter_int_eq("id", 103)
    print(f"  {high_latency}")

    # Fast sorting
    print("\nSorting by id (ascending):")
    sorted_df = df.sort_values("id")
    print(f"  {sorted_df}")
    print()


def demonstrate_random_generation():
    print("--- 3. High-Performance Random Operations ---")
    # Multi-threaded PRNGs (PCG64 and Xoshiro256++) utilizing Rayon in Rust.

    # Generate 1 million uniform floats [0.0, 1.0)
    print("Generating 1,000,000 uniform floats using Xoshiro256++...")
    cp.enable_profiling()
    uniform_array = cp.rand(1_000_000, algo="xoshiro")

    # Generate a matrix of standard normal distribution values
    print("Generating 1000x1000 normal/gaussian floats using PCG64...")
    normal_array = cp.randn((1000, 1000), algo="pcg64")

    # Use UFuncs mapped via Rayon for fast statistics
    mean_val = normal_array.mean()
    print(f"Normal ndarray Mean (Expected ~0.0): {mean_val}")
    cp.disable_profiling()

    print("\nProfile Report:")
    print(cp.profile_report())


if __name__ == "__main__":
    demonstrate_hardware_capabilities()
    demonstrate_dataframe()
    demonstrate_random_generation()
