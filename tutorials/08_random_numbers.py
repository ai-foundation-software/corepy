"""
Tutorial 08: High-Performance Random Generation
===============================================
This tutorial demonstrates the new parallelized Random Number Generation (RNG)
capabilities introduced in CorePy v0.3.0. Supported algorithms include
industry-standard PCG64 and Xoshiro256++.
"""

import corepy as cp


def main():
    print("--- CorePy High-Performance PRNGs ---")
    print("CorePy pushes PRNG generation to the Rust backend using Rayon.")
    print("This allows generating millions of random numbers extremely fast.\n")

    # 1. Uniform Distribution
    print("1. Uniform float32 in range [0.0, 1.0)")
    # Generate a matrix of size 3x3 uniform floats using the default pcg64 algo
    uniform_array = cp.random.rand((3, 3), seed=42)
    print("rand((3, 3), seed=42):")
    print(uniform_array)

    # 2. Standard Normal (Gaussian) Distribution
    print("\n2. Standard Normal float32 (mean ~0.0, std ~1.0)")
    # Generate 1 million standard normal values to test distribution speed
    cp.enable_profiling()
    normal_array = cp.random.randn((1_000, 1_000), algo="xoshiro")
    cp.disable_profiling()

    print("Generated 1,000,000 normally distributed values using Xoshiro256++.")

    # Compute fast statistics
    mean_val = normal_array.mean()
    std_val = normal_array.std()

    print(f"Calculated Mean: {mean_val} (Expected: ~0.0)")
    print(f"Calculated Std:  {std_val} (Expected: ~1.0)")

    print("\nProfiling data shows how fast parallel generation takes:")
    print(cp.profile_report())


if __name__ == "__main__":
    main()
