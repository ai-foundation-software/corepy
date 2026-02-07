#!/bin/bash
# =============================================================================
# CorePy Benchmark Script
# Supports: Linux, macOS, Windows (Git Bash/WSL)
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Performance Benchmark ==="
echo ""

# Detect CPU count
if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

# Phase 1: Clean
echo "Phase 1/4: Cleaning..."
rm -rf build
rm -rf rust/core/target

# Phase 2: Build C++ kernels
echo ""
echo "Phase 2/4: Building C++ Kernels..."
mkdir -p build
# Check if build dir exists and has CMakeCache.txt to avoid re-config if possible? 
# Actually bench.sh does a clean build usually.
cd build

if command -v uv >/dev/null 2>&1; then
    PYBIND11_CMAKE_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_CMAKE_DIR"
else
    CMAKE_ARGS=""
fi

if command -v ninja >/dev/null 2>&1; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-dev $CMAKE_ARGS
    cmake --build . --config Release
else
    cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev $CMAKE_ARGS
    cmake --build . --config Release -j $JOBS
fi
cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Phase 3: Build Rust runtime
echo ""
echo "Phase 3/4: Building Rust Runtime..."
if command -v uv >/dev/null 2>&1; then
    uv run maturin develop --release --manifest-path rust/core/Cargo.toml
else
    maturin develop --release --manifest-path rust/core/Cargo.toml
fi
echo "✅ Rust runtime built"

# Phase 4: Performance tests
echo ""
echo "Phase 4/4: Performance Verification..."
if command -v uv >/dev/null 2>&1; then
    uv run python scripts/benchmark.py
else
    python3 scripts/benchmark.py
fi

# Cleanup object files (optional)
echo ""
echo "Cleanup..."
find csrc/build -name "*.o" -type f -delete 2>/dev/null || true

echo ""
echo "=== Benchmark Complete ==="
