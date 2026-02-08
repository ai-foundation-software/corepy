#!/usr/bin/env bash
# =============================================================================
# CorePy Benchmark Script
# Usage: ./scripts/bench.sh (or via `make bench`)
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Performance Benchmark ==="
echo ""

# Check uv
command -v uv >/dev/null 2>&1 || { echo "❌ uv required"; exit 1; }

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

python_cmake_dir=$(uv run --no-sync python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="-G Ninja"
else
    GENERATOR=""
fi

uv run --no-sync cmake .. $GENERATOR -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$python_cmake_dir"
uv run --no-sync cmake --build . --config Release --parallel "$JOBS"

cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Phase 3: Build Rust runtime
echo ""
echo "Phase 3/4: Building Rust Runtime..."
uv run maturin develop --release --manifest-path rust/core/Cargo.toml
echo "✅ Rust runtime built"

# Phase 4: Performance tests
echo ""
echo "Phase 4/4: Performance Verification..."
uv run python scripts/benchmark.py

# Cleanup object files (optional)
echo ""
echo "Cleanup..."
find csrc/build -name "*.o" -type f -delete 2>/dev/null || true

echo ""
echo "=== Benchmark Complete ==="
