#!/usr/bin/env bash
# =============================================================================
# CorePy Build Script
# Usage: ./scripts/build.sh
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Build Script ==="

# Check required tools
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' is required but not installed. Please install uv."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "❌ 'cmake' is required."; exit 1; }

# Detect CPU count
if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

# Step 0: Sync dependencies
echo "Step 0/3: Syncing dependencies..."
uv sync --all-extras --group dev --no-install-project

# Step 1: Build C++ kernels
echo ""
echo "Step 1/3: Building C++ kernels..."
mkdir -p build
# Use --no-sync to avoid triggering project install (which fails before kernels are built)
python_cmake_dir=$(uv run --no-sync python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)

cd build

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="-G Ninja"
else
    GENERATOR=""
fi

uv run --no-sync cmake .. $GENERATOR -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$python_cmake_dir"
uv run --no-sync cmake --build . --config Release --parallel "$JOBS"
uv run --no-sync cmake --install . --prefix ..

# Move Metal library if present (macOS specific)
if [ -f "../default.metallib" ]; then
    mv "../default.metallib" "../corepy/default.metallib"
    echo "Moved default.metallib to corepy/ package"
fi

cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Step 2: Build Rust runtime
echo ""
echo "Step 2/3: Building Rust runtime..."
uv run maturin develop --release --manifest-path rust/core/Cargo.toml
echo "✅ Rust runtime built"

# Step 3: Verification
echo ""
echo "=== Verification ==="
uv run python scripts/verify_install.py

echo ""
echo "=== Build Complete! ==="
