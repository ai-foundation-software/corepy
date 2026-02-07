#!/bin/bash
# =============================================================================
# CorePy Build Script
# Supports: Linux, macOS, Windows (Git Bash/WSL)
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Build Script ==="
echo ""

# Check required tools
command -v cmake >/dev/null 2>&1 || { echo "❌ cmake required"; exit 1; }
command -v maturin >/dev/null 2>&1 || command -v uv >/dev/null 2>&1 || { echo "❌ maturin or uv required"; exit 1; }

# Detect CPU count for parallel builds
if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi
echo "Using $JOBS parallel jobs"

# Step 0: Install dependencies
echo "Step 0/3: Installing dependencies..."
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to sync..."
    uv sync
else
    echo "⚠️ uv not found, skipping sync"
fi

# Step 1: Build C++ kernels
echo ""
echo "Step 1/3: Building C++ kernels..."
mkdir -p build
cd build

if command -v uv >/dev/null 2>&1; then
    CMAKE_CMD="uv run cmake"
    PYBIND11_CMAKE_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    echo "Using pybind11 at: $PYBIND11_CMAKE_DIR"
    CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_CMAKE_DIR"
else
    CMAKE_CMD="cmake"
    # Fallback: hope it's in system path or user handles it
    CMAKE_ARGS=""
fi

if command -v ninja >/dev/null 2>&1; then
    $CMAKE_CMD .. -G Ninja -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS
    $CMAKE_CMD --build . --config Release
else
    $CMAKE_CMD .. -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS
    $CMAKE_CMD --build . --config Release -j $JOBS
fi

# Install artifacts locally (to corepy/ directory)
$CMAKE_CMD --install . --prefix ..

cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Step 2: Build Rust runtime
echo ""
echo "Step 2/3: Building Rust runtime..."
if command -v uv >/dev/null 2>&1; then
    echo "Using maturin to build (via uv)..."
    uv run maturin develop --release
else
    echo "Using maturin to build..."
    maturin develop --release
fi
echo "✅ Rust runtime built"

# Verification
echo ""
echo "=== Verification ==="
if command -v uv >/dev/null 2>&1; then
    uv run python scripts/verify_install.py
else
    python3 scripts/verify_install.py
fi

echo ""
echo "=== Build Complete! ==="
