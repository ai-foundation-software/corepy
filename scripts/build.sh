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

# Step 1: Build C++ kernels
echo ""
echo "Step 1/2: Building C++ kernels..."
mkdir -p csrc/build
cd csrc/build

if command -v ninja >/dev/null 2>&1; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
else
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release -j $JOBS
fi
cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Step 2: Build Rust runtime
echo ""
echo "Step 2/2: Building Rust runtime..."
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to sync and build..."
    uv sync
else
    echo "Using maturin to build..."
    maturin develop --release
fi
echo "✅ Rust runtime built"

# Verification
echo ""
echo "=== Verification ==="
if command -v uv >/dev/null 2>&1; then
    uv run python -c "
import corepy as cp
print(f'✅ corepy {cp.__version__} loaded')
print(f'✅ Backend: {cp.get_backend_policy()}')
print(f'✅ Tensor test: {cp.Tensor([1.0, 2.0, 3.0])}')
"
else
    python3 -c "
import corepy as cp
print(f'✅ corepy {cp.__version__} loaded')
print(f'✅ Backend: {cp.get_backend_policy()}')
print(f'✅ Tensor test: {cp.Tensor([1.0, 2.0, 3.0])}')
"
fi

echo ""
echo "=== Build Complete! ==="
