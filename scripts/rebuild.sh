#!/bin/bash
# =============================================================================
# CorePy Clean Rebuild Script
# Supports: Linux, macOS, Windows (Git Bash/WSL)
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Clean Rebuild ==="
echo ""

# Step 1: Clean previous builds
echo "Step 1/3: Cleaning previous builds..."
rm -rf csrc/build
rm -rf rust/core/target
rm -rf build dist *.egg-info
echo "✅ Cleaned"

# Detect CPU count
if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

# Step 2: Build C++ kernels
echo ""
echo "Step 2/3: Building C++ kernels..."
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

# Step 3: Build Rust runtime
echo ""
echo "Step 3/3: Building Rust runtime..."
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
import corepy
import numpy as np

print(f'Backend: {corepy.get_backend_policy()}')

# Test matmul
a = corepy.Tensor(np.random.randn(100, 100).astype(np.float32))
b = corepy.Tensor(np.random.randn(100, 100).astype(np.float32))
c = a.matmul(b)
print(f'Matmul test: {corepy.explain_last_dispatch()}')
"
else
    python3 -c "
import corepy
import numpy as np

print(f'Backend: {corepy.get_backend_policy()}')

# Test matmul
a = corepy.Tensor(np.random.randn(100, 100).astype(np.float32))
b = corepy.Tensor(np.random.randn(100, 100).astype(np.float32))
c = a.matmul(b)
print(f'Matmul test: {corepy.explain_last_dispatch()}')
"
fi

echo ""
echo "=== Rebuild Complete! ==="
