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
rm -rf csrc/build
rm -rf rust/core/target

# Phase 2: Build C++ kernels
echo ""
echo "Phase 2/4: Building C++ Kernels..."
mkdir -p csrc/build
cd csrc/build

if command -v ninja >/dev/null 2>&1; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-dev
    cmake --build . --config Release
else
    cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev
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
    uv run python -c "
import corepy
import numpy as np
import time

print('=== Performance Test ===')
print(f'Backend: {corepy.get_backend_policy()}')
print('')

# Test different sizes
sizes = [(100, 100), (512, 512), (1024, 1024)]
for m, n in sizes:
    a = corepy.Tensor(np.random.randn(m, n).astype(np.float32))
    b = corepy.Tensor(np.random.randn(n, m).astype(np.float32))
    
    # Warmup
    _ = a.matmul(b)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        c = a.matmul(b)
    elapsed = (time.time() - start) / 10
    
    print(f'{m}x{n}: {elapsed*1000:.2f}ms - {corepy.explain_last_dispatch()}')

print('')
print('=== Performance test complete ===')
"
else
    python3 -c "
import corepy
import numpy as np
import time

print('=== Performance Test ===')
print(f'Backend: {corepy.get_backend_policy()}')
print('')

sizes = [(100, 100), (512, 512), (1024, 1024)]
for m, n in sizes:
    a = corepy.Tensor(np.random.randn(m, n).astype(np.float32))
    b = corepy.Tensor(np.random.randn(n, m).astype(np.float32))
    
    _ = a.matmul(b)
    
    start = time.time()
    for _ in range(10):
        c = a.matmul(b)
    elapsed = (time.time() - start) / 10
    
    print(f'{m}x{n}: {elapsed*1000:.2f}ms - {corepy.explain_last_dispatch()}')

print('')
print('=== Performance test complete ===')
"
fi

# Cleanup object files (optional)
echo ""
echo "Cleanup..."
find csrc/build -name "*.o" -type f -delete 2>/dev/null || true

echo ""
echo "=== Benchmark Complete ==="
