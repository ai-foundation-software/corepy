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
echo "Phase 1/3: Cleaning..."
rm -rf build
rm -rf rust/target

# Detect OS
case "$(uname -s)" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=macOS;;
    MINGW*|MSYS*|CYGWIN*) OS=Windows;;
    *)          OS=Unknown;;
esac

# Detect GPU and set features
GPU_FEATURE=""
if [ "$OS" = "macOS" ]; then
    if [ "$(uname -m)" = "arm64" ]; then
        echo "🍎 Detected Apple Silicon (Metal)"
        GPU_FEATURE="--features metal"
    fi
elif [ "$OS" = "Linux" ] || [ "$OS" = "Windows" ]; then
    if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1 || [ -n "$CUDA_PATH" ]; then
        echo "🟩 Detected NVIDIA GPU"
        GPU_FEATURE="--features cuda"
    fi
fi

# Phase 2: Build Rust runtime
echo ""
echo "Phase 2/3: Building Rust Runtime..."
if [ -n "$GPU_FEATURE" ]; then
    echo "Using GPU features: $GPU_FEATURE"
    uv run maturin develop --release --manifest-path rust/core/Cargo.toml $GPU_FEATURE
else
    echo "Using CPU only"
    uv run maturin develop --release --manifest-path rust/core/Cargo.toml
fi
echo "✅ Rust runtime built"

# Phase 3: Performance tests
echo ""
echo "Phase 3/3: Performance Verification..."
uv run python scripts/benchmark.py

echo ""
echo "=== Benchmark Complete ==="
