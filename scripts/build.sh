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

# Step 1: Build Rust runtime
echo "Step 1/2: Building Rust runtime..."
BUILD_CMD="develop"
if [[ "$*" == *"--wheel"* ]]; then
    BUILD_CMD="build"
    echo "📦 Building distribution wheel (maturin build)"
fi

if [ -n "$GPU_FEATURE" ]; then
    echo "Using GPU features: $GPU_FEATURE"
    if [ "$BUILD_CMD" == "build" ]; then
        uv run --no-sync maturin build --release --manifest-path rust/core/Cargo.toml $GPU_FEATURE --out dist
    else
        uv run --no-sync maturin develop --release --manifest-path rust/core/Cargo.toml $GPU_FEATURE
    fi
else
    echo "Using CPU only"
    if [ "$BUILD_CMD" == "build" ]; then
        uv run --no-sync maturin build --release --manifest-path rust/core/Cargo.toml --out dist
    else
        uv run --no-sync maturin develop --release --manifest-path rust/core/Cargo.toml
    fi
fi
echo "✅ Rust runtime built"

# Step 2: Verification
echo ""
echo "=== Verification ==="
uv run --no-sync python scripts/verify_install.py

echo ""
echo "=== Build Complete! ==="
