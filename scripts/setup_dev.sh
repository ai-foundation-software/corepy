#!/usr/bin/env bash
# =============================================================================
# CorePy Developer Setup Script
# Usage: ./scripts/setup_dev.sh (or via `make install`)
# =============================================================================
set -e

# Get repo root (script may be run from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Developer Setup ==="
echo "Repository: $REPO_ROOT"
echo ""

# Detect OS
case "$(uname -s)" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=macOS;;
    MINGW*|MSYS*|CYGWIN*) OS=Windows;;
    *)          OS=Unknown;;
esac
echo "Detected OS: $OS"

# Check required tools
echo ""
echo "Checking required tools..."
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' is required. Visit https://docs.astral.sh/uv/ to install."; exit 1; }
echo "✅ uv found"

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

# Step 2: Build Rust runtime
echo ""
echo "Step 2/3: Building Rust runtime..."
if [ -n "$GPU_FEATURE" ]; then
    echo "Using GPU features: $GPU_FEATURE"
    uv run maturin develop --release --manifest-path rust/core/Cargo.toml $GPU_FEATURE
else
    echo "Using CPU only"
    uv run maturin develop --release --manifest-path rust/core/Cargo.toml
fi
echo "✅ Rust runtime built"

# Step 3: Verify installation
echo ""
echo "Step 3/3: Verifying installation..."
uv run python scripts/verify_install.py

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "  Run tests:  make test"
echo "  Build:      make build"
echo "  Benchmark:  make bench"
