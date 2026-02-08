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

command -v cmake >/dev/null 2>&1 || { echo "❌ cmake required. Install: apt/brew install cmake"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "❌ cargo required. Install: https://rustup.rs"; exit 1; }

# Step 1: Install Python dependencies
echo ""
echo "Step 1/4: Syncing dependencies..."
uv sync --all-extras --group dev --no-install-project

# Step 2: Build C++ kernels
echo ""
echo "Step 2/4: Building C++ kernels..."
mkdir -p build
cd build

python_cmake_dir=$(uv run --no-sync python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="-G Ninja"
else
    GENERATOR=""
fi

uv run --no-sync cmake .. $GENERATOR -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$python_cmake_dir"
uv run --no-sync cmake --build . --config Release

cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Step 3: Build Rust runtime
echo ""
echo "Step 3/4: Building Rust runtime..."
uv run maturin develop --release --manifest-path rust/core/Cargo.toml
echo "✅ Rust runtime built"

# Step 4: Verify installation
echo ""
echo "Step 4/4: Verifying installation..."
uv run python scripts/verify_install.py

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "  Run tests:  make test"
echo "  Build:      make build"
echo "  Benchmark:  make bench"
