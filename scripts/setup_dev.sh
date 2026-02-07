#!/bin/bash
# =============================================================================
# CorePy Developer Setup Script
# Supports: Linux, macOS, Windows (Git Bash/WSL)
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
command -v cmake >/dev/null 2>&1 || { echo "❌ cmake required. Install: apt/brew install cmake"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "❌ cargo required. Install: https://rustup.rs"; exit 1; }

if command -v uv >/dev/null 2>&1; then
    echo "✅ uv found"
    PKG_MGR="uv"
else
    echo "⚠️  uv not found, using pip"
    PKG_MGR="pip"
fi

# Step 1: Install Python dependencies
echo ""
echo "Step 1/4: Installing Python dependencies..."
if [ "$PKG_MGR" = "uv" ]; then
    uv sync --all-extras --group dev
else
    pip install -e ".[dev]" maturin scikit-build-core pybind11 cmake ninja
fi

# Step 2: Build C++ kernels
echo ""
echo "Step 2/4: Building C++ kernels..."
mkdir -p build
cd build

if command -v uv >/dev/null 2>&1; then
    PYBIND11_CMAKE_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_CMAKE_DIR"
else
    CMAKE_ARGS=""
fi

# Use Ninja if available, otherwise default generator
if command -v ninja >/dev/null 2>&1; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS
    cmake --build . --config Release
else
    cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS
    cmake --build . --config Release
fi
cd "$REPO_ROOT"
echo "✅ C++ kernels built"

# Step 3: Build Rust runtime
echo ""
echo "Step 3/4: Building Rust runtime..."
if command -v uv >/dev/null 2>&1; then
    echo "uv sync already handled the build in Step 1."
    uv run maturin develop --release
else
    maturin develop --release
fi
echo "✅ Rust runtime built"

# Step 4: Verify installation
echo ""
echo "Step 4/4: Verifying installation..."
if command -v uv >/dev/null 2>&1; then
    uv run python scripts/verify_install.py
else
    python3 scripts/verify_install.py
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "  Run tests:  uv run pytest tests/"
echo "  Build:      ./scripts/build.sh"
echo "  Benchmark:  ./scripts/bench.sh"
