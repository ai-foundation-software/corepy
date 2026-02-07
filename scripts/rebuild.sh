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

# Step 1: Clean
echo "Step 1/3: Cleaning artifacts..."
rm -rf build
rm -rf csrc/build  # legacy
rm -rf rust/core/target
rm -rf corepy.egg-info
rm -rf dist
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
find . -name "*.so" -type f -delete
find . -name "*.dylib" -type f -delete
find . -name "*.pyd" -type f -delete
find . -name "*.metallib" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Cleaned"


# Step 2: Build
echo ""
echo "Step 2/3: Building project..."
./scripts/build.sh

# Step 3: Verify
echo ""
echo "Step 3/3: Verifying installation..."
if command -v uv >/dev/null 2>&1; then
    uv run python scripts/verify_install.py
else
    python3 scripts/verify_install.py
fi

echo ""
echo "=== Rebuild Complete! ==="
