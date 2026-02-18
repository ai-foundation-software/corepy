#!/usr/bin/env bash
# =============================================================================
# CorePy Clean Rebuild Script
# Usage: ./scripts/rebuild.sh (or via `make rebuild`)
# =============================================================================
set -e

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CorePy Clean Rebuild ==="
echo ""

# Check uv
command -v uv >/dev/null 2>&1 || { echo "❌ uv required"; exit 1; }

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
find . -type d -name ".venv" -prune -o -name "*.so" -type f -print0 | xargs -0 rm -f
find . -type d -name ".venv" -prune -o -name "*.dylib" -type f -print0 | xargs -0 rm -f
find . -type d -name ".venv" -prune -o -name "*.pyd" -type f -print0 | xargs -0 rm -f
find . -type d -name ".venv" -prune -o -name "*.metallib" -type f -print0 | xargs -0 rm -f
find . -type d -name ".venv" -prune -o -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Cleaned"


# Step 2: Build
echo ""
echo "Step 2/3: Building project..."
./scripts/build.sh

# Step 3: Verify
echo ""
echo "Step 3/3: Verifying installation..."
uv run python scripts/verify_install.py

echo ""
echo "=== Rebuild Complete! ==="
