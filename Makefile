# CorePy Makefile
# enforces uv usage for all targets

.PHONY: all help install build rebuild test bench clean format lint docs verify rust-check rust-clippy rust-fmt ci coverage wheel

# Default target
all: build

help:
	@echo "CorePy Development Automation"
	@echo "============================="
	@echo "  make help      - Show this help message"
	@echo "  make install   - Install dependencies via uv sync"
	@echo "  make build     - Build Rust extensions"
	@echo "  make rebuild   - Clean and rebuild everything"
	@echo "  make test      - Run tests (uv run pytest)"
	@echo "  make bench     - Run benchmarks"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make format    - Format code with ruff"
	@echo "  make lint      - Lint code with ruff (no-sync)"
	@echo "  make ci        - Run all CI checks (lint, fmt, check, cross-check, build, test)"
	@echo "  make setup-ci  - Install all system and Rust dependencies for CI"
	@echo "  make verify    - Verify installation integrity"
	@echo "  make wheel     - Build distribution wheel (maturin build)"
	@echo "  make check-compatibility - Check development environment compatibility"

# Dependency Management
install:
	@echo "Syncing dependencies with uv..."
	uv sync --all-extras --group dev --no-install-project

# Build
build:
	@echo "Building project..."
	./scripts/build.sh

wheel:
	@echo "Building distribution wheel..."
	uv run maturin build --release --manifest-path rust/core/Cargo.toml --out dist

rebuild:
	@echo "Rebuilding project..."
	./scripts/rebuild.sh

# =============================================================================
# CI & Testing
# =============================================================================

.PHONY: test
test:
	@echo "Running tests..."
	uv run --no-sync pytest tests/ --cov=corepy --cov-report=term -v

.PHONY: check
check:
	@echo "Checking Rust compilation..."
	cargo check --manifest-path rust/core/Cargo.toml
	@echo "Verifying installation..."
	uv run --no-sync python scripts/verify_install.py


check-compatibility:
	@echo "Checking development environment compatibility..."
	@echo "Core Tools:"
	@command -v cargo >/dev/null && echo "  ✅ cargo found" || echo "  ❌ cargo NOT found (install via https://rustup.rs)"
	@command -v uv >/dev/null && echo "  ✅ uv found" || echo "  ❌ uv NOT found (install via https://docs.astral.sh/uv/getting-started/installation/)"
	@command -v python3 >/dev/null && echo "  ✅ python3 found" || echo "  ❌ python3 NOT found"
	@echo "System Dependencies:"
	@dpkg -s libopenblas-dev >/dev/null 2>&1 && echo "  ✅ libopenblas-dev found" || echo "  ❌ libopenblas-dev NOT found (run 'make setup-ci')"
	@dpkg -s gcc-aarch64-linux-gnu >/dev/null 2>&1 && echo "  ✅ gcc-aarch64-linux-gnu found" || echo "  ❌ gcc-aarch64-linux-gnu NOT found (run 'make setup-ci')"


# Code Quality
format:
	@echo "Formatting code..."
	uv run --no-sync ruff format .

lint:
	@echo "Linting code..."
	uv run --no-sync ruff check . --fix

rust-check:
	@echo "Running cargo check..."
	uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml

rust-fmt:
	@echo "Formatting Rust code..."
	uv run --no-sync cargo fmt --manifest-path rust/core/Cargo.toml
	@echo "Checking Rust format..."
	uv run --no-sync cargo fmt --manifest-path rust/core/Cargo.toml -- --check

ensure-targets:
	@echo "Ensuring Rust targets are installed..."
	@rustup target add aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu

setup-ci: ensure-targets
	@echo "Installing system dependencies for CI..."
	@if [ "$$(uname)" = "Linux" ]; then \
		if command -v apt-get >/dev/null; then \
			sudo apt-get update && sudo apt-get install -y libopenblas-dev gcc-aarch64-linux-gnu; \
		else \
			echo "⚠️  Please manually install libopenblas-dev and aarch64-linux-gnu-gcc for your distribution."; \
		fi \
	fi
	@echo "Syncing python dependencies..."
	uv sync --all-extras --group dev

rust-cross-check: ensure-targets
	@echo "Running cargo check for aarch64-unknown-linux-gnu..."
	PYO3_CROSS_PYTHON_VERSION=3.12 uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml --target aarch64-unknown-linux-gnu
	@echo "Running cargo clippy for aarch64-unknown-linux-gnu..."
	PYO3_CROSS_PYTHON_VERSION=3.12 uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml --target aarch64-unknown-linux-gnu -- -D warnings --allow clippy::missing_safety_doc
	@echo "Running cargo check for x86_64-unknown-linux-gnu..."
	PYO3_CROSS_PYTHON_VERSION=3.12 uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml --target x86_64-unknown-linux-gnu
	@echo "Running cargo clippy for x86_64-unknown-linux-gnu..."
	PYO3_CROSS_PYTHON_VERSION=3.12 uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml --target x86_64-unknown-linux-gnu -- -D warnings --allow clippy::missing_safety_doc

rust-lint:
	@echo "Running cargo clippy..."
	uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml -- -D warnings --allow clippy::missing_safety_doc

ci-prep:
	@echo "Preparing CI environment..."
	@if make check-compatibility | grep -q "❌"; then \
		make setup-ci; \
	else \
		echo "✅ Environment ready."; \
	fi

.PHONY: ci
ci: ci-prep format lint rust-fmt rust-check rust-lint rust-cross-check build test
	@echo "✅ All CI checks passed!"

# Cleanup
clean:
	@echo "Cleaning artifacts..."
	rm -rf build
	rm -rf rust/target
	rm -rf dist
	rm -f build.log
	rm -rf *.egg-info
	find . -type d -name ".venv" -prune -o -name "__pycache__" -type d -exec rm -rf {} +
	find . -type d -name ".venv" -prune -o -name "*.so" -type f -exec rm -f {} +
	find . -type d -name ".venv" -prune -o -name "*.dylib" -type f -exec rm -f {} +
	find . -type d -name ".venv" -prune -o -name "*.pyd" -type f -exec rm -f {} +
