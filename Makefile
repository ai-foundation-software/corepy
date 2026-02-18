# CorePy Makefile
# enforces uv usage for all targets

.PHONY: all help install build rebuild test bench clean format lint docs verify rust-check rust-clippy rust-fmt ci coverage

# Default target
all: build

help:
	@echo "CorePy Development Automation"
	@echo "============================="
	@echo "  make help      - Show this help message"
	@echo "  make install   - Install dependencies via uv sync"
	@echo "  make build     - Build C++ and Rust extensions"
	@echo "  make rebuild   - Clean and rebuild everything"
	@echo "  make test      - Run tests (uv run pytest)"
	@echo "  make bench     - Run benchmarks"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make format    - Format code with ruff"
	@echo "  make lint      - Lint code with ruff (no-sync)"
	@echo "  make ci        - Run all CI checks (lint, fmt, check, cross-check, build, test)"
	@echo "  make verify    - Verify installation integrity"
	@echo "  make check-compatibility - Check development environment compatibility"

# Dependency Management
install:
	@echo "Syncing dependencies with uv..."
	uv sync --all-extras --group dev --no-install-project

# Build
build:
	@echo "Building project..."
	./scripts/build.sh

rebuild:
	@echo "Rebuilding project..."
	./scripts/rebuild.sh

# Testing & Verification
test:
	@echo "Running tests..."
	uv run pytest tests/ -v

coverage:
	@echo "Running coverage..."
	uv run pytest --cov=corepy --cov-report=xml --cov-report=html tests/


bench:
	@echo "Running benchmarks..."
	./scripts/bench.sh

verify:
	@echo "Verifying installation..."
	uv run python scripts/verify_install.py


check-compatibility:
	@echo "Checking development environment compatibility..."
	@echo "Core Tools:"
	@command -v cargo >/dev/null && echo "  ✅ cargo found (system)" || echo "  ❌ cargo NOT found (system)"
	@command -v uv >/dev/null && echo "  ✅ uv found" || echo "  ❌ uv NOT found"
	@echo "Build Dependencies (inside venv):"
	@uv run bash -c "command -v cmake >/dev/null" && echo "  ✅ cmake found" || echo "  ❌ cmake NOT found"
	@uv run bash -c "command -v ninja >/dev/null" && echo "  ✅ ninja found" || echo "  ❌ ninja NOT found"
	@echo "Cross-Compilation Tools (system):"
	@command -v aarch64-linux-gnu-g++ >/dev/null && echo "  ✅ aarch64-linux-gnu-g++ found" || echo "  ⚠️  aarch64-linux-gnu-g++ NOT found (Linux ARM64 cross-compilation will be skipped)"
	@command -v x86_64-linux-gnu-g++ >/dev/null && echo "  ✅ x86_64-linux-gnu-g++ found" || echo "  ⚠️  x86_64-linux-gnu-g++ NOT found (Linux x86_64 cross-compilation will be skipped)"


# Code Quality
format:
	@echo "Formatting code..."
	uv run ruff format .

lint:
	@echo "Linting code..."
	uv run --no-sync ruff check .

rust-check:
	@echo "Running cargo check..."
	uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml

rust-fmt:
	@echo "Checking Rust format..."
	uv run --no-sync cargo fmt --manifest-path rust/core/Cargo.toml -- --check

ensure-targets:
	@echo "Ensuring Rust targets are installed..."
	@rustup target list --installed | grep -q aarch64-unknown-linux-gnu || rustup target add aarch64-unknown-linux-gnu
	@rustup target list --installed | grep -q x86_64-unknown-linux-gnu || rustup target add x86_64-unknown-linux-gnu

rust-cross-check: ensure-targets
	@if command -v aarch64-linux-gnu-g++ >/dev/null; then \
		echo "Running cargo check for aarch64-unknown-linux-gnu..."; \
		uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml --target aarch64-unknown-linux-gnu; \
		echo "Running cargo clippy for aarch64-unknown-linux-gnu..."; \
		uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml --target aarch64-unknown-linux-gnu -- -D warnings --allow clippy::missing_safety_doc; \
	else \
		echo "⚠️ Skipping aarch64-unknown-linux-gnu check (cross-compiler aarch64-linux-gnu-g++ not found)"; \
	fi
	@if command -v x86_64-linux-gnu-g++ >/dev/null; then \
		echo "Running cargo check for x86_64-unknown-linux-gnu..."; \
		uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml --target x86_64-unknown-linux-gnu; \
		echo "Running cargo clippy for x86_64-unknown-linux-gnu..."; \
		uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml --target x86_64-unknown-linux-gnu -- -D warnings --allow clippy::missing_safety_doc; \
	else \
		echo "⚠️ Skipping x86_64-unknown-linux-gnu check (cross-compiler x86_64-linux-gnu-g++ not found)"; \
	fi

rust-lint:
	@echo "Running cargo clippy..."
	uv run --no-sync cargo clippy --manifest-path rust/core/Cargo.toml -- -D warnings --allow clippy::missing_safety_doc

ci: check-compatibility lint rust-fmt rust-check rust-lint rust-cross-check build test
	@echo "✅ All CI checks passed!"

# Cleanup
clean:
	@echo "Cleaning artifacts..."
	rm -rf build
	rm -rf csrc/build
	rm -rf rust/core/target
	rm -rf dist
	rm -rf lib
	rm -rf include
	rm -f build.log
	rm -rf *.egg-info
	find . -type d -name ".venv" -prune -o -name "__pycache__" -type d -exec rm -rf {} +
	find . -type d -name ".venv" -prune -o -name "*.so" -type f -exec rm -f {} +
	find . -type d -name ".venv" -prune -o -name "*.dylib" -type f -exec rm -f {} +
	find . -type d -name ".venv" -prune -o -name "*.pyd" -type f -exec rm -f {} +
