# Corepy Development Guide

## Prerequisites

Before setting up the development environment, ensure you have:

1. **Python 3.9+**
2. **uv** package manager (will be installed automatically by setup script)
3. **Rust toolchain** (will be installed automatically by setup script)
4. **C++ compiler** (gcc/clang on Linux, Xcode on macOS, MSVC on Windows)
5. **CMake 3.15+**
6. **ninja build system**

## Quick Setup

Run the automated setup script:

```bash
make install
```

This script will:
- Install `uv` if not present
- Install Rust toolchain if not present
- Sync all Python dependencies
- Build the Rust runtime (`corepy-runtime`)
- Build C++ extensions
- Install the package in editable mode

## Manual Setup

If you prefer to set up manually:

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3. Sync Dependencies

```bash
# Install all dependencies including dev extras
uv sync --all-extras --group dev
```

### 4. Build Rust Runtime

```bash
cd rust/core
cargo build --release
cd ../..
```

### 5. Build Python Package

### 5. Build Python Package

```bash
# uv sync handles the editable install and build dependencies automatically!
uv sync --all-extras --group dev
```

## Development Workflow

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
uv run pytest --cov=corepy --cov-report=html
```

### Standard Development Commands

Use the provided `Makefile` for a standardized experience:

- `make install`: Syncs dependencies via `uv sync`.
- `make build`: Builds C++ and Rust extensions via `scripts/build.sh`.
- `make test`: Runs tests via `uv run pytest`.
- `make bench`: Runs benchmarks via `scripts/bench.sh`.
- `make clean`: Cleans artifacts.
- `make verify`: Verifies installation.

### Manual Workflow Details

#### Running Tests with Coverage
```bash
uv run pytest --cov=corepy --cov-report=html
```

#### Clean Build
```bash
make clean
make build
```

### Running Benchmarks

```bash
make bench
```

### Code Quality Tools

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy corepy
```

## Project Structure

```
corepy/
├── corepy/              # Python package
├── csrc/                # C++ source files
├── rust/                # Rust runtime
│   └── core/            # Core Rust runtime library
├── tests/               # Test suite
```

## Dependencies Overview

### Runtime Dependencies
- `typing-extensions>=4.6.0` - Type hints
- `pydantic>=2.0.0` - Data validation
- `numpy>=2.0.2` - Numerical computing

### Development Dependencies
- `pytest>=7.0` - Testing framework
- `pytest-cov>=4.0` - Coverage reporting
- `ruff>=0.1.0` - Linter and formatter
- `mypy>=1.0` - Type checker
- `build` - Build tool
- `twine` - Package publishing

### Documentation Dependencies
- `sphinx>=7.0` - Documentation generator
- `sphinx-rtd-theme>=1.3.0` - ReadTheDocs theme
- `myst-parser>=2.0.0` - Markdown parser

### Build Dependencies
- `scikit-build-core>=0.4.3` - CMake-based build
- `pybind11` - C++/Python bindings
- `cmake>=3.15` - Build system
- `ninja` - Build tool

## Common Issues

### Issue: `uv sync` removes packages

**Problem**: Running `uv sync` uninstalls packages like scipy or other manually installed packages that aren't in `pyproject.toml`.

**Solution**: This is correct behavior! `uv sync` ensures your environment matches exactly what's declared in your `pyproject.toml` dependencies. Only use packages that are:
- Declared in runtime dependencies (for the package to work)
- Declared in development dependencies (for development/testing)
- Declared in optional dependencies (for optional features)

If you need a package, add it to the appropriate section in `pyproject.toml`.

### About Maturin

**Why is maturin needed?**

Maturin is a **build tool** for Rust-Python bindings. It's required because:
1. The `corepy-runtime` component (in `rust/corepy-runtime/`) uses PyO3 to create Python bindings
2. It's declared as the build backend in `rust/corepy-runtime/pyproject.toml`
3. It builds and installs the Rust extension during development

**When is it used?**
- **Development**: Used by developers to build the Rust runtime locally
- **Release**: Used in CI/CD to build platform-specific wheels for distribution
- **End users**: NOT needed - they install pre-built wheels from PyPI

That's why maturin is in the **development dependencies** but not in runtime dependencies.

### Issue: Build fails with CMake errors

**Problem**: CMake can't find required libraries or compilers.

**Solution**: 
- Ensure you have a C++ compiler installed
- Install CMake 3.15+: `sudo apt install cmake` (Linux) or `brew install cmake` (macOS)
- Install ninja: `sudo apt install ninja-build` (Linux) or `brew install ninja` (macOS)

### Issue: Rust build fails

**Problem**: `cargo build` fails in the `rust/corepy-runtime` directory.

**Solution**:
- Update Rust toolchain: `rustup update`
- Clean and rebuild: `cargo clean && cargo build --release`

## Making Changes

### Modifying Python Code
1. Edit files in the `corepy/` directory
2. Changes are immediately available (editable install)
3. Run tests: `uv run pytest`

### Modifying C++ Code
1. Edit files in the `csrc/` directory
2. Rebuild: `make build`
3. Verify: `make verify`

### Modifying Rust Code
1. Edit files in the `rust/core/` directory
2. Rebuild: `make build`
3. Verify: `make verify`

## Contributing

See [Contributing Guide](docs/07_contributing/CONTRIBUTING.md) for detailed contribution guidelines.

## Additional Resources

- [Installation Guide](docs/01_quickstart/install.md)
- [Architecture Documentation](docs/03_architecture/architecture.md)
- [Platform Support Guide](docs/01_quickstart/platform_support.md)
