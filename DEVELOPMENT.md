# Development Guide

## 1. Project Overview
Corepy is a high-performance hybrid tensor library combining **Python** usability with **Rust** safety and **C++** kernels for raw speed. It uses a custom dispatcher to route operations to CPU (OpenBLAS/AVX2) or GPU (Metal on macOS) backends. The project uses `uv` for Python dependency management and `maturin` to build the Rust extension that binds everything together.

## 2. Prerequisites

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-dev \
    python3-venv \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    git \
    libopenblas-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup Environment
uv sync --all-extras --group dev --no-install-project
source .venv/bin/activate
```

### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Build Tools
brew install python cmake ninja openblas

# Install Xcode Command Line Tools
xcode-select --install

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup Environment
uv sync --all-extras --group dev --no-install-project
source .venv/bin/activate
```

### Windows (Minimal Setup)

To build native extensions on Windows without installing the full Visual Studio IDE, we use the standalone **Visual Studio Build Tools**.

#### Option A: Chocolatey (Recommended / Automated)
This command installs the compiler and tools without manual GUI steps.

```powershell
# Run as Administrator
choco install git python rust visualstudio2022buildtools visualstudio2022-workload-vctools -y
```

#### Option B: Winget + CLI
Allows installation without the Chocolatey package manager.

```powershell
# 1. Install Tools
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.11 -e --source winget
winget install --id Rustlang.Rustup -e --source winget

# 2. Install Build Tools (Headless / No IDE)
# This installs only the compiler components needed for Python extensions
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart"
```

#### Final Configuration
Open a new PowerShell window (non-Admin) to setup `uv`:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Setup Environment
uv sync --all-extras --group dev --no-install-project
.venv\Scripts\activate
```

## 3. Development Workflow

### Running the Project
The project is a library, so "running" usually means running scripts or the REPL with the library loaded.
```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Run a script
python examples/demo.py
```

### Running Tests
We use `pytest` for the Python test suite.
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_tensor.py
```

### Linting & Formatting
We use `ruff` for both linting and formatting.

```bash
# Format code
make format
# OR
uv run ruff format .

# Check for linting errors
make lint
# OR
uv run --no-sync ruff check .
```

### Debug Mode
To build with debug symbols (slower, but debuggable):
1.  Modify `scripts/build.sh`: change `-DCMAKE_BUILD_TYPE=Release` to `Debug`.
2.  Run `make build`.

### Environment Variables
The build system respects standard environment variables:
- `CC` / `CXX`: Specify C/C++ compiler.
- `RUSTFLAGS`: Pass flags to `cargo`.

### Database & Docker
*Currently, Corepy does not require a database or Docker containers for core development.*

## 4. Project Structure

- `corepy/`: Python source code (user-facing API).
- `rust/`: Rust runtime and compute kernels (the engine).
  - `rust/core/`: Main Rust crate.
- `csrc/`: Legacy C++ kernels (being migrated to Rust).
- `tests/`: Python test suite (`pytest`).
- `scripts/`: Build and setup scripts.
- `docs/`: Project documentation.

## 5. Troubleshooting

**"uv not found"**: Ensure `~/.cargo/bin` or `~/.local/bin` is in your PATH.
**Build fails (missing CMake)**: Ensure CMake 3.15+ is installed.
**Linker errors**: Try `make clean && make build` to remove stale artifacts.
