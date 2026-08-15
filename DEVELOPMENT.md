# Development Guide

## 1. Project Overview
Corepy-ai is a high-performance unified runtime for data, computation, and AI workflows. It combines the ease of **Python** with the safety and speed of **Rust**. The architecture consists of a Rust core (`_corepy_rust`) built with `maturin`, which provides high-performance tensor operations and dynamically selects the optimal CPU math backend (MKL, AOCL, Accelerate, or OpenBLAS).

The project uses `uv` for lightning-fast Python dependency management and `cargo` for the Rust toolchain.

## 2. Prerequisites

### Linux (Ubuntu/Debian)

Install the required system packages, including Python development headers and build essentials for the Rust extensions.

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-dev \
    python3-venv \
    build-essential \
    gcc \
    make \
    git \
    libopenblas-dev \
    gcc-aarch64-linux-gnu \
    curl
```

**Install Rust Toolchain:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Setup Environment:**
```bash
uv sync --all-extras --group dev --no-install-project
source .venv/bin/activate
```

### macOS

**Install Homebrew:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install Python and Build Tools:**
```bash
brew install python openblas
xcode-select --install
```

**Install Rust Toolchain:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Setup Environment:**
```bash
uv sync --all-extras --group dev --no-install-project
source .venv/bin/activate
```

### Windows

**1. Install Python & Tools:**
- Download and install Python from [python.org](https://www.python.org/downloads/windows/) (ensure "Add Python to PATH" is checked).
- Install Git for Windows.
- Install Rust via [rustup.rs](https://rustup.rs/).

**2. Install Visual Studio Build Tools:**
To build the Rust native extensions, you need the C++ build tools.

```powershell
# Using winget (Recommended)
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart"
```

**3. Setup Environment:**
Open a new PowerShell window and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Setup project
uv sync --all-extras --group dev --no-install-project
.venv\Scripts\activate
```

## 3. Development Workflow

### Running the Project
Since Corepy is a library, you can run examples or verify the installation using the provided scripts.

```bash
# Build the Rust extension in development mode
make build
# OR
uv run maturin develop --manifest-path rust/core/Cargo.toml

# Run an example
uv run python examples/hello_corepy.py
```

### Running Tests
We use `pytest` for testing both Python logic and Rust integration.

```bash
make test
# OR
uv run pytest tests/
```

### Linting
We use `ruff` for fast linting.

```bash
make lint
# OR
uv run ruff check .
```

### Formatting
We use `ruff` to maintain a consistent code style.

```bash
make format
# OR
uv run ruff format .
```

### Environment Variables
Corepy uses environment variables to control backend selection and threading:
- `COREPY_BACKEND`: Override the math backend (`mkl`, `aocl`, `openblas`, `rust`, `accelerate`).
- `RUSTFLAGS`: Pass additional flags to the Rust compiler.
- `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`: Control threading for specific BLAS backends.
- `RAYON_NUM_THREADS`: Control threading for the pure-Rust backend.

### Database Setup
Currently, Corepy does not require a database for development.

### Docker
*No Dockerfile detected in the core repository. Core development is performed directly on the host or in a virtualized Linux environment.*

## 4. CPU Backend & BLAS Setup

Corepy automatically selects the best available BLAS at startup based on your CPU architecture.

### Backend Priority

| Priority | Backend | Condition |
|----------|---------|----------|
| 1 | Intel MKL | `GenuineIntel` CPU + `libmkl_rt.so` found |
| 2 | AMD AOCL | `AuthenticAMD` CPU + `libblis-mt.so` found |
| 3 | Apple Accelerate | macOS aarch64 (M1/M2/M3/M4) |
| 4 | OpenBLAS | `libopenblas.so` found anywhere |
| 5 | Pure-Rust | Always available (built-in fallback) |

### Verification
Verify which backend is being used:

```bash
uv run python -c "import corepy.matmul as cm; print(cm.backend_info())"
```
