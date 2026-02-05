# 🍎 macOS Setup Guide

This guide covers setting up **Corepy** on macOS for both **Apple Silicon (M-series)** and **Intel** processors.

> [!NOTE]
> **Metal GPU Acceleration Status:**
> Metal GPU support is currently under active development. This guide will help you install the high-performance CPU version of Corepy, which utilizes **NEON** (Apple Silicon) and **AVX2** (Intel) optimizations for excellent performance.

---

## 📋 Prerequisites

Ensure you have:

1.  **macOS 12.0 (Monterey) or newer**
2.  **Python 3.10 or newer**
3.  **Xcode Command Line Tools**
4.  **Homebrew**

---

## 🚀 Setup Instructions

### 1. Install System Tools

First, install the Xcode Command Line Tools and Homebrew if you haven't already.

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add Homebrew to your shell environment (follow the instructions printed by the installer).

### 2. Install Dependencies

Install build tools and Python using Homebrew.

```bash
# Install CMake and core libraries
brew install cmake pkg-config openblas

# Install Rust (required for Corepy runtime)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3. Clone and Build Corepy

These steps apply to both **Apple Silicon** and **Intel** Macs.

```bash
# Clone the repository
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements-base.txt

# Build C++ kernels
cd csrc
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
cd ..

# Install Corepy (editable mode)
pip install -e .
```

> [!TIP]
> **Apple Silicon Users:** The build system will automatically detect your M1/M2/M3 chip and enable NEON SIMD optimizations.
> **Intel Users:** AVX2 optimizations will be enabled automatically if supported by your CPU.

---

## ✅ Verify Installation

Run the following Python script to check if Corepy allows you to inspect device information.

```python
import corepy as cp

# Check device info (uses the newly exposed API)
try:
    info = cp.get_device_info()
    print("Corepy Device Info:")
    print(f"  CPU Cores: {info.cpu_cores}")
    print(f"  Platform:  {info.platform_system}")
    print(f"  Has NEON:  {info.has_neon}")
    print(f"  Has AVX2:  {info.has_avx2}")
    
    if info.has_gpu:
        print(f"  GPUs:      {info.gpu_names}")
    else:
        print("  GPUs:      None (Metal support coming soon)")
        
except AttributeError:
    print("Error: cp.get_device_info() not found. Did you install the latest version?")
```

### Expected Output (Apple Silicon)
```text
Corepy Device Info:
  CPU Cores: 10
  Platform:  Darwin
  Has NEON:  True
  Has AVX2:  False
  GPUs:      None (Metal support coming soon)
```

---

## 🧪 Performance Benchmark

You can check basic matrix multiplication performance on your CPU.

```python
import corepy as cp
import numpy as np
import time

# Create random matrices
size = 2048
print(f"Benchmarking {size}x{size} Matrix Multiplication...")
a = cp.Tensor(np.random.rand(size, size))
b = cp.Tensor(np.random.rand(size, size))

# Force CPU backend (currently default)
cp.set_backend_policy(cp.BackendPolicy.DEFAULT)

start = time.time()
c = cp.matmul(a, b)
elapsed = time.time() - start

print(f"Time: {elapsed:.4f} seconds")
print(cp.explain_last_dispatch())
```

---

## 🔧 Troubleshooting

### "CMake not found"
Ensure Homebrew is in your PATH.
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

### "OpenBLAS not found"
If CMake complains about OpenBLAS:
```bash
brew install openblas
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
```
(On Intel Macs, replace `/opt/homebrew` with `/usr/local`).

---

## ℹ️ Platform Notes

- **Apple Silicon (ARM64)**: Corepy uses NEON instructions for vectorized operations. Performance is generally very high due to the high memory bandwidth of the Unified Memory Architecture.
- **Intel (x86_64)**: Corepy utilizes AVX2/FMA instructions. Ensure your method of installation (e.g., conda vs brew) provides compatible binaries if not building from source.
