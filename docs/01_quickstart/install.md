# 📥 Installation Guide

This guide will help you get `Corepy` installed on your computer.

## ✅ Prerequisites

Before you begin, make sure you have **Python 3.10** or newer installed.

You can check your Python version by running this in your terminal:
```bash
python --version
```
Expected output: `Python 3.10.x` (or higher).

---

## 🚀 Easy Install

### Option 1: Preferred (uv)
We recommend `uv` for modern, fast Python package management.

```bash
uv pip install corepy
```

### Option 2: Universal (pip)
Standard pip installation is fully supported on all platforms (Linux, macOS, Windows).

```bash
pip install corepy
```

### Verifying Installation
To make sure everything is working, run this simple command:

```bash
python -c "import corepy; print(corepy.__version__)"
```

---

## 🛠️ Building from Source

If you want to modify Corepy, you need **Rust** (1.70+) and **CMake** (3.15+).

### 1. Install Build Tools

*   **Rust**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
*   **CMake**:
    *   **macOS**: `brew install cmake`
    *   **Ubuntu**: `sudo apt-get install cmake build-essential`
    *   **Windows**: Install via Visual Studio Installer ("C++ CMake tools").

### 2. Clone & Build

```bash
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy

# Recommended: Use the Makefile workflow
make build && make install
```

Alternatively, you can build manually:
```bash
./scripts/build.sh
pip install -e .
```

### 3. Troubleshooting

**"xcrun: error" on macOS?**
Install command line tools: `xcode-select --install`.

**"OpenBLAS not found" on Linux?**
Corepy will fall back to generic C++ kernels, but for speed install OpenBLAS:
`sudo apt-get install libopenblas-dev`

---

## 🖥️ GPU Setup (Optional)

Corepy is **CPU-first**, meaning it works perfectly without a GPU.
However, if you have an NVIDIA GPU or Apple Silicon, Corepy can use it for significant performance improvements.

### Platform-Specific GPU Guides:

- **🐧 Linux (NVIDIA RTX)**: [GPU Setup Guide](gpu_setup.md) - For RTX 2000/3000/4000/5000 series GPUs
- **🍎 macOS (Apple Silicon)**: [macOS GPU Setup Guide](gpu_setup_macos.md) - For M1/M2/M3/M4 chips  
- **🪟 Windows (NVIDIA RTX)**: [Windows GPU Setup Guide](gpu_setup_windows.md) - For RTX GPUs on Windows
- **📋 Quick Reference**: [All Platforms Quick Reference](gpu_quick_reference.md) - Quick setup checklist for all platforms

Each guide includes step-by-step instructions for:
- Installing GPU drivers and toolkits (CUDA/Metal)
- Building Corepy with GPU support
- Verification and performance testing
- Troubleshooting common issues
