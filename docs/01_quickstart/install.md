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

## 🚀 Easy Install (For Users)

The easiest way to install Corepy is using `pip`, the standard Python package installer.

```bash
pip install corepy
```

### Verifying Installation
To make sure everything is working, run this simple command:

```bash
python -c "import corepy; print(corepy.__version__)"
```
If it prints a version number (like `0.2.0`), you are ready to go!

---

## 🛠️ Building from Source (For Contributors)

If you want to modify Corepy or see how it works under the hood, you need to build it from source. This requires a few more tools because Corepy uses **Rust** and **C++**.

### 0. Quick Setup (Automated)

We provide a script that detects your OS, installs dependencies (using `uv` or `pip`), and builds the project.

```bash
./scripts/setup_dev.sh
```

If you prefer manual setup, follow the steps below.

### 1. Install Build Tools

#### Rust
You need the Rust compiler. The easiest way is to use `rustup`:
- **Linux/macOS**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Windows**: Download `rustup-init.exe` from [rust-lang.org](https://www.rust-lang.org/tools/install).

#### C++ Compiler
- **Linux**: Install `build-essential` and `cmake`.
  ```bash
  sudo apt-get install build-essential cmake
  ```
- **macOS**: Install Xcode Command Line Tools.
  ```bash
  xcode-select --install
  # You might also need CMake
  brew install cmake
  ```
- **Windows**: Install Visual Studio with "Desktop development with C++".

### 2. Clone the Repository

Get the code from GitHub:

```bash
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy
```

### 3. Set up a Virtual Environment

It is best practice to keep your project dependencies identical to the project's.
We use `venv` to create an isolated environment.

```bash
# Create the environment named '.venv'
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 4. Install Dependencies & Build

Now, install the development tools and build the project.

```bash
# Install build tools
pip install --upgrade pip
pip install -r requirements-base.txt

# Build and install Corepy in "editable" mode
# This means changes you make to Python files happen instantly.
pip install -e .
```

### 5. Troubleshooting

**"Undefined symbols" on macOS?**
This can happen if the Rust linker can't find Python. Ensure you are using the virtual environment activation step above.

**Build fails on Linux?**
Make sure you have `cmake` installed: `cmake --version`.

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
