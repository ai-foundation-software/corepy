# 🍎 macOS Setup Guide

This guide covers setting up **Corepy** on macOS for both **Apple Silicon (M-series)** and **Intel** processors.

> [!NOTE]
> **Metal GPU Acceleration:**
> Corepy v0.2.4+ supports **Metal** acceleration on Apple Silicon.

---

## 📋 Prerequisites

1.  **macOS 12.0+**
2.  **Python 3.10+**
3.  **Xcode Command Line Tools**: `xcode-select --install`
4.  **CMake**: `brew install cmake`

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
# Install critical build tools
brew install cmake pkg-config

# Install Rust (required)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Build Corepy

```bash
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy

# Recommended build command
make build && make install
```

> [!TIP]
> **Apple Silicon:** The build system automatically detects M1/M2/M3 chips, enables NEON optimizations, and compiles Metal shaders.
> **Intel:** AVX2 optimizations are enabled automatically.

---

## ✅ Verify Metal Support

Run this Python script to verify your GPU is accessible:

```python
import corepy as cp

try:
    # Allocate a tensor on Metal
    t = cp.Tensor([1.0, 2.0], device="metal")
    print("✅ Metal GPU is ACTIVE and working!")
except Exception as e:
    print(f"❌ Metal GPU unavailable: {e}")
```

For advanced usage and performance tips, see the **[Metal GPU Guide](../05_advanced/metal_gpu.md)**.
