# Corepy

> **A high-performance Python foundation for data and AI.**
> *Correctness First. Device Aware. Built to Last.*

## 📖 What is Corepy?

Corepy is a library that helps you write fast and safe Python code for data processing and AI.

If you have ever:
- Waited too long for a Python loop to finish.
- Crashed your program because of a "Segmentation Fault" in a C extension.
- Struggled to get your code running on both a laptop (CPU) and a powerful server (GPU).

Corepy is designed to solve these problems. It combines the ease of **Python**, the raw speed of **C++**, and the safety of **Rust**.

### Key Features
- **🚀 Fast**: Uses C++ for heavy number crunching (SIMD optimized).
- **🛡️ Safe**: Uses Rust to manage memory and parallel tasks, preventing crashes.
- **💻 Everywhere**: Works on Linux, macOS (including Apple Silicon), and Windows.
- **🤖 Smart**: Automatically uses your hardware (like multiple CPU cores) without complex setup.

---

## 💻 Supported Platforms

We officially support the following operating systems:

| Platform | Architecture | Notes |
| :--- | :--- | :--- |
| **Linux** | x86_64, aarch64 | Great for servers and cloud (AWS Graviton supported). |
| **macOS** | Apple Silicon (M1/M2/M3) | Optimized for Apple's hardware. |
| **macOS** | Intel | Fully supported. |
| **Windows** | x86_64 | Works natively with standard tools. |

---

## 🛠️ Installation

### Option 1: Install via pip (Recommended)
If you just want to use the library:

```bash
pip install corepy
```

### Option 2: Install from Source (For Developers)
If you want to contribute or change the code, see our [Installation Guide](docs/install.md).

---

## ⚡ Quick Start

Here is a simple example showing how to load some data and process it safely.

```python
import corepy as cp

# 1. Load data efficiently (automatically uses parallel processing)
# This looks like normal Python, but it's powered by Rust under the hood.
data = cp.read_csv("data.csv")

# 2. Perform a calculation
# Corepy automatically selects the best way to run this on your CPU.
result = data.select("price").mean()

print(f"Average Price: {result}")
```

For more examples, see the [Usage Guide](docs/usage.md).

---

## 📚 Documentation
- [**Installation Guide**](docs/install.md): Detailed setup instructions.
- [**Usage Guide**](docs/usage.md): How to use Corepy for real work.
- [**Contributing**](CONTRIBUTING.md): How to help build Corepy.

## 🤝 Stability
Corepy is currently in **Alpha**. This means:
- We prioritize **correctness** (getting the right answer) over everything else.
- APIs might change slightly as we improve them.
