# 🪟 GPU Setup Guide - Windows (NVIDIA RTX)

This guide will help you set up **Corepy** on Windows with **NVIDIA RTX GPUs** (2000/3000/4000/5000 series) using CUDA.

> [!NOTE]
> This guide supports **RTX 2050, RTX 3060, RTX 4090**, and all other RTX 2000/3000/4000/5000 series GPUs on Windows 10/11.

---

## 📋 Prerequisites

Before you begin, ensure you have:

1. **NVIDIA RTX GPU** (2000/3000/4000/5000 series)
   - Examples: RTX 2050, RTX 2060, RTX 3060, RTX 3080, RTX 4060, RTX 4090, RTX 5090
2. **Windows 10 (64-bit) or Windows 11**
3. **Python 3.10 or newer** ([Download from python.org](https://www.python.org/downloads/))
4. **Visual Studio 2019 or 2022** (with C++ Desktop Development)
5. **Administrator privileges** (for driver and CUDA installation)

---

## 📊 Setup Process Overview

**Windows GPU Setup Steps:**
1. Install NVIDIA GPU drivers
2. Install Visual Studio with C++ support
3. Install CUDA Toolkit
4. Install CMake and Rust
5. Clone and build Corepy with CUDA support
6. Verify GPU detection

---

## 🚀 Step-by-Step Installation

### Step 1: Install NVIDIA Drivers

#### Option A: GeForce Experience (Recommended for Gaming GPUs)

1. Download [GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/)
2. Install and launch GeForce Experience
3. Go to **Drivers** tab → Click **Download**
4. Restart your computer after installation

#### Option B: Manual Driver Installation

1. Go to [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select:
   - **Product Type**: GeForce
   - **Product Series**: GeForce RTX 20/30/40/50 Series
   - **Product**: Your specific GPU model (e.g., RTX 2050)
   - **Operating System**: Windows 10/11 64-bit
3. Click **Search** → Download and install the driver
4. Restart your computer

**Verify installation:**

Open Command Prompt (Win+R, type `cmd`, press Enter) and run:
```cmd
nvidia-smi
```

You should see your GPU information (e.g., "NVIDIA GeForce RTX 2050").

---

### Step 2: Install Visual Studio

Corepy requires a C++ compiler. Visual Studio Community Edition is free:

1. Download [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
2. Run the installer
3. Select **"Desktop development with C++"** workload
4. In "Individual components", ensure these are checked:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10/11 SDK
   - C++ CMake tools for Windows
5. Click **Install** (this will take 10-30 minutes)
6. Restart your computer after installation

**Verify installation:**
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl
```

You should see the Microsoft C++ compiler version information.

---

### Step 3: Install CUDA Toolkit

Download and install CUDA Toolkit 12.3 (recommended):

1. Go to [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select:
   - **Operating System**: Windows
   - **Architecture**: x86_64
   - **Version**: 10 or 11 (your Windows version)
   - **Installer Type**: exe (network) - faster download
3. Download and run the installer
4. Choose **Custom (Advanced)** installation
5. Select:
   - ✅ CUDA Toolkit
   - ✅ CUDA Development
   - ✅ CUDA Runtime
   - ✅ CUDA Documentation (optional)
6. Click **Next** and **Install**

**Verify installation:**
```cmd
nvcc --version
```

Expected output:
```
Cuda compilation tools, release 12.3, V12.3.XXX
```

If `nvcc` is not recognized, add CUDA to your PATH:
1. Press Win+R, type `sysdm.cpl`, press Enter
2. Go to **Advanced** tab → **Environment Variables**
3. Under "System variables", find **Path** → Click **Edit**
4. Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin`
5. Click **OK** and restart Command Prompt

---

### Step 4: Install CMake and Rust

#### Install CMake

1. Download [CMake for Windows](https://cmake.org/download/) (Windows x64 Installer)
2. Run the installer
3. **Important**: Select **"Add CMake to the system PATH for all users"**
4. Click **Next** → **Install**

**Verify:**
```cmd
cmake --version
```

#### Install Rust

1. Download [rustup-init.exe](https://rustup.rs/)
2. Run the installer
3. Press Enter to proceed with default installation
4. Restart Command Prompt after installation

**Verify:**
```cmd
rustc --version
```

---

### Step 5: Clone and Build Corepy

Open **Command Prompt** or **PowerShell**:

```cmd
:: Clone the repository
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy

:: Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements-base.txt

:: Build C++ kernels with CUDA support
cd csrc

:: Find your GPU architecture:
:: RTX 2050: 75
:: RTX 3060/3070/3080/3090: 86
:: RTX 4060/4070/4080/4090: 89
:: RTX 5090: 90

:: Configure with CMake (example for RTX 2050)
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DUSE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DCMAKE_BUILD_TYPE=Release

:: Build
cmake --build build --config Release

cd ..

:: Install Corepy (editable mode)
pip install -e .
```

> [!IMPORTANT]
> Replace the `-DCMAKE_CUDA_ARCHITECTURES` value based on your GPU:
> - **RTX 2000 series** (2050, 2060, 2070, 2080): `75`
> - **RTX 3000 series** (3060, 3070, 3080, 3090): `86`
> - **RTX 4000 series** (4060, 4070, 4080, 4090): `89`
> - **RTX 5000 series** (5090): `90`
>
> To support multiple GPUs: `-DCMAKE_CUDA_ARCHITECTURES="75;86;89;90"`

---

## ✅ Verify GPU Support

After installation, verify GPU detection:

```python
import corepy as cp

# Check device information
device_info = cp.get_device_info()
print(f"GPU Count: {device_info.gpu_count}")
print(f"GPU Names: {device_info.gpu_names}")
print(f"Has GPU: {device_info.has_gpu}")
```

Expected output:
```
GPU Count: 1
GPU Names: ['CUDA Device 0']
Has GPU: True
```

---

## 🧪 Test GPU Performance

Run a benchmark to confirm GPU acceleration:

```python
import corepy as cp
import numpy as np
import time

# Create large matrices
size = 4096
a = cp.Array(np.random.rand(size, size))
b = cp.Array(np.random.rand(size, size))

# Force GPU backend
cp.set_backend_policy('gpu')

# Benchmark matrix multiplication
start = time.time()
c = cp.matmul(a, b)
gpu_time = time.time() - start

print(f"GPU Time: {gpu_time:.4f} seconds")
print(cp.explain_last_dispatch())
```

---

## 🔧 Troubleshooting

### Issue 1: "CUDA not found" during build

**Solution:**
Ensure CUDA is in your PATH. Open a new Command Prompt and check:
```cmd
echo %PATH% | findstr CUDA
nvcc --version
```

If not found, add to PATH:
1. Win+R → `sysdm.cpl`
2. Advanced → Environment Variables
3. Edit "Path" → Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin`

---

### Issue 2: "Cannot open include file: 'cuda_runtime.h'"

**Solution:**
Visual Studio can't find CUDA headers. Set the `CUDA_PATH` environment variable:

```cmd
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
```

Restart Command Prompt and rebuild.

---

### Issue 3: "MSBuild not found" or CMake error

**Solution:**
Open **Visual Studio Developer Command Prompt** instead of regular Command Prompt:
- Press **Start** → Search "Developer Command Prompt for VS 2022"
- Run the build commands from there

Or set up the environment manually:
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

---

### Issue 4: "No GPU detected" at runtime

**Solution:**
1. Verify `nvidia-smi` works
2. Check if CUDA DLLs are accessible:
   ```cmd
   where cudart64_12.dll
   ```
3. Ensure you built with `-DUSE_CUDA=ON`
4. Rebuild from scratch:
   ```cmd
   cd csrc
   rmdir /s /q build
   cmake -B build -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
   cmake --build build --config Release
   ```

---

### Issue 5: Python not found during CMake

**Solution:**
Tell CMake where Python is:
```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DUSE_CUDA=ON ^
  -DPython_ROOT_DIR="C:\Users\YourUsername\AppData\Local\Programs\Python\Python311"
```

Replace `Python311` with your Python version.

---

## 📊 GPU Architecture Reference

| GPU Series | Model Examples | Compute Capability | CMake Flag |
|:-----------|:--------------|:------------------|:-----------|
| RTX 2000 | 2050, 2060, 2070, 2080 | 7.5 | `75` |
| RTX 3000 | 3060, 3070, 3080, 3090 | 8.6 | `86` |
| RTX 4000 | 4060, 4070, 4080, 4090 | 8.9 | `89` |
| RTX 5000 | 5090 | 9.0 | `90` |

---

## 🎯 Environment Variables

Control GPU usage:

```cmd
:: Force GPU backend
set COREPY_BACKEND=gpu

:: Force CPU backend
set COREPY_BACKEND=cpu

:: Select GPU device (if multiple GPUs)
set CUDA_VISIBLE_DEVICES=0

:: Enable CUDA debugging
set CUDA_LAUNCH_BLOCKING=1
```

To make permanent, use `setx` instead of `set`, or add via System Properties.

---

## 🚢 Deploying to Another Windows PC

To use Corepy on **another Windows PC with RTX GPU**:

### Option 1: Install from Source (Recommended)

1. Follow **Steps 1-5** above on the target PC
2. Ensure same CUDA version is installed
3. Build with correct GPU architecture

### Option 2: Copy Pre-built Binaries (Advanced)

> [!CAUTION]
> This only works if both PCs have:
> - Same Windows version
> - Same CUDA version
> - Same Visual Studio version
> - Same Python version
> - Same GPU architecture

Not recommended due to DLL dependencies.

---

## 🎮 Performance Tips for Windows

1. **Disable Windows Defender real-time scanning** for your project folder (improves build speed)
2. **Use an SSD** for your project directory
3. **Close other GPU-intensive applications** (browsers, games) during benchmarking
4. **Update to latest Windows updates** for best GPU driver compatibility
5. **Monitor GPU usage** with Task Manager (Ctrl+Shift+Esc → Performance → GPU)

---

## 📚 Additional Resources

- [NVIDIA CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Visual Studio Documentation](https://docs.microsoft.com/en-us/visualstudio/)
- [Corepy Architecture Documentation](../03_architecture/architecture.md)

---

## ℹ️ Notes

- **RTX 2050** (laptop): 4GB VRAM, keep matrix sizes ≤ 8192×8192
- **RTX 3060** (desktop): 12GB VRAM, supports larger operations
- **RTX 4090**: 24GB VRAM, handles massive computations
- **Power mode**: Set to "High Performance" in Windows Power Options for best GPU performance

---

For questions or issues, please open an issue on [GitHub](https://github.com/ai-foundation-software/corepy/issues).
