#!/bin/bash
set -e

# Build C++ extension
echo "Building C++ extension..."
pip install scikit-build-core pybind11 cmake ninja
pip install --no-build-isolation -ve .

# Build Rust extension (using maturin if available, or just cargo build for now)
# Note: In a real hybrid setup, we'd merge these. For now, we build separately.
echo "Building Rust extension..."
cd rust
if command -v maturin &> /dev/null; then
    maturin develop
else
    cargo build --release
    # Symlink for dev
    if [ -f target/release/lib_corepy_rust.so ]; then
         ln -sf target/release/lib_corepy_rust.so ../corepy/_corepy_rust.so
    elif [ -f target/release/lib_corepy_rust.dylib ]; then
         ln -sf target/release/lib_corepy_rust.dylib ../corepy/_corepy_rust.so
    fi
fi
cd ..

echo "Build complete."
