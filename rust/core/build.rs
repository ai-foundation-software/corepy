// Build script to link C++ kernels
fn main() {
    // Tell cargo to link against the C++ kernel library
    // The library is built by CMake in csrc/

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let repo_root = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let build_path = repo_root.join("build").join("csrc");

    // Check if C++ library is built
    if !build_path.exists() {
        eprintln!("WARNING: C++ kernels not built!");
        eprintln!("Run: ./scripts/build.sh or check 'build' directory");
        return;
    }

    // On Windows with MSVC, the library is in a Release subdirectory
    #[cfg(target_os = "windows")]
    {
        let release_path = build_path.join("Release");
        if release_path.exists() {
            println!("cargo:rustc-link-search=native={}", release_path.display());
        } else {
            println!("cargo:rustc-link-search=native={}", build_path.display());
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-search=native={}", build_path.display());
    }

    println!("cargo:rustc-link-lib=static=corepy_kernels");

    // Link OpenBLAS (required by corepy_kernels)
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        // Try common Homebrew paths
        if let Ok(prefix) = std::env::var("LIBRARY_PATH") {
            for path in prefix.split(':') {
                if !path.is_empty() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    #[cfg(target_os = "windows")]
    {
        // Use OPENBLAS_DIR if set
        if let Ok(openblas_dir) = std::env::var("OPENBLAS_DIR") {
            println!("cargo:rustc-link-search=native={}/lib", openblas_dir);
        }
        // OpenBLAS Windows release has openblas.lib (not libopenblas.lib)
        // Runtime needs libopenblas.dll in PATH or alongside the .pyd
        println!("cargo:rustc-link-lib=dylib=openblas");
    }

    // Tell cargo to rerun if C++ code changes
    println!("cargo:rerun-if-changed=../../csrc/src");
    println!("cargo:rerun-if-changed=../../csrc/include");
}
