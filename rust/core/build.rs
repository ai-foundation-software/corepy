// Build script to link C++ kernels
fn main() {
    // Declare custom cfg for conditional compilation
    println!("cargo::rustc-check-cfg=cfg(use_accelerate)");

    // Tell cargo to link against the C++ kernel library
    // The library is built by CMake in csrc/

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let repo_root = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    // Check for explicit C++ source directory from environment
    let build_path = if let Ok(csrc_dir) = std::env::var("COREPY_CSRC_DIR") {
        std::path::PathBuf::from(csrc_dir)
    } else {
        repo_root.join("build").join("csrc")
    };

    // Check if C++ library is built
    let cpp_lib_exists = build_path.exists();

    if !cpp_lib_exists {
        // Prepare a detailed error message with diagnostic paths
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unable to determine>".to_string());

        let msg = format!(
            "C++ kernels not found at '{}'.\\n\
             Please run `./scripts/build.sh` or ensure `cmake` build has completed before compiling Rust.\\n\
             The build script expects the C++ library in `build/csrc` relative to the repository root.\\n\
             \\n\
             Diagnostic info:\\n\
             - Repository root: {}\\n\
             - Current working directory: {}\\n\
             - Expected C++ library path: {}\\n\
             - COREPY_CSRC_DIR env var: {:?}",
            build_path.display(),
            repo_root.display(),
            cwd,
            build_path.display(),
            std::env::var("COREPY_CSRC_DIR").ok()
        );

        // Emit a cargo warning so it's visible even if we panic convention changes
        println!("cargo:warning={}", msg);

        // Always panic in CI to catch build ordering issues
        if std::env::var("CI").is_ok() {
            panic!("{}", msg);
        } else {
            println!("cargo:warning=Continuing without C++ kernels (symbols will be missing at runtime).");
        }
    }

    // Only link C++ kernels if they exist
    if cpp_lib_exists {
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
    }

    // Link OpenBLAS or Accelerate based on target OS
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os == "linux" {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if target_os == "macos" {
        // On Apple Silicon (aarch64), link Accelerate framework for AMX support
        if target_arch == "aarch64" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-cfg=use_accelerate");
            println!(
                "cargo:warning=Using Apple Accelerate framework (AMX coprocessor enabled) & Metal"
            );
        } else {
            // On Intel Macs, still use OpenBLAS
            // Try common Homebrew paths
            if let Ok(prefix) = std::env::var("LIBRARY_PATH") {
                for path in prefix.split(':') {
                    if !path.is_empty() {
                        println!("cargo:rustc-link-search=native={}", path);
                    }
                }
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target_os == "windows" {
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
