// Build script for corepy-ai
//
// BLAS library selection priority (checked by file existence):
//
//   1. MKL      — if CARGO_FEATURE_MKL or libmkl_rt found in known paths
//   2. AOCL     — if CARGO_FEATURE_AOCL or libblis-mt found in known paths
//   3. Accelerate — Apple Silicon (macOS aarch64, always available)
//   4. System OpenBLAS — libopenblas.so at a system path that we VERIFY EXISTS
//
// We NEVER follow LIBRARY_PATH / env vars that could pick up numpy's private
// bundled openblas (libopenblas-<hash>.so.0).

use std::path::Path;

fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Search a list of candidate paths and return the first directory that
/// actually contains the requested filename.
fn find_lib_dir(lib_name: &str, candidates: &[&str]) -> Option<String> {
    for dir in candidates {
        if file_exists(&format!("{}/{}", dir, lib_name))
            || file_exists(&format!("{}/{}.so", dir, lib_name))
            || file_exists(&format!("{}/{}.so.0", dir, lib_name))
            || file_exists(&format!("{}/{}.so.2", dir, lib_name))
            || file_exists(&format!("{}/{}.dylib", dir, lib_name))
        {
            return Some((*dir).to_string());
        }
    }
    None
}

/// Return (dir, link_name) where link_name is the correct -l argument.
/// If only the versioned file exists (libmkl_rt.so.2), link by filename.
fn find_mkl() -> Option<(String, String)> {
    // Env var overrides (user set MKLROOT / MKL_ROOT)
    for var in &["MKLROOT", "MKL_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            for subdir in &["lib/intel64", "lib", "Library/lib"] {
                let dir = format!("{}/{}", root, subdir);
                for name in &[
                    "libmkl_rt.so",
                    "libmkl_rt.so.2",
                    "libmkl_rt.so.1",
                    "libmkl_rt.dylib",
                    "mkl_rt.lib",
                ] {
                    if file_exists(&format!("{}/{}", dir, name)) {
                        return Some((dir, name.to_string()));
                    }
                }
            }
        }
    }

    // VIRTUAL_ENV / CONDA_PREFIX — pip install mkl-devel puts .so here
    for var in &["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Ok(prefix) = std::env::var(var) {
            for dir_suffix in &["lib", "Library/lib"] {
                let dir = format!("{}/{}", prefix, dir_suffix);
                for name in &[
                    "libmkl_rt.so",
                    "libmkl_rt.so.2",
                    "libmkl_rt.so.1",
                    "mkl_rt.lib",
                ] {
                    if file_exists(&format!("{}/{}", dir, name)) {
                        return Some((dir, name.to_string()));
                    }
                }
            }
        }
    }

    // Standard system locations
    for dir in &[
        "/opt/intel/oneapi/mkl/latest/lib/intel64",
        "/opt/intel/oneapi/mkl/latest/lib",
        "/opt/intel/mkl/lib/intel64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ] {
        for name in &["libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1"] {
            if file_exists(&format!("{}/{}", dir, name)) {
                return Some(((*dir).to_string(), name.to_string()));
            }
        }
    }
    None
}

/// Emit the correct cargo link directives for a vendor library.
/// If the .so has a version suffix (e.g. .so.2), use rustc-link-arg=-l:...
/// If it's a plain .so, use rustc-link-lib=dylib=... (standard).
fn emit_vendor_lib(filename: &str) {
    let is_versioned = filename.ends_with(".so.2")
        || filename.ends_with(".so.1")
        || filename.ends_with(".so.3")
        || filename.ends_with(".so.4");
    if is_versioned {
        // Pass -l:libmkl_rt.so.2 directly to linker
        println!("cargo:rustc-link-arg=-l:{}", filename);
    } else {
        // Strip lib prefix and .so suffix for standard -l linking
        let mut name = filename.strip_prefix("lib").unwrap_or(filename);

        // For Windows .lib files we use the exact name (e.g. mkl_rt) without stripping out lib if it belongs to the core file.
        // We will strip the suffix.
        name = name
            .trim_end_matches(".so")
            .trim_end_matches(".dylib")
            .trim_end_matches(".lib")
            .trim_end_matches(".dll");

        println!("cargo:rustc-link-lib=dylib={}", name);
    }
}

fn find_aocl() -> Option<(String, String)> {
    if let Ok(root) = std::env::var("AOCL_ROOT") {
        let dir = format!("{}/lib", root);
        for name in &["libblis-mt.so", "libblis-mt.so.4", "libblis-mt.so.3"] {
            if file_exists(&format!("{}/{}", dir, name)) {
                return Some((dir, name.to_string()));
            }
        }
    }
    for dir in &[
        "/opt/aocl/lib",
        "/opt/amd/aocl/aocl-linux-gcc/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ] {
        for name in &["libblis-mt.so", "libblis-mt.so.4"] {
            if file_exists(&format!("{}/{}", dir, name)) {
                return Some(((*dir).to_string(), name.to_string()));
            }
        }
    }
    None
}

fn find_system_openblas(lib_arch: &str) -> Option<String> {
    // Only return a verified SYSTEM path — never pick up venv/site-packages.
    let system_candidates = [
        format!("/usr/lib/{}", lib_arch),
        format!("/usr/lib/{}/openblas-pthread", lib_arch),
        "/usr/lib/openblas".to_string(),
        "/usr/local/lib".to_string(),
        // Homebrew (macOS)
        "/opt/homebrew/opt/openblas/lib".to_string(),
        "/usr/local/opt/openblas/lib".to_string(),
        // conda-forge
        "/opt/conda/lib".to_string(),
    ];
    let candidate_refs: Vec<&str> = system_candidates.iter().map(|s| s.as_str()).collect();
    find_lib_dir("libopenblas", &candidate_refs)
}

fn main() {
    // ── Declare conditional cfg keys ────────────────────────────────────────
    println!("cargo::rustc-check-cfg=cfg(use_accelerate)");
    println!("cargo::rustc-check-cfg=cfg(use_mkl)");
    println!("cargo::rustc-check-cfg=cfg(use_aocl)");
    println!("cargo::rustc-check-cfg=cfg(cpp_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blas)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // ── macOS Apple Silicon: always use Accelerate ───────────────────────────
    #[cfg(target_os = "macos")]
    if target_arch == "aarch64" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-cfg=use_accelerate");
        println!("cargo:rustc-cfg=has_blas");
        println!("cargo:warning=[corepy build] Apple Silicon → Accelerate (AMX + vecLib BLAS)");
        emit_rerun();
        return;
    }

    // ── Priority 1: System OpenBLAS (Linux) ──────────────────────────────────
    if target_os == "linux" {
        let lib_arch = match target_arch.as_str() {
            "aarch64" => "aarch64-linux-gnu",
            _ => "x86_64-linux-gnu",
        };
        if let Some(dir) = find_system_openblas(lib_arch) {
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-cfg=has_blas");
            println!("cargo:warning=[corepy build] System OpenBLAS → {}", dir);
            emit_rerun();
            return;
        }
    }

    // ── Priority 2: MKL (feature gate OR auto-detected) ─────────────────────
    let force_mkl = std::env::var("CARGO_FEATURE_MKL").is_ok();
    if force_mkl
        || ((target_os == "linux" || target_os == "windows" || target_os == "macos")
            && find_mkl().is_some())
    {
        if let Some((dir, filename)) = find_mkl() {
            println!("cargo:rustc-link-search=native={}", dir);
            if target_os != "windows" {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
            }
            emit_vendor_lib(&filename);

            if target_os != "windows" {
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=m");
                println!("cargo:rustc-link-lib=dylib=dl");
            }
            println!("cargo:rustc-cfg=use_mkl");
            println!("cargo:rustc-cfg=has_blas");
            println!(
                "cargo:warning=[corepy build] Intel MKL → {} ({})",
                dir, filename
            );
            emit_rerun();
            return;
        } else if force_mkl {
            panic!(
                "[corepy build] --features mkl requested but libmkl_rt not found.\n\
                 Set MKLROOT or install MKL: uv pip install mkl-devel"
            );
        }
    }

    // ── Priority 2: AOCL (feature gate OR auto-detected) ────────────────────
    let force_aocl = std::env::var("CARGO_FEATURE_AOCL").is_ok();
    if force_aocl || (target_os == "linux" && find_aocl().is_some()) {
        if let Some((dir, filename)) = find_aocl() {
            println!("cargo:rustc-link-search=native={}", dir);
            if target_os != "windows" {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
            }
            emit_vendor_lib(&filename);

            if target_os != "windows" {
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=m");
            }
            println!("cargo:rustc-cfg=use_aocl");
            println!("cargo:rustc-cfg=has_blas");
            println!(
                "cargo:warning=[corepy build] AMD AOCL → {} ({})",
                dir, filename
            );
            emit_rerun();
            return;
        } else if force_aocl {
            panic!(
                "[corepy build] --features aocl requested but libblis-mt not found.\n\
                 Set AOCL_ROOT or download AOCL from https://www.amd.com/en/developer/aocl.html"
            );
        }
    }

    // ── Priority 3: macOS Intel — OpenBLAS via Homebrew ─────────────────────
    if target_os == "macos" {
        for brew_path in &[
            "/usr/local/opt/openblas/lib",
            "/opt/homebrew/opt/openblas/lib",
        ] {
            if file_exists(&format!("{}/libopenblas.dylib", brew_path)) {
                println!("cargo:rustc-link-search=native={}", brew_path);
                println!("cargo:rustc-link-lib=dylib=openblas");
                println!("cargo:rustc-link-lib=dylib=c++");
                println!("cargo:rustc-cfg=has_blas");
                println!(
                    "cargo:warning=[corepy build] macOS OpenBLAS → {}",
                    brew_path
                );
                emit_rerun();
                return;
            }
        }
        // Last resort on macOS: Accelerate for x86_64 too
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=use_accelerate");
        println!("cargo:rustc-cfg=has_blas");
        println!("cargo:warning=[corepy build] macOS fallback → Accelerate");
        emit_rerun();
        return;
    }

    // ── Windows fallback ─────────────────────────────────────────────────────
    if target_os == "windows" {
        if let Ok(openblas_dir) = std::env::var("OPENBLAS_DIR") {
            let mut link_name = "openblas";
            // Add both root and lib to search path, since the .lib might be at root
            println!("cargo:rustc-link-search=native={}", openblas_dir);
            println!("cargo:rustc-link-search=native={}/lib", openblas_dir);

            // On Windows, the DLL might be named libopenblas.dll or openblas.dll
            // The import library is libopenblas.lib at the root or lib/
            if file_exists(&format!("{}/libopenblas.lib", openblas_dir))
                || file_exists(&format!("{}/lib/libopenblas.lib", openblas_dir))
            {
                link_name = "libopenblas";
            }
            println!("cargo:rustc-link-lib=dylib={}", link_name);
            println!("cargo:rustc-cfg=has_blas");
            println!(
                "cargo:warning=[corepy build] Windows → {} (from OPENBLAS_DIR)",
                link_name
            );
            emit_rerun();
            return;
        }
    }

    // ── No BLAS found: warn but don't crash — pure-Rust fallback will be used ─
    println!(
        "cargo:warning=[corepy build] No BLAS library found! Pure-Rust fallback only.\n\
         Install one of: `uv pip install mkl-devel`  or  `sudo apt install libopenblas-dev`"
    );
    emit_rerun();
}

fn emit_rerun() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=MKL_ROOT");
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
}
