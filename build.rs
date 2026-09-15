//! Compiles the CUDA vector-add implementation with `nvcc` into a static
//! archive and links it, together with the CUDA runtime, into the executable.
//!
//! The spike deliberately uses no build-time crates: a short explicit
//! invocation keeps the native build easy to review and works offline.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by Cargo");

    // Fixed by the development environment; overridable for other machines.
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".to_string());
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cu_source = format!("{manifest_dir}/cuda/vector_add.cu");
    let object = PathBuf::from(&out_dir).join("vector_add.o");
    let archive = PathBuf::from(&out_dir).join("libgpugeno_cuda.a");

    let status = Command::new(&nvcc)
        .args([
            "-c",
            cu_source.as_str(),
            "-o",
            object.to_str().expect("object path is valid UTF-8"),
            "-std=c++17",
            "-O3",
            // Development machine: three RTX 3060 GPUs (compute capability 8.6).
            "-arch=sm_86",
            "-Xcompiler",
            "-fPIC",
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to run {nvcc}: {error}"));
    if !status.success() {
        panic!("nvcc failed with {status}");
    }

    let status = Command::new("ar")
        .args([
            "rcs",
            archive.to_str().expect("archive path is valid UTF-8"),
            object.to_str().expect("object path is valid UTF-8"),
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to run ar: {error}"));
    if !status.success() {
        panic!("ar failed with {status}");
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    // Static archive with the C ABI implementation, then the shared CUDA
    // runtime and C++ runtime that the archive references.
    println!("cargo:rustc-link-lib=static=gpugeno_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    // Re-run the native build when its inputs or the chosen toolchain change.
    println!("cargo:rerun-if-changed=cuda/vector_add.cu");
    println!("cargo:rerun-if-changed=cuda/vector_add.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
