//! Builds the native CUDA implementation and the direct-Vulkan flagstat shader.
//!
//! CUDA is compiled with explicit `nvcc`/`ar` invocations. The Vulkan-specific
//! WGSL source is validated and translated to embedded SPIR-V by `naga`, so a
//! system shader compiler is not a runtime or build-time prerequisite.

use std::path::{Path, PathBuf};
use std::process::Command;

fn compile_vulkan_shader(manifest_dir: &str, out_dir: &str) {
    let source_path = Path::new(manifest_dir).join("src/vulkan_flagstat.wgsl");
    let source = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", source_path.display()));
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
        panic!(
            "failed to parse {}: {}",
            source_path.display(),
            error.emit_to_string(&source)
        )
    });
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("failed to validate {}: {error}", source_path.display()));
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "flagstat".to_string(),
    };
    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))
        .unwrap_or_else(|error| panic!("failed to compile {}: {error}", source_path.display()));
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let output = Path::new(out_dir).join("vulkan_flagstat.spv");
    std::fs::write(&output, bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output.display()));
}

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by Cargo");
    compile_vulkan_shader(&manifest_dir, &out_dir);

    // Fixed by the development environment; overridable for other machines.
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".to_string());
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let sources = [
        ("cuda/gpugeno_cuda.cu", "gpugeno_cuda.o"),
        ("cuda/flagstat.cu", "flagstat.o"),
    ];
    let mut objects = Vec::with_capacity(sources.len());

    for (source, object_name) in sources {
        let source = format!("{manifest_dir}/{source}");
        let object = PathBuf::from(&out_dir).join(object_name);
        let status = Command::new(&nvcc)
            .args([
                "-c",
                source.as_str(),
                "-o",
                object.to_str().expect("object path is valid UTF-8"),
                "-std=c++17",
                "-O3",
                // Development machine: RTX 3060 (compute capability 8.6).
                "-arch=sm_86",
                "-Xcompiler",
                "-fPIC",
            ])
            .status()
            .unwrap_or_else(|error| panic!("failed to run {nvcc}: {error}"));
        if !status.success() {
            panic!("nvcc failed for {source} with {status}");
        }
        objects.push(object);
    }

    let archive = PathBuf::from(&out_dir).join("libgpugeno_cuda.a");
    match std::fs::remove_file(&archive) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("failed to remove stale {}: {error}", archive.display()),
    }
    let mut ar = Command::new("ar");
    ar.args([
        "rcs",
        archive.to_str().expect("archive path is valid UTF-8"),
    ]);
    ar.args(&objects);
    let status = ar
        .status()
        .unwrap_or_else(|error| panic!("failed to run ar: {error}"));
    if !status.success() {
        panic!("ar failed with {status}");
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-lib=static=gpugeno_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    for input in [
        "cuda/gpugeno_cuda.cu",
        "cuda/gpugeno_cuda.h",
        "cuda/flagstat.cu",
        "cuda/flagstat.cuh",
        "src/vulkan_flagstat.wgsl",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={input}");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
