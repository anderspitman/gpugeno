# Development, verification, and benchmarking

**Purpose:** Record the current environment, canonical input, build and verification commands, device policy, benchmark semantics, and test expectations.
**Read when:** Building, testing, reproducing the checkpoint, changing device behavior, or collecting performance evidence.
**Caution:** Hardware observations are dated. Confirm them before relying on device counts, paths, or tool availability.

## Contents

- [Development machine and canonical input](#development-machine-and-canonical-input)
- [Reproduce the current checkpoint](#reproduce-the-current-checkpoint)
- [Rocky Linux 8-compatible release builds](#rocky-linux-8-compatible-release-builds)
- [Platform and devices](#platform-and-devices)
- [Benchmark meaning](#benchmark-meaning)
- [Testing](#testing)
- [Recording new evidence](#recording-new-evidence)

## Development machine and canonical input

Observed on 2026-09-15:

- Linux
- Three NVIDIA GeForce RTX 3060 GPUs, 12,288 MiB each
- NVIDIA driver 550.163.01
- CUDA toolkit 12.4; `nvcc` is `/usr/local/cuda/bin/nvcc`
- Vulkan instance 1.4.328; NVIDIA devices expose Vulkan 1.3.277
- Rust stable is installed under `/home/agent/.cargo/bin` (`rustc` and `cargo` 1.98.1), but that directory was not present in the observed `PATH`; `rustfmt` and `clippy` components were added during the spike
- `samtools` was not visible on `PATH`

The later controlled campaigns found two CUDA-visible RTX 3060 devices rather than the earlier three; device 0 remained available. They also found samtools 1.24 at `/usr/local/bin/samtools`. Treat device counts and tool visibility as dated observations and enumerate them again before a new campaign.

The canonical initial real-world input is:

- `/agents/shadowfax/data/HG002_chr22.bam` — 1,635,811,603 bytes
- `/agents/shadowfax/data/HG002_chr22.bam.bai` — 144,864 bytes

A read-only metadata inspection on 2026-09-15 found:

- 81,884 BGZF members, including the 28-byte empty EOF member
- 5,324,522,133 total uncompressed bytes
- 3,102 nonzero BAI linear-index entries and 2,284 unique virtual offsets
- 2,281 unique compressed block offsets represented by those linear entries
- A maximum compressed gap of 3,926,229 bytes between linear anchors
- The first linear anchor exactly matched the first alignment record for this file
- The last linear anchor was 89,055 compressed bytes before the BGZF EOF member

This supports trying linear-index anchors in a later flagstat experiment: this input has enough anchors for many CUDA workgroups, and no individual observed gap approaches a tentative hundreds-of-megabytes batch size. It does not yet prove whole-file coverage for arbitrary valid BAI files.

Observed additionally on 2026-09-17:

- `wgpu` adapter 0 was `AMD Radeon RX 6600 (RADV NAVI23)`, Vulkan discrete GPU, Mesa/RADV 25.2.7
- The canonical `wgpu` workload completed exactly on that AMD adapter; CUDA remained available separately through NVIDIA devices

The development environment may require the full CUDA and Vulkan development toolchains. Avoiding development dependencies is not yet a prototype goal.

## Reproduce the current checkpoint

From `/home/agent/gpugeno`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"

cargo fmt --all -- --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings

cargo run --release --example cuda_vector_add -- \
  --device 0 --elements 1048576

cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304

cargo run --release --example vulkan_flagstat_spike -- --device 0

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend vulkan --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 \
  --benchmark --validate

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 \
  --benchmark --validate

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 \
  --benchmark --validate
```

All commands passed at the current checkpoint. `gpugeno flagstat` defaults to `wgpu`; `cuda`, `vulkan`, and `wgpu` are all explicit public backend choices, with Vulkan selecting a physical-device enumeration index and refusing CPU/software devices. The examples are temporary integration/regression diagnostics.

## Rocky Linux 8-compatible release builds

The supported portable-release baseline is currently **x86-64 Rocky Linux 8 with glibc 2.28, CUDA 12.4, and a CUDA `sm_86`/`compute_86` code target**. Here, “portable” means that the executable is linked against the Rocky Linux 8 userspace ABI rather than the newer development-host ABI. It does not yet mean CUDA-free or GPU-architecture-independent: CUDA compilation/linkage remains unconditional, the executable requires `libcudart.so.12` even when a portable backend is selected, and `build.rs` still emits only `sm_86` cubins plus `compute_86` PTX.

[`packaging/rockylinux8/Containerfile`](../packaging/rockylinux8/Containerfile) uses the NVIDIA CUDA 12.4.1 Rocky Linux 8 development image, Rust 1.98.1, and the locked Cargo graph. Pulling NVIDIA's base image is subject to the NVIDIA Deep Learning Container License displayed by that image. The image build fails if the resulting executable requires a glibc symbol newer than `GLIBC_2.28` or no longer records `libcudart.so.12` as a dynamic dependency. No GPU is needed to compile. The final local image is also based on the matching Rocky Linux 8 CUDA runtime and contains the Vulkan loader, but the exported executable is the primary release artifact.

Build and export it from the repository root with Podman:

```bash
./scripts/build-rocky8-release.sh
```

The default output is ignored by Git and contains:

```text
dist/rockylinux8-x86_64/gpugeno
dist/rockylinux8-x86_64/gpugeno.sha256
dist/rockylinux8-x86_64/release-info.txt
```

The wrapper also leaves the runnable local image `localhost/gpugeno-release:rockylinux8`. Pass one positional path to select another output directory. `GPUGENO_RELEASE_IMAGE` overrides the image tag, and `GPUGENO_PODMAN_BUILD_ARGS` supplies simple whitespace-separated extra build flags such as `--pull=always`. `release-info.txt` records the Git revision, whether the source tree was dirty, the Cargo lockfile digest, compiler versions, glibc ceiling, and direct dynamic dependencies. Normal rootless Podman needs no privilege. In a restricted nested environment where user namespaces are unavailable but non-interactive sudo is authorized, use:

```bash
GPUGENO_PODMAN_SUDO=1 ./scripts/build-rocky8-release.sh
```

Verify the exported checksum from inside its directory because the checksum deliberately contains only the artifact basename:

```bash
(
  cd dist/rockylinux8-x86_64
  sha256sum -c gpugeno.sha256
)
```

Before deployment, run `ldd ./gpugeno` on the destination and require every dependency to resolve. A native Rocky Linux 8 destination needs the matching CUDA 12 runtime (`libcudart.so.12`) and `libstdc++`; with NVIDIA's RHEL 8 CUDA repository configured, the relevant packages are normally `cuda-cudart-12-4` and `libstdc++`. CUDA execution additionally needs an NVIDIA driver compatible with CUDA 12.4 and either an `sm_86` GPU or a newer architecture on which the driver can JIT the embedded `compute_86` PTX. Direct Vulkan or `wgpu` execution needs `libvulkan.so.1` (normally `vulkan-loader`) and a hardware Vulkan ICD; the direct backend requires Vulkan 1.1. The NVIDIA driver does not by itself provide the CUDA runtime. Install the runtime package or configure the dynamic loader to find `/usr/local/cuda-12.4/lib64`.

The container build deliberately does not run `cargo test`: the complete suite contains hardware tests requiring real Vulkan adapters. Continue to run the normal verification suite on a GPU-equipped development host, then smoke-test the exported executable on the actual Rocky Linux 8 destination with the intended backend and representative BAM/BAI input.

The first verified container build on 2026-09-23 produced a 7.7 MiB executable whose maximum required glibc symbol was `GLIBC_2.28`. `ldd` resolved all dependencies inside the Rocky Linux 8.9 runtime image, a no-argument launch reached the CLI usage error normally, and `cuobjdump` showed the two expected `sm_86` cubins and two `compute_86` PTX payloads. The exported executable also completed the canonical CUDA/device-0 `--validate` run on the development host with the exact established counters. This establishes the artifact's build/linkage and one representative CUDA execution; a smoke test on the actual destination is still required.

## Platform and devices

- **Decided:** Linux is the current development platform. CUDA development targets NVIDIA, while the portable Vulkan and `wgpu` paths have also been validated on AMD Radeon RX 6600 hardware.
- **Decided:** Single-GPU execution only for now.
- **Deferred:** Multi-GPU. Flagstat batches are naturally reducible, but robust scheduling, per-device resource pools, adapter identity across APIs, timing, and failure handling would add cross-cutting complexity. The batch interface should not make future multi-GPU scheduling unnecessarily difficult.

## Benchmark meaning

- **Decided:** The main interest is performance differences among CUDA, direct Vulkan, and `wgpu`, not total disk-to-output latency.
- **Decided:** Measure host-to-device upload, kernel execution, and device-to-host readback separately. Also report aggregate GPU-stage time where useful.
- **Decided:** Kernel time must not include transfer time in the headline kernel metric.
- **Decided:** Begin with one normal streaming execution of each batch. Resident-buffer repetition and synthetic microbenchmarks are deferred.
- **Decided:** Timing output should be opt-in and should not contaminate normal samtools-style `stdout`; benchmark information belongs on `stderr` initially.
- **Working decision:** Prefer GPU timestamp mechanisms for kernel timing. Any host-timed fallback must be labeled rather than presented as equivalent.
- **Decided:** Each backend may be tuned for its own best performance.

## Testing

- **Decided:** Automated tests should be self-contained and must not require samtools or the old libshadowfax executable.
- **Working decision:** Generate small valid synthetic BAM/BAI fixtures during tests and compare exact counters to golden values.
- **Recommended fixture coverage:** all relevant flag bits, QC-pass/fail separation, primary/secondary/supplementary precedence, paired combinations, MAPQ 4/5 boundary, records crossing BGZF blocks, repeated/empty BAI intervals, batch boundaries, and trailing unmapped records.
- External comparisons with libshadowfax or samtools may exist as optional developer tools, not required tests.
- A public CPU backend is deferred. A small host classifier may be used internally as a test oracle if it remains simple.

## Recording new evidence

- Label one-off observations as smoke data; do not present them as benchmark distributions.
- For controlled benchmarks, record the executable or commit, input, hardware/adapter, command matrix, warmup and measured counts, ordering, cache conditions, validation policy, failures, and whether any run was discarded.
- Keep normal result output on `stdout` and benchmark/device metadata on `stderr`.
- Preserve methodology defects and limitations in the report.
- Store durable conclusions and sufficient reproduction detail in [`benchmarks.md`](benchmarks.md); raw logs need not become repository dependencies.
- Update [`README.md`](README.md) only with the current consequence and a link to the detailed report.
