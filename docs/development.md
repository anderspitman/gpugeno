# Development, verification, and benchmarking

**Purpose:** Record the current environment, canonical input, build and verification commands, device policy, benchmark semantics, and test expectations.
**Read when:** Building, testing, reproducing the checkpoint, changing device behavior, or collecting performance evidence.
**Caution:** Hardware observations are dated. Confirm them before relying on device counts, paths, or tool availability.

## Contents

- [Development machine and canonical input](#development-machine-and-canonical-input)
- [Reproduce the current checkpoint](#reproduce-the-current-checkpoint)
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
