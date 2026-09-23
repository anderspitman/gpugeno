# gpugeno documentation

**Last updated:** 2026-09-23
**Current checkpoint:** Bounded CPU/GPU overlap is implemented, reviewed, and retained after a controlled comparison of five backend/device combinations. Exact representative correctness is established for CUDA, Direct Vulkan, and native `wgpu` on NVIDIA and AMD hardware. A verified Rocky Linux 8 release build now provides the glibc 2.28 deployment baseline.
**Current work:** The documentation reorganization is complete. No subsequent implementation slice is approved.

This file is the canonical entry point for both people and coding agents. Read it completely before working in the repository. Detailed documents are intentionally not all mandatory; use the required-reading list and documentation map below.

## Current work

The project record is organized as this short shared entry point plus routed architecture, development, semantic, experiment, benchmark, and history documents. The reorganization changed no source code, implementation decision, or project scope.

No implementation task is currently approved. Do not infer one from the deferred possibilities. Present the smallest relevant options and ask the project owner one focused question at a time until a slice is selected.

### Required reading

There is no task-specific required reading while no task is approved. Read this file completely, then use the documentation map for any proposal or question under discussion.

When a task is selected, replace this paragraph with an explicit list of the documents or sections needed to perform it. Do not make every historical report required by default.

## Documentation map

| If you need to… | Read… |
|---|---|
| Understand the current system or change orchestration, batching, ownership, a backend, or the CLI | [`architecture.md`](architecture.md) |
| Build, test, reproduce the checkpoint, select devices, or run a benchmark | [`development.md`](development.md) |
| Change flagstat classification, counters, output, BAM field access, BAI work discovery, or semantic fixtures | [`flagstat.md`](flagstat.md) |
| Understand why an implementation technique exists or revisit an implementation experiment | [`experiments.md`](experiments.md), only the relevant section |
| Interpret or repeat the backend or overlap performance campaigns | [`benchmarks.md`](benchmarks.md), only the relevant campaign |
| Revisit a superseded direction or understand the decision sequence | [`history.md`](history.md), only the relevant section |
| Learn how agents must work in and maintain this repository | [`AGENTS.md`](../AGENTS.md) |

Large documents begin with their own contents and reading guidance. Follow links to particular headings; do not read an entire historical document unless the task genuinely spans it.

## Maintaining this documentation

Documentation maintenance is part of every task. The complete repository instructions are in [`AGENTS.md`](../AGENTS.md#maintaining-the-documentation); the essential rules are:

1. Update documentation in the same task as the code, evidence, or decision it describes.
2. Keep the current checkpoint and current task here accurate.
3. Give detailed facts one authoritative home. A summary elsewhere must link to it rather than reproduce it.
4. Put a conclusion and its current consequences before detailed evidence.
5. Keep present-tense architecture separate from dated experiments and superseded history.
6. Preserve material caveats, negative results, methodology defects, and reasons behind decisions.
7. Use stable file-and-heading links rather than line numbers.
8. Update the required-reading list whenever a task is approved or handed off.
9. Prefer the existing small document set over creating a new file without a clear disclosure boundary.
10. Do not let routine changes, raw logs, or speculative future work accumulate here.

Humans are encouraged to review the same entry point and routed documents that agents use. If required reading is irrelevant, duplicated, stale, or growing without bound, fix the routing or move the detail rather than expecting every future reader to absorb it.

## Project goal

Build a command-line prototype named **`gpugeno`** for comparing GPU implementations of operations over BAM data.

The original target was portable pileup derived from CuBayes. The first operation is **flagstat**, because its small fixed-field classifier establishes and compares CUDA, direct Vulkan, and native WebGPU/`wgpu` compute paths before the project attempts pileup.

The general data path is:

```text
seekable BAM + BAI
    -> bounded reads of BGZF blocks
    -> CPU decompression with persistent libdeflate workers
    -> decompressed BAM batches plus record-aligned work spans
    -> selected GPU backend
    -> partial flagstat counters
    -> host reduction and samtools-style text
```

The public backend selector is:

```text
--backend cuda|vulkan|wgpu
```

Backends may be optimized independently. This compares practical implementations, not literal shader translations.

### Purpose and evidence standard

- This is a research and funding demonstration of realistic CUDA bioinformatics workloads ported to cross-platform GPU APIs, not a production tool for external bioinformatics users.
- Cross-platform evidence is the primary priority. CUDA-only progress does not establish portability.
- Every backend must execute a genuine GPU workload over representative data. It may not hide a CPU implementation or silently fall back.
- Representative output must agree across backends and with independent checks. Memory safety, bounded resources, complete normal-file processing, and honest transfer/kernel/readback timing remain required.
- Production behavior for every malformed BAM, pathological BAI, unusual oversized span, exhaustive samtools edge case, and sophisticated retry policy is deferred unless needed for the demonstration corpus or to avoid an unrealistic advantage.
- Actual non-NVIDIA execution has been demonstrated on an AMD Radeon RX 6600 through both Direct Vulkan and native `wgpu`/Vulkan with exact canonical counters.

## Current checkpoint

The repository contains a bounded streaming Rust/CUDA/Direct-Vulkan/`wgpu` flagstat prototype. `wgpu` is the default backend; all three public backends execute genuine GPU classifiers and must fail rather than silently fall back.

The latest technical checkpoint is the reviewed bounded CPU/GPU overlap orchestration:

- metadata and stream construction begin on the main thread;
- one named producer constructs canonical host batches;
- a zero-capacity rendezvous permits at most two complete canonical host batches;
- the consumer retains one synchronous GPU resource slot;
- every cancellation path disconnects the receiver before joining the producer;
- disconnect alone is not clean EOF;
- existing stage timings remain work sums that may overlap and must not be added to infer wall time.

A controlled warm-cache campaign compared the preserved no-overlap binary, the reviewed overlap candidate, and samtools 1.24 across CUDA/NVIDIA, Direct Vulkan AMD/NVIDIA, and `wgpu` AMD/NVIDIA. All outputs and coverage fields remained exact. Every candidate row improved both external elapsed and program wall, so overlap is retained despite increased batch-building, host-staging, and some GPU-stage work under contention. See [`benchmarks.md`](benchmarks.md#bounded-cpugpu-overlap-implementation-and-performance-evaluation).

A checked-in Podman release workflow now builds against Rocky Linux 8/glibc 2.28 with CUDA 12.4 and Rust 1.98.1, rejects a newer glibc requirement, and exports the executable with a checksum and toolchain/dependency record. The first build passed its linkage checks inside the Rocky Linux 8 runtime image. This solves the requested old-userspace build baseline, not the separate CUDA-free packaging gap; see [`development.md`](development.md#rocky-linux-8-compatible-release-builds).

No implementation slice after this checkpoint is approved.

## Critical constraints

- Input is currently valid, seekable, coordinate-sorted BGZF-compressed BAM with a BAI index. CSI, SAM, CRAM, and non-seekable input are deferred.
- Processing must remain bounded-memory streaming; complete BAM loading or decompression is not acceptable.
- The default uncompressed batch cap is 256 MiB and the default decompression worker count is eight; both have CLI controls where documented.
- All backends receive deterministic canonical batches and spans. Backend-specific kernels and resource layouts may differ.
- Requested unavailable, CPU, or software GPU adapters fail clearly. No backend selector silently changes implementation.
- Normal output remains samtools-style flagstat text on `stdout`; benchmark and device metadata belong on `stderr`.
- H2D, kernel, and D2H timing must remain separate. Host-timed fallbacks must be labeled, and overlapping stage totals are not a wall-time decomposition.
- `cubayes/` and `libshadowfax/` are read-only references unless the owner explicitly approves a reference edit.
- Only the next explicitly selected small experiment is committed. Deferred possibilities are not a roadmap.

## Current known risks and deferred boundaries

- CUDA compilation and linkage remain unconditional. The Rocky Linux 8 recipe provides an old-glibc-compatible CUDA-enabled artifact, but the nominally portable backends still do not have a CUDA-free build.
- Complete coverage is demonstrated on the canonical HG002 chromosome 22 BAM, not every unusual or sparse BAI. An indivisible span larger than the configured cap fails clearly rather than being skipped.
- Available Vulkan hardware exercised coherent memory and GPU timestamp paths. Non-coherent mapping and host-synchronized timing fallbacks were reviewed but not selected by the smoke devices.
- A same-process direct-Vulkan/`wgpu` hardware-test SIGSEGV is mitigated by a crate-local test lock, not root-caused for concurrent production API use.
- Cancellation cannot forcibly interrupt a native libdeflate call that is permanently stuck; producer join waits for the active bounded call to return or unwind.
- The portable overlap candidates remained slower than samtools in the controlled warm-cache comparison, although all improved materially over their no-overlap baselines.
- Pileup, CUDA-free packaging, backend tuning, multiple GPU slots, multi-GPU operation, direct mapped decompression, and broader format compatibility remain possible future experiments, not approved work.

Read the linked architecture, benchmark, or history section before acting on one of these boundaries.

## Repository status and references

The root is a Git repository on branch `main`. Rust owns the CLI, streaming pipeline, and portable backends; CUDA host and kernel code is compiled with `nvcc` behind a narrow C ABI. The detailed source map and ownership model are in [`architecture.md`](architecture.md).

The canonical representative input is `/agents/shadowfax/data/HG002_chr22.bam` with its adjacent `.bai`. Current environment, hardware, and verification commands are in [`development.md`](development.md).

Reference repositories:

- `cubayes/`: clean CuBayes reference clone; historically recorded at `9687f167bdca43d19d379ed83cd52b98983fce6a`.
- `libshadowfax/`: clean experimental reference fork containing CUDA flagstat; historically recorded at `2db65c0b27000e5302ac57f0d9ef2be878662ee8`.

Their role and semantic authority are described in [`flagstat.md`](flagstat.md) and [`history.md`](history.md).
