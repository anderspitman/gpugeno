# Current architecture and decisions

**Purpose:** Describe how the current system works and the decisions that constrain implementation.
**Read when:** Changing orchestration, bounded streaming, backend ownership, GPU resource use, the CLI, or cross-backend behavior.
**Orientation:** This is a present-tense document. Dated implementation provenance belongs in [`experiments.md`](experiments.md), and performance evidence belongs in [`benchmarks.md`](benchmarks.md).

## Contents

- [System overview](#system-overview)
- [Repository and source ownership](#repository-and-source-ownership)
- [Scope and compatibility](#scope-and-compatibility)
- [Input, batching, and overlap](#input-batching-and-overlap)
- [CUDA flagstat path](#cuda-flagstat-path)
- [Native `wgpu` path](#native-wgpu-path)
- [Direct Vulkan path](#direct-vulkan-path)
- [Language and backend organization](#language-and-backend-organization)
- [CLI](#cli)

## System overview

```text
seekable BAM + BAI
    -> metadata and physical flagstat anchors
    -> bounded BGZF framing and persistent libdeflate workers
    -> canonical decompressed BAM batches and record-aligned spans
    -> selected CUDA, Direct Vulkan, or wgpu GPU classifier
    -> per-span partial counters
    -> checked host reduction and samtools-style output
```

The main thread opens the indexed stream and then moves it to a named producer. A zero-capacity rendezvous overlaps construction of the next host batch with validation and one synchronous backend slot. The protocol permits at most two complete canonical host batches: one consumer-owned batch and one completed producer-owned batch blocked in `send`. Receiver disconnection precedes producer join on cancellation, and channel disconnection without an explicit EOF message is an error.

`IndexedBamBatch.data` is the operation-neutral canonical byte stream. All backends receive deterministic batches and span starts, while retaining independent kernels, resource layouts, workgroup choices, and timing mechanisms.

## Repository and source ownership

The root now contains a bounded streaming Rust/CUDA/Direct-Vulkan/`wgpu` flagstat prototype:

- `Cargo.toml` and `Cargo.lock`
- `build.rs`: invokes `nvcc`/`ar` for CUDA and uses build-time `naga` to validate and compile the dedicated direct-Vulkan WGSL shader to embedded SPIR-V
- `src/main.rs` and `src/batch_producer.rs`: explicit CUDA/Direct Vulkan/`wgpu` CLI orchestration plus the bounded rendezvous producer that overlaps next-batch construction with one synchronous backend slot
- `src/lib.rs`: safe Rust owner around the opaque CUDA C context, including flagstat, vector-add, and raw upload operations
- `src/wgpu_backend.rs` and `src/flagstat.wgsl`: native `wgpu` adapter/device ownership, direct canonical-byte upload through one grow-only synchronized resource slot, reusable timestamps/readback, and the real portable compute classifier
- `src/vulkan_backend.rs` and `src/vulkan_flagstat.wgsl`: public direct-`ash` Vulkan ownership, one synchronized grow-only resource slot, coherent/non-coherent mapped-memory handling, transfer/compute barriers, per-stage dispatch/readback, timestamps, and the dedicated SPIR-V classifier
- `src/bgzf.rs`: validated BGZF framing, the sequential prefix diagnostic, virtual offsets, and persistent bounded libdeflate workers
- `src/bam.rs`: BAM header parsing, flagstat counters/text, and independent host classifier
- `src/bai.rs`: bounded BAI parsing that preserves coordinate-bearing repeated linear work items and derives the flagstat physical-anchor view
- `src/indexed_batch.rs`: deterministic bounded virtual-offset planning, parallel member decompression/reordering, and disjoint physical batch streaming
- `cuda/gpugeno_cuda.h` and `cuda/gpugeno_cuda.cu`: public C ABI, reusable CUDA context/buffers, transfer/timing orchestration, and temporary vector-add/upload support
- `cuda/flagstat.cuh` and `cuda/flagstat.cu`: internal launch declaration plus the flagstat classifier and bounded CUDA record-walk kernel
- `examples/cuda_vector_add.rs`, `examples/bam_upload.rs`, and `examples/vulkan_flagstat_spike.rs`: temporary integration/regression diagnostics
- `AGENTS.md`: repository workflow and documentation-maintenance instructions for coding agents
- `docs/README.md`: mandatory shared entry point with the current checkpoint, current work, constraints, risks, and reading routes
- `docs/architecture.md`, `docs/development.md`, `docs/flagstat.md`, `docs/experiments.md`, `docs/benchmarks.md`, and `docs/history.md`: progressively disclosed current reference, evidence, and history
- `cubayes/`: a clean CuBayes reference clone
- `libshadowfax/`: a clean experimental fork containing the CUDA flagstat implementation
- `.gitignore`: ignores build output, local reference clones, editor swap files, and alignment/index data

The root is a Git repository on branch `main`. The implementation, project metadata, and routed documentation are tracked. The latest implementation checkpoint is the reviewed bounded CPU/GPU overlap orchestration built on the public direct-Vulkan whole-file backend; the prior synthetic Vulkan checkpoint is preserved in [`experiments.md`](experiments.md#superseded-implementation-history-direct-vulkan-synthetic-flagstat-integration-spike). `cubayes/` and `libshadowfax/` remain separate ignored reference repositories.

Reference revisions and locations at the time of this update:

- `cubayes/`: branch `main`, commit `9687f167bdca43d19d379ed83cd52b98983fce6a`; upstream `git@gitlab.com:shadowfaxbio/cubayes`
- `libshadowfax/`: branch `main`, commit `2db65c0b27000e5302ac57f0d9ef2be878662ee8`; this environment's origin is `/agents/shadowfax/origin/libshadowfax/`

## Scope and compatibility

- **Decided:** Start with flagstat, not pileup.
- **Decided:** The initial semantic and algorithmic reference is `libshadowfax/lib/shadowfax/flagstat.cu`.
- **Decided:** CuBayes remains the authoritative reference for the broader BAM/pileup project, but mainline `cubayes/` has no flagstat kernel.
- **Decided:** Match the useful flagstat behavior closely; do not reproduce accidental bugs in the old stream wrappers.
- **Decided:** Default result output should look like `samtools flagstat`.
- **Observed:** The canonical `HG002_chr22.bam` flagstat output is byte-identical to samtools 1.24 and all three GPU backends. Broader version/corpus compatibility remains deferred; canonical agreement does not establish every samtools edge case.

## Input, batching, and overlap

- **Decided:** Initial input is a valid, seekable, coordinate-sorted BGZF-compressed BAM with a BAI index.
- **Decided:** BAI only. CSI, SAM, CRAM, and non-seekable input are deferred.
- **Decided:** Processing must be bounded-memory streaming; loading or decompressing the complete BAM is not acceptable.
- **Decided:** Use libdeflate on persistent CPU workers rather than implementing GPU BGZF decompression. `--threads N` controls the shared indexed stream, and the evidence-based default is 8.
- **Decided:** Keep worker in-flight data bounded: at most one compressed member is assigned to each worker, while completed out-of-order data remains inside the bounded outer batch. Every worker owns and reuses exactly one decompressor for its lifetime.
- **Decided:** Use record-aligned BAI virtual offsets to expose many independent GPU work spans. Do not begin with a CPU-generated offset for every BAM record.
- **Working decision:** All GPU backends should receive deterministic, identical outer batches and spans. A backend may subdivide them internally.
- **Decided:** Use a fixed 256 MiB default uncompressed batch cap and expose `--max-uncompressed-bytes`.
- **Decided:** Whole-file orchestration opens the indexed stream on the main thread, then moves it to one named producer before backend construction. A zero-capacity rendezvous delivers explicit batch/error/EOF messages while validation and one synchronous backend slot remain on the main thread.
- **Decided:** The overlap envelope permits at most two complete canonical host batches. Receiver disconnect precedes producer join on every cancellation path, and disconnect alone is never clean EOF.
- **Decided:** Existing stage timings remain work sums that may now overlap. Benchmark output reports producer/consumer wait telemetry and explicitly forbids summing overlapping stages to infer wall time.

## CUDA flagstat path

- **Decided:** Process the complete representative BAM through bounded streaming batches; this is not another prefix-only experiment.
- **Decided:** Deliver a CUDA-only `gpugeno flagstat` path with samtools-style output and separate H2D, kernel, and D2H measurements when benchmarking is requested.
- **Decided:** Take architectural inspiration from current CuBayes pileup and classifier semantics from libshadowfax flagstat, but do not copy known skip/final-batch defects or unsafe assumptions. `cubayes/src/cubayes_main.cu` plus `cubayes/lib/cubayes/pipeline.h` are the current orchestration reference. The project owner reports that `cubayes_main_actor.cu` is a faster parallel experiment suspected of crash/freeze bugs; it is not a correctness or lifecycle reference, and any idea taken from it requires independent validation.
- **Decided:** Preserve BAI linear entries as coordinate-bearing work items in a reusable shared representation. Flagstat may derive sorted, deduplicated physical anchors to count each BAM record exactly once. Future pileup must be able to retain repeated offsets and genomic windows because overlap semantics differ.
- **Decided:** Separate metadata/work planning, bounded BGZF decompression plus virtual-to-batch offset translation, backend execution, and result reduction. Keep only bounded batch data resident so worker pools, double buffering, or stage overlap can be added without replacing the operation contract.
- **Decided:** Add explicit first-record and physical-end coverage for header-adjacent and trailing/unindexed records. Never silently skip zero-length, oversized, or final spans. The first implementation may clearly reject an anchor gap that cannot fit the configured batch.
- **Decided:** Use the safer bounded record-walk ideas in newer CuBayes pileup code where useful rather than requiring a literal copy of libshadowfax's duplicate `nth_read` traversal.
- **Decided:** Validate the real CUDA totals against a small independent Rust classifier used as an oracle, not exposed as a public CPU backend.
- **Historical slice boundary:** parallel libdeflate, pipeline overlap, Vulkan, and `wgpu` were explicit non-goals of the original CUDA slice; they were implemented in later approved slices where documented. Pinned memory, multi-GPU, CSI/SAM/CRAM, and pileup remain outside the current implementation.

## Native `wgpu` path

- **Decided:** `wgpu` is now the default, while `--backend cuda`, `--backend vulkan`, and `--backend wgpu` remain explicit. Direct Vulkan is a public backend and uses no fallback; no selector silently changes backend.
- **Decided:** Numeric `--device` means the index in `wgpu`'s enumerated adapter list for the `wgpu` backend. The selected adapter name, underlying API, device type, driver, and timing source are always reported on `stderr`.
- **Decided:** Reject adapters reported as CPU devices. This prevents an explicit GPU backend request from quietly becoming llvmpipe or another software implementation.
- **Decided:** Upload arbitrary BAM bytes directly from canonical `IndexedBamBatch.data` into WGSL-readable `array<u32>` storage. The mapped upload copy clears only the final zero to three padding bytes; a uniform carries the separate logical byte count. Report the eliminated packing metric as zero, mapped staging writes, and resource growth/binding setup separately rather than folding them into transfer or kernel time.
- **Decided:** The WGSL kernel uses one 128-lane workgroup per existing span, a bounded shared record-offset table, workgroup atomic `u32` counters, and one 32-counter partial per span. The host widens and reduces partials into the common `u64` counters.
- **Decided:** When both required timestamp features exist, H2D copies, the compute pass, and D2H copies use GPU timestamps. Otherwise the same three stages are separately submitted, synchronized, host-timed, and labeled `host-synchronized`.
- **Completed optimization boundary:** `IndexedBamBatch.data` remains canonical. One synchronized slot reuses grow-only operation-neutral BAM/upload and span buffers separately from flagstat's pipeline, parameters, results/statuses, timestamps, and readbacks. BGZF decompression remains independent of `wgpu` memory.
- **Observed resolved performance concern:** the first `wgpu` representative run spent about 4.6 seconds packing bytes and 2.2 seconds filling fresh staging buffers; after parallel libdeflate, repeated runs still spent roughly 3.7–3.8 seconds packing and 1.9–2.1 seconds staging. The completed raw-upload/reuse task eliminated packing and reduced the paired staging observation to 314 ms without hiding either cost in GPU timing.
- **Historical slice boundary:** pipeline overlap, parallel decompression, non-NVIDIA validation, and Direct Vulkan were outside the original `wgpu` slice and were addressed later. Pileup, browser execution, and CUDA tuning remain deferred.

## Direct Vulkan path

- **Decided:** `--backend vulkan` is public and consumes every streamed batch's canonical `IndexedBamBatch.data` and `span_starts`. `--device N` is a nonnegative Vulkan physical-device enumeration index; selected metadata (index, name, type, vendor/device IDs, API version, timing source) is always printed to `stderr`. CPU, `OTHER`, and known software renderer devices fail without fallback.
- **Decided:** `VulkanContext` owns one synchronized grow-only slot containing device-side BAM/span/parameter/result/status buffers, matching host-visible transfer buffers, and one descriptor pool/set binding only the five device buffers. Device memory prefers `DEVICE_LOCAL`; transfer memory requires `HOST_VISIBLE` and prefers `HOST_COHERENT`. Allocation size is tracked separately from buffer capacity, and whole-allocation flush/invalidate handles non-coherent mappings.
- **Decided:** Capacity classes are bounded next powers of two clamped to `maxStorageBufferRange`; only pairs whose required logical size exceeds their capacity are replaced. The 16-byte parameter pair is created once. The previous fence is waited before descriptor/buffer growth, and descriptor pools are rebuilt only after a bound device buffer changes or no set exists.
- **Decided:** The timestamp path records one command buffer/submission with six query timestamps around upload copies, the compute dispatch, and readback copies, with explicit transfer→compute and compute→transfer buffer barriers. The fallback uses three separately synchronized submissions and reports `host-synchronized`; no combined host interval is presented as three GPU stages. H2D/kernel/D2H exclude host staging/setup.
- **Decided:** The dedicated 128-lane Vulkan classifier retains the proven precedence, little-endian byte extraction, bounded offset table, checks before every fixed-field read, and statuses 1/2/3. Any nonzero status fails the backend call with span, numeric status, and description; no partial counters are reduced.
- **Decided:** Benchmark output separately labels Vulkan host staging writes, resource setup/growth, zero removed packing, and `raw-little-endian` mode. No Vulkan fallback to `wgpu`, CUDA, or a host classifier is permitted.

## Language and backend organization

- **Working decision:** Rust owns the eventual CLI, streaming pipeline, common types, and backend abstraction.
- **Decided for the integration spike:** CUDA host code and kernels are compiled by `nvcc` into a static object/archive, linked into the Rust executable, and exposed through a narrow C ABI. This is not a separately installed or distributed C library.
- **Rationale:** Native `wgpu` is best supported from Rust; `ash` provides direct Vulkan access; CUDA has a stable C-facing host boundary. This appears less risky than using WebGPU from C or C++.
- **Considered but not selected:** Compile `.cu` to PTX/cubin/fatbin, embed it in the Rust binary, and manage CUDA directly from Rust through a Driver API crate. This remains technically possible, but the project owner prefers C APIs and chose the C host boundary for the spike.
- **Decided:** A CUDA-enabled prototype may require a complete CUDA development environment.
- **Decided:** `wgpu` means a native command-line backend initially, not browser execution. Keep compute/input boundaries separate enough that a browser host is not needlessly prevented later.
- **Decided:** Backends may use different kernels, workgroup sizes, reductions, buffer layouts, and other optimizations as long as the results agree.

## CLI

The intended shape is:

```text
gpugeno flagstat INPUT.bam [OPTIONS]
```

Current option decisions:

- `--backend cuda|vulkan|wgpu`; default is now `wgpu`. Direct Vulkan is available and selects a Vulkan physical-device enumeration index.
- `--device`; one selected GPU per invocation, initially defaulting to device 0.
- `--max-uncompressed-bytes`; both paths share the 256 MiB default.
- `--threads N`; positive decompression-worker count, default 8.
- `--benchmark`; timing output is opt-in.

An explicitly requested unavailable backend must fail clearly. It must never silently fall back to another backend, because that would invalidate comparisons.

The CUDA and `wgpu` flagstat stages are both available. Explicit selection is retained for reproducible comparisons even though omitted `--backend` now selects `wgpu`.

## Additional current contracts

Platform/device policy, benchmark meaning, and testing requirements are maintained in [`development.md`](development.md). Flagstat field semantics, counter precedence, output, and index-based work discovery are maintained in [`flagstat.md`](flagstat.md).
