# gpugeno design and project memory

**Last updated:** 2026-09-21
**Current phase:** the bounded CPU/GPU overlap implementation-and-correctness checkpoint is complete and has passed full-agent concurrency review. One named producer owns the existing indexed stream and hands each batch through a zero-capacity rendezvous to the main-thread synchronous backend, retaining one GPU slot and at most two complete canonical host batches. Deterministic lifecycle/equivalence tests and five complete CUDA/Direct Vulkan/`wgpu` correctness preflights passed exactly. The full controlled before/after/samtools performance campaign remains a separate unapproved task.

## First-class fresh-agent workflow

This project intentionally has **no persistent coordinator agent**. `design.md` is the handoff boundary. A fresh coding agent should need only the instruction “read `design.md` and continue.” Every agent is responsible for leaving the repository and this document ready for the next fresh agent.

### Start of every agent session

1. Read this document completely before editing.
2. Run `git status --short`, inspect recent `git log`, and verify that the documented current state still matches the checkout.
3. Inspect only the source and reference files needed for the approved task; do not reload the full historical codebase by default.
4. If this document says no next slice is approved, do not infer one. Present the smallest relevant options and ask the project owner one focused question at a time until one slice is selected.
5. Record the selected slice and its explicit non-goals here before or alongside implementation so an interrupted session is recoverable.

### During work

- Work in small vertical experiments. Do not silently absorb likely future tasks into the current one.
- A session may complete several small tasks only when each is explicitly selected from evidence produced by the previous task.
- Treat `cubayes/` and `libshadowfax/` as read-only references unless the project owner explicitly decides otherwise.
- Investigate reversible implementation details independently. Ask before changing product scope, compatibility, benchmark meaning, public interfaces, or expensive architecture.
- Test normal behavior, relevant boundaries, and native error/resource-lifetime paths—not just the successful path.
- Keep commits small and independently understandable.

### End of every task or session

1. Review the complete diff and run the documented verification commands.
2. Update this file in the same task with:
   - current phase and repository state;
   - progress checklist;
   - exact implementation and commands;
   - observed measurements, labeled as smoke data or benchmarks;
   - decisions and rationale;
   - disproven assumptions, defects found during review, and remaining risks;
   - the next approved slice, or an explicit statement that none is approved.
3. Remove stale present-tense instructions. Preserve useful superseded decisions in the history section instead of leaving contradictory “current” guidance.
4. Commit code and documentation together unless the task is blocked. Leave a clean working tree. If blocked or intentionally uncommitted, state exactly why and what remains both here and in the final response.
5. Stop at the approved boundary. The next fresh agent must be able to continue from this document without access to prior chat transcripts.

### Current handoff

The bounded overlap implementation-and-correctness checkpoint is complete in commit history. `DisjointBamStream::open` remains on the main thread so metadata/open error order and `anchor_count` are unchanged; the opened stream then moves to one named producer started immediately before main-thread backend construction. `sync_channel(0)` carries explicit `Batch`, `StreamError`, and `Eof` messages. A disconnect is never EOF. Validation, all three backend contexts/calls, reduction, accounting, and output remain on the main thread, and every backend retains one synchronous resource slot.

The rendezvous channel permits exactly the consumer's current complete batch plus one producer-owned completed next batch blocked at send; a buffered capacity-one channel was rejected because it could permit three. At the 256 MiB cap this adds a second approximately 256 MiB logical canonical host batch, subject to `Vec` capacity, metadata, worker, backend upload, and device-buffer caveats. On cancellation, the receiver is dropped before producer join. Stream, validation, backend, and consumer root errors are preserved; producer panic or an unexpected exit kind is appended as secondary cleanup/protocol context. Producer teardown explicitly drops the stream and joins its persistent BGZF workers before reporting lifetime. A non-panicking RAII fallback prevents detachment during unwind.

Existing stage metrics retain their work-sum meanings. Benchmark mode additionally reports first/later/EOF consumer waits, first/later/terminal producer rendezvous waits, and producer lifetime, plus an explicit warning that batch build, validation, backend host stages, and GPU stages overlap and must not be summed into wall. Normal stdout is unchanged.

The implementation has passed deterministic ordering, strict two-live-item backpressure, stream error, cancellation-while-building/sending, producer panic, protocol-disconnect, compile-time `Send`, and real sequential-versus-rendezvous BAM/BAI equivalence tests. Five complete `--benchmark --validate` preflights on CUDA/NVIDIA, Direct Vulkan AMD/NVIDIA, and `wgpu` AMD/NVIDIA retained exact canonical coverage, host counters, byte-identical output, and SHA-256. These validation-enabled runs are correctness smoke observations, not performance evidence; the full before/after/samtools campaign remains deferred and is not approved. Exact details are in **Completed bounded CPU/GPU overlap implementation and correctness checkpoint**.

## Purpose of this document

This is the durable memory for `gpugeno`. Its goal is to let a future coding agent reconstruct the project as closely as practical without needing the original conversation.

It must preserve three kinds of information:

1. **Current truth:** what exists now, what we are building next, and how close it is to working.
2. **Intent and rationale:** goals, constraints, interfaces, and why important choices were made.
3. **History:** assumptions that proved wrong, approaches that were superseded, and evidence learned during implementation.

This is a living design record, not a fixed long-range roadmap. The project is deliberately experimental. Only the next small slice should be treated as committed; later work should be selected after measuring and learning from that slice.

### How to update it

Every coding agent working on this project should read this file first and update it as part of the same change when it learns something material.

- Keep **Current phase**, **Current state**, **Current decisions**, and **Current handoff** accurate.
- Append concise entries to **Decision and discovery history**; do not erase useful history when a decision changes.
- Mark an old decision as superseded and link it to the replacement.
- Record measured facts with enough detail to reproduce them: command, input, hardware, and relevant result.
- Distinguish decisions from untested hypotheses. Do not turn an implementation convenience into a project requirement without discussion.
- Ask the project owner one focused question at a time when product scope, compatibility, benchmark meaning, or a costly tradeoff is unclear.
- Do not bloat this file with raw logs, ordinary refactors, abandoned code sketches, or speculative future task lists. Summarize what a later agent needs to reason correctly.
- Update progress and unresolved risks after each completed experiment before proposing the next slice.

Status words used below:

- **Decided:** explicitly selected as the current direction.
- **Working decision:** the best current choice, but expected to be validated experimentally.
- **Hypothesis:** not yet demonstrated.
- **Deferred:** intentionally outside the current slice, not necessarily rejected forever.
- **Observed:** verified from source, tools, or a completed experiment.

## Project goal

Build a command-line prototype named **`gpugeno`** for comparing GPU implementations of operations over BAM data.

The original target was portable pileup derived from CuBayes. The first operation is now **flagstat**, because its small fixed-field classifier is a better way to establish and compare the CUDA, direct Vulkan, and native WebGPU/`wgpu` compute paths before attempting pileup.

The general data path is:

```text
seekable BAM + BAI
    -> bounded reads of BGZF blocks
    -> CPU decompression with multiple libdeflate workers
    -> decompressed BAM batches plus record-aligned work spans
    -> selected GPU backend
    -> partial flagstat counters
    -> host reduction and samtools-style text
```

The intended backend selector is:

```text
--backend cuda|vulkan|wgpu
```

Backends are allowed to be optimized independently. This is a comparison of practical implementations, not a requirement to run transliterations of one identical shader.

### Prototype purpose and evidence standard

- **Decided:** This is a research/funding demonstration of porting realistic CUDA bioinformatics workloads to cross-platform GPU APIs, not a production tool intended for external bioinformatics users.
- **Decided:** Cross-platform evidence is now the primary project priority. CUDA-only pileup progress does not by itself address the proposal's portability claim.
- **Decided:** A backend must execute a genuine GPU classifier/pileup workload with representative data. It must not hide a CPU implementation behind a backend selector or silently fall back.
- **Decided:** Results should agree on the canonical representative input and targeted synthetic fixtures so comparisons remain credible. Memory safety, bounded resources, complete normal-file processing, and honest transfer/kernel/readback timing remain required.
- **Deferred:** Production-grade behavior for every malformed BAM, pathological BAI, unusual oversized span, exhaustive samtools compatibility detail, and sophisticated retry/recovery. Implement these only when needed to avoid crashes on the demonstration corpus or a materially unrealistic performance advantage.
- **Observed:** Portability has now been demonstrated on actual non-NVIDIA hardware: the complete canonical workload ran through native `wgpu`/Vulkan on an AMD Radeon RX 6600 with exact counters. A CUDA-free build artifact is still required for convenient deployment to machines without the CUDA toolkit/runtime.

## Current state

### Repository

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
- `idea.md`: the original pileup-oriented idea
- `design.md`: this document
- `cubayes/`: a clean CuBayes reference clone
- `libshadowfax/`: a clean experimental fork containing the CUDA flagstat implementation
- `.gitignore`: ignores build output, local reference clones, editor swap files, and alignment/index data

The root is a Git repository on branch `main`. The implementation, project metadata, and this design record are tracked. The latest implementation checkpoint is the reviewed bounded CPU/GPU overlap orchestration built on the public direct-Vulkan whole-file backend; the prior synthetic Vulkan checkpoint is preserved below as superseded history. `cubayes/` and `libshadowfax/` remain separate ignored reference repositories.

Reference revisions and locations at the time of this update:

- `cubayes/`: branch `main`, commit `9687f167bdca43d19d379ed83cd52b98983fce6a`; upstream `git@gitlab.com:shadowfaxbio/cubayes`
- `libshadowfax/`: branch `main`, commit `2db65c0b27000e5302ac57f0d9ef2be878662ee8`; this environment's origin is `/agents/shadowfax/origin/libshadowfax/`

### Development machine

Observed on 2026-09-15:

- Linux
- Three NVIDIA GeForce RTX 3060 GPUs, 12,288 MiB each
- NVIDIA driver 550.163.01
- CUDA toolkit 12.4; `nvcc` is `/usr/local/cuda/bin/nvcc`
- Vulkan instance 1.4.328; NVIDIA devices expose Vulkan 1.3.277
- Rust stable is installed under `/home/agent/.cargo/bin` (`rustc` and `cargo` 1.98.1), but that directory was not present in the observed `PATH`; `rustfmt` and `clippy` components were added during the spike
- `samtools` was not visible on `PATH`

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

### Reproduce the current checkpoint

From `/home/agent/gpugeno`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"

cargo fmt --check
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

### Progress

- [x] Narrowed the first operation from pileup to flagstat.
- [x] Located and inspected the CUDA reference kernel.
- [x] Selected the initial platform, input assumptions, CLI direction, and benchmark meaning.
- [x] Decided to precede the first BAM vertical slice with a minimal Rust/CUDA integration spike.
- [x] Selected a statically linked CUDA C API rather than Rust-managed embedded PTX for the spike.
- [x] Create the Rust project and run the CUDA vector-add spike.
- [x] Review the spike, correct error-path resource/lifetime issues, and record what it proved.
- [x] Commit the reviewed spike and project metadata.
- [x] Decide the first true vertical slice: decompress one bounded BGZF batch with libdeflate in Rust and upload it to CUDA.
- [x] Implement and review the bounded BGZF upload slice.
- [x] Decide the next experiment from its results: a CUDA-only whole-file flagstat streaming vertical slice.
- [x] Implement the reusable indexed bounded-batch layer and CUDA flagstat path.
- [x] Validate exact counters on `HG002_chr22.bam` against an independent host classifier.
- [x] Implement and validate the native real-WGSL `wgpu` flagstat backend.
- [x] Select bounded parallel libdeflate decompression as the next experiment.
- [x] Implement and benchmark `--threads N` with persistent bounded workers.
- [x] Select raw `wgpu` upload and reusable buffers as the next optimization.
- [x] Eliminate full host packing, reuse one synchronized resource slot, and benchmark the result.
- [x] Implement, review, and verify the approved complete public Direct Vulkan whole-file flagstat backend.
- [x] Implement and fully review bounded CPU/GPU overlap with a strict two-host-batch rendezvous and exact five-backend correctness evidence.
- [ ] Run the deferred controlled no-overlap/overlap/samtools performance campaign after owner approval.

## Current decisions

### Scope and compatibility

- **Decided:** Start with flagstat, not pileup.
- **Decided:** The initial semantic and algorithmic reference is `libshadowfax/lib/shadowfax/flagstat.cu`.
- **Decided:** CuBayes remains the authoritative reference for the broader BAM/pileup project, but mainline `cubayes/` has no flagstat kernel.
- **Decided:** Match the useful flagstat behavior closely; do not reproduce accidental bugs in the old stream wrappers.
- **Decided:** Default result output should look like `samtools flagstat`.
- **Observed:** The canonical `HG002_chr22.bam` flagstat output is byte-identical to samtools 1.24 and all three GPU backends. Broader version/corpus compatibility remains deferred; canonical agreement does not establish every samtools edge case.

### Input and streaming

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

### Completed CUDA flagstat vertical slice

- **Decided:** Process the complete representative BAM through bounded streaming batches; this is not another prefix-only experiment.
- **Decided:** Deliver a CUDA-only `gpugeno flagstat` path with samtools-style output and separate H2D, kernel, and D2H measurements when benchmarking is requested.
- **Decided:** Take architectural inspiration from current CuBayes pileup and classifier semantics from libshadowfax flagstat, but do not copy known skip/final-batch defects or unsafe assumptions. `cubayes/src/cubayes_main.cu` plus `cubayes/lib/cubayes/pipeline.h` are the current orchestration reference. The project owner reports that `cubayes_main_actor.cu` is a faster parallel experiment suspected of crash/freeze bugs; it is not a correctness or lifecycle reference, and any idea taken from it requires independent validation.
- **Decided:** Preserve BAI linear entries as coordinate-bearing work items in a reusable shared representation. Flagstat may derive sorted, deduplicated physical anchors to count each BAM record exactly once. Future pileup must be able to retain repeated offsets and genomic windows because overlap semantics differ.
- **Decided:** Separate metadata/work planning, bounded BGZF decompression plus virtual-to-batch offset translation, backend execution, and result reduction. Keep only bounded batch data resident so worker pools, double buffering, or stage overlap can be added without replacing the operation contract.
- **Decided:** Add explicit first-record and physical-end coverage for header-adjacent and trailing/unindexed records. Never silently skip zero-length, oversized, or final spans. The first implementation may clearly reject an anchor gap that cannot fit the configured batch.
- **Decided:** Use the safer bounded record-walk ideas in newer CuBayes pileup code where useful rather than requiring a literal copy of libshadowfax's duplicate `nth_read` traversal.
- **Decided:** Validate the real CUDA totals against a small independent Rust classifier used as an oracle, not exposed as a public CPU backend.
- **Explicit non-goals:** parallel libdeflate workers, pinned memory, pipeline overlap, Vulkan, `wgpu`, multi-GPU, CSI/SAM/CRAM, pileup itself, and performance tuning beyond stage timings.

### Completed native `wgpu` flagstat vertical slice

- **Decided:** `wgpu` is now the default, while `--backend cuda`, `--backend vulkan`, and `--backend wgpu` remain explicit. Direct Vulkan is a public backend and uses no fallback; no selector silently changes backend.
- **Decided:** Numeric `--device` means the index in `wgpu`'s enumerated adapter list for the `wgpu` backend. The selected adapter name, underlying API, device type, driver, and timing source are always reported on `stderr`.
- **Decided:** Reject adapters reported as CPU devices. This prevents an explicit GPU backend request from quietly becoming llvmpipe or another software implementation.
- **Decided:** Upload arbitrary BAM bytes directly from canonical `IndexedBamBatch.data` into WGSL-readable `array<u32>` storage. The mapped upload copy clears only the final zero to three padding bytes; a uniform carries the separate logical byte count. Report the eliminated packing metric as zero, mapped staging writes, and resource growth/binding setup separately rather than folding them into transfer or kernel time.
- **Decided:** The WGSL kernel uses one 128-lane workgroup per existing span, a bounded shared record-offset table, workgroup atomic `u32` counters, and one 32-counter partial per span. The host widens and reduces partials into the common `u64` counters.
- **Decided:** When both required timestamp features exist, H2D copies, the compute pass, and D2H copies use GPU timestamps. Otherwise the same three stages are separately submitted, synchronized, host-timed, and labeled `host-synchronized`.
- **Completed optimization boundary:** `IndexedBamBatch.data` remains canonical. One synchronized slot reuses grow-only operation-neutral BAM/upload and span buffers separately from flagstat's pipeline, parameters, results/statuses, timestamps, and readbacks. BGZF decompression remains independent of `wgpu` memory.
- **Observed resolved performance concern:** the first `wgpu` representative run spent about 4.6 seconds packing bytes and 2.2 seconds filling fresh staging buffers; after parallel libdeflate, repeated runs still spent roughly 3.7–3.8 seconds packing and 1.9–2.1 seconds staging. The completed raw-upload/reuse task eliminated packing and reduced the paired staging observation to 314 ms without hiding either cost in GPU timing.
- **Explicit non-goals for the completed `wgpu` slice:** pileup, browser execution, pipeline overlap, parallel decompression, CUDA tuning, and acquiring non-NVIDIA hardware. Direct Vulkan was subsequently promoted by the public backend slice below.

### Superseded direct-Vulkan synthetic flagstat spike decisions

- **Historical decision:** The synthetic spike kept direct Vulkan outside `gpugeno flagstat --backend vulkan`; that boundary is superseded by the public backend slice below. Its numeric `--device` semantics remain the Vulkan physical-device enumeration index.
- **Decided:** Accept discrete, integrated, or virtual GPU device types, but reject `CPU`, `OTHER`, and known software-renderer names rather than silently using software. An invalid index reports every enumerated device.
- **Historical implementation:** The spike used small per-call buffers and descriptor pools. The public backend now uses one synchronized grow-only resource slot and rebuilds descriptors only when a bound device buffer changes.
- **Decided:** Keep a dedicated Vulkan WGSL source and compile it reproducibly to embedded SPIR-V 1.3 with build-time `naga` 30.0.1. Installed `glslc`, `glslangValidator`, and SPIR-V tools were absent, so no external compiler/runtime dependency was added.
- **Decided:** Select HOST_VISIBLE memory compatible with each buffer, prefer HOST_COHERENT, and explicitly flush/invalidate the entire allocation for non-coherent memory. Descriptor ranges are exact logical/padded buffer sizes and checked against device limits.
- **Historical timing:** The spike used a two-query compute-only timestamp or one synchronized host interval. The public backend uses six timestamps for H2D/kernel/D2H or three separately synchronized host submissions, with valid-bit masking and `timestampPeriod` conversion.
- **Historical explicit non-goals:** whole-file BAM/BAI streaming, public direct-Vulkan backend selection, production backend/resource abstractions, benchmark comparisons, overlap/double buffering/tuning, pileup, CUDA-free packaging, and reference-repository edits. These were the synthetic-spike boundary and are superseded only for the approved public backend slice; the remaining non-goals are retained in the current handoff.

### Completed public Direct Vulkan whole-file flagstat backend

- **Decided:** `--backend vulkan` is public and consumes every streamed batch's canonical `IndexedBamBatch.data` and `span_starts`. `--device N` is a nonnegative Vulkan physical-device enumeration index; selected metadata (index, name, type, vendor/device IDs, API version, timing source) is always printed to `stderr`. CPU, `OTHER`, and known software renderer devices fail without fallback.
- **Decided:** `VulkanContext` owns one synchronized grow-only slot containing device-side BAM/span/parameter/result/status buffers, matching host-visible transfer buffers, and one descriptor pool/set binding only the five device buffers. Device memory prefers `DEVICE_LOCAL`; transfer memory requires `HOST_VISIBLE` and prefers `HOST_COHERENT`. Allocation size is tracked separately from buffer capacity, and whole-allocation flush/invalidate handles non-coherent mappings.
- **Decided:** Capacity classes are bounded next powers of two clamped to `maxStorageBufferRange`; only pairs whose required logical size exceeds their capacity are replaced. The 16-byte parameter pair is created once. The previous fence is waited before descriptor/buffer growth, and descriptor pools are rebuilt only after a bound device buffer changes or no set exists.
- **Decided:** The timestamp path records one command buffer/submission with six query timestamps around upload copies, the compute dispatch, and readback copies, with explicit transfer→compute and compute→transfer buffer barriers. The fallback uses three separately synchronized submissions and reports `host-synchronized`; no combined host interval is presented as three GPU stages. H2D/kernel/D2H exclude host staging/setup.
- **Decided:** The dedicated 128-lane Vulkan classifier retains the proven precedence, little-endian byte extraction, bounded offset table, checks before every fixed-field read, and statuses 1/2/3. Any nonzero status fails the backend call with span, numeric status, and description; no partial counters are reduced.
- **Decided:** Benchmark output separately labels Vulkan host staging writes, resource setup/growth, zero removed packing, and `raw-little-endian` mode. No Vulkan fallback to `wgpu`, CUDA, or a host classifier is permitted.

### Completed first vertical slice boundaries

- **Decided:** Process only one bounded prefix batch, not the entire BAM.
- **Working default:** Cap the batch at 256 MiB of uncompressed bytes, configurable by the temporary example.
- **Decided:** Discover complete BGZF members and decompress them sequentially with one reused Rust `libdeflater::Decompressor`. Worker pools and stage overlap are later concerns.
- **Decided:** Upload the resulting contiguous bytes through the existing CUDA context and report the CUDA-event H2D time.
- **Decided:** This slice has no GPU compute kernel, readback verification, BAI parsing, BAM header/record parsing, or full-file streaming.

### Language and backend organization

- **Working decision:** Rust owns the eventual CLI, streaming pipeline, common types, and backend abstraction.
- **Decided for the integration spike:** CUDA host code and kernels are compiled by `nvcc` into a static object/archive, linked into the Rust executable, and exposed through a narrow C ABI. This is not a separately installed or distributed C library.
- **Rationale:** Native `wgpu` is best supported from Rust; `ash` provides direct Vulkan access; CUDA has a stable C-facing host boundary. This appears less risky than using WebGPU from C or C++.
- **Considered but not selected:** Compile `.cu` to PTX/cubin/fatbin, embed it in the Rust binary, and manage CUDA directly from Rust through a Driver API crate. This remains technically possible, but the project owner prefers C APIs and chose the C host boundary for the spike.
- **Decided:** A CUDA-enabled prototype may require a complete CUDA development environment.
- **Decided:** `wgpu` means a native command-line backend initially, not browser execution. Keep compute/input boundaries separate enough that a browser host is not needlessly prevented later.
- **Decided:** Backends may use different kernels, workgroup sizes, reductions, buffer layouts, and other optimizations as long as the results agree.

### CLI

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

### Platform and devices

- **Decided:** Initial platform is Linux on NVIDIA GPUs.
- **Decided:** Single-GPU execution only for now.
- **Deferred:** Multi-GPU. Flagstat batches are naturally reducible, but robust scheduling, per-device resource pools, adapter identity across APIs, timing, and failure handling would add cross-cutting complexity. The batch interface should not make future multi-GPU scheduling unnecessarily difficult.

### Benchmark meaning

- **Decided:** The main interest is performance differences among CUDA, direct Vulkan, and `wgpu`, not total disk-to-output latency.
- **Decided:** Measure host-to-device upload, kernel execution, and device-to-host readback separately. Also report aggregate GPU-stage time where useful.
- **Decided:** Kernel time must not include transfer time in the headline kernel metric.
- **Decided:** Begin with one normal streaming execution of each batch. Resident-buffer repetition and synthetic microbenchmarks are deferred.
- **Decided:** Timing output should be opt-in and should not contaminate normal samtools-style `stdout`; benchmark information belongs on `stderr` initially.
- **Working decision:** Prefer GPU timestamp mechanisms for kernel timing. Any host-timed fallback must be labeled rather than presented as equivalent.
- **Decided:** Each backend may be tuned for its own best performance.

### Testing

- **Decided:** Automated tests should be self-contained and must not require samtools or the old libshadowfax executable.
- **Working decision:** Generate small valid synthetic BAM/BAI fixtures during tests and compare exact counters to golden values.
- **Recommended fixture coverage:** all relevant flag bits, QC-pass/fail separation, primary/secondary/supplementary precedence, paired combinations, MAPQ 4/5 boundary, records crossing BGZF blocks, repeated/empty BAI intervals, batch boundaries, and trailing unmapped records.
- External comparisons with libshadowfax or samtools may exist as optional developer tools, not required tests.
- A public CPU backend is deferred. A small host classifier may be used internally as a test oracle if it remains simple.

## Completed prerequisite: Rust/CUDA integration spike

This was deliberately a technical spike rather than a true BAM-to-result vertical slice.

### Goal

Prove the smallest useful Rust/CUDA integration path with vector addition:

```text
Rust host vectors -> statically linked CUDA C API -> H2D copies
                  -> vector-add kernel -> D2H copy -> Rust validation
```

### Required behavior

1. Create the root Rust project named `gpugeno`.
2. Have the Cargo build invoke `nvcc` for a `.cu` implementation and link the resulting static native code plus the CUDA runtime into the Rust executable.
3. Expose a small, explicitly C-compatible API. C++ name mangling or C++ types must not cross the boundary.
4. Select one CUDA device by numeric index.
5. Allocate reusable CUDA state and device buffers on the CUDA side.
6. Add two large `f32` vectors on the GPU, return the result to Rust, and validate it deterministically in Rust.
7. Use explicit status/error reporting across the C API; native code must not terminate the process for a recoverable error.
8. Measure upload, kernel, and readback separately, preferably with CUDA events on one stream.
9. Exercise a large enough vector to make allocation, transfer, and launch behavior visible; 16,777,216 elements is the current suggested default, not yet a compatibility requirement.
10. Update this document with the exact build mechanism, C declarations, command, result, timings, and any ownership or error-handling problems.

### Keep it small

Do not add BAM, BAI, BGZF, libdeflate, `wgpu`, Vulkan, a general backend trait, asynchronous pipeline machinery, or speculative production abstractions to this spike.

The spike succeeds when a clean build produces one Rust program that selects an RTX 3060, launches the statically linked CUDA vector-add kernel, validates the complete result, reports useful timings, and exits cleanly.

The harness will be a **temporary Cargo example**, not a `gpugeno` subcommand or supported CLI surface. The intended invocation shape is:

```bash
cargo run --release --example cuda_vector_add -- --device 0 --elements 16777216
```

It may be deleted once the CUDA integration decisions have been carried into real functionality.

### Implemented result

The spike uses this boundary:

```text
Rust-owned vectors and safe `CudaContext`
    -> fixed `extern "C"` API
    -> CUDA C++ context owning one stream, six events, and reusable buffers
    -> 256-thread bounds-checked CUDA vector-add kernel
```

`build.rs` runs `/usr/local/cuda/bin/nvcc` with C++17, `-O3`, `-arch=sm_86`, and PIC, archives the object with `ar`, and links `libgpugeno_cuda.a`, shared `libcudart`, and `libstdc++` into the Rust example. `NVCC` and `CUDA_HOME` override the defaults and are Cargo rebuild inputs. There is no project shared library, embedded PTX, CUDA Driver API crate, bindgen, or Cargo dependency.

The C API has three functions: create an opaque context for a numeric device, run vector addition with host pointers/count/timings/error buffer, and destroy the context. Calls return explicit status codes and NUL-terminated messages; C++ exceptions are caught at the boundary. The Rust wrapper owns destruction with `Drop` and preinitializes output metadata before unsafe calls.

Review found and corrected two native error-path issues: a context leak if stream creation failed, and immediate returns that could leave asynchronous work touching borrowed Rust slices. Post-enqueue failures now best-effort synchronize before returning the original error.

Verified commands:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example cuda_vector_add -- --device 0 --elements 16777216
```

All passed, and every output element was validated exactly. An independently repeated 16,777,216-element sample on device 0 measured H2D 10.772 ms, kernel 0.726 ms, and D2H 24.918 ms. These are smoke observations, not benchmark results. Device 99 correctly failed with status 1 and reported that only three devices were available. The C symbols were confirmed unmangled in the executable, which dynamically links CUDA runtime 12 and `libstdc++`.

Current limitations are intentional: `sm_86` is hardcoded for the RTX 3060 development target; CUDA runtime and C++ runtime are shared system dependencies; buffers grow exactly to the requested capacity and do not shrink; the context is single-thread-oriented; and this temporary example is not a supported CLI.

## Completed first vertical slice: bounded BGZF decompression and CUDA upload

This was the first real-data path through Rust, libdeflate, and CUDA.

### Goal

Prove this real-data path with one bounded prefix of a valid BAM:

```text
BAM file prefix
    -> discover complete BGZF members
    -> decompress each member through libdeflate from Rust
    -> concatenate at most about 256 MiB of uncompressed bytes
    -> upload the contiguous byte buffer through the CUDA C API
    -> report counts, byte sizes, CPU timing, and H2D timing
```

### Required behavior

1. Add a temporary Cargo example, tentatively `bam_upload`, accepting an input BAM path, `--device`, and `--max-uncompressed-bytes`.
2. Read the input incrementally. Do not load the compressed BAM or decompressed file in full.
3. Parse enough gzip/BGZF framing to validate the gzip magic/method, mandatory FEXTRA/`BC` subfield, `BSIZE`, block bounds, and trailer `ISIZE`.
4. Include only complete BGZF members whose cumulative uncompressed size fits the configured batch cap. A single legal BGZF member must be supportable.
5. Reuse one `libdeflater::Decompressor` and call its gzip decompression path so gzip integrity and output size are checked.
6. Concatenate decompressed members in file order into one Rust-owned byte vector.
7. Extend the existing opaque CUDA context with one narrow upload operation that owns/reuses a raw device-byte buffer and returns H2D timing. Keep the vector-add example working.
8. Synchronize before returning so the Rust input borrow is no longer referenced by CUDA. Preserve the spike's explicit status/error-buffer conventions and error-path lifetime safety.
9. Print the number of blocks, compressed and uncompressed bytes, batch-building/decompression timing, selected device, and CUDA H2D timing.
10. Run a small bounded test and a 256 MiB test against `/agents/shadowfax/data/HG002_chr22.bam`.

### Explicit non-goals

- No GPU kernel, checksum, reduction, or readback
- No BAI or BAM semantic parsing
- No full-file processing
- No libdeflate worker pool or pipeline overlap
- No supported production CLI or backend trait
- No Vulkan, `wgpu`, pileup, or flagstat

The operation proves successful CUDA submission and synchronization, not byte identity on the device; content verification through GPU compute is deliberately deferred.

### Implemented result

`libdeflater` 1.26.0 is now the only Rust runtime dependency. `read_bgzf_prefix` incrementally reads complete members from the file start, validates gzip/BGZF framing (`FEXTRA`, `BC`, `BSIZE`, optional-header bounds, trailer and `ISIZE`), reuses one `Decompressor`, and passes each full member to `gzip_decompress` so libdeflate verifies gzip integrity and CRC. Members are concatenated in order only when their complete uncompressed output fits the configured cap. Determining whether the next member fits requires reading that compressed member to inspect its trailer; an excluded member is neither counted nor decompressed.

The CUDA context now owns a reusable raw-byte device buffer in addition to the vector-add buffers. The new C upload operation reuses the existing stream and H2D event pair, synchronizes before returning, and preserves the established status/error-buffer and post-enqueue lifetime rules. Rust exposes it as `CudaContext::upload(&[u8])`. No device pointer is exposed and no GPU compute or readback occurs.

The temporary command is:

```bash
cargo run --release --example bam_upload -- INPUT.bam \
  --device 0 --max-uncompressed-bytes 268435456
```

Seven self-contained tests cover ordered concatenation, cap boundaries, first-member rejection, a 65,536-byte output member, missing `BC`, truncation, and canonical EOF handling. Independent verification passed `cargo fmt --check`, `cargo test`, `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`, and the vector-add regression.

Observed real-data results on device 0:

```text
4 MiB cap:
  blocks=64 compressed_bytes=1,216,766 uncompressed_bytes=4,162,185
  batch_build=6.353 ms libdeflate=4.695 ms CUDA_H2D=0.470 ms

256 MiB cap:
  blocks=4,128 compressed_bytes=81,511,192 uncompressed_bytes=268,435,257
  batch_build=401.879 ms libdeflate=314.678 ms CUDA_H2D=21.024 ms
```

These are smoke observations, not stable benchmarks. The uploaded source is an ordinary pageable Rust `Vec<u8>`; pinned-memory optimization was intentionally deferred. The exact canonical 28-byte BAM EOF marker terminates input and is excluded from counts. A malformed five-byte gzip prefix failed cleanly with compressed-offset context.

## Completed CUDA whole-file flagstat vertical slice

### Implemented streaming structure

The production-shaped command is now:

```bash
gpugeno flagstat INPUT.bam \
  [--backend cuda] [--device N] [--bai INPUT.bam.bai] \
  [--max-uncompressed-bytes N] [--benchmark] [--validate]
```

Normal `stdout` is the 16-line samtools-style summary. `--benchmark` writes metadata, batch, and separate CUDA-event H2D/kernel/D2H measurements to `stderr`. `--validate` classifies the same exact physical byte stream with the independent Rust implementation and requires all 32 counters to match. An unavailable explicit backend fails rather than falling back; only CUDA exists in this slice.

The host path is deliberately layered for future pileup:

1. `bai.rs` retains every nonzero linear-index entry as `(coordinate, virtual_offset)`, including repeated offsets. It separately derives a sorted/deduplicated physical flagstat view.
2. BAM header parsing supplies an explicit first-record anchor. The BGZF EOF location supplies an explicit physical end anchor, covering header-adjacent and trailing unindexed records.
3. `DisjointBamStream` reads/framing-plans one bounded batch at a time and sends its BGZF members through persistent bounded libdeflate workers. Results are reassembled in member sequence before exposure. An interior boundary member can be decompressed by both adjacent batches, but retained bytes are sliced into disjoint logical ranges.
4. Every batch retains a BGZF-member map and can translate retained virtual offsets into 32-bit batch-relative positions. This is intended to support a later pileup planner without forcing pileup to use flagstat's deduplicated/disjoint work semantics.
5. The CUDA call uploads bytes plus span starts, launches one 128-thread block per physical span, reads one 32-counter result per span, and reduces those partials on the host.

The kernel preserves libshadowfax's classifier and QC pass/fail layout but uses the safer record-walk shape from newer CuBayes pileup: thread 0 builds a bounded shared table of at most 128 validated record starts, then lanes classify distinct records. Every record checks the four-byte size, minimum 32-byte BAM core, and span end before fixed fields are read. Malformed/non-record-aligned spans return per-span status rather than allowing an out-of-bounds access. The native C boundary also validates span ordering and retains the established synchronization rule on all post-enqueue failures.

A span larger than the configured batch cap is currently rejected with its virtual offset; it is never skipped. A 4 MiB real-data run exercised this failure at virtual offset `169285607368`.

### Automated and regression verification

The self-contained suite now has 14 tests covering the previous BGZF cases plus BAI repeated-entry preservation/deduplication, classifier precedence and MAPQ 4/5 behavior, formatting, partial records, virtual-offset translation across a duplicated boundary member, exact retained bytes across bounded batches, and explicit rejection of an indivisible oversized span.

Verified commands:

```bash
cargo fmt --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 \
  --max-uncompressed-bytes 268435456 \
  --benchmark --validate
```

All passed. The vector-add and upload examples remain working regression diagnostics.

### Real-data correctness result

The 256 MiB run emitted 2,284 spans from 2,285 physical anchors in 20 bounded batches. It classified 5,324,198,102 alignment-stream bytes; together with the 324,031-byte decompressed BAM header this exactly accounts for the previously measured 5,324,522,133 uncompressed bytes. The CUDA and Rust oracle counters matched exactly:

```text
10633980 + 0 in total (QC-passed reads + QC-failed reads)
10633980 + 0 primary
0 + 0 secondary
0 + 0 supplementary
0 + 0 duplicates
0 + 0 primary duplicates
10457612 + 0 mapped (98.34% : N/A)
10457612 + 0 primary mapped (98.34% : N/A)
10633980 + 0 paired in sequencing
5317145 + 0 read1
5316835 + 0 read2
10254964 + 0 properly paired (96.44% : N/A)
10281244 + 0 with itself and mate mapped
176368 + 0 singletons (1.66% : N/A)
22494 + 0 with mate mapped to a different chr
22392 + 0 with mate mapped to a different chr (mapQ>=5)
```

This validates exact coverage on the representative file, including the final tail, but is not yet an external samtools/libshadowfax compatibility result: neither samtools nor pysam is installed, and the old executable was not used.

### Smoke measurements

These are single-run smoke observations on device 0, not stable benchmarks:

```text
256 MiB cap:
  batches=20 spans=2,284 blocks_decompressed=82,360
  logical_bytes=5,324,198,102 compressed_bytes_read=1,645,336,143
  metadata=0.534 ms batch_build=7,093.901 ms
  H2D=422.262 ms kernel=81.795 ms D2H=0.418 ms GPU_stage=504.475 ms
  host_validation=941.771 ms wall=8,673.428 ms

16 MiB cap:
  batches=339 spans=2,284 blocks_decompressed=87,700
  logical_bytes=5,324,198,102 compressed_bytes_read=1,750,900,852
  metadata=0.435 ms batch_build=7,302.088 ms
  H2D=427.934 ms kernel=1,118.026 ms D2H=6.532 ms GPU_stage=1,552.492 ms
  host_validation=553.388 ms wall=9,504.120 ms
```

The logical bytes and all counters were identical at both batch sizes. Boundary-member rereads explain compressed bytes exceeding the 1,635,811,603-byte file and increasing with more batches. The dramatic summed kernel-time increase across 339 small launches shows that batch policy is part of meaningful backend comparison even with the same 2,284 spans. Sequential batch construction remains by far the largest measured stage, but parallel decompression and overlap remain intentionally unimplemented.

### CUDA source split and device-activity verification

The misleading monolithic names were removed. `cuda/gpugeno_cuda.cu` now owns the C ABI, context, transfers, timing, and diagnostic vector-add path; the actual flagstat classifier and `flagstat_kernel` are in `cuda/flagstat.cu`, with only an internal launch declaration in `cuda/flagstat.cuh`. `build.rs` compiles both translation units and archives both objects.

Review found a native-build defect during this split: `ar rcs` updates an existing archive but does not remove members whose object names disappeared. The first incremental split build therefore still linked the stale old `vector_add.o`, even though a clean build would not. `build.rs` now removes the old archive before recreating it. `nm -C target/release/gpugeno` then confirmed that the executable contains `gpugeno_launch_flagstat` and a kernel symbol attributed to `flagstat.cu`.

A post-fix 16 MiB run was sampled approximately every 66 ms with:

```bash
nvidia-smi --id=0 \
  --query-gpu=utilization.gpu,memory.used \
  --format=csv,noheader,nounits
```

During the 9.07-second command, 134 of 138 samples reported nonzero GPU utilization, with a maximum observed 32% and 126 MiB observed device memory. The command again returned 10,633,980 reads and CUDA-event totals of H2D 477.776 ms, kernel 1,118.912 ms, and D2H 7.216 ms. This independently confirms execution on GPU 0. It also explains misleading casual observation: `examples/bam_upload` intentionally launches no kernel, the vector-add example's kernel is sub-millisecond at its small invocation, and the 256 MiB flagstat run submits short GPU bursts separated by CPU decompression.

### Remaining risks

- Exact coverage is demonstrated for `HG002_chr22.bam`, not every valid sparse or unusual BAI. Oversized anchor gaps currently fail clearly.
- The host oracle is a separate Rust implementation, but there is no external samtools result yet.
- The shared index and BGZF mapping preserve what pileup needs, but pileup still needs an explicit policy for left-overlapping reads, right-side data extent, and a completion signal such as the status used by current CuBayes prescan/pileup.
- At this CUDA-only checkpoint, batch construction remained stage-sequential with GPU execution. The later bounded rendezvous producer superseded that orchestration while retaining one GPU slot. Boundary members are still intentionally reread for simplicity.

## Completed native `wgpu` whole-file flagstat vertical slice

### Implementation

`wgpu` 30.0.1 now owns a native adapter, device, queue, and compiled WGSL compute pipeline in Rust. The CLI defaults to `wgpu` and retains explicit `--backend cuda`; requesting the unimplemented direct `vulkan` backend fails without fallback. For `wgpu`, `--device N` selects exactly adapter `N` from `Instance::enumerate_adapters(Backends::all())`. CPU adapters are rejected, and an unavailable index reports the complete enumerated adapter list. Adapter name, underlying API, device type, driver, and timing source are printed to `stderr` without contaminating flagstat `stdout`.

The backend consumes the existing immutable `IndexedBamBatch::data` and `span_starts` directly. Host code packs arbitrary BAM bytes into little-endian `u32` words because portable WGSL storage arrays do not expose byte elements. The real shader in `src/flagstat.wgsl` uses byte extraction from those words, one 128-lane workgroup per existing physical span, a 128-entry workgroup record table built by lane 0, bounds checks before fixed-field reads, and 32 workgroup atomic `u32` counters. Each span writes one partial; Rust checks every shader status, widens partials, and reduces into the shared `u64` `FlagstatCounters`. The bounded 256 MiB default means a partial cannot approach `u32` overflow on valid minimum-sized BAM records.

Timing is explicit. On adapters with both `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS`, six GPU timestamps delimit staging-to-device copies, the compute pass, and device-to-map-visible readback copies. On other adapters, those stages use separate submit/wait intervals and are labeled `host-synchronized`, not represented as GPU timestamps. CPU byte packing, mapped staging-buffer writes, and per-batch resource/binding setup are reported separately. These setup costs are intentionally not hidden inside H2D or kernel numbers.

Two new tests cover byte packing and actual WGSL execution over representative flags, two physical spans, all 32 counters, and a malformed final record status. The GPU result is compared exactly with the independent host classifier. The complete suite now has 16 tests.

### Verification and representative evidence

Verified on 2026-09-15:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 \
  --max-uncompressed-bytes 268435456 \
  --benchmark --validate

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 \
  --max-uncompressed-bytes 268435456 \
  --benchmark --validate

cmp /tmp/gpugeno-wgpu-256.out /tmp/gpugeno-cuda-regression.out
```

All passed. The `wgpu` run selected adapter 0, `NVIDIA GeForce RTX 3060`, reported `api=Vulkan`, `device_type=DiscreteGpu`, NVIDIA driver `550.163.01`, and `timing_source=gpu-timestamps`. It processed the same 20 batches, 2,284 spans, 5,324,198,102 logical alignment bytes, and 1,645,336,143 compressed bytes read as CUDA. `--validate` found all 32 counters equal to the host oracle, and the full 16-line output was byte-identical to the CUDA run. This proves a real non-CUDA software/API path on the representative workload; because the physical GPU was still NVIDIA, it does not yet prove the proposal's non-NVIDIA hardware claim.

At that earlier checkpoint, explicit `--backend vulkan` failed clearly with no fallback; this behavior was superseded by the public backend slice below. The `wgpu` invalid-index and llvmpipe rejection checks remained valid.

### Smoke measurements

The representative 256 MiB `wgpu` run reported the following single-run smoke data, not stable benchmark results:

```text
batches=20 spans=2,284 blocks_decompressed=82,360
logical_bytes=5,324,198,102 compressed_bytes_read=1,645,336,143
metadata=0.459 ms batch_build=7,520.867 ms
GPU-timestamp H2D=304.395 ms kernel=137.716 ms D2H=0.269 ms GPU_stage=442.380 ms
host packing=4,664.635 ms staging writes=2,133.792 ms resource setup=124.040 ms
host validation=890.541 ms wall=18,986.954 ms
```

The immediately following CUDA regression run produced identical counters and coverage with CUDA-event H2D 404.244 ms, kernel 81.633 ms, and D2H 0.459 ms. This is regression and smoke evidence, not a fair final API benchmark: the first portable implementation allocates resources per batch and its deliberately reported packing/staging work makes its wall time much larger. The existing shared batch policy remains unchanged.

### Remaining `wgpu` risks

- Actual non-NVIDIA execution is now observed on AMD Radeon RX 6600 through RADV/Vulkan with exact representative output. Broader vendor/OS coverage is still untested.
- **Packaging portability gap:** `build.rs` still invokes `nvcc` unconditionally, and even `--backend wgpu` uses an executable dynamically linked to `libcudart.so.12` and `libstdc++`. A normal AMD/Intel machine without the CUDA toolkit therefore cannot yet build or launch the portable backend. Make CUDA compilation/linkage optional before distributing a genuinely standalone portable artifact.
- The default test suite now executes a small real `wgpu` dispatch and therefore requires at least one hardware adapter at enumerated index 0 in addition to the project's existing CUDA build-tool requirement.
- Timestamp fallback code compiled and is explicit but was not exercised on this timestamp-capable adapter.
- The initial implementation's fresh per-batch buffers and full byte-to-word packing pass were removed by the completed raw-upload/reuse optimization. The `wgpu` GPU slot remains intentionally synchronized; direct mapped decompression and multiple in-flight GPU submissions remain deferred. Shared host next-batch construction was later overlapped through the bounded rendezvous producer.
- As before, representative correctness does not imply production handling of every malformed BAM or pathological BAI.

## Completed bounded parallel libdeflate experiment

### Implementation and bounds

`gpugeno flagstat` now accepts public `--threads N`; zero and non-integer values fail clearly, and the evidence-selected default is 8. `DisjointBamStream` constructs one worker pool when opened and retains it across all outer batches. Each named OS worker constructs one `libdeflater::Decompressor`, reuses it for every assigned BGZF member, and exits when the stream drops.

The coordinator still performs BGZF framing and virtual-offset planning in compressed-file order. It assigns at most one member to each available worker. Per-worker input queues have capacity one, the shared completion queue has capacity `N`, and scheduling never has more than `N` members in flight. Results carry deterministic member sequence numbers; the coordinator stores out-of-order completions only inside the current outer batch and appends them in sequence. Thus the memory envelope remains the configured outer batch (plus vector capacity and bounded `O(N * 65536)` worker input/output), rather than an unbounded job or result backlog. No worker runs concurrently with backend processing of the previous batch: pipeline overlap was not added.

Framing was split from decompression without changing validation: the coordinator still validates complete gzip/BGZF headers, `BC`/`BSIZE`, bounds, trailer `ISIZE`, and EOF behavior, while every worker sends the complete member through libdeflate's gzip decoder for integrity/CRC checking. The one-thread setting uses the same worker/reordering path and is the baseline rather than a separate sequential implementation. A synthetic regression compares one and four workers and requires byte data, spans, virtual endpoints, block maps, decompressed-block counts, and compressed-byte counts to match exactly. Additional tests cover zero-worker rejection and a worker-side libdeflate integrity error with clean shutdown; the complete suite has 18 tests.

### Representative benchmark and default

Measured on 2026-09-15 on the documented 24-CPU, RTX 3060 development host with the 256 MiB outer cap. The CUDA scaling command was run three times per count without host validation; the table reports medians and the complete observed `batch_build` range. Output from every run was byte-compared. `batch_build` includes serial framing/planning, worker decompression, ordering, and batch assembly; it is not pure libdeflate CPU time.

```bash
for threads in 1 2 4 8 16; do
  for run in 1 2 3; do
    target/release/gpugeno flagstat /agents/shadowfax/data/HG002_chr22.bam \
      --backend cuda --device 0 --max-uncompressed-bytes 268435456 \
      --threads "$threads" --benchmark
  done
done
```

| threads | median batch build (ms) | observed range (ms) | median wall (ms) | speedup in batch build vs 1 |
|---:|---:|---:|---:|---:|
| 1 | 7,855.456 | 7,684.294–8,103.806 | 8,453.127 | 1.00x |
| 2 | 4,253.306 | 4,203.895–4,348.457 | 4,819.814 | 1.85x |
| 4 | 2,594.613 | 2,587.489–2,622.137 | 3,173.895 | 3.03x |
| 8 | 1,781.002 | 1,705.099–1,835.169 | 2,375.257 | 4.41x |
| 16 | 1,763.352 | 1,757.441–1,800.883 | 2,342.313 | 4.45x |

Eight is the default: it cuts median batch construction by 77.3% versus one worker and reaches 99.0% of the 16-worker median throughput while using half as many workers. Sixteen workers improved the CUDA median by only 17.650 ms (1.0%), well below a useful default tradeoff and consistent with serial framing/assembly and scheduling becoming limiting.

Each `wgpu` count was also run once with `--benchmark --validate`. This is corroborating backend/correctness evidence, not a stable timing distribution:

| threads | batch build (ms) | wall (ms) | host validation |
|---:|---:|---:|:---|
| 1 | 8,575.292 | 18,680.318 | exact match |
| 2 | 4,424.127 | 14,209.735 | exact match |
| 4 | 2,605.307 | 12,592.374 | exact match |
| 8 | 1,799.249 | 11,511.666 | exact match |
| 16 | 1,842.368 | 11,785.995 | exact match |

All ten representative thread/backend combinations emitted the same 16-line output. Every run retained exactly 20 batches, 2,284 spans from 2,285 anchors, 82,360 decompressed members, 5,324,198,102 logical alignment bytes, and 1,645,336,143 compressed bytes read. Each validated `wgpu` run matched all 32 host-oracle counters; the explicit CUDA one-thread validation also matched. This demonstrates that worker completion order does not affect batches, virtual mappings, spans, bytes, or counters. Current `wgpu` host packing remained 3,722.703–3,808.510 ms and staging writes 1,935.667–2,057.679 ms across those runs; this task did not optimize or hide them.

### Verification and remaining risks

The completed verification commands were:

```bash
cargo fmt --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304

cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
```

Risks and limits:

- Scaling plateaus between 8 and 16 workers on this host. Framing, ordered concatenation, per-member channels, allocation, and storage are still serial/coordinator costs; this experiment does not isolate them as pure decompression timing.
- The static default of 8 may oversubscribe a smaller machine. `--threads` is the deliberate tuning escape hatch; no topology-aware policy was added from evidence on only one host.
- The one-worker path includes OS-thread/channel/reordering overhead and is a common-path baseline, not a promise to equal the old direct sequential implementation's timing.
- Out-of-order outputs are bounded by the outer batch, but `Vec` capacity can exceed logical bytes according to Rust's growth policy. The queue/job envelope itself is strictly bounded by worker count.
- Worker shutdown joins all threads. Normal, decoder-error, oversized-span, and representative paths terminate, but forced cancellation of a hypothetical stuck native libdeflate call is not implemented.
- That experiment did not change the existing unusual-BAI, external-samtools, non-NVIDIA, or CUDA-packaging risks. It deliberately did not overlap CPU/GPU stages or alter the then-current `wgpu` packing/staging path, GPU kernels, pileup, or reference repositories; packing/staging was optimized in the subsequent completed slice.

## Completed `wgpu` raw-upload and reusable single-slot optimization

### Implementation and lifetime model

`IndexedBamBatch.data: Vec<u8>` is unchanged and remains the canonical operation-neutral BAM stream. `WgpuContext::flagstat` no longer allocates or fills a `Vec<u32>`. It maps the slot's reusable upload buffer, copies the canonical bytes once, and zeroes only the final zero to three bytes needed to align the GPU copy to a four-byte storage word. WGSL continues to extract little-endian bytes from `array<u32>`, while the existing parameter uniform carries `data.len()` rather than the padded allocation/copy size. The representative WGSL test now begins with a 37-byte record and ends on a non-word boundary, exercising unaligned record offsets and final-word padding against the independent host classifier.

The context owns one explicit `WgpuBufferSlot`. Its operation-neutral `WgpuInputSlot` contains BAM device/upload and span device/upload pairs. A separate flagstat slot contains parameter device/upload, partial-result/status device buffers, matching mapped readbacks, reusable timestamp query/resolve/readback resources, and the bind group; the pipeline is also flagstat state rather than shared input state. This prevents flagstat's output shape from becoming the future pileup input contract. A future slot pool can replicate the slot, but this task deliberately keeps one slot and fully synchronizes/maps results before reuse.

Every variable-sized resource is grow-only. Capacities use the next power-of-two size class, clamped to the adapter's storage/buffer limit; the 256 MiB outer cap therefore uses one 256 MiB BAM device buffer and one 256 MiB upload buffer rather than repeated near-cap growth. Logical copy/readback sizes remain per batch. A size class is less than twice the requested size, so retained capacity is bounded; during a later growth the replaced old and new resources may coexist transiently while handles are dropped. Timestamp resources and the 16-byte parameter pair are created once. Bind groups are rebuilt only when a bound resource grows. No decompressor writes into mapped GPU memory, and no overlap or second slot was introduced.

Timing labels remain distinct. `wgpu_host_packing=0.000 ms` means the conversion pass does not exist, not that hidden work was moved; `wgpu_upload_mode=raw-little-endian` makes that explicit. `wgpu_staging_write` includes map completion, the one canonical-byte copy, span/parameter writes, and final padding. `wgpu_resource_setup` includes slot capacity checks, actual growth allocations, timestamp creation, and required bind-group rebuilds. H2D/kernel/D2H remain GPU timestamp intervals on the tested adapter and do not include those host stages.

### Paired representative benchmark

A before run was captured immediately before editing, and the after run used the final size-class implementation. Both are single release-mode benchmark observations with host validation enabled—not medians or a timing distribution—on adapter 0 (`NVIDIA GeForce RTX 3060`, Vulkan, NVIDIA driver `550.163.01`, `gpu-timestamps`). Both used the canonical BAM, the established 256 MiB cap, and eight decompression workers:

```bash
target/release/gpugeno flagstat /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 \
  --benchmark --validate
```

| measured stage | before (ms) | after (ms) | paired change |
|---|---:|---:|---:|
| full host packing | 3,808.327 | 0.000 | eliminated |
| mapped staging writes | 1,919.175 | 314.014 | -83.6% |
| resource setup/growth | 116.167 | 92.337 | -20.5% |
| H2D GPU timestamp | 249.666 | 222.228 | observation only |
| kernel GPU timestamp | 113.489 | 51.198 | observation only |
| D2H GPU timestamp | 0.237 | 0.124 | observation only |
| aggregate GPU stage | 363.391 | 273.550 | -24.7%, run variance not isolated |
| batch build | 1,800.176 | 1,750.537 | observation only |
| host validation | 872.492 | 858.281 | observation only |
| wall | 11,509.907 | 4,785.535 | -58.4% |

The separately reported packing+staging+setup sum fell from 5,843.669 ms to 406.351 ms (93.0%). Only packing removal and host resource feeding are attributed to this change; the lower GPU timestamp and batch/validation observations are not claimed as kernel or decompression improvements. Wall includes adapter/context initialization, submission/map waits, validation, and other orchestration not represented by the summed stage labels.

Both NVIDIA runs retained exactly 20 batches, 2,284 spans from 2,285 anchors, 82,360 decompressed members, 5,324,198,102 logical alignment bytes, and 1,645,336,143 compressed bytes read. Both matched all host-oracle counters. Their 16-line outputs were byte-identical with SHA-256 `dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113`.

A subsequent independent full run used `AMD Radeon RX 6600 (RADV NAVI23)`, `api=Vulkan`, `device_type=DiscreteGpu`, Mesa/RADV 25.2.7, and `timing_source=gpu-timestamps`. It retained the same coverage, matched the host oracle, and produced the same output SHA-256. Its single-run smoke observations were batch build 2,467.152 ms, staging writes 355.333 ms, resource setup 23.914 ms, H2D 380.895 ms, kernel 58.435 ms, D2H 0.049 ms, GPU stage 439.379 ms, host validation 911.785 ms, and wall 5,574.969 ms. This is the first actual non-NVIDIA hardware evidence; it is not a controlled vendor benchmark because host conditions and adapter differ from the paired NVIDIA run.

### Verification, regressions, and remaining risks

The final verification set was:

```bash
cargo fmt --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release --example cuda_vector_add -- \
  --device 0 --elements 1048576
cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
```

All 19 tests passed. The CUDA representative regression retained the same coverage and exact host match, and its output was byte-identical to final `wgpu`; its CUDA-event times were H2D 384.492 ms, kernel 81.556 ms, D2H 0.469 ms, and wall 3,248.793 ms. The CUDA vector-add and bounded-upload diagnostics also passed. These are regression observations, not a refreshed cross-API benchmark.

Remaining risks and limits:

- The optimization was validated in its paired measurement on the documented NVIDIA/Vulkan adapter and independently on AMD/RADV/Vulkan. It does not change the CUDA packaging gap or establish broader vendor/OS coverage.
- The mapped-upload approach depends on `wgpu`'s portable buffer mapping/copy semantics, but only the timestamp-capable path was exercised. Host-synchronized timing fallback still compiles and retains its labels but was not run.
- Power-of-two size classes trade bounded retained slack (less than one requested size) for avoiding repeated near-cap growth. A growth can transiently retain old and new handles; this remains bounded but is not a hard process-RSS measurement.
- A single slot intentionally serializes map/write, GPU work, and readback. There is no decompression-to-mapped-memory path, overlap, double buffering, multiple in-flight batches, or kernel change.
- Packing is reported as zero because the pass was deleted; final padding and the raw byte copy are included in staging time. Comparing only the zero packing field while ignoring staging would be misleading.

## Superseded implementation history: direct-Vulkan synthetic flagstat integration spike

### Implementation and ownership

`src/vulkan_spike.rs` was the deliberately narrow predecessor to the production backend. `ash` dynamically loaded Vulkan, selected one physical device, ran per-call resources, and proved the classifier and timestamp semantics before the public slice. Its per-dispatch buffers and descriptor pool were intentionally not a production resource-reuse design; the implementation now lives in `src/vulkan_backend.rs`.

Five host-visible buffers hold padded canonical BAM bytes, `u32` span starts, per-span 32-counter partials, per-span statuses, and a 16-byte parameter block. Memory selection respects each buffer's memory-type bitmask, prefers HOST_COHERENT, and supports non-coherent types by flushing or invalidating the complete mapped allocation (`VK_WHOLE_SIZE`, offset zero), satisfying atom alignment without assuming coherency. Bindings use exact nonzero ranges and are checked against storage/dispatch/workgroup limits. The shader checks every `block_size` before fixed BAM fields, reports status 1/2/3 for truncated size/small core/span overrun, uses 128 workgroup lanes and 32 atomic partials, and preserves the established flagstat precedence. Rust reads all statuses and partials, widens each through `FlagstatCounters::from_u32_flat`, and leaves policy to the focused caller.

`src/vulkan_flagstat.wgsl` is a dedicated direct-Vulkan source. Build-time `naga` validates it and emits SPIR-V 1.3 to `OUT_DIR`; `include_bytes!` embeds that artifact and runtime checks its word shape/magic before `vkCreateShaderModule`. This was selected because no `glslc`, `glslangValidator`, `spirv-as`, or `spirv-val` was installed, while `naga` 30.0.1 was already locked transitively by `wgpu`. There is no runtime shader compiler or external build tool.

- The synthetic test/example and its AMD/NVIDIA smoke observations remain useful provenance. The production tests retain the four-record/two-span oracle, malformed-status/reuse, growth, invalid-index, and software-rejection coverage while adding whole-file streaming.

When `timestampComputeAndGraphics` and nonzero queue-family timestamp valid bits are available, command-buffer timestamps bracket the dispatch. Readback masks wrapping subtraction to the reported valid-bit width and converts ticks using `timestampPeriod`. Otherwise timing is the synchronized submit/fence interval and is labeled `host-synchronized`. All tested physical GPUs supported timestamps, so the fallback is compiled and reviewed but not hardware-exercised.

### Verification and smoke observations

Final verification used:

```bash
cargo fmt --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release --example vulkan_flagstat_spike -- --device 0
cargo run --release --example vulkan_flagstat_spike -- --device 1
cargo run --release --example vulkan_flagstat_spike -- --device 2
cargo run --release --example vulkan_flagstat_spike -- --device 4294967295  # expected failure
cargo run --release --example vulkan_flagstat_spike -- --device 3           # expected llvmpipe rejection

cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
```

The two focused tests brought the suite to 21 tests. Device 0 was AMD Radeon RX 6600/RADV and reported a 0.023840 ms GPU-timestamp interval for the four-record dispatch. Devices 1 and 2 were NVIDIA GeForce RTX 3060 and reported 0.008992 ms and 0.008864 ms. These are smoke observations of successful timestamp exercise, not benchmarks or a cross-device comparison. Invalid index `4294967295` failed with all four physical devices listed, and device 3 (`llvmpipe`, CPU) failed as a prohibited software fallback.

The CUDA vector-add and bounded-upload regressions passed. The complete CUDA and AMD/`wgpu` representative runs each processed 20 batches, 2,284 spans, and 5,324,198,102 logical bytes, matched the host oracle, and retained output SHA-256 `dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113`. These were regression smoke runs, not refreshed comparative benchmarks.

One initial default-parallel `cargo test` run terminated with SIGSEGV while the new direct-Vulkan tests and the existing real-`wgpu` GPU test could execute concurrently against adapter 0. Every test passed with one test thread. A crate-local test mutex now serializes only the three hardware-driver tests; two subsequent ordinary parallel `cargo test` runs passed. This avoids conflating driver/process concurrency with classifier correctness and does not serialize production code.

### Superseded spike boundary and observations

- The predecessor had only synthetic dispatches and per-call allocation; whole-file streaming, public selection, and grow-only production resources are documented in the completed backend section below.
- AMD RADV and NVIDIA proprietary drivers were exercised on Linux during the spike. No validation layer was installed, no other OS/vendor was tested, and the non-coherent memory and host-timing fallback branches were not selected by available GPUs; those remain production risks.
- The same-process uncoordinated direct-Vulkan/`wgpu` test crash was mitigated by the crate-local hardware test lock, not root-caused. Concurrent production use of both APIs remains outside this slice.
- The predecessor's two-query compute-only timing was intentionally not comparable to streaming backend timings; the public implementation now measures all three GPU stages.

## Completed public Direct Vulkan whole-file flagstat backend

### Implementation and ownership

The approved slice is complete. `src/vulkan_spike.rs` was promoted to `src/vulkan_backend.rs`, exposed from `src/lib.rs`, and the historical diagnostic example now imports the production module. `BackendChoice::Vulkan` and `BackendContext::Vulkan(Box<VulkanContext>)` are wired into `gpugeno flagstat`; `wgpu` remains the default. A Vulkan numeric device is parsed as a nonnegative value before conversion to `u32`, selects exactly the Vulkan physical-device enumeration index, prints index/name/type/vendor ID/device ID/API version/timing source to `stderr`, and rejects `CPU`, `OTHER`, llvmpipe/lavapipe/SwiftShader/software-renderer names without fallback.

`VulkanContext` owns the loader/instance, selected physical/logical device and compute queue, one resettable command pool/buffer, one signaled/reusable fence, pipeline/layouts, descriptor layout, and optional six-query timestamp pool. It owns exactly one `ResourceSlot`: device-side BAM (`STORAGE_BUFFER|TRANSFER_DST`), spans (`STORAGE_BUFFER|TRANSFER_DST`), 16-byte parameters (`UNIFORM_BUFFER|TRANSFER_DST`), partial counters (`STORAGE_BUFFER|TRANSFER_SRC`), and statuses (`STORAGE_BUFFER|TRANSFER_SRC`), plus matching HOST_VISIBLE transfer buffers. The descriptor pool/set binds only those five device buffers. Pair capacities are grow-only bounded power-of-two classes clamped to `maxStorageBufferRange`; only a pair whose required logical size exceeds capacity is recreated, and the parameter pair is created once. Descriptor pools are cleared before bound-buffer replacement and rebuilt only when needed. The fence is waited before every reuse/growth, so old resources are not destroyed while submitted work can reference them.

Device memory selection prefers compatible `DEVICE_LOCAL` memory; upload/readback selection requires `HOST_VISIBLE` and prefers `HOST_COHERENT`. `OwnedBuffer` tracks allocation size separately from buffer capacity. Host writes copy canonical BAM bytes once and zero only the final 0–3 bytes needed for a complete storage word; span/parameter words and all readback words use explicit little-endian conversion. Non-coherent mappings flush/invalidate the whole allocation at offset zero. Descriptor ranges are nonzero exact capacities and all size/dispatch/storage/uniform limits are checked before submission.

The timestamped command buffer resets six queries, writes timestamps before/after upload copies, inserts explicit transfer-write→compute-read barriers for BAM/spans/parameters, timestamps immediately before/after dispatch, inserts compute-write→transfer-read barriers for counters/statuses, timestamps before/after readback copies, submits once, and waits the fence before mapping results. Query pairs 0–1, 2–3, and 4–5 produce H2D, kernel, and D2H using valid-bit masking, wraparound subtraction, and `timestampPeriod`. Devices without the required timestamp capability use three separately synchronized upload, compute, and readback submissions timed independently on the host and labeled `host-synchronized`; no combined host interval is reported as three GPU-equivalent stages. Pre-submit failures reset command state; post-submit failures wait/device-idle best effort and poison the context if synchronization itself fails.

The dedicated `src/vulkan_flagstat.wgsl` and build-time `naga` SPIR-V 1.3 generation remain unchanged semantically: little-endian extraction, the 128-lane bounded record walk, classifier precedence, fixed-field checks, and statuses 1/2/3 are preserved. Every span writes its full 32-word result and status. A nonzero status fails `VulkanContext::flagstat` with span number, numeric status, and `shader_status_description`; the CLI adds the virtual batch range and never reduces partial success.

### Tests and verification

Focused coverage now includes pure size-class and timestamp-wrap helpers; four representative records in two spans against per-span and full host oracles; malformed final record status followed by a valid same-context dispatch; differently sized small/large/small calls forcing slot growth and proving no stale results; invalid device index; CPU/software rejection when enumerated; and the existing crate-local `GPU_TEST_LOCK` serialization with the real `wgpu` test. The suite has 23 tests.

The required verification commands passed:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --all -- --check
cargo test
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
git diff --check

cargo run --release --example vulkan_flagstat_spike -- --device 0
cargo run --release --example vulkan_flagstat_spike -- --device 4294967295  # expected failure
cargo run --release --example vulkan_flagstat_spike -- --device 3           # expected llvmpipe rejection
cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example bam_upload -- \
  /agents/shadowfax/data/HG002_chr22.bam \
  --device 0 --max-uncompressed-bytes 4194304
```

The Vulkan diagnostic passed on AMD device 0 and NVIDIA devices 1 and 2. The invalid `u32::MAX` invocation listed all four enumerated devices and failed; llvmpipe device 3 failed as CPU/software. CUDA diagnostics passed with 1,048,576 validated vector elements and a 4 MiB upload (`blocks=64`, `compressed_bytes=1,216,766`, `uncompressed_bytes=4,162,185`).

The complete representative commands all passed with `--validate` and `--benchmark`:

```bash
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend vulkan --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend wgpu --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
cargo run --release -- flagstat \
  /agents/shadowfax/data/HG002_chr22.bam \
  --backend cuda --device 0 --threads 8 \
  --max-uncompressed-bytes 268435456 --benchmark --validate
```

On this run Vulkan and `wgpu` device/adapter 0 were AMD Radeon RX 6600 (RADV NAVI23), Vulkan, Mesa/RADV 25.2.7, with GPU timestamps; CUDA device 0 was the documented NVIDIA GeForce RTX 3060. Every run processed 20 batches, 2,284 spans, 82,360 decompressed blocks, 5,324,198,102 logical bytes, and 1,645,336,143 compressed bytes read. Each host validation matched all 32 counters. The three complete stdout files were byte-identical and each had SHA-256 `dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113`.

Single-run smoke observations (not benchmark distributions) were:

```text
Vulkan AMD: batch_build=1,778.745 ms; staging_write=627.670 ms; resource_setup=4.068 ms;
            H2D=376.705 ms; kernel=58.688 ms; D2H=0.309 ms; GPU_stage=435.702 ms;
            host_validation=888.950 ms; wall=4,569.153 ms; timing_source=gpu-timestamps
wgpu AMD:   batch_build=1,797.847 ms; staging_write=338.265 ms; resource_setup=23.802 ms;
            H2D=381.251 ms; kernel=59.105 ms; D2H=0.051 ms; GPU_stage=440.407 ms;
            host_validation=891.679 ms; wall=4,370.806 ms; timing_source=gpu-timestamps
CUDA NVIDIA:batch_build=1,822.838 ms; H2D=330.987 ms; kernel=81.770 ms; D2H=0.485 ms;
            GPU_stage=413.242 ms; host_validation=878.378 ms; wall=3,233.978 ms
```

### Review findings and remaining risks

Review corrected the historical per-call ownership design rather than carrying it into production: descriptor lifetime is now before bound-buffer lifetime, all ten slot buffers are paired and bounded, resource growth occurs only after synchronization, transfer/compute barriers cover each relevant buffer, and timestamp query ordering now places the upload end query before the transfer→compute barrier and the compute start query after it. Review also found and fixed native-endian readback conversion, missing public status failure propagation, and CLI parsing that could not represent `u32::MAX`; the final code uses explicit little-endian conversion, fails malformed spans, and accepts the required invalid-index diagnostic.

Remaining risks are bounded and explicit: available hardware selected the GPU-timestamp/coherent paths, so the three-submission host-timing fallback and non-coherent flush/invalidate branch were reviewed but not selected by smoke devices; no Vulkan validation layer or other OS/vendor was tested; unusual sparse BAI and oversized-span behavior retain the existing clear-error policy; samtools was unavailable during this implementation slice but was installed and compared after the controlled backend benchmark below; CUDA build/linkage remains unconditional; and the mixed Vulkan/`wgpu` driver SIGSEGV risk is mitigated only for tests by `GPU_TEST_LOCK`, not root-caused for concurrent production API use. No multiple slots, overlap, double buffering, allocator, pileup, packaging change, or reference-repository edit was added. No next slice is approved.

## Completed HG002 three-backend performance comparison

### Approval, device enumeration, and selected matrix

Completed 2026-09-18 on the Linux development host as the owner-approved documentation-only comparison of all three public GPU backends. No implementation code, `cubayes/`, or `libshadowfax/` was changed. The fixed workload was `/agents/shadowfax/data/HG002_chr22.bam` with its default adjacent BAI, a 268,435,456-byte (256 MiB) maximum uncompressed batch, eight decompression workers, the release binary, benchmark mode, and one complete streaming pass per invocation.

The existing diagnostics and failure listings enumerated these devices; software and duplicate-API entries were not selected:

- CUDA reported two available devices (the existing `--device 3` failure said `2 available`); `nvidia-smi` identified CUDA device 0 as NVIDIA GeForce RTX 3060, 12,288 MiB, PCI bus `00000000:00:06.0`, and device 1 as the same model on `00000000:00:08.0`. The NVIDIA driver was 550.163.01 and the installed CUDA toolkit compiler reported 12.4.131. The selected CUDA row is device 0, timed with CUDA events.
- Direct Vulkan's existing invalid-index listing was: physical 0 AMD Radeon RX 6600 (RADV NAVI23), physical 1 NVIDIA GeForce RTX 3060, physical 2 NVIDIA GeForce RTX 3060, and physical 3 `llvmpipe` CPU. The selected AMD row is physical device 0, vendor/device `0x1002/0x73ff`, API 1.4.318; the selected NVIDIA row is the first hardware device, physical device 1, vendor/device `0x10de/0x2504`, API 1.3.277. Both reported `gpu-timestamps`; the driver stack was RADV/Mesa 25.2.7 for AMD and NVIDIA 550.163.01 for NVIDIA.
- `wgpu`'s existing invalid-index listing was: adapter 0 AMD/RADV/Vulkan `DiscreteGpu`, adapter 1 NVIDIA/Vulkan `DiscreteGpu`, adapter 2 NVIDIA/Vulkan `DiscreteGpu`, adapter 3 `llvmpipe`/Vulkan `Cpu`, and adapter 4 NVIDIA `.../PCIe/SSE2`/GL `Other`. The selected AMD row is adapter 0 (`api=Vulkan`, driver `radv`, `Mesa 25.2.7`); the selected NVIDIA row is the first hardware Vulkan adapter, adapter 1 (`api=Vulkan`, driver `NVIDIA`, `550.163.01`). Both reported `gpu-timestamps`. No GL duplicate or software adapter was benchmarked.

The five matrix combinations, in the required order, were: CUDA NVIDIA device 0; Direct Vulkan AMD physical device 0; `wgpu` AMD/Vulkan adapter 0; Direct Vulkan NVIDIA physical device 1; and `wgpu` NVIDIA/Vulkan adapter 1. The two NVIDIA APIs expose the same RTX 3060 model and device ID, but exact physical-board identity across CUDA and Vulkan enumeration was not established; the comparison is therefore labeled same-model rather than same-board.

### Commands, run counts, and controls

The binary was built once before sampling:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --all-targets
```

Each invocation used the binary directly, with the backend/device substituted from the matrix:

```bash
target/release/gpugeno flagstat /agents/shadowfax/data/HG002_chr22.bam \
  --backend BACKEND --device DEVICE \
  --max-uncompressed-bytes 268435456 --threads 8 --benchmark [--validate]
```

There were exactly five correctness preflights (one per matrix row), with `--validate`; their timings were excluded from statistics because host validation materially changes wall time. There was then exactly one non-validating warmup per row and seven non-validating measured invocations per row: 5 preflights, 5 warmups, and 35 measured passes (7 per combination). Measured rounds were single-process and deterministic rotating order; round starts were matrix positions 0, 1, 2, 3, 4, 0, and 1, so no backends ran concurrently. The measured logs preserve each stdout/stderr pair under `/tmp/gpugeno-hg002-measured-r*-<combination>.{out,err}`; corresponding preflight and warmup logs are under `/tmp/gpugeno-hg002-preflight-*` and `/tmp/gpugeno-hg002-warmup-*`.

The filesystem/page cache was left warm after preflight and warmup. Linux caches were not dropped, root was not requested, and no cold-I/O claim is made. The comparison is a warm-cache, one-process-at-a-time comparison with fixed eight-worker decompression, one synchronized backend resource slot, and no CPU/GPU overlap.

### Correctness and coverage

All five preflights exited 0, reported `result=exact-match`, and reported exactly 20 batches, 2,284 spans, 2,285 anchors, 82,360 decompressed blocks, 5,324,198,102 logical bytes, and 1,645,336,143 compressed bytes read. Their stdout files were byte-identical. The established stdout SHA-256 was:

```text
dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113
```

Every warmup and all 35 measured invocations also exited 0, retained the exact standard coverage fields and this stdout hash, and no invocation was discarded or rerun. The seven-run statistics below use only the 35 measured invocations; measured runs intentionally omitted `--validate`.

### Absolute measured results

All cells below are milliseconds and use `median (observed min–max)` over seven measured runs. `Host staging write`, `resource setup`, and `packing` are the separately reported backend fields; a dash means that backend does not report that field. `wgpu` and Direct Vulkan packing are explicitly zero because their raw-little-endian paths have no packing pass.

| combination | metadata | batch_build | H2D | kernel | D2H | GPU_stage | wall | host staging write | resource setup | packing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUDA NVIDIA device 0 | 0.464 (0.437–0.638) | 1,775.435 (1,748.024–1,819.206) | 364.390 (357.740–404.042) | 81.634 (81.584–81.848) | 0.505 (0.448–0.558) | 446.530 (440.036–486.219) | 2,340.233 (2,303.095–2,386.931) | — | — | — |
| Direct Vulkan AMD physical 0 | 0.464 (0.413–0.560) | 1,796.264 (1,781.760–1,822.923) | 376.592 (376.263–377.054) | 58.944 (58.643–59.245) | 0.065 (0.064–0.322) | 435.757 (434.984–436.429) | 3,708.951 (3,585.649–3,810.317) | 634.994 (617.030–645.349) | 2.943 (2.879–3.076) | 0.000 (0.000–0.000) |
| `wgpu` AMD/Vulkan adapter 0 | 0.487 (0.450–0.681) | 1,816.767 (1,791.657–1,841.774) | 380.991 (380.904–381.463) | 59.042 (58.909–59.283) | 0.050 (0.050–0.051) | 440.223 (439.880–440.554) | 3,517.816 (3,468.799–3,568.672) | 347.363 (340.039–351.915) | 23.818 (23.065–24.393) | 0.000 (0.000–0.000) |
| Direct Vulkan NVIDIA physical 1 | 0.465 (0.436–0.688) | 1,780.941 (1,747.514–1,828.577) | 221.652 (221.165–222.143) | 51.053 (51.030–51.065) | 0.052 (0.050–0.052) | 272.769 (272.250–273.229) | 3,317.404 (3,186.935–3,386.769) | 312.026 (309.496–313.348) | 74.874 (71.729–83.616) | 0.000 (0.000–0.000) |
| `wgpu` NVIDIA/Vulkan adapter 1 | 0.514 (0.422–0.556) | 1,780.620 (1,752.549–1,808.351) | 221.772 (221.363–222.178) | 51.220 (51.197–51.235) | 0.130 (0.128–0.136) | 273.116 (272.713–273.549) | 3,292.456 (3,232.865–3,349.036) | 324.501 (321.930–327.588) | 82.237 (78.045–87.211) | 0.000 (0.000–0.000) |

Derived throughput uses the fixed 5,324,198,102 logical bytes and each median time: `bytes / (median milliseconds × 10^6)`, in decimal GB/s.

| combination | kernel throughput | aggregate GPU-stage throughput | wall throughput |
|---|---:|---:|---:|
| CUDA NVIDIA device 0 | 65.220 GB/s | 11.923 GB/s | 2.275 GB/s |
| Direct Vulkan AMD physical 0 | 90.326 GB/s | 12.218 GB/s | 1.435 GB/s |
| `wgpu` AMD/Vulkan adapter 0 | 90.176 GB/s | 12.094 GB/s | 1.513 GB/s |
| Direct Vulkan NVIDIA physical 1 | 104.288 GB/s | 19.519 GB/s | 1.605 GB/s |
| `wgpu` NVIDIA/Vulkan adapter 1 | 103.948 GB/s | 19.494 GB/s | 1.617 GB/s |

### RTX 3060-class three-backend view

Ratios are `row median / Direct Vulkan NVIDIA median` for the same metric; `1.000` is the Direct Vulkan baseline, values below 1 mean lower time/faster, and values above 1 mean higher time/slower. The rows are same-model comparisons, not proven same-board comparisons.

| backend/device | kernel median (range) | kernel ratio | GPU_stage median (range) | GPU-stage ratio | wall median (range) | wall ratio |
|---|---:|---:|---:|---:|---:|---:|
| Direct Vulkan, physical 1 | 51.053 (51.030–51.065) | 1.000 | 272.769 (272.250–273.229) | 1.000 | 3,317.404 (3,186.935–3,386.769) | 1.000 |
| CUDA, device 0 | 81.634 (81.584–81.848) | 1.599 | 446.530 (440.036–486.219) | 1.637 | 2,340.233 (2,303.095–2,386.931) | 0.705 |
| `wgpu`, adapter 1 | 51.220 (51.197–51.235) | 1.003 | 273.116 (272.713–273.549) | 1.001 | 3,292.456 (3,232.865–3,349.036) | 0.992 |

### AMD RX 6600 Direct Vulkan versus `wgpu`

Ratios use the same `row median / Direct Vulkan AMD median` definition. Host feeding is kept separate from GPU timings. The `host-feed median` is the median of each run's staging-plus-setup fields and is a descriptive derived value, not a replacement for either separately reported field.

| backend/device | kernel median (range) | kernel ratio | GPU_stage median (range) | GPU-stage ratio | wall median (range) | wall ratio | host staging write | resource setup | packing | host-feed median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct Vulkan, physical 0 | 58.944 (58.643–59.245) | 1.000 | 435.757 (434.984–436.429) | 1.000 | 3,708.951 (3,585.649–3,810.317) | 1.000 | 634.994 (617.030–645.349) | 2.943 (2.879–3.076) | 0.000 (0.000–0.000) | 637.934 (619.939–648.425) |
| `wgpu`, adapter 0 | 59.042 (58.909–59.283) | 1.002 | 440.223 (439.880–440.554) | 1.010 | 3,517.816 (3,468.799–3,568.672) | 0.948 | 347.363 (340.039–351.915) | 23.818 (23.065–24.393) | 0.000 (0.000–0.000) | 371.181 (364.432–375.965) |

### Interpretation, defects, and limitations

- CUDA is available only on NVIDIA in this matrix; there is no AMD CUDA row. The AMD differences therefore cannot be attributed to API choice alone, and the RTX view also combines different host/API/driver stacks.
- Direct Vulkan and `wgpu` on both selected portable devices use Vulkan underneath. Their AMD comparison measures abstraction/runtime/resource-path differences over the same underlying API family, not Vulkan versus a different low-level driver API. The backends remain independently optimized and have different host staging implementations.
- All selected devices exposed GPU timestamps, so H2D, kernel, and D2H are the primary stage comparison. No host-synchronized timing row was substituted or mixed into these tables. H2D is the largest GPU-stage component in every median row (about 222–381 ms), while kernel is about 51–82 ms and D2H is negligible; the GPU-stage totals are not kernel timings.
- Shared `batch_build` is the largest named time in every row (about 1.776–1.817 s). Backend-specific host feeding is also material: Direct Vulkan AMD staging is 635 ms, versus 347 ms for `wgpu` AMD; on NVIDIA, Vulkan and `wgpu` staging are about 312 and 325 ms, with resource setup around 75 and 82 ms. Packing is zero for both raw-little-endian paths. These are measured bottleneck observations, not causal proof from seven samples.
- Program `wall` starts inside the process before metadata and backend construction and includes metadata, decompression/batch construction, API context/resource work, transfers, compute, readback, and output orchestration. It excludes shell/process-launch measurement; measured runs also exclude timed host validation because they did not use `--validate`. The CUDA wall ratio below one despite its slower GPU-stage ratio is therefore not an API-only result: wall includes different orchestration and the exact physical NVIDIA board identity was not proven. Likewise, the AMD `wgpu` wall ratio below one accompanies lower measured host feeding while its GPU-stage ratio is slightly above one; this is an observation, not a causal attribution.
- The two NVIDIA Vulkan enumerations and CUDA enumeration showed RTX 3060 devices, but no cross-API PCI/UUID identity proof was available from the existing diagnostics. The same-model label is intentional. Software entries (`llvmpipe` and the `wgpu` GL duplicate) were excluded.
- Seven observations per combination provide variability context, not a statistically rigorous hardware benchmarking campaign or a claim of significance. This is a controlled seven-run, warm-cache comparison, not cold storage behavior or a definitive benchmark distribution. Fixed eight-worker decompression, one synchronized slot, no overlap, and the shared batch policy are part of the result.
- No invocation failed, differed, or required an outlier discard. A notable inventory discrepancy is that the current CUDA installation exposes two RTX 3060 devices, whereas older project memory described three; the required device 0 remained available. The raw logs are ephemeral evidence at `/tmp/gpugeno-hg002-preflight-*`, `/tmp/gpugeno-hg002-warmup-*`, and `/tmp/gpugeno-hg002-measured-*`; they are not a durable repository dependency.

### Exploratory samtools baseline and optimization direction

After the controlled backend comparison, `samtools 1.24` became available at `/usr/local/bin/samtools`. A small follow-up used one warmup and five warm-cache shell-elapsed runs of `samtools flagstat -@ 8 /agents/shadowfax/data/HG002_chr22.bam`, followed by the same one-warmup/five-run sequence for the release `gpugeno` CUDA command with eight workers and the 256 MiB cap. All outputs were byte-identical to each other and to the established SHA-256. Bash `time` reported samtools median 2.399 s (2.382–2.412) and gpugeno CUDA median 2.350 s (2.322–2.373), making gpugeno 2.0% lower in this exploratory sample. The tools were not interleaved and there were only five observations, so this establishes current warm-cache parity rather than a definitive win.

The evidence-based leading optimization hypothesis was bounded CPU/GPU pipeline overlap, not classifier-kernel tuning. The selected implementation uses a producer-owned next host batch blocked at a rendezvous while retaining one synchronized GPU slot. The prior medians were about 1.78–1.82 s for batch construction, 0.273–0.447 s for the GPU stage, 0.31–0.32 s for portable NVIDIA host staging, and only 0.051–0.082 s for the kernel. Kernel-only tuning could recover little compared with work potentially hidden by overlap. The bounded implementation and correctness evidence now exist; its performance payoff remains unmeasured until the separate controlled campaign.

For one-shot portable CLI latency, overlap alone may not close the entire samtools gap. On NVIDIA, roughly 0.83–0.88 s of the `wgpu`/Direct Vulkan median wall is not represented by metadata, batch build, staging, resource setup, and GPU-stage sums; adapter/device/pipeline initialization and other orchestration are likely contributors but have not been isolated. A following portable optimization should first instrument initialization, then evaluate Vulkan pipeline caching or context reuse where the invocation model permits it. Direct decompression into mapped upload memory is lower priority until overlap shows that host staging remains exposed rather than hidden. No optimization slice was approved by this analysis.

## Completed bounded CPU/GPU overlap implementation and correctness checkpoint

### Implementation and bounded lifetime model

Commit `d192f9c` added the binary-internal `src/batch_producer.rs` and rewired only shared orchestration in `src/main.rs`; backend internals and reference repositories were unchanged. The main thread still performs metadata, anchor planning, and `DisjointBamStream::open`, then moves the open stream into `gpugeno-batch-producer` through `std::thread::Builder`. The producer starts before backend creation, so first-batch construction can overlap some device/context startup, but the zero-capacity rendezvous prevents it from beginning batch two until the main thread receives batch one. Steady state overlaps construction of N+1 with validation, host staging, synchronous GPU work, and reduction for N. Fill and drain remain exposed.

The producer protocol has explicit batch, stream-error, and EOF messages over `sync_channel(0)`. The rendezvous has no queued element: at most one complete batch is owned by the consumer and one completed next batch remains producer-owned while `send` blocks. The producer cannot construct a third. This increases logical canonical host batch storage from roughly one 256 MiB batch to two, not a process-RSS hard bound: `Vec` slack, member/span metadata, eight bounded worker payloads/stacks/libdeflate state, and existing portable upload/device buffers are additional.

Every normal/error path disconnects the receiver before joining. Backend-construction, validation, backend, and injected consumer failures retain their root error while expected `ReceiverDropped` cancellation is ignored. Stream errors are moved intact and require the matching `ErrorReported` exit. Channel disconnect without explicit EOF is a protocol error. Producer panic is the root when no earlier error exists and secondary cleanup context otherwise. Full-agent review tightened `cancel_with_root` so an unexpected non-`ReceiverDropped` cancellation exit is also reported as secondary protocol context rather than hidden. `BatchProducer::Drop` disconnects and joins without panicking as a last-resort unwind guard. Cancellation cannot forcibly interrupt a native libdeflate call already executing; join waits for the current bounded `next_batch` to finish or unwind.

### Timing semantics and tests

Existing `batch_build`, validation, backend host-stage, H2D/kernel/D2H, and wall fields retain their prior definitions. New benchmark-only fields are `consumer_first_batch_wait`, `consumer_next_batch_wait`, `consumer_eof_wait`, `producer_first_send_wait`, `producer_backpressure_wait`, `producer_terminal_send_wait`, and `producer_lifetime`. Producer lifetime includes explicit stream/worker teardown. Output states that these overlapping work sums must not be added to infer wall; no unobservable “overlap saved” number is emitted.

The binary test target now has 11 producer/orchestration tests in addition to 23 library tests. They cover compile-time `Send` assertions without unsafe impls; ordered batches and explicit EOF; a deterministic peak of exactly two live items with the third not built during blocked send; errors before/after a batch; consumer-root cancellation while blocked sending; disconnect while building; producer panic; disconnect-not-EOF; unexpected cancellation exit context; real synthetic BAM/BAI sequential-versus-producer equality for every deterministic batch field except time plus concatenated bytes/counters; and open/corrupt-BGZF errors. Two consecutive normal test-suite runs passed.

### Complete correctness evidence

The pre-implementation release binary is preserved ephemerally as `/tmp/gpugeno-no-overlap-f955eea` with SHA-256 `16bb88703fd03ca1a6b70d902c67445c939cf441fa557d84c24b9ccb367b6f07`. Final checks passed `cargo fmt --all -- --check`, two `cargo test` runs, `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`, and `git diff --check`. An invalid CUDA device after producer startup failed without output or a hang, exercising backend-construction cancellation.

One complete `--benchmark --validate` canonical preflight passed on each of CUDA device 0, Direct Vulkan physical devices 0/1, and `wgpu` adapters 0/1. Every run reported 20 batches, 2,284 spans, 2,285 anchors, 82,360 decompressed blocks, 5,324,198,102 logical bytes, 1,645,336,143 compressed bytes read, and an exact host match. All five stdout files were byte-identical with SHA-256 `dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113`.

The validation-enabled smoke runs showed nonnegative/plausible overlap telemetry but are not suitable for speedup conclusions because validation competes with the producer. First-batch consumer wait was 36.019 ms for CUDA and at most 0.009 ms on the portable rows; summed later-batch wait ranged from 80.524 to 1,084.756 ms; first-send wait ranged from 0.005 to 460.981 ms; and producer lifetime ranged from 2,852.750 to 3,380.549 ms. The data demonstrate the intended rendezvous activity, not whether unvalidated wall time improved.

### Remaining boundary

No multiple GPU slots/submissions, buffered or unbounded queue, direct mapped decompression, backend/kernel tuning, pileup, multi-GPU, packaging change, async runtime, coverage-policy change, or reference edit was added. The known inability to force-cancel a permanently stuck native decompressor and the pre-existing BGZF worker-panic robustness caveat remain. A separate controlled baseline/candidate/samtools performance campaign is the next relevant experiment, but it is not approved; no performance claim should be made from this checkpoint.

## Superseded candidate: GPU per-block byte sums

This candidate was not implemented. The project owner selected the more ambitious whole-file flagstat slice instead.

## Flagstat semantic reference

The initial source of truth is:

- `libshadowfax/lib/shadowfax/flagstat.cu`
- Public counter structure: `libshadowfax/lib/shadowfax/shadowfax.h`
- Text output example: `libshadowfax/src/shadowfax_flagstat.c`
- GPU record helpers: `libshadowfax/lib/cubayes/krnl_common.cuh`
- BAI work-item generation: `libshadowfax/lib/cubayes/partitioner.h`

### BAM fields read by flagstat

Offsets below are relative to the start of a BAM alignment record including its four-byte `block_size` field:

- `block_size`: byte 0, little-endian `u32`
- `ref_id`: byte 4, little-endian `i32`
- `mapq`: byte 13, `u8`
- `flag`: byte 18, little-endian `u16`
- `next_ref_id`: byte 24, little-endian `i32`

Flag bits used:

| Name | Value |
|---|---:|
| paired | `0x001` |
| proper pair | `0x002` |
| unmapped | `0x004` |
| mate unmapped | `0x008` |
| read 1 | `0x040` |
| read 2 | `0x080` |
| secondary | `0x100` |
| QC fail | `0x200` |
| duplicate | `0x400` |
| supplementary | `0x800` |

Reverse-strand bits exist but are not used by this flagstat classifier.

### Counters

Each counter is maintained separately for QC-passed (`QCFAIL` clear) and QC-failed (`QCFAIL` set) records:

1. `n_reads`
2. `n_mapped`
3. `n_pair_all`
4. `n_pair_map`
5. `n_pair_good`
6. `n_sgltn`
7. `n_read1`
8. `n_read2`
9. `n_dup`
10. `n_diffchr`
11. `n_diffhigh`
12. `n_secondary`
13. `n_supp`
14. `n_primary`
15. `n_pmapped`
16. `n_pdup`

Classification order from the reference kernel:

```text
count total read
if secondary:
    count secondary
else if supplementary:
    count supplementary
else:
    count primary
    if paired:
        count paired in sequencing
        if proper-pair and read is mapped: count properly paired
        count read1/read2 bits
        if mate unmapped and read mapped: count singleton
        if read and mate both mapped:
            count pair mapped
            if ref_id != next_ref_id:
                count different chromosome
                if mapq >= 5: count different chromosome with mapq >= 5
    if read mapped: count primary mapped
    if duplicate: count primary duplicate
if read mapped: count mapped
if duplicate: count duplicate
```

Secondary takes precedence over supplementary if both bits are present, matching the reference's `if`/`else if` structure.

### Text output

The intended line order is the current samtools-style 16-line summary:

```text
P + F in total (QC-passed reads + QC-failed reads)
P + F primary
P + F secondary
P + F supplementary
P + F duplicates
P + F primary duplicates
P + F mapped (PCT : PCT)
P + F primary mapped (PCT : PCT)
P + F paired in sequencing
P + F read1
P + F read2
P + F properly paired (PCT : PCT)
P + F with itself and mate mapped
P + F singletons (PCT : PCT)
P + F with mate mapped to a different chr
P + F with mate mapped to a different chr (mapQ>=5)
```

Percentages use two decimal places and `N/A` for a zero denominator. Denominators are total reads for `mapped`, primary reads for `primary mapped`, and paired-in-sequencing reads for `properly paired` and `singletons`.

## GPU work-discovery direction

### What the reference does

`libshadowfax` creates work items from nonzero BAI linear-index virtual offsets, which generally occur at 16,384-base reference intervals. The CUDA launch uses one block per work span and 128 threads per block.

Within a span, each thread calls `nth_read(current_start, end, threadIdx.x)`. This means lanes duplicate some variable-length record traversal. After processing a round, an `atomicMax` advances a shared byte offset to the end of the furthest record. The process repeats until the span is exhausted, then thread-local counters are atomically reduced into one result for the block.

The duplicate traversal is acceptable because it avoids a serial host pass that constructs one offset for every record, and many BAI spans can occupy the GPU concurrently.

### Current portable direction

- All three completed GPU backends preserve index-anchored device-side record discovery.
- Treat a BAI virtual offset as `(compressed BGZF offset << 16) | uncompressed offset`.
- Convert virtual offsets into offsets in each decompressed batch.
- Deduplicate or otherwise handle repeated BAI offsets without dropping data.
- Add explicit start/end coverage so records before the first useful index anchor and unindexed trailing records are included.
- Keep batches below the offset range required by future Vulkan/`wgpu` implementations; 32-bit batch-relative offsets are the likely portable denominator.
- Reduce final totals into host `u64` counters. The completed `wgpu` path uses bounded per-span 32-bit atomic partials and widens them during host reduction.

### Historical work-discovery assumptions and outcomes

These were hypotheses before the whole-file CUDA and `wgpu` experiments. Their current outcomes are noted explicitly:

1. **Observed on the representative BAM:** BAI supplied 2,285 physical anchors and 2,284 spans, enough to occupy both kernels.
2. **Observed on the representative BAM:** explicit first/end anchors partitioned all 5,324,198,102 alignment-stream bytes exactly once, including the tail.
3. **Still unproven generally:** sparse or repeated BAI intervals may expose a span larger than the configured cap; the current path reports an error rather than scanning or skipping.
4. **Superseded implementation idea:** neither backend uses the old reference's duplicate `nth_read` traversal; both use a bounded lane-0 record-offset table.
5. **Observed and improved:** CPU libdeflate and transfers feed measurable kernels. Sequential batch construction initially dominated, but eight persistent workers reduced its median from 7.855 seconds to 1.781 seconds. The original `wgpu` packing/staging bottleneck was subsequently reduced from 5.844 seconds to 0.406 seconds for packing+staging+setup in the paired observation.
6. **Not directly evaluated:** the Rust host plus CUDA C ABI worked, and native Rust `wgpu` integrated cleanly, but no all-C/C++ three-API host was built for comparison.

## Known concerns in the old reference

These are source observations, not yet reproduced test failures:

- Mainline CuBayes does not contain a flagstat implementation; the kernel exists only in `libshadowfax`.
- The old `libshadowfax` stream wrappers appear capable of returning `done`/`NULL` when a batch end reaches the EOF sentinel, potentially discarding the final batch. The new path does not use those wrappers; explicit physical-end coverage accounted for the full representative stream.
- The old partitioner sometimes skips a single oversized window or zero-byte spans. `gpugeno` instead rejects an indivisible oversized span and its tests verify that behavior; it never silently advances past one.
- The old CPU decompression fallback allocates and frees a libdeflate decompressor for every BGZF block. `gpugeno` now uses persistent workers with one reused decompressor per worker.
- The old default path uses nvCOMP. `gpugeno` intentionally moves decompression to CPU libdeflate so that all GPU backends can share the same decompressed input path.

## Decision and discovery history

### Initial idea

The project began as a cross-platform port of CuBayes pileup. The proposed pipeline was chunked BAM reads, BGZF discovery, parallel libdeflate decompression, GPU pileup, and result readback. Candidate GPU APIs were CUDA, Vulkan, and WebGPU. Rust versus C/C++ was initially open.

### Why flagstat became first

Flagstat was selected as the first operation because it only needs fixed BAM fields and counter reductions. It exercises input, decompression, transfers, GPU dispatch, reductions, backend selection, and benchmarking without first solving portable pileup's CIGAR traversal and large output layout.

An early assumption that mainline CuBayes had a flagstat kernel was incorrect. The implementation was found in the separately cloned `libshadowfax`, an older experimental CuBayes fork built around a C API and Python wrapper.

### Compatibility target changed

The discussion initially considered exact samtools compatibility. Past experience suggested some samtools behavior had been difficult to match, but the details were not remembered and may have involved mpileup depth/order behavior. The current target is therefore the `libshadowfax` flagstat classifier first, with closer samtools investigation later. Default text should still resemble samtools.

### Index value was reconsidered

It was initially stated that sorting and BAI were unnecessary for flagstat because each alignment is classified independently and BGZF blocks already decompress independently. That was incomplete: BAM records may cross BGZF boundaries, and BAI virtual offsets provide known record-aligned anchors that let many GPU workgroups discover records without a serial host framing pass. The corrected direction is to use BAI for parallel work partitioning, not because flagstat is coordinate-dependent.

### Language direction

Rust was selected as the working host-language direction because native `wgpu` support is strongest there and direct Vulkan is available through `ash`. CUDA support is considered sufficiently mature when existing `.cu` kernels are compiled with `nvcc` and exposed through a narrow C ABI; writing CUDA kernels in Rust is not required.

### Benchmark scope

End-to-end time is secondary. Upload, kernel, and readback must be measured separately, with kernel execution treated as its own primary comparison. Initial measurements use one real streaming pass. Backends may be independently optimized.

### Roadmap was intentionally reduced

A detailed sequence of CUDA flagstat, pipelining, benchmark work, `wgpu`, Vulkan, and tuning slices was proposed. The project owner rejected committing to that sequence because early experiments are likely to invalidate assumptions. A full end-to-end CUDA flagstat path was then proposed as the first vertical slice and was also judged too thick: it combined Rust/CUDA linkage, FFI, BGZF, libdeflate, BAI, record coverage, the flagstat kernel, output, and timing.

At that point, the approved immediate work had been reduced to the Rust/CUDA vector-add integration spike. The BGZF byte-sum path was only a candidate, not an approved task. The subsequent decision to choose an even thinner bounded upload slice is recorded below.

### CUDA packaging choice

Two meanings of “put CUDA in the Rust program” were compared. One is to compile CUDA host code and kernels into a static native archive linked into the final Rust executable, exposing a C ABI; no separate library is shipped. The other is to embed PTX/cubin/fatbin and manage the CUDA Driver API from Rust. The project owner prefers C APIs, so the statically linked C boundary was retained for the integration spike.

### Rust/CUDA spike outcome

The integration spike validated the chosen packaging direction: Rust can own host data and a safe context wrapper while `nvcc`-compiled CUDA host/kernel code is statically included behind a C API. Build invalidation, device selection, transfers, CUDA-event timing, exact result validation, native errors, and RAII destruction all worked on the RTX 3060.

The first implementation passed normal-path tests, but review identified two subtle error-path defects: a partial-construction leak and possible asynchronous access to Rust-borrowed buffers after an error return. Both were corrected before acceptance. This is evidence that later native APIs must be reviewed specifically for partial resource construction and host-buffer lifetimes, not just successful execution.

### First vertical slice selected

After the CUDA integration spike, the project owner deliberately split the proposed BGZF-plus-GPU-checksum experiment again. The approved slice stops after sequential Rust/libdeflate decompression of one bounded BAM prefix and a synchronized CUDA upload. GPU compute and readback are deferred so BGZF/libdeflate integration and the upload boundary can be evaluated independently.

### Bounded BGZF upload outcome

The first vertical slice confirmed that a real BAM prefix can be incrementally framed, decompressed through one reused Rust/libdeflate object, concatenated under an uncompressed-byte cap, and uploaded through the existing CUDA C boundary. The 256 MiB experiment completed successfully without BAI or BAM semantic parsing.

One practical observation is that H2D from pageable Rust memory was about 21 ms for roughly 256 MiB, while summed libdeflate calls were about 315 ms. These single-run numbers do not establish a bottleneck, but they provide a baseline for deciding whether the next experiment should add GPU content verification, parallel decompression, or pinned memory.

### CUDA whole-file flagstat outcome

The project owner chose to skip the proposed checksum/readback micro-slices and integrate flagstat directly, while emphasizing that the host path must remain a streaming foundation for pileup. Inspection of both current CuBayes pileup and libshadowfax changed the implementation detail: the classifier semantics came from libshadowfax, but record discovery within each CUDA block uses the newer CuBayes-style thread-0 shared offset table with explicit bounds checks rather than duplicate `nth_read` traversal.

The resulting bounded stream successfully converted BAI virtual offsets to decompressed batch positions, included explicit first/end anchors, and counted the complete representative BAM. Deduplicated anchors are only an operation-specific flagstat view; the source BAI representation keeps repeated coordinate-bearing windows because those repeats can be meaningful for pileup overlap. The 256 MiB and 16 MiB runs produced identical exact counters and alignment-stream byte totals, disproving the concern that the old wrapper's final-batch loss was inherent to index-anchored processing.

The experiment also showed that batching cannot be treated as backend-neutral overhead without care: splitting the same 2,284 spans across 339 launches raised summed CUDA kernel event time from about 82 ms to about 1,118 ms. Sequential decompression/batch building remained about 7.1–7.3 seconds and dominated both runs. These are smoke observations, not final optimization conclusions.

### Demonstration scope clarified

The project owner clarified that gpugeno supports a funding proposal's cross-platform claim rather than aiming to become an externally used bioinformatics application. This supersedes the implicit production-quality standard that had begun to drive discussion of rare BAI gaps, exhaustive overlap recovery, and malformed input. Representative correctness is still necessary—otherwise performance could come from doing less work—but it is a means of making the portability comparison believable.

This shifted the immediate priority away from additional CUDA pileup correctness machinery. With a complete CUDA flagstat baseline already available, native `wgpu` flagstat was the smallest useful proof that the same realistic operation could execute without CUDA. The initial run used Vulkan on NVIDIA; a later independent run completed the same canonical workload exactly on an AMD Radeon RX 6600 through RADV/Vulkan, satisfying the actual non-NVIDIA execution goal. Direct Vulkan and pileup remain relevant later.

### Native `wgpu` flagstat outcome

The selected portability slice succeeded without changing the indexed stream: the same batches and span starts feed a real WGSL compute pipeline, and the complete representative output matched both the host oracle and CUDA exactly. The first development-machine run selected Vulkan on an RTX 3060 and exposed GPU timestamps for all three measured GPU stages. A later independent run selected an AMD Radeon RX 6600 via RADV/Vulkan and again produced exact canonical output, establishing both a non-CUDA API implementation and actual non-NVIDIA execution.

The experiment also disproved any assumption that portable byte-addressing and reduction would be free host work. Packing 5.3 GiB of BAM bytes into WGSL-readable words and writing fresh staging buffers consumed several seconds, all now reported explicitly. Reuse/direct packing are optimization candidates, not correctness changes. CPU/software adapters are visible in enumeration (llvmpipe was adapter 3) and must be rejected to keep backend claims honest.

### Parallel libdeflate outcome

The shared indexed stream now uses deterministic parallel member decompression rather than one direct decompressor. Separating framing/planning from inflation allowed the coordinator to enforce the same virtual-offset boundaries before worker completion, while sequence-tagged results preserved member order and every observable batch property. A bounded one-job-per-worker scheduler was sufficient; pipeline overlap was not needed to expose decompression scaling.

On the representative 24-CPU host, median CUDA batch construction fell from 7.855 seconds with one worker to 1.781 seconds with eight. Sixteen workers reached only 1.763 seconds, so eight became the default rather than spending twice the workers for about 1% further improvement. The `wgpu` checks showed the same decompression trend and exact counters, while its separate packing/staging costs remained visible and dominant after decompression sped up.

### Raw `wgpu` upload and reusable-slot outcome

The portable host-feeding cost was not inherent to WGSL byte access. BAM is already a byte stream in the little-endian representation consumed by the shader, so the full `Vec<u32>` materialization was redundant. Copying the raw bytes into a reusable mapped upload buffer and clearing only the final partial word preserved exact counters, including a new deliberately unaligned synthetic record case.

Resource reuse benefited from bounded size classes rather than exact growth: near-cap batches can differ slightly in size, so exact grow-only allocation may still recreate a 256 MiB pair several times. Next-power-of-two classes, clamped to device limits, made the established 256 MiB run settle after its first input allocation while retaining a less-than-two-times capacity bound. The final paired observation cut host packing/staging/setup from 5.844 seconds to 0.406 seconds and wall from 11.510 seconds to 4.786 seconds. It did not justify adding overlap or coupling decompression to GPU memory; one synchronized slot remains the current architecture.

### Direct-Vulkan synthetic flagstat outcome

The direct API integration worked with a much smaller boundary than a third production backend: one `ash` context and per-call resources ran the same bounded record-walk/classifier semantics over a two-span synthetic stream, exposed per-span statuses, and exactly matched the common host oracle on AMD/RADV and NVIDIA. Build-time WGSL-to-SPIR-V with the already-used `naga` version was more reproducible in this environment than introducing an absent system shader tool, while still yielding a dedicated embedded Vulkan module.

Review concentrated on lessons from the CUDA spike: every partially created native object had to gain a cleanup owner, mapped non-coherent memory needed explicit whole-allocation flush/invalidate handling, submitted commands could not outlive per-call buffers on ordinary error returns, and every fallible Vulkan call needed operation context. Queue timestamp valid bits—not merely query-pool creation—control timestamp support and masking. The first ordinary parallel full-suite run also exposed a same-process driver concurrency SIGSEGV when direct Vulkan and real `wgpu` tests could overlap; serializing only hardware tests made repeated normal runs stable. This did not establish that mixed-API concurrency is safe. The later approved public backend retained the crate-local test lock and kept concurrent production API use outside scope.

### CuBayes actor implementation is not authoritative

The project owner clarified that `cubayes/src/cubayes_main_actor.cu` is a faster parallel implementation suspected of bugs that can crash or freeze. It must not define gpugeno's correctness, retry, synchronization, ownership, or shutdown behavior. The non-actor `cubayes_main.cu` and `pipeline.h`, together with the pileup/prescan kernels, are the appropriate current reference.

Both paths use prescan status and trailing-window walkback, but their no-progress policies differ materially. The non-actor pipeline stops when the first window is incomplete rather than discarding it. The actor contains an explicit `SKIP` branch that advances one work item when no region completes; that may lose a difficult region and must not be copied. Even the non-actor stop behavior is not automatically the desired gpugeno policy: gpugeno should report a clear bounded-resource error or deliberately retry with a justified larger envelope, rather than hang, silently skip, or return partial success.

### Project naming

The prototype was initially called `sfxproto`. Before application code was created, it was renamed to **`gpugeno`**. The existing checkout paths containing `shadowfax` and the `libshadowfax` reference repository retain their names; they are not the application name.

### Other decisions

- Native CLI first; avoid unnecessary barriers to a future browser host.
- Linux/NVIDIA first.
- BAI only.
- Three public GPU backends are now available; no public CPU backend.

- `wgpu` is the default backend; CUDA remains explicitly selectable.
- Requested unavailable backends fail; no silent fallback.
- Fixed default batch size with a user override.
- Self-contained automated tests.
- Full development toolchains are acceptable for this prototype.
- CUDA first, because the existing kernel provides the clearest eventual flagstat baseline.
- The vector-add integration harness is a temporary Cargo example and may be removed once real CUDA functionality supersedes it.
- Single GPU now; multi-GPU deferred despite three available GPUs.
- Initial performance corpus is `HG002_chr22.bam`.
- Program name is `gpugeno`, and flagstat is a subcommand to leave room for later operations.

## Deferred possibilities, not a committed roadmap

Now that CUDA, native `wgpu`, bounded parallel decompression, raw-upload/resource reuse, actual AMD/Vulkan validation, the complete public direct-Vulkan backend, and the bounded overlap implementation/correctness checkpoint are complete, plausible later experiments include the controlled overlap performance evaluation, optional CUDA packaging, pileup, or backend tuning. None is currently approved.

When revisiting portable backends, preserve these general intentions unless evidence changes them:

- Runtime selection through `--backend cuda|vulkan|wgpu`.
- Common semantic counters and deterministic input coverage.
- Backend-specific optimization rather than forced identical kernels.
- Report the actual adapter and underlying API used by `wgpu`; on Linux/NVIDIA it may itself use Vulkan.
- Keep filesystem/decompression orchestration out of the compute contract so a future browser host remains plausible.

## Immediate task status

The bounded CPU/GPU overlap implementation and five-backend correctness checkpoint are complete and accepted after full-agent review. The full controlled baseline/candidate/samtools performance campaign is not approved; wait for the owner rather than inferring it as the next task.
