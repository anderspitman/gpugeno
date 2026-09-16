# gpugeno design and project memory

**Last updated:** 2026-09-15  
**Current phase:** the Rust/CUDA spike and the bounded BGZF/libdeflate-to-CUDA upload slice are complete and coordinator-reviewed; no next slice is approved.

## Fresh-agent handoff

If a new coding agent is told only to read this document and continue, it should do the following:

1. Treat this document as the project memory and source of current product intent.
2. Verify the repository and toolchain state, because those observations may have changed since the last update.
3. Do not start another implementation slice without project-owner approval. Review the completed bounded-upload results and select the next smallest experiment.
4. Put new `gpugeno` application code at the project root. Treat `cubayes/` and `libshadowfax/` as read-only reference repositories unless the project owner explicitly decides otherwise.
5. Prefer the smallest experiment that answers one uncertainty; do not pre-design later pipeline stages.
6. Investigate implementation details independently when they do not change product behavior. If a consequential choice or contradiction remains, ask the project owner one focused question at a time.
7. After an approved experiment, update this document with the exact implementation, commands, measurements, and discoveries before proposing further work.

The leading next candidate is GPU-side per-block byte sums, but it is not approved.

## Purpose of this document

This is the durable memory for `gpugeno`. Its goal is to let a future coding agent reconstruct the project as closely as practical without needing the original conversation.

It must preserve three kinds of information:

1. **Current truth:** what exists now, what we are building next, and how close it is to working.
2. **Intent and rationale:** goals, constraints, interfaces, and why important choices were made.
3. **History:** assumptions that proved wrong, approaches that were superseded, and evidence learned during implementation.

This is a living design record, not a fixed long-range roadmap. The project is deliberately experimental. Only the next small slice should be treated as committed; later work should be selected after measuring and learning from that slice.

### How to update it

Every coding agent working on this project should read this file first and update it as part of the same change when it learns something material.

- Keep **Current state**, **Current decisions**, and the **immediate work** section accurate.
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

## Current state

### Repository

The root now contains a minimal Rust crate and temporary CUDA spike:

- `Cargo.toml` and `Cargo.lock`
- `build.rs`: invokes `nvcc` and `ar`, then links the native archive, CUDA runtime, and C++ runtime
- `src/lib.rs`: safe Rust owner around the opaque CUDA C context, including vector-add and raw-byte upload operations
- `src/bgzf.rs`: bounded incremental BGZF framing and sequential libdeflate decompression
- `cuda/vector_add.h` and `cuda/vector_add.cu`: C ABI, CUDA context/upload logic, and temporary vector-add kernel
- `examples/cuda_vector_add.rs`: temporary CUDA integration harness
- `examples/bam_upload.rs`: temporary bounded BAM-prefix decompression/upload harness
- `idea.md`: the original pileup-oriented idea
- `design.md`: this document
- `cubayes/`: a clean CuBayes reference clone
- `libshadowfax/`: a clean experimental fork containing the CUDA flagstat implementation
- `.gitignore`: ignores `/target` and alignment/index files

The root is a Git repository on branch `main`. The spike, project metadata, and this design record are tracked. `cubayes/` and `libshadowfax/` remain separate nested reference repositories and are ignored by the root repository.

Reference revisions at the time of this update:

- `cubayes/`: branch `main`, commit `9687f167bdca43d19d379ed83cd52b98983fce6a`
- `libshadowfax/`: branch `main`, commit `2db65c0b27000e5302ac57f0d9ef2be878662ee8`

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

The development environment may require the full CUDA and Vulkan development toolchains. Avoiding development dependencies is not a prototype goal.

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
- [ ] Decide the next experiment from its results.
- [ ] Eventually complete an end-to-end CUDA flagstat run and validate `HG002_chr22.bam`.

## Current decisions

### Scope and compatibility

- **Decided:** Start with flagstat, not pileup.
- **Decided:** The initial semantic and algorithmic reference is `libshadowfax/lib/shadowfax/flagstat.cu`.
- **Decided:** CuBayes remains the authoritative reference for the broader BAM/pileup project, but mainline `cubayes/` has no flagstat kernel.
- **Decided:** Match the useful flagstat behavior closely; do not reproduce accidental bugs in the old stream wrappers.
- **Decided:** Default result output should look like `samtools flagstat`.
- **Deferred:** Investigating exact compatibility with a particular samtools release. Tests previously found difficult-to-match samtools behavior, but the details are not currently remembered and may have concerned mpileup rather than flagstat.

### Input and streaming

- **Decided:** Initial input is a valid, seekable, coordinate-sorted BGZF-compressed BAM with a BAI index.
- **Decided:** BAI only. CSI, SAM, CRAM, and non-seekable input are deferred.
- **Decided:** Processing must be bounded-memory streaming; loading or decompressing the complete BAM is not acceptable.
- **Decided:** Use libdeflate on CPU workers rather than implementing GPU BGZF decompression.
- **Decided:** Use record-aligned BAI virtual offsets to expose many independent GPU work spans. Do not begin with a CPU-generated offset for every BAM record.
- **Working decision:** All GPU backends should receive deterministic, identical outer batches and spans. A backend may subdivide them internally.
- **Decided:** Use a fixed default batch size and expose a batch-size argument. The actual default has not been selected.

### First vertical slice boundaries

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

- `--backend cuda|vulkan|wgpu`; target default is `wgpu` once that backend exists.
- `--device`; one selected GPU per invocation, initially defaulting to device 0.
- A batch-size override; spelling and units remain to be finalized.
- `--benchmark`; timing output is opt-in.

An explicitly requested unavailable backend must fail clearly. It must never silently fall back to another backend, because that would invalidate comparisons.

During a CUDA-only flagstat stage, requiring `--backend cuda` would be acceptable even though the eventual default is `wgpu`.

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

Coordinator review found and corrected two native error-path issues: a context leak if stream creation failed, and immediate returns that could leave asynchronous work touching borrowed Rust slices. Post-enqueue failures now best-effort synchronize before returning the original error.

Verified commands:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo run --release --example cuda_vector_add -- --device 0 --elements 1048576
cargo run --release --example cuda_vector_add -- --device 0 --elements 16777216
```

All passed, and every output element was validated exactly. A coordinator-run 16,777,216-element sample on device 0 measured H2D 10.772 ms, kernel 0.726 ms, and D2H 24.918 ms. These are smoke observations, not benchmark results. Device 99 correctly failed with status 1 and reported that only three devices were available. The C symbols were confirmed unmangled in the executable, which dynamically links CUDA runtime 12 and `libstdc++`.

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

Seven self-contained tests cover ordered concatenation, cap boundaries, first-member rejection, a 65,536-byte output member, missing `BC`, truncation, and canonical EOF handling. Coordinator verification passed `cargo fmt --check`, `cargo test`, `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`, and the vector-add regression.

Coordinator-observed real-data results on device 0:

```text
4 MiB cap:
  blocks=64 compressed_bytes=1,216,766 uncompressed_bytes=4,162,185
  batch_build=6.353 ms libdeflate=4.695 ms CUDA_H2D=0.470 ms

256 MiB cap:
  blocks=4,128 compressed_bytes=81,511,192 uncompressed_bytes=268,435,257
  batch_build=401.879 ms libdeflate=314.678 ms CUDA_H2D=21.024 ms
```

These are smoke observations, not stable benchmarks. The uploaded source is an ordinary pageable Rust `Vec<u8>`; pinned-memory optimization was intentionally deferred. The exact canonical 28-byte BAM EOF marker terminates input and is excluded from counts. A malformed five-byte gzip prefix failed cleanly with compressed-offset context.

## Later candidate: GPU per-block byte sums

After this slice, a likely next experiment is to upload block offsets/lengths and compute per-block wrapping byte sums, comparing them with host sums. It is not yet approved.

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

- Preserve index-anchored GPU-side record discovery when the first CUDA flagstat experiment is approved.
- Treat a BAI virtual offset as `(compressed BGZF offset << 16) | uncompressed offset`.
- Convert virtual offsets into offsets in each decompressed batch.
- Deduplicate or otherwise handle repeated BAI offsets without dropping data.
- Add explicit start/end coverage so records before the first useful index anchor and unindexed trailing records are included.
- Keep batches below the offset range required by future Vulkan/`wgpu` implementations; 32-bit batch-relative offsets are the likely portable denominator.
- Reduce final totals into host `u64` counters. WebGPU's portable integer/atomic constraints may require bounded 32-bit partial counters later; this has not yet been validated.

### Important unproven assumptions

These are hypotheses to test during an eventual flagstat experiment, not settled facts:

1. BAI supplies enough distinct record-aligned anchors on the representative BAM to keep the GPU occupied.
2. The chosen anchors can partition the entire physical record stream exactly once, including long unindexed tails.
3. Sparse or repeated BAI intervals can be handled without needing a normal-case serial host record scan.
4. Duplicate traversal in the reference algorithm performs well enough to be a useful baseline.
5. CPU libdeflate and transfer can feed the flagstat kernel fast enough for kernel differences to be measurable.
6. Rust plus a CUDA C ABI is less costly than hosting all three APIs from C/C++.

If an assumption fails, record the evidence here before replacing the design.

## Known concerns in the old reference

These are source observations, not yet reproduced test failures:

- Mainline CuBayes does not contain a flagstat implementation; the kernel exists only in `libshadowfax`.
- The old `libshadowfax` stream wrappers appear capable of returning `done`/`NULL` when a batch end reaches the EOF sentinel, potentially discarding the final batch. This must be tested, and accidental wrapper behavior must not define compatibility.
- The old partitioner sometimes skips a single oversized window or zero-byte spans. That behavior is unsafe as a whole-file correctness contract and must not be copied without proving complete coverage.
- The old CPU decompression fallback allocates and frees a libdeflate decompressor for every BGZF block. `gpugeno` should prefer persistent worker-owned decompressors.
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

The first implementation passed normal-path tests but coordinator review identified two subtle error-path defects: a partial-construction leak and possible asynchronous access to Rust-borrowed buffers after an error return. Both were corrected before acceptance. This is evidence that later native APIs must be reviewed specifically for partial resource construction and host-buffer lifetimes, not just successful execution.

### First vertical slice selected

After the CUDA integration spike, the project owner deliberately split the proposed BGZF-plus-GPU-checksum experiment again. The approved slice stops after sequential Rust/libdeflate decompression of one bounded BAM prefix and a synchronized CUDA upload. GPU compute and readback are deferred so BGZF/libdeflate integration and the upload boundary can be evaluated independently.

### Bounded BGZF upload outcome

The first vertical slice confirmed that a real BAM prefix can be incrementally framed, decompressed through one reused Rust/libdeflate object, concatenated under an uncompressed-byte cap, and uploaded through the existing CUDA C boundary. The 256 MiB experiment completed successfully without BAI or BAM semantic parsing.

One practical observation is that H2D from pageable Rust memory was about 21 ms for roughly 256 MiB, while summed libdeflate calls were about 315 ms. These single-run numbers do not establish a bottleneck, but they provide a baseline for deciding whether the next experiment should add GPU content verification, parallel decompression, or pinned memory.

### Project naming

The prototype was initially called `sfxproto`. Before application code was created, it was renamed to **`gpugeno`**. The existing checkout paths containing `shadowfax` and the `libshadowfax` reference repository retain their names; they are not the application name.

### Other decisions

- Native CLI first; avoid unnecessary barriers to a future browser host.
- Linux/NVIDIA first.
- BAI only.
- Three public GPU backends eventually; no public CPU backend for now.
- `wgpu` should eventually be the default backend.
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

After the approved bounded upload slice, likely options include a GPU checksum experiment, CUDA flagstat, native `wgpu`, direct Vulkan, deeper streaming overlap, CUDA tuning, a CPU reference backend, or returning to pileup. The next choice should depend on observed correctness problems and timing breakdowns.

When revisiting portable backends, preserve these general intentions unless evidence changes them:

- Runtime selection through `--backend cuda|vulkan|wgpu`.
- Common semantic counters and deterministic input coverage.
- Backend-specific optimization rather than forced identical kernels.
- Report the actual adapter and underlying API used by `wgpu`; on Linux/NVIDIA it may itself use Vulkan.
- Keep filesystem/decompression orchestration out of the compute contract so a future browser host remains plausible.

## Immediate unresolved questions

Do not answer all of these speculatively. Resolve them when the relevant experiment reaches the decision point, asking the project owner when behavior or scope is affected.

1. Should the next experiment add GPU per-block byte sums and readback verification, parallel libdeflate workers, or pinned host memory?
2. For eventual flagstat, which BAI offsets safely form disjoint whole-file anchors: linear entries only, chunk boundaries too, or a validated combination?
3. How should an unusually large span with no intermediate BAI anchor be split while preserving bounded memory and GPU parallelism?
4. What fixed default BAM batch size should the later streaming pipeline use?
5. Which timing and synchronization boundaries will remain comparable among CUDA, Vulkan, and `wgpu`?
6. What exact expected totals should be recorded for `HG002_chr22.bam` after independently validating them?
7. How much malformed-input validation belongs on the host before GPU dispatch? Valid input is assumed, but GPU out-of-bounds access is never acceptable.
