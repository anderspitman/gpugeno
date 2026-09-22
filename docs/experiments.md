# Implementation experiments

**Purpose:** Preserve completed implementation experiments, their boundaries, evidence, review findings, and remaining risks.
**Read when:** Revisiting a technique, changing an area whose ownership was established by an experiment, or checking why a current architectural decision exists.
**Reading strategy:** Start with the relevant section only. For current behavior, prefer [`architecture.md`](architecture.md); these reports describe the checkpoint and scope at the time of each experiment.

## Contents

- [Rust/CUDA integration spike](#completed-prerequisite-rustcuda-integration-spike) — validated the statically linked C ABI, ownership, error, and timing boundary.
- [Bounded BGZF decompression and CUDA upload](#completed-first-vertical-slice-bounded-bgzf-decompression-and-cuda-upload) — established bounded real-data framing, libdeflate decompression, and synchronized upload.
- [CUDA whole-file flagstat](#completed-cuda-whole-file-flagstat-vertical-slice) — established complete indexed streaming, classifier correctness, and explicit first/end coverage.
- [Native `wgpu` whole-file flagstat](#completed-native-wgpu-whole-file-flagstat-vertical-slice) — demonstrated the real portable shader path and exposed host packing/staging costs.
- [Bounded parallel libdeflate](#completed-bounded-parallel-libdeflate-experiment) — selected eight persistent bounded workers from measured scaling.
- [`wgpu` raw upload and reusable slot](#completed-wgpu-raw-upload-and-reusable-single-slot-optimization) — eliminated packing and retained one grow-only synchronized slot.
- [Direct-Vulkan synthetic spike](#superseded-implementation-history-direct-vulkan-synthetic-flagstat-integration-spike) — established native ownership, shader compilation, mapping, status, and timing behavior before production integration.
- [Public Direct Vulkan whole-file backend](#completed-public-direct-vulkan-whole-file-flagstat-backend) — promoted the spike into the complete third public backend.

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
