# Performance evidence

**Purpose:** Preserve controlled performance campaigns, exact methodology, measurements, interpretation, defects, and limitations.
**Read when:** Making performance claims, repeating a campaign, changing benchmark semantics, or selecting an optimization based on existing evidence.
**Reading strategy:** Read the conclusion and limitations of the relevant campaign first; load the command matrix and tables only when the task depends on them.

## Contents

- [HG002 three-backend performance comparison](#completed-hg002-three-backend-performance-comparison) — the controlled no-overlap baseline across five backend/device combinations; GPU-stage behavior was similar for the two portable APIs, while host feeding and startup materially affected wall time.
- [Bounded CPU/GPU overlap implementation and performance evaluation](#bounded-cpugpu-overlap-implementation-and-performance-evaluation) — exact output was retained and all five combinations improved external elapsed and program wall, so the strict two-host-batch overlap design was retained.
- [Comparative samtools/gpugeno hotspot profiling](#comparative-samtoolsgpugeno-hotspot-profiling) — flat CPU profiles show decompression dominates both tools and identify gpugeno's ordered canonical-batch copy as the strongest narrow optimization candidate.

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

The evidence-based leading optimization hypothesis was bounded CPU/GPU pipeline overlap, not classifier-kernel tuning. The selected implementation uses a producer-owned next host batch blocked at a rendezvous while retaining one synchronized GPU slot. The prior medians were about 1.78–1.82 s for batch construction, 0.273–0.447 s for the GPU stage, 0.31–0.32 s for portable NVIDIA host staging, and only 0.051–0.082 s for the kernel. Kernel-only tuning could recover little compared with work potentially hidden by overlap. The bounded implementation and correctness evidence now exist, and the subsequent controlled campaign below measures its payoff.

For one-shot portable CLI latency, overlap alone may not close the entire samtools gap. On NVIDIA, roughly 0.83–0.88 s of the `wgpu`/Direct Vulkan median wall is not represented by metadata, batch build, staging, resource setup, and GPU-stage sums; adapter/device/pipeline initialization and other orchestration are likely contributors but have not been isolated. A following portable optimization should first instrument initialization, then evaluate Vulkan pipeline caching or context reuse where the invocation model permits it. Direct decompression into mapped upload memory is lower priority until overlap shows that host staging remains exposed rather than hidden. No optimization slice was approved by this analysis.

## Bounded CPU/GPU overlap implementation and performance evaluation

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

### Controlled bounded-overlap performance evaluation

**Date/status:** Completed 2026-09-21 as a documentation-only benchmark. No source, test, Cargo, reference-repository, or binary file was modified after the reviewed checkpoint. The release candidate was built once at `ed23c176edaf542f9ee448de8658f0edc847115d` and copied to the immutable run path.

#### Method, binaries, and exact matrix

The preserved no-overlap executable and candidate hashes were:

```text
/tmp/gpugeno-no-overlap-f955eea
16bb88703fd03ca1a6b70d902c67445c939cf441fa557d84c24b9ccb367b6f07

/tmp/gpugeno-overlap-ed23c17
f3f45850b80bd8525729488dd56512ea76f1ead37a298b6a0124e2dd7046acb1
```

The candidate build/copy command was:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --all-targets
cp -p target/release/gpugeno /tmp/gpugeno-overlap-ed23c17
```

The common workload was `/agents/shadowfax/data/HG002_chr22.bam` with its default adjacent BAI, `--max-uncompressed-bytes 268435456`, `--threads 8`, warm filesystem/page cache, one process at a time, and no `--validate` in warmups or measured runs. The exact canonical command order was:

| paired canonical slots | combination | baseline command | overlap-candidate command |
|---:|---|---|---|
| 1 / 6 | CUDA NVIDIA device 0 | `/tmp/gpugeno-no-overlap-f955eea flagstat /agents/shadowfax/data/HG002_chr22.bam --backend cuda --device 0 --max-uncompressed-bytes 268435456 --threads 8 --benchmark` | `/tmp/gpugeno-overlap-ed23c17 flagstat /agents/shadowfax/data/HG002_chr22.bam --backend cuda --device 0 --max-uncompressed-bytes 268435456 --threads 8 --benchmark` |
| 2 / 7 | Direct Vulkan AMD physical 0 | same command with `--backend vulkan --device 0` | same command with candidate executable and `--backend vulkan --device 0` |
| 3 / 8 | `wgpu` AMD/Vulkan adapter 0 | same command with `--backend wgpu --device 0` | same command with candidate executable and `--backend wgpu --device 0` |
| 4 / 9 | Direct Vulkan NVIDIA physical 1 | same command with `--backend vulkan --device 1` | same command with candidate executable and `--backend vulkan --device 1` |
| 5 / 10 | `wgpu` NVIDIA/Vulkan adapter 1 | same command with `--backend wgpu --device 1` | same command with candidate executable and `--backend wgpu --device 1` |
| 11 | samtools | — | `/usr/local/bin/samtools flagstat -@ 8 /agents/shadowfax/data/HG002_chr22.bam` |

For the five candidate correctness preflights, the candidate commands above added `--validate`; one samtools preflight used the exact samtools command in slot 11. There were exactly six correctness preflights (five candidate plus samtools), eleven warmups (the five baseline commands, five candidate commands, then samtools in the table's canonical order), and 77 measured invocations (seven per command). Measured round `r` rotated the eleven-command list by `(r - 1) mod 11` positions while preserving cyclic order. The raw logs and analysis scripts are ephemeral under `/tmp/gpugeno-overlap-perf-20260921`.

Each invocation used Bash's reserved `time` with the tested separation shape:

```bash
TIMEFORMAT='%R'
{ time COMMAND >RUN.out 2>RUN.err; } 2>RUN.time
```

Before warmups, this wrapper was tested on a non-campaign command that emitted distinct `wrapper-stdout` and `wrapper-stderr` lines. Its `.time` file contained exactly one numeric value (`0.001` seconds), proving that stdout, command stderr, and timing stderr were separate. Every warmup and measured invocation then had its own `.out`, `.err`, `.time`, and `.status`; all 88 campaign invocations exited zero and passed immediate hash/coverage/metric checks.

There was one preflight-only wrapper mistake. The first CUDA preflight captured `preflight-cuda-nvidia0.time` (`2.458` seconds), but the wrapper used for the next four candidate preflights omitted the outer `2>RUN.time` redirection, so their timing values escaped to the session output and no four `.time` files exist. Preflight timing is excluded from every statistic; the existing four stdout/stderr/status files were preserved and independently validated for exit zero, metadata, exact coverage/hash, exact match, overlap fields, and the timing-relationship warning. They were not rerun or reconstructed. The samtools preflight used the corrected wrapper and captured `2.510` seconds. This is a methodology defect in excluded evidence, not a failed or replaced measured sample.

Current metadata captured in `metadata-nvidia-smi.txt` reported NVIDIA driver `550.163.01`, CUDA `12.4`, and two visible NVIDIA GeForce RTX 3060 12,288 MiB devices: device 0 at PCI bus `00000000:00:06.0` and device 1 at `00000000:00:08.0`. Direct Vulkan preflight stderr reported physical device 0 as AMD Radeon RX 6600 (RADV NAVI23), vendor/device `0x1002/0x73ff`, API `1.4.318`, and physical device 1 as NVIDIA GeForce RTX 3060, vendor/device `0x10de/0x2504`, API `1.3.277`; both used `gpu-timestamps`. `wgpu` adapter 0 reported AMD/RADV, underlying `api=Vulkan`, driver `radv`, `Mesa 25.2.7`, and adapter 1 reported NVIDIA, underlying `api=Vulkan`, driver `NVIDIA`, `550.163.01`; both used `gpu-timestamps`. CUDA stage fields use the established CUDA-event timing path; the CUDA benchmark stderr does not print a separate timing-source metadata line. `samtools --version` recorded samtools and htslib `1.24`.

#### Correctness and independent checks

All five candidate preflights exited zero, reported `result=exact-match`, and reported exactly 20 batches, 2,284 spans, 2,285 anchors, 82,360 decompressed blocks, 5,324,198,102 logical bytes, and 1,645,336,143 compressed bytes read. All five candidate stdout files were byte-identical to the samtools preflight and to the established hash:

```text
dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113
```

Every warmup and all 77 measured invocations exited zero, had that exact stdout hash, and retained the standard gpugeno coverage. Measured and warmup stderr omitted `result=exact-match` because they omitted validation. An independent Python standard-library parser verified the exact seven-row count for every command, the prescribed rotation, all 77 hashes, all 70 gpugeno coverage records, absence of overlap fields and the warning in baseline stderr, presence of all seven nonnegative overlap fields and the warning in candidate stderr, the expected Vulkan/`wgpu` timing sources, and no malformed elapsed file. It also checked every printed `GPU_stage` against `H2D + kernel + D2H` within 0.01 ms for printed rounding. No measured row was discarded.

#### External elapsed

Shell elapsed is reported in milliseconds here as median (observed min–max) over seven measured rows for each command. These are the external process boundaries used for samtools comparisons.

| command | external elapsed median (min–max) |
|---|---:|
| CUDA NVIDIA device 0 baseline | 2765.000 (2574.000–2802.000) |
| Direct Vulkan AMD physical 0 baseline | 4316.000 (4219.000–4398.000) |
| `wgpu` AMD/Vulkan adapter 0 baseline | 4063.000 (3990.000–4169.000) |
| Direct Vulkan NVIDIA physical 1 baseline | 3945.000 (3829.000–4013.000) |
| `wgpu` NVIDIA/Vulkan adapter 1 baseline | 3922.000 (3831.000–4021.000) |
| CUDA NVIDIA device 0 candidate | 2201.000 (2111.000–2282.000) |
| Direct Vulkan AMD physical 0 candidate | 3150.000 (3026.000–3280.000) |
| `wgpu` AMD/Vulkan adapter 0 candidate | 3099.000 (3083.000–3264.000) |
| Direct Vulkan NVIDIA physical 1 candidate | 3167.000 (3106.000–3281.000) |
| `wgpu` NVIDIA/Vulkan adapter 1 candidate | 3207.000 (3101.000–3294.000) |
| samtools 1.24 | 2477.000 (2431.000–2508.000) |

#### Absolute measured results

All cells below are milliseconds and are median (observed min–max) over seven measured rows. `host staging` means Vulkan `vulkan_host_staging_write` or `wgpu_staging_write`; `resource setup` and `packing` are the separately reported backend fields. A dash is not applicable. Portable raw-little-endian paths report packing as zero.

| row | program wall | batch_build | H2D | kernel | D2H | GPU_stage | host staging | resource setup | packing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUDA NVIDIA device 0 baseline | 2695.682 (2505.075–2739.969) | 1860.078 (1846.229–1884.189) | 607.740 (447.889–666.044) | 81.758 (81.650–81.953) | 0.476 (0.434–0.501) | 689.972 (530.059–748.404) | — | — | — |
| Direct Vulkan AMD physical 0 baseline | 3939.817 (3859.300–4014.912) | 1916.432 (1894.093–1966.900) | 376.216 (375.712–377.571) | 59.252 (58.822–59.357) | 0.064 (0.063–0.066) | 435.605 (434.598–436.940) | 749.816 (746.014–775.288) | 2.997 (2.883–41.207) | 0.000 (0.000–0.000) |
| `wgpu` AMD/Vulkan adapter 0 baseline | 3748.638 (3657.252–3769.746) | 1903.999 (1894.996–1953.851) | 381.011 (380.663–381.706) | 59.089 (58.803–59.200) | 0.051 (0.050–0.307) | 440.169 (439.679–441.022) | 469.766 (446.492–507.444) | 29.551 (29.019–33.788) | 0.000 (0.000–0.000) |
| Direct Vulkan NVIDIA physical 1 baseline | 3554.339 (3465.702–3602.513) | 1915.300 (1901.860–1940.162) | 223.332 (222.463–224.949) | 51.042 (51.017–51.062) | 0.050 (0.049–0.052) | 274.416 (273.573–276.033) | 412.240 (405.488–414.003) | 76.984 (73.923–79.447) | 0.000 (0.000–0.000) |
| `wgpu` NVIDIA/Vulkan adapter 1 baseline | 3511.543 (3446.009–3604.674) | 1909.085 (1880.365–1924.836) | 223.642 (222.913–228.240) | 51.215 (51.198–51.254) | 0.130 (0.125–0.141) | 275.022 (274.251–279.594) | 423.257 (416.389–427.694) | 82.360 (80.787–84.438) | 0.000 (0.000–0.000) |
| CUDA NVIDIA device 0 candidate | 2125.743 (2042.840–2211.102) | 2090.774 (2014.048–2172.059) | 723.412 (590.248–842.711) | 81.844 (81.757–81.934) | 0.727 (0.630–0.765) | 806.064 (672.777–925.234) | — | — | — |
| Direct Vulkan AMD physical 0 candidate | 2821.450 (2717.624–2926.575) | 2252.018 (2235.002–2302.065) | 377.883 (377.663–378.743) | 59.005 (58.809–59.313) | 0.064 (0.064–0.065) | 437.207 (436.833–437.616) | 903.686 (897.569–920.836) | 1.496 (1.328–8.362) | 0.000 (0.000–0.000) |
| `wgpu` AMD/Vulkan adapter 0 candidate | 2780.914 (2771.731–2942.601) | 2324.772 (2256.597–2378.058) | 384.775 (383.467–386.573) | 59.080 (58.616–59.344) | 0.051 (0.050–0.051) | 443.441 (442.638–445.478) | 585.544 (554.673–632.724) | 37.641 (34.886–39.857) | 0.000 (0.000–0.000) |
| Direct Vulkan NVIDIA physical 1 candidate | 2800.717 (2771.604–2909.953) | 2241.958 (2198.520–2290.276) | 244.124 (241.431–247.917) | 51.047 (51.004–51.087) | 0.051 (0.050–0.052) | 295.221 (292.569–298.972) | 513.021 (506.410–527.052) | 97.530 (85.151–100.954) | 0.000 (0.000–0.000) |
| `wgpu` NVIDIA/Vulkan adapter 1 candidate | 2825.821 (2741.617–2896.996) | 2309.439 (2128.669–2354.439) | 242.809 (234.812–247.132) | 51.229 (51.193–51.264) | 0.132 (0.126–0.139) | 294.133 (286.171–298.489) | 519.822 (475.361–536.023) | 104.779 (94.672–113.053) | 0.000 (0.000–0.000) |

#### Before/after and samtools tables

External elapsed is measured by the shell and is reported in milliseconds here; program `wall` is the internal benchmark field. The external speedup is `baseline external median / candidate external median`, with values above 1 faster. External and wall reduction are `(baseline - candidate) / baseline * 100`. Batch-build, GPU-stage, and host-staging ratios are `candidate median / baseline median`; a ratio above 1 exposes increased work or contention rather than assuming unchanged costs.

| combination | baseline external | candidate external | external speedup | external reduction | baseline wall | candidate wall | wall speedup | wall reduction | batch-build ratio | GPU-stage ratio | host-staging ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUDA NVIDIA device 0 | 2765.000 (2574.000–2802.000) | 2201.000 (2111.000–2282.000) | 1.256x | 20.40% | 2695.682 (2505.075–2739.969) | 2125.743 (2042.840–2211.102) | 1.268x | 21.14% | 1.124x | 1.168x | — |
| Direct Vulkan AMD physical 0 | 4316.000 (4219.000–4398.000) | 3150.000 (3026.000–3280.000) | 1.370x | 27.02% | 3939.817 (3859.300–4014.912) | 2821.450 (2717.624–2926.575) | 1.396x | 28.39% | 1.175x | 1.004x | 1.205x |
| `wgpu` AMD/Vulkan adapter 0 | 4063.000 (3990.000–4169.000) | 3099.000 (3083.000–3264.000) | 1.311x | 23.73% | 3748.638 (3657.252–3769.746) | 2780.914 (2771.731–2942.601) | 1.348x | 25.82% | 1.221x | 1.007x | 1.246x |
| Direct Vulkan NVIDIA physical 1 | 3945.000 (3829.000–4013.000) | 3167.000 (3106.000–3281.000) | 1.246x | 19.72% | 3554.339 (3465.702–3602.513) | 2800.717 (2771.604–2909.953) | 1.269x | 21.20% | 1.171x | 1.076x | 1.244x |
| `wgpu` NVIDIA/Vulkan adapter 1 | 3922.000 (3831.000–4021.000) | 3207.000 (3101.000–3294.000) | 1.223x | 18.23% | 3511.543 (3446.009–3604.674) | 2825.821 (2741.617–2896.996) | 1.243x | 19.53% | 1.210x | 1.069x | 1.228x |

Candidate overlap telemetry is also in milliseconds, median (observed min–max):

| candidate | consumer_first_batch_wait | consumer_next_batch_wait | consumer_eof_wait | producer_first_send_wait | producer_backpressure_wait | producer_terminal_send_wait | producer_lifetime |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUDA NVIDIA device 0 | 18.534 (15.008–24.919) | 1158.690 (1113.758–1228.276) | 0.008 (0.007–0.011) | 0.006 (0.005–0.006) | 0.112 (0.109–0.121) | 32.711 (26.239–36.592) | 2124.758 (2041.739–2209.958) |
| Direct Vulkan AMD physical 0 | 0.006 (0.005–0.009) | 619.321 (595.454–634.272) | 0.007 (0.006–0.009) | 461.294 (303.461–516.054) | 0.119 (0.113–0.124) | 58.197 (57.893–58.871) | 2820.462 (2716.754–2925.666) |
| `wgpu` AMD/Vulkan adapter 0 | 0.007 (0.005–0.007) | 870.259 (856.211–980.464) | 0.007 (0.005–0.009) | 387.530 (337.680–481.112) | 8.570 (0.116–18.067) | 44.826 (43.541–47.204) | 2780.009 (2770.690–2941.682) |
| Direct Vulkan NVIDIA physical 1 | 0.007 (0.006–0.017) | 1057.507 (1032.790–1098.736) | 0.007 (0.005–0.008) | 453.863 (380.648–518.393) | 38.299 (24.121–40.356) | 34.920 (34.614–36.053) | 2799.844 (2770.660–2908.918) |
| `wgpu` NVIDIA/Vulkan adapter 1 | 0.007 (0.006–0.009) | 1064.788 (1032.507–1083.167) | 0.006 (0.006–0.009) | 452.328 (357.232–527.523) | 60.539 (44.589–65.115) | 35.071 (34.482–36.118) | 2824.745 (2740.582–2896.003) |

Against samtools, the ratio is `candidate external median / samtools external median`; below 1 is faster. Percent difference is `(samtools median - candidate median) / samtools median * 100`, so positive means candidate faster and negative means candidate slower.

| candidate | candidate external median (min–max) ms | samtools external median (min–max) ms | candidate/samtools ratio | percent faster (+) / slower (−) |
|---|---:|---:|---:|---:|
| CUDA NVIDIA device 0 | 2201.000 (2111.000–2282.000) | 2477.000 (2431.000–2508.000) | 0.889x | 11.14% |
| Direct Vulkan AMD physical 0 | 3150.000 (3026.000–3280.000) | 2477.000 (2431.000–2508.000) | 1.272x | -27.17% |
| `wgpu` AMD/Vulkan adapter 0 | 3099.000 (3083.000–3264.000) | 2477.000 (2431.000–2508.000) | 1.251x | -25.11% |
| Direct Vulkan NVIDIA physical 1 | 3167.000 (3106.000–3281.000) | 2477.000 (2431.000–2508.000) | 1.279x | -27.86% |
| `wgpu` NVIDIA/Vulkan adapter 1 | 3207.000 (3101.000–3294.000) | 2477.000 (2431.000–2508.000) | 1.295x | -29.47% |

#### Interpretation, decision, and remaining risks

- External elapsed and program wall improved for every backend/device pair. External reductions were 20.40% CUDA, 27.02% Direct Vulkan AMD, 23.73% `wgpu` AMD, 19.72% Direct Vulkan NVIDIA, and 18.23% `wgpu` NVIDIA. Program-wall reductions were 21.14%, 28.39%, 25.82%, 21.20%, and 19.53%, respectively. This is the observed wall/elapsed outcome; no inferred `overlap_saved` value is computed or claimed.
- CUDA was faster than samtools in these seven warm-cache observations: candidate median 2.201 s versus 2.477 s, ratio 0.889x and 11.14% faster. The observed CUDA range (2.111–2.282 s) was below the samtools range (2.431–2.508 s), but seven observations do not establish statistical significance, general superiority, or cold-I/O behavior. Every portable candidate still trailed samtools in this comparison by 25.11–29.47%. Overlap nevertheless moved each portable path materially closer: candidate medians fell to 3.099–3.207 s from 3.922–4.316 s baseline.
- `consumer_next_batch_wait` was substantial for every candidate (medians 619.321–1,158.690 ms), indicating that the consumer often waited for producer next-batch construction: producer starvation remained visible rather than all CPU work being hidden. `producer_backpressure_wait` was small for CUDA and AMD Direct Vulkan (0.112 and 0.119 ms), but reached 8.570 ms on AMD `wgpu`, 38.299 ms on NVIDIA Direct Vulkan, and 60.539 ms on NVIDIA `wgpu`; those waits show cases where producer completion was hidden behind consumer-side work. These are telemetry interpretations, not an additive wall decomposition.
- Candidate `batch_build` increased by 12.4%–22.1% versus baseline (ratios 1.124, 1.175, 1.221, 1.171, and 1.210), consistent with CPU/memory contention from overlap. Portable host staging also increased by 20.5%–24.6%; packing remained exactly zero on both raw-little-endian portable paths. GPU-stage ratios were 1.168 CUDA, 1.004 AMD Direct Vulkan, 1.007 AMD `wgpu`, 1.076 NVIDIA Direct Vulkan, and 1.069 NVIDIA `wgpu`. The kernel medians themselves remained near their backend-specific prior values; H2D and host feeding absorbed much of the exposed contention. The Direct Vulkan AMD baseline resource-setup range reached 41.207 ms, and that observation was retained rather than discarded.
- Fill/drain effects are visible. CUDA's first consumer wait was 18.534 ms while its first producer send wait was only 0.006 ms; portable first producer send waits were 387.530–461.294 ms, consistent with producer work being ready while backend initialization/first receive became available. `consumer_eof_wait` was only 0.006–0.008 ms, while terminal-send waits were 32.711–58.197 ms; these small final waits and producer lifetime values show a finite drain/teardown phase rather than steady-state-only timing. The candidate starts the producer before backend construction, so first-send values include initialization interactions.
- Overlap is retained. It preserved exact representative correctness, reduced external elapsed and program wall in all five rows, and increases logical canonical host storage to at most two complete batches under the strict rendezvous. This is not a process-RSS hard bound: vector slack, metadata, eight worker states/payloads, and backend upload/device resources remain additional. The result is favorable even though some work-sum fields regress; overlapping work sums cannot be added to predict wall time.
- CUDA exists only on NVIDIA in this matrix. Portable AMD/NVIDIA rows are useful cross-vendor evidence, but Direct Vulkan and `wgpu` both use Vulkan underneath; `wgpu` is not a different low-level driver API here. The two NVIDIA API enumerations expose the same RTX 3060 model, but exact cross-API physical-board identity was not established, so this is a same-model rather than same-board comparison. Fixed eight workers, a 256 MiB cap, one synchronized GPU slot, warm cache, one process, and seven observations constrain generality. External shell elapsed and program wall have different boundaries; external elapsed is the metric used for samtools comparison. No following slice is approved.

### Remaining boundary

No multiple GPU slots/submissions, buffered or unbounded queue, direct mapped decompression, backend/kernel tuning, pileup, multi-GPU, packaging change, async runtime, coverage-policy change, or reference edit was added. The known inability to force-cancel a permanently stuck native decompressor and the pre-existing BGZF worker-panic robustness caveat remain. The controlled performance campaign is complete and documented above; overlap is retained, but no subsequent implementation slice is approved.

## Comparative samtools/gpugeno hotspot profiling

### Conclusion and scope

Completed 2026-09-23 as the owner-approved profiling-only comparison of samtools 1.24, gpugeno CUDA device 0, gpugeno `wgpu` AMD adapter 0, and gpugeno `wgpu` NVIDIA adapter 1. No implementation, parameter sweep, Direct Vulkan profile, public-interface change, or reference-repository edit was included.

Both samtools and gpugeno are decompression-dominated in CPU-active samples on the canonical workload. The strongest gpugeno-specific target is the ordered copy from worker-owned decompressed BGZF members into the canonical `PendingBatch.data`: it accounted for approximately 15–16% of flat process samples on every gpugeno path, and existing overlap telemetry independently shows that next-batch production remains exposed on the critical path. This supports proposing a narrow copy-removal experiment; it does **not** predict a wall-time improvement.

The worker's zero-filled decompression output is a separate, smaller hypothesis. Worker-side `memset` accounted for approximately 6% of process samples, but safely avoiding initialization may require a different ownership/raw-libdeflate design than merely eliminating the assembly copy. The full `memcpy`/`memset` totals are not removable work: `wgpu` has a separate mapped-staging copy, and compressed-member input construction contributes additional initialization.

### Profiling facilities and method

`perf`, Valgrind/Callgrind, GDB-family stack tools, `strace`, eBPF profilers, `uftrace`, and comparable CPU profilers were absent. Kernel policy was `perf_event_paranoid=3`. Nsight Systems 2026.1.3 reported both `perf_event_open` and CPU sampling unavailable, so it could not provide CPU instruction-pointer or call-stack samples without a policy or privilege change. No installation, `sudo`, or policy change was attempted.

The campaign therefore used two complementary sources:

1. A campaign-local `LD_PRELOAD` profiler installed process-wide `ITIMER_PROF`/`SIGPROF`, captured x86-64 instruction pointers and thread IDs into preallocated storage, and symbolized against final process mappings after exit. It was validated on balanced and deliberately asymmetric multi-thread fixtures. This provides repeated **flat CPU-active sample shares**, not call stacks, blocked time, absolute CPU accounting, or wall attribution.
2. Nsight Systems ran with CPU sampling disabled and OS-runtime tracing enabled. Its long-wait stacks were used only to interpret blocking and worker/driver activity, not as compute profiles or additive wall-time totals.

The sampler's material limitations are part of the result. Process-wide signals can coalesce and their thread delivery is statistical; requested periods of 1,000 microseconds in round one and 250 microseconds in rounds two and three still produced similar sample counts. The handler's libc `syscall(SYS_gettid)` is not guaranteed POSIX async-signal-safe, teardown has a latent race with an already-running handler, and final-map symbolization cannot reliably cover unloaded/JIT mappings or address reuse. No malformed or dropped campaign record was observed, leading shares were stable across rounds, and two validation fixtures supported coarse thread attribution. The Python resource wrapper also inherited the preload and output path; the target won the exclusive output creation in every retained run, but a target crash could have made this setup misleading. Future use should preload the target directly and record period/PID in the sample header.

The optimized symbolized gpugeno artifact was built from `a984568deb1a0348f8595e48102c375fbd7823b1` with release optimization, full debuginfo, no stripping, forced frame pointers, and an external `nvcc` wrapper adding device line information plus host debug/frame-pointer flags:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
CARGO_TARGET_DIR=/tmp/gpugeno-profile-20260922-lead/build-symbolized \
CARGO_INCREMENTAL=0 \
CARGO_PROFILE_RELEASE_DEBUG=2 \
CARGO_PROFILE_RELEASE_STRIP=none \
RUSTFLAGS="-C force-frame-pointers=yes" \
NVCC=/tmp/gpugeno-profile-20260922-lead/artifact/nvcc-profile-wrapper \
  cargo build --release --bin gpugeno
```

The retained ephemeral artifact was `/tmp/gpugeno-profile-20260922-lead/artifact/gpugeno-profile`, SHA-256 `4b716c2d4a63547ba1251d85aaf0ea75294fa4f604f5a0ad5a9bac90181fc637`. Its debug/frame-pointer configuration differs from the controlled performance binary, so elapsed observations from this campaign are smoke context only and are not merged with the earlier performance tables.

The exact target commands were:

```bash
INPUT=/agents/shadowfax/data/HG002_chr22.bam
BIN=/tmp/gpugeno-profile-20260922-lead/artifact/gpugeno-profile

/usr/local/bin/samtools flagstat -@ 8 "$INPUT"
"$BIN" flagstat "$INPUT" --backend cuda --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 --benchmark
"$BIN" flagstat "$INPUT" --backend wgpu --device 0 \
  --max-uncompressed-bytes 268435456 --threads 8 --benchmark
"$BIN" flagstat "$INPUT" --backend wgpu --device 1 \
  --max-uncompressed-bytes 268435456 --threads 8 --benchmark
```

Commands ran serially in the listed order: one unprofiled warmup/preflight per command, three flat-profile rounds per command, then one supplemental Nsight OS-runtime trace per command. The flat-profile periods were recorded in the campaign narrative but not embedded in each raw header or per-run command record, a reproducibility defect. Raw logs, profiler source, analysis, and traces are ephemeral under `/tmp/gpugeno-profile-20260922-lead`; the profiler shared-object SHA-256 was `11b24cc7c3dfc31f8538b0f62ac5ee1f3c0a3d98496f6109810c746b6b15c5f1`.

### Correctness and flat hotspot evidence

All 20 target invocations—four preflights, twelve flat profiles, and four OS-runtime traces—produced the established stdout SHA-256:

```text
dae9929278b2da62aec0393030a63dfcafa32a26dfd218242c037075c98cf113
```

All saved gpugeno stderr retained 20 batches, 2,284 spans, 2,285 anchors, 82,360 decompressed blocks, 5,324,198,102 logical bytes, and 1,645,336,143 compressed bytes read. The 16 preflight/flat wrapper records explicitly contain status zero; complete reports, empty collection stderr, complete target stderr, and exact target output establish successful completion of the four Nsight rows, though they lack separate captured target-status files.

The table aggregates three flat-profile rounds per command. Percentages are exclusive flat leaf/module shares of retained CPU-active samples.

| command | retained samples | dominant evidence |
|---|---:|---|
| samtools 1.24 | 4,516 | libz module 92.45%; unresolved internal libz 74.71%; exported `crc32_z` 17.03%; samtools executable 1.99%; `flagstat_loop` 0.31% |
| gpugeno CUDA 0 | 3,034 | `deflate_decompress_bmi2` 46.04%; all `memcpy` 15.43%; all `memset` 8.73%; libcuda module 6.03% |
| gpugeno `wgpu` AMD 0 | 3,026 | `deflate_decompress_bmi2` 43.82%; all `memcpy` 20.29%; all `memset` 8.82% |
| gpugeno `wgpu` NVIDIA 1 | 3,047 | `deflate_decompress_bmi2` 44.77%; all `memcpy` 19.72%; all `memset` 8.07% |

Samtools' system libz was stripped, so module attribution is stronger than its unresolved internal function labels. Gpugeno's libdeflate leaf was stable at 42.20–48.49% across paths and rounds. It is real work in the optimized BMI2 path, not evidence of an obvious local classifier defect or a reason to replace libdeflate.

Thread-role attribution used creation order, raw TID patterns, source-level function mix, and independent named-thread evidence from Nsight. It separates the actionable shared copy from unrelated memory work:

| path | worker output `memset` / all samples | producer `memcpy` / all samples | producer `memset` / all samples | `wgpu` main `memcpy` / all samples |
|---|---:|---:|---:|---:|
| CUDA | 6.66% | 15.23% | 2.08% | — |
| `wgpu` AMD | 6.35% | 15.66% | 2.12% | 4.53% |
| `wgpu` NVIDIA | 5.81% | 15.43% | 1.84% | 4.00% |

Source inspection matches these roles. `decompress_member` creates `vec![0u8; member.uncompressed_len]`; `DisjointBamStream::store_completed` then appends each ordered retained member slice with `pending.data.extend_from_slice(source)`. Separately, compressed input framing grows a member buffer with `resize(member_len, 0)` before `read_exact_at`, and `wgpu` copies canonical bytes into mapped staging memory. The roughly 5.324 GB canonical assembly volume makes the producer attribution strong, but without sampled call stacks it remains source-correlated caller inference.

Profiled gpugeno runs reported 866.872–1,340.142 ms of summed `consumer_next_batch_wait`. The prior controlled overlap campaign's corresponding medians for the same three profiled paths were 870.259–1,158.690 ms. These observations establish that producer completion is exposed for nonzero intervals; they do not show how much of that wait comes from the copy or convert overlapping work sums into a wall-time estimate.

Nsight's long OS-runtime events were dominated by samtools thread-pool condition waits and gpugeno worker/driver futex or condition waits. Corrected event totals were 4,356 samtools, 33,748 CUDA, 37,851 `wgpu` AMD, and 25,815 `wgpu` NVIDIA. Because durations sum across threads, much of this is expected parked-worker and driver-helper activity. It is not evidence for removing synchronization or for a lock-contention optimization.

### Recommendation and boundaries

A future owner-approved experiment may test eliminating the ordered member-to-`PendingBatch.data` assembly copy through direct writes into provably disjoint regions of an eventual canonical allocation, or an equivalently bounded design. Avoiding worker output initialization must be specified and measured as a separate sub-hypothesis rather than assumed to follow from copy removal.

Any design must preserve the strict two-complete-canonical-batch bound, deterministic canonical bytes/spans/block maps, one GPU resource slot, cap and terminal-member behavior, decompression-error safety, and producer cancellation/join protocol. No partially initialized bytes may become visible after failure, panic, or cancellation. Relevant tests include first-member nonzero offsets, terminal truncation inside a member, out-of-order completion, oversized spans, decompression failure, consumer cancellation, and exact sequential/producer equivalence. Explicit bytes/time telemetry should distinguish canonical assembly, destination initialization, and work merely moved elsewhere.

Flat CPU shares cannot be converted into Amdahl estimates under parallel workers, CPU/GPU overlap, and memory-bandwidth contention. The profiles do not support GPU/classifier tuning, worker-count changes, synchronization removal, or libdeflate replacement. No further profiling is required before deciding whether to approve the narrow design experiment, but that implementation is not approved by this campaign.
