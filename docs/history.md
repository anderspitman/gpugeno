# Project history and superseded directions

**Purpose:** Preserve the decision sequence, superseded assumptions, completed checkpoint progression, and rationale that could prevent future agents from repeating mistakes.
**Read when:** Reconsidering scope or architecture, interpreting a superseded statement, or asking why the project took its current direction.
**Reading strategy:** This is not current task guidance. Begin with [`README.md`](README.md) and current documents, then read only the relevant historical section.

## Contents

- [Completed checkpoint sequence](#completed-checkpoint-sequence)
- [Superseded byte-sum candidate](#superseded-candidate-gpu-per-block-byte-sums)
- [Decision and discovery history](#decision-and-discovery-history)
- [Deferred possibilities](#deferred-possibilities-not-a-committed-roadmap)
- [Documentation consolidation](#documentation-consolidation)

## Completed checkpoint sequence

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
- [x] Run the deferred controlled no-overlap/overlap/samtools performance campaign after owner approval.
- [x] Consolidate project status, plans, history, and the original idea into the sole `project.md` document.
- [x] Replace the monolithic project record with `AGENTS.md` and a small routed `docs/` set for human-readable progressive disclosure.
- [x] Add and verify a Podman-based x86-64 Rocky Linux 8 release build with an enforced glibc 2.28 ceiling and exported release metadata.

## Superseded candidate: GPU per-block byte sums

This candidate was not implemented. The project owner selected the more ambitious whole-file flagstat slice instead.

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

### Rocky Linux 8 release packaging

The first checked-in release workflow builds the complete CUDA-linked executable inside an x86-64 NVIDIA CUDA 12.4.1 Rocky Linux 8 container with Rust 1.98.1 and the locked Cargo graph. It enforces the Rocky Linux 8 `GLIBC_2.28` ceiling during the image build and exports a checksum plus toolchain/dependency metadata. The first verified artifact resolved all dynamic dependencies in the matching Rocky Linux 8 runtime image and retained the intended CUDA `sm_86` cubins and `compute_86` PTX. This addresses old-userspace ABI compatibility without changing program semantics or claiming CUDA-free packaging; current commands and runtime constraints are maintained in [`development.md`](development.md#rocky-linux-8-compatible-release-builds).

## Deferred possibilities, not a committed roadmap

Now that CUDA, native `wgpu`, bounded parallel decompression, raw-upload/resource reuse, actual AMD/Vulkan validation, the complete public direct-Vulkan backend, the bounded overlap implementation/correctness checkpoint, and its controlled performance evaluation are complete, plausible later experiments include optional CUDA packaging, pileup, or backend tuning. None is currently approved.

When revisiting portable backends, preserve these general intentions unless evidence changes them:

- Runtime selection through `--backend cuda|vulkan|wgpu`.
- Common semantic counters and deterministic input coverage.
- Backend-specific optimization rather than forced identical kernels.
- Report the actual adapter and underlying API used by `wgpu`; on Linux/NVIDIA it may itself use Vulkan.
- Keep filesystem/decompression orchestration out of the compute contract so a future browser host remains plausible.

## Documentation consolidation

The original idea and accumulated project memory were first consolidated from `idea.md` and `design.md` into a sole `project.md`. As that file grew to 1,571 lines, 20,590 words, and roughly 149 KiB, requiring every agent to read it became counterproductive.

That monolithic record was then replaced by a short shared entry point in [`README.md`](README.md) and routed current architecture, development procedures, flagstat semantics, implementation experiments, performance campaigns, and history. Repository-level [`AGENTS.md`](../AGENTS.md) makes the entry point and maintenance rules explicit. The detailed evidence was preserved while task-specific readers gained progressive disclosure; no compatibility pointer to the removed monolith is retained because agents start from the repository instructions.
