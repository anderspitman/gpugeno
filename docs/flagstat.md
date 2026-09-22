# Flagstat semantics and work discovery

**Purpose:** Define the current flagstat semantic contract, BAM fields, counters, output, index-based GPU work discovery, and relevant reference hazards.
**Read when:** Changing a classifier, shader, counter reduction, output formatter, semantic fixture, BAM fixed-field access, span planning, or coverage policy.
**Authority:** `libshadowfax` is the initial classifier reference; current gpugeno behavior and tests govern the implemented contract. Historical reference defects are not compatibility requirements.

## Contents

- [Semantic reference](#flagstat-semantic-reference)
- [GPU work-discovery direction](#gpu-work-discovery-direction)
- [Known concerns in the old reference](#known-concerns-in-the-old-reference)

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
