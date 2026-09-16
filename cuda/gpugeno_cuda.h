/* C ABI for the gpugeno CUDA implementation.
 *
 * This header must stay strictly C-compatible: no C++ types, no name
 * mangling, and no exceptions may cross this boundary. The implementation
 * (`gpugeno_cuda.cu` plus operation-specific sources) is C++ compiled by
 * nvcc, but only the declarations below
 * are exposed to Rust.
 *
 * Status codes returned by the entry points:
 *   0  success
 *   1  invalid argument (null pointer, zero or overflowing size, bad device)
 *   2  CUDA runtime API failure
 *   3  CUDA device memory allocation failure
 *   4  element count exceeds the largest launchable grid
 *   5  unexpected C++ exception caught at the ABI boundary
 *
 * Error buffers: whenever `error_capacity` is nonzero, `error_message` is
 * always left NUL-terminated (empty on success, a description on failure).
 * The native code never terminates the process for these errors.
 */
#ifndef GPUGENO_CUDA_VECTOR_ADD_H
#define GPUGENO_CUDA_VECTOR_ADD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque native context: one selected CUDA device, one stream, three timing
 * event pairs, reusable vector-add device buffers, and reusable raw-byte,
 * span-offset, flagstat-result, and status buffers. Created with
 * gpugeno_cuda_create and released with gpugeno_cuda_destroy. */
struct gpugeno_cuda_context;

/* Per-stage timings measured with CUDA events on the context stream, in
 * milliseconds. */
struct gpugeno_cuda_timings {
    /* Both host-to-device uploads, timed together. */
    float h2d_ms;
    /* Kernel execution only; transfer time is excluded. */
    float kernel_ms;
    /* Device-to-host readback. */
    float d2h_ms;
};

/* Raw-byte upload timing measured with the context's H2D event pair. */
struct gpugeno_cuda_upload_timings {
    float h2d_ms;
};

/* Flagstat counters. Each two-element field is [QC-passed, QC-failed]. */
struct gpugeno_flagstat_counts {
    uint64_t n_reads[2];
    uint64_t n_mapped[2];
    uint64_t n_pair_all[2];
    uint64_t n_pair_map[2];
    uint64_t n_pair_good[2];
    uint64_t n_sgltn[2];
    uint64_t n_read1[2];
    uint64_t n_read2[2];
    uint64_t n_dup[2];
    uint64_t n_diffchr[2];
    uint64_t n_diffhigh[2];
    uint64_t n_secondary[2];
    uint64_t n_supp[2];
    uint64_t n_primary[2];
    uint64_t n_pmapped[2];
    uint64_t n_pdup[2];
};

/* Selects `device` and creates a context. Returns 0 on success and stores
 * the context in *out_context; on failure returns nonzero, leaves
 * *out_context null (when it is not null), and fills the error buffer. */
int gpugeno_cuda_create(
    int device,
    struct gpugeno_cuda_context **out_context,
    char *error_message,
    size_t error_capacity);

/* Computes output[i] = a[i] + b[i] on the context device for element_count
 * elements and reports per-stage timings. Synchronizes the stream before
 * returning, so `output` and `out_timings` are valid on return. Returns 0 on
 * success, nonzero on failure. */
int gpugeno_cuda_vector_add(
    struct gpugeno_cuda_context *context,
    const float *a,
    const float *b,
    float *output,
    size_t element_count,
    struct gpugeno_cuda_timings *out_timings,
    char *error_message,
    size_t error_capacity);

/* Uploads byte_count raw bytes into a context-owned reusable device buffer.
 * The H2D copy is timed with CUDA events on the existing stream. The stream
 * is synchronized before success is returned, so data is no longer borrowed
 * by CUDA on return. Null pointers and zero byte_count are rejected. */
int gpugeno_cuda_upload(
    struct gpugeno_cuda_context *context,
    const unsigned char *data,
    size_t byte_count,
    struct gpugeno_cuda_upload_timings *out_timings,
    char *error_message,
    size_t error_capacity);

/* Classifies record-aligned spans in one decompressed BAM batch. span_starts
 * contains span_count strictly increasing uint32 offsets; the final span ends
 * at byte_count. One result and status byte are copied back per span. Status
 * zero means success. The stream is synchronized before return. */
int gpugeno_cuda_flagstat(
    struct gpugeno_cuda_context *context,
    const unsigned char *data,
    size_t byte_count,
    const uint32_t *span_starts,
    size_t span_count,
    struct gpugeno_flagstat_counts *out_counts,
    unsigned char *out_status,
    struct gpugeno_cuda_timings *out_timings,
    char *error_message,
    size_t error_capacity);

/* Releases all native resources owned by the context. Accepts null. */
void gpugeno_cuda_destroy(struct gpugeno_cuda_context *context);

#ifdef __cplusplus
}
#endif

#endif /* GPUGENO_CUDA_VECTOR_ADD_H */
