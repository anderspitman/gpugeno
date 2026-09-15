/* C ABI for the gpugeno CUDA vector-add integration spike.
 *
 * This header must stay strictly C-compatible: no C++ types, no name
 * mangling, and no exceptions may cross this boundary. The implementation
 * (vector_add.cu) is C++ compiled by nvcc, but only the declarations below
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

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque native context: one selected CUDA device, one stream, three timing
 * event pairs, and reusable device buffers. Created with
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

/* Releases all native resources owned by the context. Accepts null. */
void gpugeno_cuda_destroy(struct gpugeno_cuda_context *context);

#ifdef __cplusplus
}
#endif

#endif /* GPUGENO_CUDA_VECTOR_ADD_H */
