#ifndef GPUGENO_FLAGSTAT_CUH
#define GPUGENO_FLAGSTAT_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Internal C++ launch boundary used by gpugeno_cuda.cu. The public Rust/C
// boundary remains declared in gpugeno_cuda.h.
cudaError_t gpugeno_launch_flagstat(
    const unsigned char *data,
    std::size_t byte_count,
    const std::uint32_t *span_starts,
    std::size_t span_count,
    unsigned long long *results,
    unsigned char *statuses,
    cudaStream_t stream);

#endif  // GPUGENO_FLAGSTAT_CUH
