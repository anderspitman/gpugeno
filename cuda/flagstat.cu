// CUDA flagstat classifier and bounded BAM record-walk kernel.
//
// Classification semantics follow libshadowfax flagstat. Record discovery is
// inspired by the newer CuBayes pileup walk: thread 0 constructs a bounded
// shared offset table, then lanes classify distinct records.

#include "flagstat.cuh"
#include "gpugeno_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned kFlagstatThreads = 128;
constexpr unsigned kFlagstatValues = 32;
constexpr std::uint16_t kFlagPaired = 0x001;
constexpr std::uint16_t kFlagProperPair = 0x002;
constexpr std::uint16_t kFlagUnmapped = 0x004;
constexpr std::uint16_t kFlagMateUnmapped = 0x008;
constexpr std::uint16_t kFlagRead1 = 0x040;
constexpr std::uint16_t kFlagRead2 = 0x080;
constexpr std::uint16_t kFlagSecondary = 0x100;
constexpr std::uint16_t kFlagQcFail = 0x200;
constexpr std::uint16_t kFlagDuplicate = 0x400;
constexpr std::uint16_t kFlagSupplementary = 0x800;

static_assert(sizeof(gpugeno_flagstat_counts) == kFlagstatValues * sizeof(std::uint64_t));

__device__ std::uint16_t load_u16(const unsigned char *value) {
    return static_cast<std::uint16_t>(value[0]) |
           (static_cast<std::uint16_t>(value[1]) << 8);
}

__device__ std::uint32_t load_u32(const unsigned char *value) {
    return static_cast<std::uint32_t>(value[0]) |
           (static_cast<std::uint32_t>(value[1]) << 8) |
           (static_cast<std::uint32_t>(value[2]) << 16) |
           (static_cast<std::uint32_t>(value[3]) << 24);
}

__device__ void count_flagstat_record(const unsigned char *record,
                                      unsigned long long *stats) {
    const std::uint16_t flag = load_u16(record + 18);
    const unsigned w = (flag & kFlagQcFail) != 0 ? 1 : 0;
#define GPUGENO_COUNT(field) atomicAdd(stats + 2 * (field) + w, 1ULL)
    GPUGENO_COUNT(0);  // n_reads
    if ((flag & kFlagSecondary) != 0) {
        GPUGENO_COUNT(11);
    } else if ((flag & kFlagSupplementary) != 0) {
        GPUGENO_COUNT(12);
    } else {
        GPUGENO_COUNT(13);  // n_primary
        if ((flag & kFlagPaired) != 0) {
            const std::int32_t reference_id = static_cast<std::int32_t>(load_u32(record + 4));
            const std::int32_t next_reference_id =
                static_cast<std::int32_t>(load_u32(record + 24));
            const unsigned char mapq = record[13];
            GPUGENO_COUNT(2);  // n_pair_all
            if ((flag & kFlagProperPair) != 0 && (flag & kFlagUnmapped) == 0) {
                GPUGENO_COUNT(4);
            }
            if ((flag & kFlagRead1) != 0) GPUGENO_COUNT(6);
            if ((flag & kFlagRead2) != 0) GPUGENO_COUNT(7);
            if ((flag & kFlagMateUnmapped) != 0 && (flag & kFlagUnmapped) == 0) {
                GPUGENO_COUNT(5);
            }
            if ((flag & kFlagUnmapped) == 0 && (flag & kFlagMateUnmapped) == 0) {
                GPUGENO_COUNT(3);
                if (reference_id != next_reference_id) {
                    GPUGENO_COUNT(9);
                    if (mapq >= 5) GPUGENO_COUNT(10);
                }
            }
        }
        if ((flag & kFlagUnmapped) == 0) GPUGENO_COUNT(14);
        if ((flag & kFlagDuplicate) != 0) GPUGENO_COUNT(15);
    }
    if ((flag & kFlagUnmapped) == 0) GPUGENO_COUNT(1);
    if ((flag & kFlagDuplicate) != 0) GPUGENO_COUNT(8);
#undef GPUGENO_COUNT
}

// One block owns one exact physical record span. Every record is bounds
// checked before the classifier reads fixed BAM fields.
__global__ void flagstat_kernel(const unsigned char *data, std::size_t byte_count,
                                const std::uint32_t *span_starts,
                                std::size_t span_count,
                                unsigned long long *results,
                                unsigned char *statuses) {
    const std::size_t span = blockIdx.x;
    if (span >= span_count) return;
    const unsigned tid = threadIdx.x;
    const std::uint32_t begin = span_starts[span];
    const std::uint32_t end = span + 1 < span_count
                                  ? span_starts[span + 1]
                                  : static_cast<std::uint32_t>(byte_count);
    const std::uint32_t span_size = end - begin;
    const unsigned char *span_data = data + begin;

    __shared__ std::uint32_t record_offsets[kFlagstatThreads];
    __shared__ std::uint32_t next_offset;
    __shared__ std::uint32_t cursor;
    __shared__ unsigned record_count;
    __shared__ unsigned char error;
    __shared__ unsigned long long stats[kFlagstatValues];

    for (unsigned index = tid; index < kFlagstatValues; index += blockDim.x) stats[index] = 0;
    if (tid == 0) {
        cursor = 0;
        error = 0;
    }
    __syncthreads();

    while (true) {
        if (tid == 0) {
            std::uint32_t offset = cursor;
            unsigned count = 0;
            while (count < kFlagstatThreads && offset < span_size) {
                if (span_size - offset < 4) {
                    error = 1;
                    break;
                }
                const std::uint32_t block_size = load_u32(span_data + offset);
                if (block_size < 32) {
                    error = 2;
                    break;
                }
                if (block_size > span_size - offset - 4) {
                    error = 3;
                    break;
                }
                record_offsets[count++] = offset;
                offset += 4 + block_size;
            }
            record_count = count;
            next_offset = offset;
        }
        __syncthreads();
        if (error != 0 || record_count == 0) break;
        if (tid < record_count) count_flagstat_record(span_data + record_offsets[tid], stats);
        __syncthreads();
        if (tid == 0) cursor = next_offset;
        __syncthreads();
    }

    if (tid < kFlagstatValues) results[span * kFlagstatValues + tid] = stats[tid];
    if (tid == 0) statuses[span] = error;
}

}  // namespace

cudaError_t gpugeno_launch_flagstat(const unsigned char *data, std::size_t byte_count,
                                    const std::uint32_t *span_starts,
                                    std::size_t span_count,
                                    unsigned long long *results,
                                    unsigned char *statuses,
                                    cudaStream_t stream) {
    flagstat_kernel<<<static_cast<unsigned int>(span_count), kFlagstatThreads, 0, stream>>>(
        data, byte_count, span_starts, span_count, results, statuses);
    return cudaGetLastError();
}
