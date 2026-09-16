// CUDA C++ host wrapper and kernels for gpugeno.
//
// Compiled by nvcc (see build.rs) and linked statically into the Rust
// executable. Only the extern "C" functions declared in vector_add.h cross
// the ABI boundary: C++ exceptions are caught there, and ordinary CUDA
// errors are reported through the error buffer instead of terminating the
// process.

#include "vector_add.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>

// Complete definition of the opaque context declared in vector_add.h. It
// owns one stream, three timing event pairs, reusable vector-add buffers, and
// a reusable raw-byte upload buffer. It must stay at global scope so the
// extern "C" signatures match the header.
struct gpugeno_cuda_context {
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_end = nullptr;
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_output = nullptr;
    std::size_t capacity_elements = 0;
    unsigned char *device_bytes = nullptr;
    std::size_t capacity_bytes = 0;
    std::uint32_t *device_span_starts = nullptr;
    unsigned long long *device_flagstat_results = nullptr;
    unsigned char *device_flagstat_status = nullptr;
    std::size_t capacity_spans = 0;
};

namespace {

constexpr int kStatusOk = 0;
constexpr int kStatusInvalidArgument = 1;
constexpr int kStatusCudaError = 2;
constexpr int kStatusOutOfMemory = 3;
constexpr int kStatusUnsupported = 4;
constexpr int kStatusUnexpectedException = 5;

constexpr unsigned kBlockThreads = 256;
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

// One block owns one exact physical record span. Thread 0 constructs a
// bounded shared offset table as in newer CuBayes pileup record walking; all
// lanes then classify distinct records. This avoids the older flagstat
// kernel's repeated nth-record traversal while retaining its semantics.
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

__global__ void vector_add_kernel(const float *a, const float *b, float *output,
                                  std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = a[index] + b[index];
    }
}

// Writes "gpugeno_cuda error <code>: <message>" into error_message when
// possible. A nonzero-capacity buffer is always left NUL-terminated, even if
// the message does not fit.
void write_error(char *error_message, std::size_t error_capacity, int code, const char *message) {
    if (error_message == nullptr || error_capacity == 0) {
        return;
    }
    const int written =
        std::snprintf(error_message, error_capacity, "gpugeno_cuda error %d: %s", code, message);
    if (written < 0) {
        error_message[0] = '\0';
        return;
    }
    error_message[error_capacity - 1] = '\0';
}

int report_plain(int code, const char *message, char *error_message, std::size_t error_capacity) {
    write_error(error_message, error_capacity, code, message);
    return code;
}

int report_cuda_failure(const char *operation, cudaError_t error, int code, char *error_message,
                        std::size_t error_capacity) {
    char buffer[256];
    const char *cuda_message = cudaGetErrorString(error);
    if (cuda_message == nullptr || cuda_message[0] == '\0') {
        std::snprintf(buffer, sizeof buffer, "%s: CUDA error %d", operation,
                      static_cast<int>(error));
    } else {
        std::snprintf(buffer, sizeof buffer, "%s: %s", operation, cuda_message);
    }
    write_error(error_message, error_capacity, code, buffer);
    return code;
}

// Checks a CUDA Runtime call, converting any failure into a C API error
// return. Intended for use inside functions whose error-buffer parameters
// are named error_message / error_capacity.
#define GPUGENO_CUDA_CHECK(operation, status_code, cuda_call)                             \
    do {                                                                                  \
        const cudaError_t gpugeno_cuda_status = (cuda_call);                              \
        if (gpugeno_cuda_status != cudaSuccess) {                                         \
            return report_cuda_failure(operation, gpugeno_cuda_status, (status_code),     \
                                       error_message, error_capacity);                    \
        }                                                                                 \
    } while (false)

// Failure handler for calls made after stream work that reads or writes
// borrowed host memory may be enqueued: Rust's a / b / output slices are
// only valid until the FFI call returns, so the stream is drained
// best-effort (result ignored) before the original useful error is reported.
int report_failure_after_enqueue(const gpugeno_cuda_context *context, const char *operation,
                                 cudaError_t error, int code, char *error_message,
                                 std::size_t error_capacity) {
    cudaStreamSynchronize(context->stream);
    return report_cuda_failure(operation, error, code, error_message, error_capacity);
}

// Like GPUGENO_CUDA_CHECK, but for calls made after host-touching work may be
// enqueued: failures drain the stream first. Requires a `context` in scope.
#define GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(operation, status_code, cuda_call)               \
    do {                                                                                  \
        const cudaError_t gpugeno_cuda_status = (cuda_call);                              \
        if (gpugeno_cuda_status != cudaSuccess) {                                         \
            return report_failure_after_enqueue(context, operation, gpugeno_cuda_status,  \
                                                (status_code), error_message,             \
                                                error_capacity);                          \
        }                                                                                 \
    } while (false)

int create_context_impl(int device, struct gpugeno_cuda_context **out_context, char *error_message,
                        std::size_t error_capacity) {
    if (error_message != nullptr && error_capacity != 0) {
        error_message[0] = '\0';
    }
    if (out_context == nullptr) {
        return report_plain(kStatusInvalidArgument, "out_context is null", error_message,
                            error_capacity);
    }
    *out_context = nullptr;

    int device_count = 0;
    GPUGENO_CUDA_CHECK("cudaGetDeviceCount", kStatusCudaError, cudaGetDeviceCount(&device_count));
    if (device < 0 || device >= device_count) {
        char buffer[128];
        std::snprintf(buffer, sizeof buffer, "device index %d is out of range (%d available)",
                      device, device_count);
        return report_plain(kStatusInvalidArgument, buffer, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK("cudaSetDevice", kStatusCudaError, cudaSetDevice(device));

    gpugeno_cuda_context *context = new (std::nothrow) gpugeno_cuda_context();
    if (context == nullptr) {
        return report_plain(kStatusOutOfMemory, "failed to allocate the context structure",
                            error_message, error_capacity);
    }
    context->device = device;

    // Explicit check (not the macro): on failure the freshly allocated context
    // must be freed rather than leaked.
    const cudaError_t stream_error = cudaStreamCreate(&context->stream);
    if (stream_error != cudaSuccess) {
        delete context;
        return report_cuda_failure("cudaStreamCreate", stream_error, kStatusCudaError,
                                   error_message, error_capacity);
    }

    cudaEvent_t *event_slots[6] = {&context->h2d_start,  &context->h2d_end, &context->kernel_start,
                                   &context->kernel_end, &context->d2h_start, &context->d2h_end};
    for (std::size_t i = 0; i < sizeof event_slots / sizeof event_slots[0]; ++i) {
        const cudaError_t error = cudaEventCreateWithFlags(event_slots[i], cudaEventDefault);
        if (error != cudaSuccess) {
            for (std::size_t j = 0; j < i; ++j) {
                cudaEventDestroy(*event_slots[j]);
            }
            cudaStreamDestroy(context->stream);
            delete context;
            return report_cuda_failure("cudaEventCreateWithFlags", error, kStatusCudaError,
                                       error_message, error_capacity);
        }
    }

    *out_context = context;
    return kStatusOk;
}

int vector_add_impl(struct gpugeno_cuda_context *context, const float *a, const float *b,
                    float *output, std::size_t element_count,
                    struct gpugeno_cuda_timings *out_timings, char *error_message,
                    std::size_t error_capacity) {
    if (error_message != nullptr && error_capacity != 0) {
        error_message[0] = '\0';
    }
    if (context == nullptr || a == nullptr || b == nullptr || output == nullptr ||
        out_timings == nullptr) {
        return report_plain(kStatusInvalidArgument, "a required pointer is null", error_message,
                            error_capacity);
    }
    if (element_count == 0) {
        return report_plain(kStatusInvalidArgument, "element_count must be greater than zero",
                            error_message, error_capacity);
    }
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return report_plain(kStatusInvalidArgument, "element_count overflows the byte size",
                            error_message, error_capacity);
    }
    const std::size_t max_elements =
        static_cast<std::size_t>(kBlockThreads) * std::numeric_limits<unsigned int>::max();
    if (element_count > max_elements) {
        return report_plain(kStatusUnsupported, "element_count exceeds the maximum grid size",
                            error_message, error_capacity);
    }

    const std::size_t bytes = element_count * sizeof(float);
    GPUGENO_CUDA_CHECK("cudaSetDevice", kStatusCudaError, cudaSetDevice(context->device));

    if (element_count > context->capacity_elements) {
        // Allocate the replacement before releasing the current buffers so a
        // failed allocation cannot leave the context without usable buffers.
        float *next_a = nullptr;
        float *next_b = nullptr;
        float *next_output = nullptr;
        cudaError_t allocation = cudaMalloc(&next_a, bytes);
        if (allocation == cudaSuccess) {
            allocation = cudaMalloc(&next_b, bytes);
        }
        if (allocation == cudaSuccess) {
            allocation = cudaMalloc(&next_output, bytes);
        }
        if (allocation != cudaSuccess) {
            cudaFree(next_a);
            cudaFree(next_b);
            cudaFree(next_output);
            return report_cuda_failure("cudaMalloc", allocation, kStatusOutOfMemory,
                                       error_message, error_capacity);
        }
        cudaFree(context->device_a);
        cudaFree(context->device_b);
        cudaFree(context->device_output);
        context->device_a = next_a;
        context->device_b = next_b;
        context->device_output = next_output;
        context->capacity_elements = element_count;
    }

    // Upload both inputs; the H2D timing span covers both copies.
    GPUGENO_CUDA_CHECK("cudaEventRecord (h2d start)", kStatusCudaError,
                       cudaEventRecord(context->h2d_start, context->stream));
    cudaError_t transfer =
        cudaMemcpyAsync(context->device_a, a, bytes, cudaMemcpyHostToDevice, context->stream);
    if (transfer == cudaSuccess) {
        transfer = cudaMemcpyAsync(context->device_b, b, bytes, cudaMemcpyHostToDevice,
                                   context->stream);
    }
    if (transfer != cudaSuccess) {
        // The first copy may already be enqueued when the second fails to
        // submit; drain before returning so it cannot outlive the Rust borrow.
        return report_failure_after_enqueue(context, "cudaMemcpyAsync (H2D)", transfer,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (h2d end)", kStatusCudaError,
                                     cudaEventRecord(context->h2d_end, context->stream));

    // Kernel; the kernel timing span covers execution only.
    const unsigned int blocks =
        static_cast<unsigned int>((element_count + kBlockThreads - 1) / kBlockThreads);
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (kernel start)", kStatusCudaError,
                                     cudaEventRecord(context->kernel_start, context->stream));
    vector_add_kernel<<<blocks, kBlockThreads, 0, context->stream>>>(context->device_a,
                                                                     context->device_b,
                                                                     context->device_output,
                                                                     element_count);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        // The kernel did not run, but the uploads are still pending.
        return report_failure_after_enqueue(context, "vector_add_kernel launch", launch_error,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (kernel end)", kStatusCudaError,
                                     cudaEventRecord(context->kernel_end, context->stream));

    // Readback; the D2H timing span covers the copy only.
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (d2h start)", kStatusCudaError,
                                     cudaEventRecord(context->d2h_start, context->stream));
    transfer = cudaMemcpyAsync(output, context->device_output, bytes, cudaMemcpyDeviceToHost,
                               context->stream);
    if (transfer != cudaSuccess) {
        return report_failure_after_enqueue(context, "cudaMemcpyAsync (D2H)", transfer,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (d2h end)", kStatusCudaError,
                                     cudaEventRecord(context->d2h_end, context->stream));

    // Make the output and the timings valid before returning. This sync is
    // already the best-effort drain for the phase, so its failure is reported
    // as-is.
    GPUGENO_CUDA_CHECK("cudaStreamSynchronize", kStatusCudaError,
                       cudaStreamSynchronize(context->stream));

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (H2D)", kStatusCudaError,
                       cudaEventElapsedTime(&h2d_ms, context->h2d_start, context->h2d_end));
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (kernel)", kStatusCudaError,
                       cudaEventElapsedTime(&kernel_ms, context->kernel_start,
                                            context->kernel_end));
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (D2H)", kStatusCudaError,
                       cudaEventElapsedTime(&d2h_ms, context->d2h_start, context->d2h_end));

    out_timings->h2d_ms = h2d_ms;
    out_timings->kernel_ms = kernel_ms;
    out_timings->d2h_ms = d2h_ms;
    return kStatusOk;
}

int upload_impl(struct gpugeno_cuda_context *context, const unsigned char *data,
                std::size_t byte_count, struct gpugeno_cuda_upload_timings *out_timings,
                char *error_message, std::size_t error_capacity) {
    if (error_message != nullptr && error_capacity != 0) {
        error_message[0] = '\0';
    }
    if (context == nullptr || data == nullptr || out_timings == nullptr) {
        return report_plain(kStatusInvalidArgument, "a required pointer is null", error_message,
                            error_capacity);
    }
    if (byte_count == 0) {
        return report_plain(kStatusInvalidArgument, "byte_count must be greater than zero",
                            error_message, error_capacity);
    }

    GPUGENO_CUDA_CHECK("cudaSetDevice", kStatusCudaError, cudaSetDevice(context->device));

    if (byte_count > context->capacity_bytes) {
        // Preserve the old allocation until its replacement has succeeded.
        unsigned char *next_bytes = nullptr;
        const cudaError_t allocation = cudaMalloc(&next_bytes, byte_count);
        if (allocation != cudaSuccess) {
            return report_cuda_failure("cudaMalloc (raw byte buffer)", allocation,
                                       kStatusOutOfMemory, error_message, error_capacity);
        }
        const cudaError_t release = cudaFree(context->device_bytes);
        if (release != cudaSuccess) {
            cudaFree(next_bytes);
            return report_cuda_failure("cudaFree (old raw byte buffer)", release,
                                       kStatusCudaError, error_message, error_capacity);
        }
        context->device_bytes = next_bytes;
        context->capacity_bytes = byte_count;
    }

    GPUGENO_CUDA_CHECK("cudaEventRecord (upload start)", kStatusCudaError,
                       cudaEventRecord(context->h2d_start, context->stream));
    const cudaError_t transfer = cudaMemcpyAsync(context->device_bytes, data, byte_count,
                                                  cudaMemcpyHostToDevice, context->stream);
    if (transfer != cudaSuccess) {
        return report_failure_after_enqueue(context, "cudaMemcpyAsync (raw H2D)", transfer,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE("cudaEventRecord (upload end)", kStatusCudaError,
                                     cudaEventRecord(context->h2d_end, context->stream));

    // Synchronize before success so CUDA cannot retain a reference to the
    // borrowed Rust slice. On failure, make one more best-effort drain while
    // preserving the useful error from the original synchronization call.
    const cudaError_t synchronization = cudaStreamSynchronize(context->stream);
    if (synchronization != cudaSuccess) {
        return report_failure_after_enqueue(context, "cudaStreamSynchronize (raw upload)",
                                            synchronization, kStatusCudaError, error_message,
                                            error_capacity);
    }

    float h2d_ms = 0.0f;
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (raw H2D)", kStatusCudaError,
                       cudaEventElapsedTime(&h2d_ms, context->h2d_start, context->h2d_end));
    out_timings->h2d_ms = h2d_ms;
    return kStatusOk;
}

int flagstat_impl(struct gpugeno_cuda_context *context, const unsigned char *data,
                  std::size_t byte_count, const std::uint32_t *span_starts,
                  std::size_t span_count, struct gpugeno_flagstat_counts *out_counts,
                  unsigned char *out_status, struct gpugeno_cuda_timings *out_timings,
                  char *error_message, std::size_t error_capacity) {
    if (error_message != nullptr && error_capacity != 0) error_message[0] = '\0';
    if (context == nullptr || data == nullptr || span_starts == nullptr || out_counts == nullptr ||
        out_status == nullptr || out_timings == nullptr) {
        return report_plain(kStatusInvalidArgument, "a required flagstat pointer is null",
                            error_message, error_capacity);
    }
    if (byte_count == 0 || byte_count > std::numeric_limits<std::uint32_t>::max()) {
        return report_plain(kStatusInvalidArgument,
                            "flagstat byte_count must fit a nonzero uint32 range", error_message,
                            error_capacity);
    }
    if (span_count == 0 || span_count > std::numeric_limits<unsigned int>::max()) {
        return report_plain(kStatusInvalidArgument, "invalid flagstat span_count", error_message,
                            error_capacity);
    }
    if (span_count > std::numeric_limits<std::size_t>::max() /
                         sizeof(gpugeno_flagstat_counts) ||
        span_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
        return report_plain(kStatusInvalidArgument, "flagstat buffer size overflows size_t",
                            error_message, error_capacity);
    }
    if (span_starts[0] != 0 || span_starts[span_count - 1] >= byte_count) {
        return report_plain(kStatusInvalidArgument, "invalid flagstat span boundary",
                            error_message, error_capacity);
    }
    for (std::size_t span = 1; span < span_count; ++span) {
        if (span_starts[span - 1] >= span_starts[span]) {
            return report_plain(kStatusInvalidArgument,
                                "flagstat span starts are not strictly increasing", error_message,
                                error_capacity);
        }
    }

    GPUGENO_CUDA_CHECK("cudaSetDevice", kStatusCudaError, cudaSetDevice(context->device));

    if (byte_count > context->capacity_bytes) {
        unsigned char *replacement = nullptr;
        const cudaError_t allocation = cudaMalloc(&replacement, byte_count);
        if (allocation != cudaSuccess) {
            return report_cuda_failure("cudaMalloc (flagstat bytes)", allocation,
                                       kStatusOutOfMemory, error_message, error_capacity);
        }
        const cudaError_t release = cudaFree(context->device_bytes);
        if (release != cudaSuccess) {
            cudaFree(replacement);
            return report_cuda_failure("cudaFree (old flagstat bytes)", release,
                                       kStatusCudaError, error_message, error_capacity);
        }
        context->device_bytes = replacement;
        context->capacity_bytes = byte_count;
    }

    if (span_count > context->capacity_spans) {
        std::uint32_t *next_offsets = nullptr;
        unsigned long long *next_results = nullptr;
        unsigned char *next_status = nullptr;
        cudaError_t allocation = cudaMalloc(&next_offsets, span_count * sizeof(std::uint32_t));
        if (allocation == cudaSuccess) {
            allocation = cudaMalloc(&next_results,
                                    span_count * sizeof(gpugeno_flagstat_counts));
        }
        if (allocation == cudaSuccess) {
            allocation = cudaMalloc(&next_status, span_count);
        }
        if (allocation != cudaSuccess) {
            cudaFree(next_offsets);
            cudaFree(next_results);
            cudaFree(next_status);
            return report_cuda_failure("cudaMalloc (flagstat buffers)", allocation,
                                       kStatusOutOfMemory, error_message, error_capacity);
        }
        cudaFree(context->device_span_starts);
        cudaFree(context->device_flagstat_results);
        cudaFree(context->device_flagstat_status);
        context->device_span_starts = next_offsets;
        context->device_flagstat_results = next_results;
        context->device_flagstat_status = next_status;
        context->capacity_spans = span_count;
    }

    GPUGENO_CUDA_CHECK("cudaEventRecord (flagstat H2D start)", kStatusCudaError,
                       cudaEventRecord(context->h2d_start, context->stream));
    cudaError_t transfer = cudaMemcpyAsync(context->device_bytes, data, byte_count,
                                            cudaMemcpyHostToDevice, context->stream);
    if (transfer == cudaSuccess) {
        transfer = cudaMemcpyAsync(context->device_span_starts, span_starts,
                                   span_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                   context->stream);
    }
    if (transfer != cudaSuccess) {
        return report_failure_after_enqueue(context, "cudaMemcpyAsync (flagstat H2D)", transfer,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(
        "cudaEventRecord (flagstat H2D end)", kStatusCudaError,
        cudaEventRecord(context->h2d_end, context->stream));

    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(
        "cudaEventRecord (flagstat kernel start)", kStatusCudaError,
        cudaEventRecord(context->kernel_start, context->stream));
    flagstat_kernel<<<static_cast<unsigned int>(span_count), kFlagstatThreads, 0,
                      context->stream>>>(context->device_bytes, byte_count,
                                         context->device_span_starts, span_count,
                                         context->device_flagstat_results,
                                         context->device_flagstat_status);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return report_failure_after_enqueue(context, "flagstat_kernel launch", launch_error,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(
        "cudaEventRecord (flagstat kernel end)", kStatusCudaError,
        cudaEventRecord(context->kernel_end, context->stream));

    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(
        "cudaEventRecord (flagstat D2H start)", kStatusCudaError,
        cudaEventRecord(context->d2h_start, context->stream));
    transfer = cudaMemcpyAsync(out_counts, context->device_flagstat_results,
                               span_count * sizeof(gpugeno_flagstat_counts),
                               cudaMemcpyDeviceToHost, context->stream);
    if (transfer == cudaSuccess) {
        transfer = cudaMemcpyAsync(out_status, context->device_flagstat_status, span_count,
                                   cudaMemcpyDeviceToHost, context->stream);
    }
    if (transfer != cudaSuccess) {
        return report_failure_after_enqueue(context, "cudaMemcpyAsync (flagstat D2H)", transfer,
                                            kStatusCudaError, error_message, error_capacity);
    }
    GPUGENO_CUDA_CHECK_AFTER_ENQUEUE(
        "cudaEventRecord (flagstat D2H end)", kStatusCudaError,
        cudaEventRecord(context->d2h_end, context->stream));
    GPUGENO_CUDA_CHECK("cudaStreamSynchronize (flagstat)", kStatusCudaError,
                       cudaStreamSynchronize(context->stream));

    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (flagstat H2D)", kStatusCudaError,
                       cudaEventElapsedTime(&out_timings->h2d_ms, context->h2d_start,
                                            context->h2d_end));
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (flagstat kernel)", kStatusCudaError,
                       cudaEventElapsedTime(&out_timings->kernel_ms, context->kernel_start,
                                            context->kernel_end));
    GPUGENO_CUDA_CHECK("cudaEventElapsedTime (flagstat D2H)", kStatusCudaError,
                       cudaEventElapsedTime(&out_timings->d2h_ms, context->d2h_start,
                                            context->d2h_end));

    for (std::size_t span = 0; span < span_count; ++span) {
        if (out_status[span] != 0) {
            char buffer[160];
            std::snprintf(buffer, sizeof buffer,
                          "malformed or non-record-aligned BAM span %zu (kernel status %u)", span,
                          static_cast<unsigned>(out_status[span]));
            return report_plain(kStatusInvalidArgument, buffer, error_message, error_capacity);
        }
    }
    return kStatusOk;
}

void destroy_context(struct gpugeno_cuda_context *context) {
    if (context == nullptr) {
        return;
    }
    // The context API is void, so there is no error channel here; failures are
    // swallowed rather than propagated or made fatal.
    cudaSetDevice(context->device);
    cudaStreamSynchronize(context->stream);
    cudaEventDestroy(context->h2d_start);
    cudaEventDestroy(context->h2d_end);
    cudaEventDestroy(context->kernel_start);
    cudaEventDestroy(context->kernel_end);
    cudaEventDestroy(context->d2h_start);
    cudaEventDestroy(context->d2h_end);
    cudaFree(context->device_a);
    cudaFree(context->device_b);
    cudaFree(context->device_output);
    cudaFree(context->device_bytes);
    cudaFree(context->device_span_starts);
    cudaFree(context->device_flagstat_results);
    cudaFree(context->device_flagstat_status);
    cudaStreamDestroy(context->stream);
    delete context;
}

}  // namespace

extern "C" int gpugeno_cuda_create(int device, struct gpugeno_cuda_context **out_context,
                                   char *error_message, std::size_t error_capacity) {
    try {
        return create_context_impl(device, out_context, error_message, error_capacity);
    } catch (const std::exception &exception) {
        write_error(error_message, error_capacity, kStatusUnexpectedException, exception.what());
        return kStatusUnexpectedException;
    } catch (...) {
        write_error(error_message, error_capacity, kStatusUnexpectedException,
                    "unexpected C++ exception");
        return kStatusUnexpectedException;
    }
}

extern "C" int gpugeno_cuda_vector_add(struct gpugeno_cuda_context *context, const float *a,
                                       const float *b, float *output, std::size_t element_count,
                                       struct gpugeno_cuda_timings *out_timings,
                                       char *error_message, std::size_t error_capacity) {
    try {
        return vector_add_impl(context, a, b, output, element_count, out_timings, error_message,
                               error_capacity);
    } catch (const std::exception &exception) {
        write_error(error_message, error_capacity, kStatusUnexpectedException, exception.what());
        return kStatusUnexpectedException;
    } catch (...) {
        write_error(error_message, error_capacity, kStatusUnexpectedException,
                    "unexpected C++ exception");
        return kStatusUnexpectedException;
    }
}

extern "C" int gpugeno_cuda_upload(struct gpugeno_cuda_context *context,
                                   const unsigned char *data, std::size_t byte_count,
                                   struct gpugeno_cuda_upload_timings *out_timings,
                                   char *error_message, std::size_t error_capacity) {
    try {
        return upload_impl(context, data, byte_count, out_timings, error_message, error_capacity);
    } catch (const std::exception &exception) {
        // Defensively drain if an unexpected exception ever occurs after the
        // H2D submission; no borrowed Rust memory may outlive this call.
        if (context != nullptr) {
            cudaStreamSynchronize(context->stream);
        }
        write_error(error_message, error_capacity, kStatusUnexpectedException, exception.what());
        return kStatusUnexpectedException;
    } catch (...) {
        if (context != nullptr) {
            cudaStreamSynchronize(context->stream);
        }
        write_error(error_message, error_capacity, kStatusUnexpectedException,
                    "unexpected C++ exception");
        return kStatusUnexpectedException;
    }
}

extern "C" int gpugeno_cuda_flagstat(
    struct gpugeno_cuda_context *context, const unsigned char *data, std::size_t byte_count,
    const std::uint32_t *span_starts, std::size_t span_count,
    struct gpugeno_flagstat_counts *out_counts, unsigned char *out_status,
    struct gpugeno_cuda_timings *out_timings, char *error_message,
    std::size_t error_capacity) {
    try {
        return flagstat_impl(context, data, byte_count, span_starts, span_count, out_counts,
                             out_status, out_timings, error_message, error_capacity);
    } catch (const std::exception &exception) {
        if (context != nullptr) cudaStreamSynchronize(context->stream);
        write_error(error_message, error_capacity, kStatusUnexpectedException, exception.what());
        return kStatusUnexpectedException;
    } catch (...) {
        if (context != nullptr) cudaStreamSynchronize(context->stream);
        write_error(error_message, error_capacity, kStatusUnexpectedException,
                    "unexpected C++ exception");
        return kStatusUnexpectedException;
    }
}

extern "C" void gpugeno_cuda_destroy(struct gpugeno_cuda_context *context) {
    try {
        destroy_context(context);
    } catch (...) {
        // Nothing to do: the C API has no error channel, and no exception may
        // cross the boundary into Rust.
    }
}
