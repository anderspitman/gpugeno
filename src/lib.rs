//! Direct Vulkan owns the production `ash` backend; the historical diagnostic
//! example keeps its `vulkan_flagstat_spike` name but imports this module.
//!
//! The native host side is `cuda/gpugeno_cuda.cu`, and operation-specific
//! kernels such as flagstat live in their own CUDA sources. They are compiled by `nvcc` (see
//! `build.rs`) and linked into the executable. It exposes a narrow C ABI;
//! this crate wraps that ABI so callers never touch `unsafe`.

pub mod bai;
pub mod bam;
pub mod bgzf;
pub mod indexed_batch;
pub mod vulkan_backend;
pub mod wgpu_backend;

use bam::FlagstatCounters;

#[cfg(test)]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Timings reported by a native CUDA vector-add call, in milliseconds.
///
/// Layout matches `struct gpugeno_cuda_timings` in `cuda/gpugeno_cuda.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaTimings {
    /// Host-to-device upload of both input vectors, timed together.
    pub h2d_ms: f32,
    /// Vector-add kernel execution only (excludes transfers).
    pub kernel_ms: f32,
    /// Device-to-host readback of the result.
    pub d2h_ms: f32,
}

/// Timing reported by a synchronized raw-byte upload, in milliseconds.
///
/// Layout matches `struct gpugeno_cuda_upload_timings` in
/// `cuda/gpugeno_cuda.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaUploadTimings {
    /// Host-to-device copy measured with CUDA events.
    pub h2d_ms: f32,
}

#[derive(Debug)]
pub struct CudaFlagstatBatch {
    pub span_counts: Vec<FlagstatCounters>,
    pub timings: CudaTimings,
}

/// An error reported by the native CUDA boundary.
#[derive(Debug)]
pub enum CudaError {
    /// Argument validation failed before the native call.
    InvalidArgument(String),
    /// The native call returned a nonzero status.
    Native {
        /// Native status code: 1 invalid argument, 2 CUDA runtime error,
        /// 3 out of device memory, 4 size beyond the launchable grid,
        /// 5 unexpected C++ exception caught at the boundary.
        code: i32,
        /// Native-provided human-readable message, when one was supplied.
        message: String,
    },
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Native { code, message } => write!(f, "native CUDA error {code}: {message}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Owner of one opaque native CUDA context.
///
/// The context is released by the native `gpugeno_cuda_destroy` call in
/// `Drop`, so a dropped or failed owner cannot leak native resources.
pub struct CudaContext {
    raw: *mut std::os::raw::c_void,
}

impl CudaContext {
    /// Creates a native context on `device`, which the native side selects
    /// before any other CUDA work.
    pub fn create(device: i32) -> Result<Self, CudaError> {
        let mut raw: *mut std::os::raw::c_void = std::ptr::null_mut();
        let mut error_message = [0u8; 1024];
        let status = unsafe {
            ffi::gpugeno_cuda_create(
                device,
                &mut raw as *mut *mut std::os::raw::c_void,
                error_message.as_mut_ptr().cast::<std::os::raw::c_char>(),
                error_message.len(),
            )
        };
        if status != 0 {
            return Err(CudaError::Native {
                code: status,
                message: c_message(&error_message),
            });
        }
        if raw.is_null() {
            return Err(CudaError::Native {
                code: status,
                message: "native call succeeded but returned a null context".to_string(),
            });
        }
        Ok(Self { raw })
    }

    /// Adds `a` and `b` elementwise on the GPU, writing the result into
    /// `output`.
    ///
    /// All three slices must have the same nonzero length. On success the
    /// stream is synchronized by the native side and per-stage timings are
    /// returned.
    pub fn vector_add(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) -> Result<CudaTimings, CudaError> {
        let elements = a.len();
        if elements == 0 || elements != b.len() || elements != output.len() {
            return Err(CudaError::InvalidArgument(format!(
                "a, b, and output must have the same nonzero length (got {elements}, {}, {})",
                b.len(),
                output.len()
            )));
        }
        if elements > usize::MAX / std::mem::size_of::<f32>() {
            return Err(CudaError::InvalidArgument(format!(
                "element count {elements} overflows the byte size"
            )));
        }

        // Pre-initialized so a failed native call can never expose an
        // uninitialized value.
        let mut timings = CudaTimings::default();
        let mut error_message = [0u8; 1024];
        let status = unsafe {
            ffi::gpugeno_cuda_vector_add(
                self.raw,
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
                elements,
                &mut timings,
                error_message.as_mut_ptr().cast::<std::os::raw::c_char>(),
                error_message.len(),
            )
        };
        if status != 0 {
            return Err(CudaError::Native {
                code: status,
                message: c_message(&error_message),
            });
        }
        Ok(timings)
    }

    /// Uploads a nonempty byte slice into the context's reusable raw device
    /// buffer.
    ///
    /// The native side synchronizes its stream before returning, so CUDA no
    /// longer references `data` when this method completes.
    pub fn upload(&self, data: &[u8]) -> Result<CudaUploadTimings, CudaError> {
        if data.is_empty() {
            return Err(CudaError::InvalidArgument(
                "upload data must not be empty".to_string(),
            ));
        }

        let mut timings = CudaUploadTimings::default();
        let mut error_message = [0u8; 1024];
        let status = unsafe {
            ffi::gpugeno_cuda_upload(
                self.raw,
                data.as_ptr(),
                data.len(),
                &mut timings,
                error_message.as_mut_ptr().cast::<std::os::raw::c_char>(),
                error_message.len(),
            )
        };
        if status != 0 {
            return Err(CudaError::Native {
                code: status,
                message: c_message(&error_message),
            });
        }
        Ok(timings)
    }

    /// Classifies one bounded decompressed BAM batch. Every offset starts a
    /// disjoint record span; the final span ends at `data.len()`.
    pub fn flagstat(
        &self,
        data: &[u8],
        span_starts: &[u32],
    ) -> Result<CudaFlagstatBatch, CudaError> {
        if data.is_empty() || data.len() > u32::MAX as usize {
            return Err(CudaError::InvalidArgument(
                "flagstat data must fit a nonempty u32 byte range".to_string(),
            ));
        }
        if span_starts.is_empty()
            || span_starts[0] != 0
            || span_starts.windows(2).any(|pair| pair[0] >= pair[1])
            || usize::try_from(*span_starts.last().unwrap()).unwrap() >= data.len()
        {
            return Err(CudaError::InvalidArgument(
                "flagstat span starts must begin at zero, increase strictly, and lie within data"
                    .to_string(),
            ));
        }

        let mut span_counts = vec![FlagstatCounters::default(); span_starts.len()];
        let mut statuses = vec![0u8; span_starts.len()];
        let mut timings = CudaTimings::default();
        let mut error_message = [0u8; 1024];
        let status = unsafe {
            ffi::gpugeno_cuda_flagstat(
                self.raw,
                data.as_ptr(),
                data.len(),
                span_starts.as_ptr(),
                span_starts.len(),
                span_counts.as_mut_ptr(),
                statuses.as_mut_ptr(),
                &mut timings,
                error_message.as_mut_ptr().cast::<std::os::raw::c_char>(),
                error_message.len(),
            )
        };
        if status != 0 {
            return Err(CudaError::Native {
                code: status,
                message: c_message(&error_message),
            });
        }
        Ok(CudaFlagstatBatch {
            span_counts,
            timings,
        })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { ffi::gpugeno_cuda_destroy(self.raw) };
    }
}

/// Converts a NUL-terminated native error buffer into a Rust string.
fn c_message(buffer: &[u8]) -> String {
    let end = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

mod ffi {
    //! Raw C ABI declarations for `cuda/gpugeno_cuda.h`. Keep in sync with that
    //! header: signatures, status codes, and the timings layout.

    use super::{CudaTimings, CudaUploadTimings};
    use crate::bam::FlagstatCounters;
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        pub(crate) fn gpugeno_cuda_create(
            device: c_int,
            out_context: *mut *mut c_void,
            error_message: *mut c_char,
            error_capacity: usize,
        ) -> c_int;

        pub(crate) fn gpugeno_cuda_vector_add(
            context: *mut c_void,
            a: *const f32,
            b: *const f32,
            output: *mut f32,
            element_count: usize,
            out_timings: *mut CudaTimings,
            error_message: *mut c_char,
            error_capacity: usize,
        ) -> c_int;

        pub(crate) fn gpugeno_cuda_upload(
            context: *mut c_void,
            data: *const u8,
            byte_count: usize,
            out_timings: *mut CudaUploadTimings,
            error_message: *mut c_char,
            error_capacity: usize,
        ) -> c_int;

        pub(crate) fn gpugeno_cuda_flagstat(
            context: *mut c_void,
            data: *const u8,
            byte_count: usize,
            span_starts: *const u32,
            span_count: usize,
            out_counts: *mut FlagstatCounters,
            out_status: *mut u8,
            out_timings: *mut CudaTimings,
            error_message: *mut c_char,
            error_capacity: usize,
        ) -> c_int;

        pub(crate) fn gpugeno_cuda_destroy(context: *mut c_void);
    }
}
