//! Safe Rust boundary for the statically linked CUDA vector-add spike.
//!
//! The native side is `cuda/vector_add.cu`, compiled by `nvcc` (see
//! `build.rs`) and linked into the executable. It exposes a narrow C ABI;
//! this crate wraps that ABI so callers never touch `unsafe`.

/// Timings reported by a native CUDA vector-add call, in milliseconds.
///
/// Layout matches `struct gpugeno_cuda_timings` in `cuda/vector_add.h`.
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
    //! Raw C ABI declarations for `cuda/vector_add.h`. Keep in sync with that
    //! header: signatures, status codes, and the timings layout.

    use super::CudaTimings;
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

        pub(crate) fn gpugeno_cuda_destroy(context: *mut c_void);
    }
}
