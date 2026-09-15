//! Temporary Rust/CUDA integration spike.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example cuda_vector_add -- --device 0 --elements 16777216
//! ```
//!
//! The example builds two deterministic Rust-owned `f32` vectors, adds them
//! on the selected CUDA device through the statically linked C ABI, reads
//! the complete result back, validates every element, and prints per-stage
//! timings. It exits nonzero with a message on any failure.

use gpugeno::CudaContext;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("cuda_vector_add failed: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    device: i32,
    elements: usize,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let a = (0..args.elements).map(value_a).collect::<Vec<f32>>();
    let b = (0..args.elements).map(value_b).collect::<Vec<f32>>();
    let mut output = vec![0.0f32; args.elements];

    let context = CudaContext::create(args.device).map_err(|error| error.to_string())?;
    let timings = context
        .vector_add(&a, &b, &mut output)
        .map_err(|error| error.to_string())?;
    validate(&a, &b, &output)?;

    println!(
        "cuda_vector_add: OK device={} elements={} ({} MiB per vector), all elements validated exactly",
        args.device,
        args.elements,
        args.elements * std::mem::size_of::<f32>() / (1024 * 1024),
    );
    println!(
        "cuda_vector_add: H2D={:.3} ms  kernel={:.3} ms  D2H={:.3} ms",
        timings.h2d_ms, timings.kernel_ms, timings.d2h_ms
    );
    Ok(())
}

/// Deterministic value at `index` in vector `a`: a small nonnegative integer.
///
/// All values and their sums stay far below 2^24, so they are exact in `f32`
/// and validation can use strict floating-point equality.
fn value_a(index: usize) -> f32 {
    (index % 1000) as f32
}

/// Deterministic value at `index` in vector `b`: a small nonnegative integer.
fn value_b(index: usize) -> f32 {
    ((index / 1000) % 1000) as f32
}

/// Checks every element of `output` against the exactly computable sum.
fn validate(a: &[f32], b: &[f32], output: &[f32]) -> Result<(), String> {
    for (index, &actual) in output.iter().enumerate() {
        let expected = a[index] + b[index];
        if actual != expected {
            return Err(format!(
                "element {index} mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

/// Parses only `--device N` and `--elements N`.
///
/// `--device` defaults to 0, matching the project's initial device default;
/// `--elements` is required because a sizeless run would prove nothing.
fn parse_args() -> Result<Args, String> {
    let mut device: Option<i32> = None;
    let mut elements: Option<usize> = None;
    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        let value = raw
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--device" => {
                device = Some(
                    value
                        .parse::<i32>()
                        .map_err(|_| format!("--device: not an integer: {value}"))?,
                );
            }
            "--elements" => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("--elements: not an unsigned integer: {value}"))?;
                if parsed == 0 {
                    return Err("--elements must be at least 1".to_string());
                }
                if parsed > usize::MAX / std::mem::size_of::<f32>() {
                    return Err(format!("--elements is too large: {parsed}"));
                }
                elements = Some(parsed);
            }
            other => {
                return Err(format!(
                    "unsupported argument {other:?}; this spike only accepts --device N and --elements N"
                ));
            }
        }
    }
    Ok(Args {
        device: device.unwrap_or(0),
        elements: elements.ok_or("--elements N is required")?,
    })
}
