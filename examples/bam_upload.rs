//! Temporary bounded BGZF-to-CUDA upload example.
//!
//! ```text
//! cargo run --release --example bam_upload -- INPUT.bam \
//!   --device 0 --max-uncompressed-bytes 268435456
//! ```
//!
//! This builds one prefix batch with sequential libdeflate calls and performs
//! one synchronized H2D copy. It intentionally launches no GPU kernel and
//! performs no readback.

use gpugeno::bgzf::read_bgzf_prefix;
use gpugeno::CudaContext;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MAX_UNCOMPRESSED_BYTES: usize = 256 * 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("bam_upload failed: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    input: PathBuf,
    device: i32,
    max_uncompressed_bytes: usize,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let build_start = Instant::now();
    let batch = read_bgzf_prefix(&args.input, args.max_uncompressed_bytes)
        .map_err(|error| error.to_string())?;
    let build_time = build_start.elapsed();
    if batch.data.is_empty() {
        return Err("the BGZF prefix produced no bytes to upload".to_string());
    }

    let context = CudaContext::create(args.device).map_err(|error| error.to_string())?;
    let upload = context
        .upload(&batch.data)
        .map_err(|error| error.to_string())?;

    println!("bam_upload: OK device={}", args.device);
    println!(
        "bam_upload: blocks={} compressed_bytes={} uncompressed_bytes={}",
        batch.blocks,
        batch.compressed_bytes,
        batch.data.len()
    );
    println!(
        "bam_upload: batch_build={:.3} ms libdeflate={:.3} ms CUDA_H2D={:.3} ms",
        build_time.as_secs_f64() * 1000.0,
        batch.inflate_time.as_secs_f64() * 1000.0,
        upload.h2d_ms
    );
    Ok(())
}

/// Parses exactly `INPUT.bam [--device N] [--max-uncompressed-bytes N]`.
fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut device = None;
    let mut max_uncompressed_bytes = None;
    let mut raw = std::env::args().skip(1);

    while let Some(argument) = raw.next() {
        match argument.as_str() {
            "--device" => {
                if device.is_some() {
                    return Err("--device may be specified only once".to_string());
                }
                let value = raw
                    .next()
                    .ok_or_else(|| "--device requires a value".to_string())?;
                device = Some(
                    value
                        .parse::<i32>()
                        .map_err(|_| format!("--device: not an integer: {value}"))?,
                );
            }
            "--max-uncompressed-bytes" => {
                if max_uncompressed_bytes.is_some() {
                    return Err("--max-uncompressed-bytes may be specified only once".to_string());
                }
                let value = raw
                    .next()
                    .ok_or_else(|| "--max-uncompressed-bytes requires a value".to_string())?;
                let parsed = value.parse::<usize>().map_err(|_| {
                    format!("--max-uncompressed-bytes: not an unsigned integer: {value}")
                })?;
                if parsed == 0 {
                    return Err("--max-uncompressed-bytes must be at least 1".to_string());
                }
                max_uncompressed_bytes = Some(parsed);
            }
            option if option.starts_with('-') => {
                return Err(format!(
                    "unsupported argument {option:?}; expected INPUT.bam [--device N] [--max-uncompressed-bytes N]"
                ));
            }
            path => {
                if input.is_some() {
                    return Err(format!("unexpected second input path {path:?}"));
                }
                input = Some(PathBuf::from(path));
            }
        }
    }

    Ok(Args {
        input: input.ok_or_else(|| "INPUT.bam is required".to_string())?,
        device: device.unwrap_or(0),
        max_uncompressed_bytes: max_uncompressed_bytes.unwrap_or(DEFAULT_MAX_UNCOMPRESSED_BYTES),
    })
}
