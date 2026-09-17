use gpugeno::bai::{flagstat_anchors, read_bai};
use gpugeno::bam::{
    classify_records, default_bai_path, format_flagstat, read_bam_header, FlagstatCounters,
};
use gpugeno::bgzf::data_end_virtual_offset;
use gpugeno::indexed_batch::DisjointBamStream;
use gpugeno::wgpu_backend::WgpuContext;
use gpugeno::CudaContext;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_BATCH_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_THREADS: usize = 8;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gpugeno: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    input: PathBuf,
    bai: Option<PathBuf>,
    backend: BackendChoice,
    device: i32,
    max_uncompressed_bytes: usize,
    threads: usize,
    benchmark: bool,
    validate: bool,
}

#[derive(Clone, Copy)]
enum BackendChoice {
    Cuda,
    Wgpu,
}

impl BackendChoice {
    fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

enum BackendContext {
    Cuda(CudaContext),
    Wgpu(Box<WgpuContext>),
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let wall_start = Instant::now();
    let bai_path = args.bai.unwrap_or_else(|| default_bai_path(&args.input));

    let metadata_start = Instant::now();
    let header = read_bam_header(&args.input).map_err(|error| error.to_string())?;
    let index = read_bai(&bai_path).map_err(|error| error.to_string())?;
    if index.reference_count != header.reference_count {
        return Err(format!(
            "BAM has {} references but {} has {}",
            header.reference_count,
            bai_path.display(),
            index.reference_count
        ));
    }
    let data_end = data_end_virtual_offset(&args.input).map_err(|error| error.to_string())?;
    let anchors = flagstat_anchors(header.first_record, data_end, &index.linear_work_items)
        .map_err(|error| error.to_string())?;
    let metadata_time = metadata_start.elapsed();

    let mut stream = DisjointBamStream::open(
        &args.input,
        anchors,
        args.max_uncompressed_bytes,
        args.threads,
    )
    .map_err(|error| error.to_string())?;
    let anchor_count = stream.anchor_count();
    let mut context = match args.backend {
        BackendChoice::Cuda => BackendContext::Cuda(
            CudaContext::create(args.device).map_err(|error| error.to_string())?,
        ),
        BackendChoice::Wgpu => {
            let adapter_index = u32::try_from(args.device)
                .map_err(|_| "--device must be nonnegative for wgpu".to_string())?;
            let context = WgpuContext::create(adapter_index).map_err(|error| error.to_string())?;
            let info = context.adapter_info();
            eprintln!(
                "gpugeno wgpu: adapter_index={} name={:?} api={:?} device_type={:?} driver={:?} driver_info={:?} timing_source={}",
                info.index,
                info.name,
                info.api,
                info.device_type,
                info.driver,
                info.driver_info,
                context.timing_source()
            );
            BackendContext::Wgpu(Box::new(context))
        }
    };

    let mut gpu_total = FlagstatCounters::default();
    let mut host_total = FlagstatCounters::default();
    let mut batches = 0u64;
    let mut spans = 0u64;
    let mut blocks = 0u64;
    let mut logical_bytes = 0u64;
    let mut compressed_bytes_read = 0u64;
    let mut batch_build_ms = 0.0f64;
    let mut h2d_ms = 0.0f64;
    let mut kernel_ms = 0.0f64;
    let mut d2h_ms = 0.0f64;
    let mut packing_ms = 0.0f64;
    let mut staging_ms = 0.0f64;
    let mut setup_ms = 0.0f64;
    let mut host_validation_ms = 0.0f64;

    while let Some(batch) = stream.next_batch().map_err(|error| error.to_string())? {
        if args.validate {
            let validation_start = Instant::now();
            let batch_host = classify_records(&batch.data).map_err(|error| {
                format!(
                    "host validation failed for virtual range {}..{}: {error}",
                    batch.virtual_start.raw(),
                    batch.virtual_end.raw()
                )
            })?;
            host_validation_ms += validation_start.elapsed().as_secs_f64() * 1000.0;
            host_total.add_assign(&batch_host);
        }

        match &mut context {
            BackendContext::Cuda(context) => {
                let result =
                    context
                        .flagstat(&batch.data, &batch.span_starts)
                        .map_err(|error| {
                            format!(
                                "CUDA flagstat failed for virtual range {}..{}: {error}",
                                batch.virtual_start.raw(),
                                batch.virtual_end.raw()
                            )
                        })?;
                for partial in &result.span_counts {
                    gpu_total.add_assign(partial);
                }
                h2d_ms += f64::from(result.timings.h2d_ms);
                kernel_ms += f64::from(result.timings.kernel_ms);
                d2h_ms += f64::from(result.timings.d2h_ms);
            }
            BackendContext::Wgpu(context) => {
                let result =
                    context
                        .flagstat(&batch.data, &batch.span_starts)
                        .map_err(|error| {
                            format!(
                                "wgpu flagstat failed for virtual range {}..{}: {error}",
                                batch.virtual_start.raw(),
                                batch.virtual_end.raw()
                            )
                        })?;
                for partial in &result.span_counts {
                    gpu_total.add_assign(partial);
                }
                packing_ms += result.timings.packing_ms;
                staging_ms += result.timings.staging_ms;
                setup_ms += result.timings.setup_ms;
                h2d_ms += result.timings.h2d_ms;
                kernel_ms += result.timings.kernel_ms;
                d2h_ms += result.timings.d2h_ms;
            }
        }

        batches += 1;
        spans += batch.span_count() as u64;
        blocks += batch.blocks_decompressed as u64;
        logical_bytes += batch.data.len() as u64;
        compressed_bytes_read += batch.compressed_bytes_read;
        batch_build_ms += batch.build_time.as_secs_f64() * 1000.0;
    }

    if spans + 1 != anchor_count as u64 {
        return Err(format!(
            "stream emitted {spans} spans for {anchor_count} physical anchors"
        ));
    }
    if args.validate && gpu_total != host_total {
        return Err(format!(
            "{} flagstat counters disagree with the host oracle\nGPU: {gpu_total:#?}\nhost: {host_total:#?}",
            args.backend.name()
        ));
    }

    print!("{}", format_flagstat(&gpu_total));
    if args.benchmark {
        eprintln!(
            "gpugeno benchmark: backend={} device={} threads={} batches={} spans={} anchors={} blocks_decompressed={} logical_bytes={} compressed_bytes_read={}",
            args.backend.name(),
            args.device,
            args.threads,
            batches,
            spans,
            anchor_count,
            blocks,
            logical_bytes,
            compressed_bytes_read
        );
        eprintln!(
            "gpugeno benchmark: metadata={:.3} ms batch_build={batch_build_ms:.3} ms H2D={h2d_ms:.3} ms kernel={kernel_ms:.3} ms D2H={d2h_ms:.3} ms GPU_stage={:.3} ms wall={:.3} ms",
            metadata_time.as_secs_f64() * 1000.0,
            h2d_ms + kernel_ms + d2h_ms,
            wall_start.elapsed().as_secs_f64() * 1000.0
        );
        if matches!(args.backend, BackendChoice::Wgpu) {
            eprintln!(
                "gpugeno benchmark: wgpu_host_packing={packing_ms:.3} ms wgpu_staging_write={staging_ms:.3} ms wgpu_resource_setup={setup_ms:.3} ms wgpu_upload_mode=raw-little-endian"
            );
        }
        if args.validate {
            eprintln!(
                "gpugeno benchmark: host_validation={host_validation_ms:.3} ms result=exact-match"
            );
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    if raw.next().as_deref() != Some("flagstat") {
        return Err(usage());
    }

    let mut input = None;
    let mut bai = None;
    let mut device = None;
    let mut max_uncompressed_bytes = None;
    let mut threads = None;
    let mut backend = None;
    let mut benchmark = false;
    let mut validate = false;

    while let Some(argument) = raw.next() {
        match argument.as_str() {
            "--backend" => {
                set_once(&mut backend, raw.next(), "--backend")?;
            }
            "--device" => {
                let value = required_value(&mut raw, "--device")?;
                if device.is_some() {
                    return Err("--device may be specified only once".to_string());
                }
                device = Some(
                    value
                        .parse::<i32>()
                        .map_err(|_| format!("--device is not an integer: {value}"))?,
                );
            }
            "--bai" => {
                let value = required_value(&mut raw, "--bai")?;
                if bai.replace(PathBuf::from(value)).is_some() {
                    return Err("--bai may be specified only once".to_string());
                }
            }
            "--max-uncompressed-bytes" => {
                let value = required_value(&mut raw, "--max-uncompressed-bytes")?;
                if max_uncompressed_bytes.is_some() {
                    return Err("--max-uncompressed-bytes may be specified only once".to_string());
                }
                let parsed = value.parse::<usize>().map_err(|_| {
                    format!("--max-uncompressed-bytes is not an unsigned integer: {value}")
                })?;
                if parsed == 0 || parsed > u32::MAX as usize {
                    return Err(format!(
                        "--max-uncompressed-bytes must be between 1 and {}",
                        u32::MAX
                    ));
                }
                max_uncompressed_bytes = Some(parsed);
            }
            "--threads" => {
                let value = required_value(&mut raw, "--threads")?;
                if threads.is_some() {
                    return Err("--threads may be specified only once".to_string());
                }
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("--threads is not a positive integer: {value}"))?;
                if parsed == 0 {
                    return Err("--threads must be greater than zero".to_string());
                }
                threads = Some(parsed);
            }
            "--benchmark" => benchmark = true,
            "--validate" => validate = true,
            option if option.starts_with('-') => {
                return Err(format!("unsupported option {option}\n{}", usage()));
            }
            path => {
                if input.replace(PathBuf::from(path)).is_some() {
                    return Err(format!("unexpected second input path {path}"));
                }
            }
        }
    }

    let backend =
        match backend.as_deref().unwrap_or("wgpu") {
            "cuda" => BackendChoice::Cuda,
            "wgpu" => BackendChoice::Wgpu,
            "vulkan" => return Err(
                "backend \"vulkan\" is unavailable; direct Vulkan is not implemented (no fallback)"
                    .to_string(),
            ),
            value => {
                return Err(format!(
                    "unsupported backend {value:?}; expected cuda, vulkan, or wgpu"
                ));
            }
        };

    Ok(Args {
        input: input.ok_or_else(usage)?,
        bai,
        backend,
        device: device.unwrap_or(0),
        max_uncompressed_bytes: max_uncompressed_bytes.unwrap_or(DEFAULT_BATCH_BYTES),
        threads: threads.unwrap_or(DEFAULT_THREADS),
        benchmark,
        validate,
    })
}

fn required_value(raw: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    raw.next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn set_once(
    destination: &mut Option<String>,
    value: Option<String>,
    option: &str,
) -> Result<(), String> {
    if destination.is_some() {
        return Err(format!("{option} may be specified only once"));
    }
    *destination = Some(value.ok_or_else(|| format!("{option} requires a value"))?);
    Ok(())
}

fn usage() -> String {
    "usage: gpugeno flagstat INPUT.bam [--backend cuda|wgpu] [--device N] [--bai INPUT.bam.bai] [--max-uncompressed-bytes N] [--threads N] [--benchmark] [--validate]".to_string()
}
