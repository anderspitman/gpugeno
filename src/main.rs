mod batch_producer;

use batch_producer::{BatchProducer, ProducerExitKind, ProducerMessage};
use gpugeno::bai::{flagstat_anchors, read_bai};
use gpugeno::bam::{
    classify_records, default_bai_path, format_flagstat, read_bam_header, FlagstatCounters,
};
use gpugeno::bgzf::data_end_virtual_offset;
use gpugeno::indexed_batch::{DisjointBamStream, IndexedBamBatch};
use gpugeno::vulkan_backend::VulkanContext;
use gpugeno::wgpu_backend::WgpuContext;
use gpugeno::CudaContext;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

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
    device: u64,
    max_uncompressed_bytes: usize,
    threads: usize,
    benchmark: bool,
    validate: bool,
}

#[derive(Clone, Copy)]
enum BackendChoice {
    Cuda,
    Vulkan,
    Wgpu,
}

impl BackendChoice {
    fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Wgpu => "wgpu",
        }
    }
}

enum BackendContext {
    Cuda(CudaContext),
    Vulkan(Box<VulkanContext>),
    Wgpu(Box<WgpuContext>),
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let wall_start = Instant::now();
    let bai_path = args
        .bai
        .clone()
        .unwrap_or_else(|| default_bai_path(&args.input));

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

    let stream = DisjointBamStream::open(
        &args.input,
        anchors,
        args.max_uncompressed_bytes,
        args.threads,
    )
    .map_err(|error| error.to_string())?;
    let anchor_count = stream.anchor_count();

    // The producer starts before backend construction.  The zero-capacity
    // rendezvous means it can finish only the first batch during construction;
    // it cannot create a second complete batch until the consumer receives the
    // first one.
    let producer = BatchProducer::spawn(stream)
        .map_err(|error| format!("failed to start batch producer: {error}"))?;
    let mut context = match create_backend(args.backend, args.device) {
        Ok(context) => context,
        Err(error) => return Err(batch_producer::cancel_with_root(producer, error)),
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
    let mut vulkan_staging_ms = 0.0f64;
    let mut vulkan_setup_ms = 0.0f64;
    let mut host_validation_ms = 0.0f64;
    let mut consumer_first_batch_wait = Duration::ZERO;
    let mut consumer_next_batch_wait = Duration::ZERO;
    let mut consumer_eof_wait = Duration::ZERO;

    let producer_stats = loop {
        let receive_start = Instant::now();
        let message = match producer.recv() {
            Ok(message) => message,
            Err(_) => return Err(producer_disconnect_error(producer)),
        };
        let receive_wait = receive_start.elapsed();

        match message {
            ProducerMessage::Batch(batch) => {
                if batches == 0 {
                    consumer_first_batch_wait += receive_wait;
                } else {
                    consumer_next_batch_wait += receive_wait;
                }
                let processed = match process_batch(&args, &mut context, batch) {
                    Ok(processed) => processed,
                    Err(error) => {
                        return Err(batch_producer::cancel_with_root(producer, error));
                    }
                };
                if let Some(host_counts) = &processed.host_counts {
                    host_total.add_assign(host_counts);
                }
                gpu_total.add_assign(&processed.gpu_counts);
                batches += 1;
                spans += processed.spans;
                blocks += processed.blocks;
                logical_bytes += processed.logical_bytes;
                compressed_bytes_read += processed.compressed_bytes_read;
                batch_build_ms += processed.batch_build_ms;
                h2d_ms += processed.h2d_ms;
                kernel_ms += processed.kernel_ms;
                d2h_ms += processed.d2h_ms;
                packing_ms += processed.packing_ms;
                staging_ms += processed.staging_ms;
                setup_ms += processed.setup_ms;
                vulkan_staging_ms += processed.vulkan_staging_ms;
                vulkan_setup_ms += processed.vulkan_setup_ms;
                host_validation_ms += processed.host_validation_ms;
            }
            ProducerMessage::StreamError(error) => {
                return Err(stream_error_after_join(producer, error.to_string()));
            }
            ProducerMessage::Eof => {
                consumer_eof_wait += receive_wait;
                let exit = match producer.disconnect_and_join() {
                    Ok(exit) => exit,
                    Err(panic) => return Err(panic.to_string()),
                };
                if exit.kind != ProducerExitKind::Finished {
                    return Err(format!(
                        "batch producer protocol ended with {:?} after explicit EOF",
                        exit.kind
                    ));
                }
                break exit.stats;
            }
        }
    };

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
        eprintln!(
            "gpugeno benchmark: consumer_first_batch_wait={:.3} ms consumer_next_batch_wait={:.3} ms consumer_eof_wait={:.3} ms producer_first_send_wait={:.3} ms producer_backpressure_wait={:.3} ms producer_terminal_send_wait={:.3} ms producer_lifetime={:.3} ms",
            duration_ms(consumer_first_batch_wait),
            duration_ms(consumer_next_batch_wait),
            duration_ms(consumer_eof_wait),
            duration_ms(producer_stats.first_batch_send_wait),
            duration_ms(producer_stats.later_batch_send_wait),
            duration_ms(producer_stats.terminal_send_wait),
            duration_ms(producer_stats.lifetime),
        );
        eprintln!(
            "gpugeno benchmark: timing_relationship=batch_build+host_validation+backend_host_stages+GPU_stages_can_overlap_and_must_not_be_summed_to_infer_wall_time"
        );
        if matches!(args.backend, BackendChoice::Vulkan) {
            eprintln!(
                "gpugeno benchmark: vulkan_host_staging_write={vulkan_staging_ms:.3} ms vulkan_resource_setup={vulkan_setup_ms:.3} ms vulkan_packing=0.000 ms vulkan_upload_mode=raw-little-endian"
            );
        }
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

struct ProcessedBatch {
    gpu_counts: FlagstatCounters,
    host_counts: Option<FlagstatCounters>,
    spans: u64,
    blocks: u64,
    logical_bytes: u64,
    compressed_bytes_read: u64,
    batch_build_ms: f64,
    h2d_ms: f64,
    kernel_ms: f64,
    d2h_ms: f64,
    packing_ms: f64,
    staging_ms: f64,
    setup_ms: f64,
    vulkan_staging_ms: f64,
    vulkan_setup_ms: f64,
    host_validation_ms: f64,
}

fn process_batch(
    args: &Args,
    context: &mut BackendContext,
    batch: IndexedBamBatch,
) -> Result<ProcessedBatch, String> {
    let virtual_start = batch.virtual_start.raw();
    let virtual_end = batch.virtual_end.raw();
    let mut host_counts = None;
    let mut host_validation_ms = 0.0;
    if args.validate {
        let validation_start = Instant::now();
        let batch_host = classify_records(&batch.data).map_err(|error| {
            format!(
                "host validation failed for virtual range {virtual_start}..{virtual_end}: {error}"
            )
        })?;
        host_validation_ms = duration_ms(validation_start.elapsed());
        host_counts = Some(batch_host);
    }

    let mut gpu_counts = FlagstatCounters::default();
    let mut h2d_ms = 0.0;
    let mut kernel_ms = 0.0;
    let mut d2h_ms = 0.0;
    let mut packing_ms = 0.0;
    let mut staging_ms = 0.0;
    let mut setup_ms = 0.0;
    let mut vulkan_staging_ms = 0.0;
    let mut vulkan_setup_ms = 0.0;

    match context {
        BackendContext::Cuda(context) => {
            let result = context
                .flagstat(&batch.data, &batch.span_starts)
                .map_err(|error| {
                    format!(
                    "CUDA flagstat failed for virtual range {virtual_start}..{virtual_end}: {error}"
                )
                })?;
            for partial in &result.span_counts {
                gpu_counts.add_assign(partial);
            }
            h2d_ms += f64::from(result.timings.h2d_ms);
            kernel_ms += f64::from(result.timings.kernel_ms);
            d2h_ms += f64::from(result.timings.d2h_ms);
        }
        BackendContext::Vulkan(context) => {
            let result = context
                .flagstat(&batch.data, &batch.span_starts)
                .map_err(|error| {
                    format!(
                        "Vulkan flagstat failed for virtual range {virtual_start}..{virtual_end}: {error}"
                    )
                })?;
            for partial in &result.span_counts {
                gpu_counts.add_assign(partial);
            }
            vulkan_staging_ms += result.timings.staging_ms;
            vulkan_setup_ms += result.timings.setup_ms;
            h2d_ms += result.timings.h2d_ms;
            kernel_ms += result.timings.kernel_ms;
            d2h_ms += result.timings.d2h_ms;
        }
        BackendContext::Wgpu(context) => {
            let result = context
                .flagstat(&batch.data, &batch.span_starts)
                .map_err(|error| {
                    format!(
                        "wgpu flagstat failed for virtual range {virtual_start}..{virtual_end}: {error}"
                    )
                })?;
            for partial in &result.span_counts {
                gpu_counts.add_assign(partial);
            }
            packing_ms += result.timings.packing_ms;
            staging_ms += result.timings.staging_ms;
            setup_ms += result.timings.setup_ms;
            h2d_ms += result.timings.h2d_ms;
            kernel_ms += result.timings.kernel_ms;
            d2h_ms += result.timings.d2h_ms;
        }
    }

    Ok(ProcessedBatch {
        gpu_counts,
        host_counts,
        spans: batch.span_count() as u64,
        blocks: batch.blocks_decompressed as u64,
        logical_bytes: batch.data.len() as u64,
        compressed_bytes_read: batch.compressed_bytes_read,
        batch_build_ms: duration_ms(batch.build_time),
        h2d_ms,
        kernel_ms,
        d2h_ms,
        packing_ms,
        staging_ms,
        setup_ms,
        vulkan_staging_ms,
        vulkan_setup_ms,
        host_validation_ms,
    })
}

fn create_backend(choice: BackendChoice, device: u64) -> Result<BackendContext, String> {
    match choice {
        BackendChoice::Cuda => Ok(BackendContext::Cuda(
            CudaContext::create(
                i32::try_from(device)
                    .map_err(|_| "--device is too large for CUDA's device index".to_string())?,
            )
            .map_err(|error| error.to_string())?,
        )),
        BackendChoice::Vulkan => {
            let device_index = u32::try_from(device)
                .map_err(|_| "--device must fit a nonnegative Vulkan u32 index".to_string())?;
            let context = VulkanContext::create(device_index).map_err(|error| error.to_string())?;
            let info = context.device_info();
            eprintln!(
                "gpugeno vulkan: physical_device_index={} name={:?} device_type={:?} vendor_id=0x{:04x} device_id=0x{:04x} api_version={}.{}.{} timing_source={}",
                info.index,
                info.name,
                info.device_type,
                info.vendor_id,
                info.device_id,
                info.api_version >> 22,
                (info.api_version >> 12) & 0x3ff,
                info.api_version & 0xfff,
                context.timing_source()
            );
            Ok(BackendContext::Vulkan(Box::new(context)))
        }
        BackendChoice::Wgpu => {
            let adapter_index = u32::try_from(device)
                .map_err(|_| "--device must fit a nonnegative wgpu u32 index".to_string())?;
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
            Ok(BackendContext::Wgpu(Box::new(context)))
        }
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn producer_disconnect_error(producer: BatchProducer) -> String {
    match producer.disconnect_and_join() {
        Ok(exit) => format!(
            "batch producer channel disconnected without EOF or stream error (exit={:?})",
            exit.kind
        ),
        Err(panic) => panic.to_string(),
    }
}

fn stream_error_after_join(producer: BatchProducer, root: String) -> String {
    match producer.disconnect_and_join() {
        Ok(exit) if exit.kind == ProducerExitKind::ErrorReported => root,
        Ok(exit) => format!(
            "{root}; secondary producer protocol error: expected ErrorReported, got {:?}",
            exit.kind
        ),
        Err(panic) => format!("{root}; secondary cleanup error: {panic}"),
    }
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
                        .parse::<u64>()
                        .map_err(|_| format!("--device must be a nonnegative integer: {value}"))?,
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

    let backend = match backend.as_deref().unwrap_or("wgpu") {
        "cuda" => BackendChoice::Cuda,
        "vulkan" => BackendChoice::Vulkan,
        "wgpu" => BackendChoice::Wgpu,
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
    "usage: gpugeno flagstat INPUT.bam [--backend cuda|vulkan|wgpu] [--device N] [--bai INPUT.bam.bai] [--max-uncompressed-bytes N] [--threads N] [--benchmark] [--validate]".to_string()
}
