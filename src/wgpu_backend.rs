//! Native `wgpu` implementation of the flagstat classifier.
//!
//! One WGSL workgroup owns each existing record-aligned span. The shader emits
//! bounded `u32` partial counters, which are widened to the shared host `u64`
//! representation after readback.

use crate::bam::FlagstatCounters;
use std::sync::mpsc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const COUNTERS_PER_SPAN: usize = 32;
const TIMESTAMP_COUNT: u32 = 6;

#[derive(Debug, Clone)]
pub struct WgpuAdapterInfo {
    pub index: u32,
    pub name: String,
    pub api: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuTimingSource {
    GpuTimestamps,
    HostSynchronized,
}

impl std::fmt::Display for WgpuTimingSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuTimestamps => formatter.write_str("gpu-timestamps"),
            Self::HostSynchronized => formatter.write_str("host-synchronized"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WgpuTimings {
    /// CPU time spent packing BAM bytes into portable little-endian u32 words.
    pub packing_ms: f64,
    /// CPU time spent creating and filling mapped staging buffers.
    pub staging_ms: f64,
    /// CPU time spent creating per-batch GPU resources and bindings.
    pub setup_ms: f64,
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    pub source: WgpuTimingSource,
}

#[derive(Debug)]
pub struct WgpuFlagstatBatch {
    pub span_counts: Vec<FlagstatCounters>,
    pub timings: WgpuTimings,
}

#[derive(Debug)]
pub struct WgpuError(String);

impl WgpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for WgpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for WgpuError {}

pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_info: WgpuAdapterInfo,
    limits: wgpu::Limits,
    timestamp_mode: bool,
}

impl WgpuContext {
    /// Selects exactly the enumerated wgpu adapter at `adapter_index`.
    /// Software/CPU adapters are rejected rather than becoming a hidden CPU
    /// fallback for an explicitly requested GPU backend.
    pub fn create(adapter_index: u32) -> Result<Self, WgpuError> {
        pollster::block_on(Self::create_async(adapter_index))
    }

    async fn create_async(adapter_index: u32) -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let available = adapters
            .iter()
            .enumerate()
            .map(|(index, adapter)| {
                let info = adapter.get_info();
                format!(
                    "{index}: {} ({:?}, {:?})",
                    info.name, info.backend, info.device_type
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let adapter = adapters
            .into_iter()
            .nth(adapter_index as usize)
            .ok_or_else(|| {
                WgpuError::new(format!(
                    "wgpu adapter {adapter_index} is unavailable; enumerated adapters: [{}]",
                    available
                ))
            })?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(WgpuError::new(format!(
                "wgpu adapter {adapter_index} ({}) is a CPU/software adapter; refusing GPU-backend fallback",
                info.name
            )));
        }

        let supported_features = adapter.features();
        let timestamp_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamp_mode = supported_features.contains(timestamp_features);
        let required_features = if timestamp_mode {
            timestamp_features
        } else {
            wgpu::Features::empty()
        };
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpugeno wgpu device"),
                required_features,
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|error| WgpuError::new(format!("failed to create wgpu device: {error}")))?;

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpugeno flagstat WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("flagstat.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpugeno flagstat pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("flagstat"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = error_scope.pop().await {
            return Err(WgpuError::new(format!(
                "failed to compile wgpu flagstat pipeline: {error}"
            )));
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_info: WgpuAdapterInfo {
                index: adapter_index,
                name: info.name,
                api: info.backend,
                device_type: info.device_type,
                driver: info.driver,
                driver_info: info.driver_info,
            },
            limits,
            timestamp_mode,
        })
    }

    pub fn adapter_info(&self) -> &WgpuAdapterInfo {
        &self.adapter_info
    }

    pub fn timing_source(&self) -> WgpuTimingSource {
        if self.timestamp_mode {
            WgpuTimingSource::GpuTimestamps
        } else {
            WgpuTimingSource::HostSynchronized
        }
    }

    pub fn flagstat(
        &self,
        data: &[u8],
        span_starts: &[u32],
    ) -> Result<WgpuFlagstatBatch, WgpuError> {
        validate_input(data, span_starts)?;
        let span_count = span_starts.len();
        let result_bytes = span_count
            .checked_mul(COUNTERS_PER_SPAN)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| WgpuError::new("wgpu result buffer size overflow"))?;
        let status_bytes = span_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| WgpuError::new("wgpu status buffer size overflow"))?;
        let packed_bytes = data.len().div_ceil(4) * 4;
        self.check_limits(packed_bytes, result_bytes, status_bytes, span_count)?;

        let packing_start = Instant::now();
        let packed_data = pack_bytes(data);
        let parameters = [data.len() as u32, span_count as u32, 0, 0];
        let packing_ms = elapsed_ms(packing_start);

        let staging_start = Instant::now();
        let data_staging =
            self.staging_buffer("BAM upload staging", bytemuck::cast_slice(&packed_data));
        let spans_staging =
            self.staging_buffer("span upload staging", bytemuck::cast_slice(span_starts));
        let params_staging = self.staging_buffer(
            "parameter upload staging",
            bytemuck::cast_slice(&parameters),
        );
        let staging_ms = elapsed_ms(staging_start);

        let setup_start = Instant::now();
        let data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BAM data"),
            size: packed_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spans_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("span starts"),
            size: std::mem::size_of_val(span_starts) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat parameters"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat partial counters"),
            size: result_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let statuses_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat statuses"),
            size: status_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result_readback =
            self.readback_buffer("flagstat counter readback", result_bytes as u64);
        let status_readback = self.readback_buffer("flagstat status readback", status_bytes as u64);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flagstat bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, &data_buffer),
                binding(1, &spans_buffer),
                binding(2, &results_buffer),
                binding(3, &statuses_buffer),
                binding(4, &params_buffer),
            ],
        });
        let setup_ms = elapsed_ms(setup_start);

        let stage_times = if self.timestamp_mode {
            self.run_timestamped(
                span_count as u32,
                &bind_group,
                &data_staging,
                &spans_staging,
                &params_staging,
                &data_buffer,
                &spans_buffer,
                &params_buffer,
                &results_buffer,
                &statuses_buffer,
                &result_readback,
                &status_readback,
                result_bytes as u64,
                status_bytes as u64,
            )?
        } else {
            self.run_host_timed(
                span_count as u32,
                &bind_group,
                &data_staging,
                &spans_staging,
                &params_staging,
                &data_buffer,
                &spans_buffer,
                &params_buffer,
                &results_buffer,
                &statuses_buffer,
                &result_readback,
                &status_readback,
                result_bytes as u64,
                status_bytes as u64,
            )?
        };

        let result_words = map_u32_buffer(&self.device, &result_readback, result_bytes)?;
        let statuses = map_u32_buffer(&self.device, &status_readback, status_bytes)?;
        if let Some((span, status)) = statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| *status != 0)
        {
            let description = match status {
                1 => "trailing bytes shorter than block_size",
                2 => "block_size smaller than BAM core",
                3 => "record extends beyond span",
                _ => "unknown shader status",
            };
            return Err(WgpuError::new(format!(
                "WGSL flagstat rejected span {span} with status {status} ({description})"
            )));
        }
        let (counter_chunks, counter_remainder) = result_words.as_chunks::<COUNTERS_PER_SPAN>();
        debug_assert!(counter_remainder.is_empty());
        let span_counts = counter_chunks
            .iter()
            .map(|values| FlagstatCounters::from_u32_flat(values))
            .collect();

        Ok(WgpuFlagstatBatch {
            span_counts,
            timings: WgpuTimings {
                packing_ms,
                staging_ms,
                setup_ms,
                h2d_ms: stage_times[0],
                kernel_ms: stage_times[1],
                d2h_ms: stage_times[2],
                source: self.timing_source(),
            },
        })
    }

    fn check_limits(
        &self,
        data_bytes: usize,
        result_bytes: usize,
        status_bytes: usize,
        span_count: usize,
    ) -> Result<(), WgpuError> {
        if span_count > self.limits.max_compute_workgroups_per_dimension as usize {
            return Err(WgpuError::new(format!(
                "batch needs {span_count} workgroups but adapter limit is {}",
                self.limits.max_compute_workgroups_per_dimension
            )));
        }
        let max_binding = self.limits.max_storage_buffer_binding_size as usize;
        let max_buffer = self.limits.max_buffer_size as usize;
        for (name, size) in [
            ("BAM data", data_bytes),
            ("partial counters", result_bytes),
            ("statuses", status_bytes),
        ] {
            if size > max_binding || size > max_buffer {
                return Err(WgpuError::new(format!(
                    "{name} buffer needs {size} bytes but adapter limits are max_storage_buffer_binding_size={} and max_buffer_size={}",
                    self.limits.max_storage_buffer_binding_size, self.limits.max_buffer_size
                )));
            }
        }
        Ok(())
    }

    fn staging_buffer(&self, label: &'static str, contents: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn readback_buffer(&self, label: &'static str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_timestamped(
        &self,
        span_count: u32,
        bind_group: &wgpu::BindGroup,
        data_staging: &wgpu::Buffer,
        spans_staging: &wgpu::Buffer,
        params_staging: &wgpu::Buffer,
        data_buffer: &wgpu::Buffer,
        spans_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        results_buffer: &wgpu::Buffer,
        statuses_buffer: &wgpu::Buffer,
        result_readback: &wgpu::Buffer,
        status_readback: &wgpu::Buffer,
        result_bytes: u64,
        status_bytes: u64,
    ) -> Result<[f64; 3], WgpuError> {
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flagstat stage timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: TIMESTAMP_COUNT,
        });
        let query_resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat timestamp resolve"),
            size: u64::from(TIMESTAMP_COUNT) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback = self.readback_buffer(
            "flagstat timestamp readback",
            u64::from(TIMESTAMP_COUNT) * 8,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamped flagstat commands"),
            });
        encoder.write_timestamp(&query_set, 0);
        encode_uploads(
            &mut encoder,
            data_staging,
            spans_staging,
            params_staging,
            data_buffer,
            spans_buffer,
            params_buffer,
        );
        encoder.write_timestamp(&query_set, 1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flagstat compute"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(span_count, 1, 1);
        }
        encoder.write_timestamp(&query_set, 4);
        encode_readbacks(
            &mut encoder,
            results_buffer,
            statuses_buffer,
            result_readback,
            status_readback,
            result_bytes,
            status_bytes,
        );
        encoder.write_timestamp(&query_set, 5);
        encoder.resolve_query_set(&query_set, 0..TIMESTAMP_COUNT, &query_resolve, 0);
        encoder.copy_buffer_to_buffer(
            &query_resolve,
            0,
            &query_readback,
            0,
            u64::from(TIMESTAMP_COUNT) * 8,
        );
        self.queue.submit([encoder.finish()]);
        let words = map_u64_buffer(&self.device, &query_readback, TIMESTAMP_COUNT as usize * 8)?;
        let period_ns = f64::from(self.queue.get_timestamp_period());
        Ok([
            ticks_ms(words[0], words[1], period_ns),
            ticks_ms(words[2], words[3], period_ns),
            ticks_ms(words[4], words[5], period_ns),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn run_host_timed(
        &self,
        span_count: u32,
        bind_group: &wgpu::BindGroup,
        data_staging: &wgpu::Buffer,
        spans_staging: &wgpu::Buffer,
        params_staging: &wgpu::Buffer,
        data_buffer: &wgpu::Buffer,
        spans_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        results_buffer: &wgpu::Buffer,
        statuses_buffer: &wgpu::Buffer,
        result_readback: &wgpu::Buffer,
        status_readback: &wgpu::Buffer,
        result_bytes: u64,
        status_bytes: u64,
    ) -> Result<[f64; 3], WgpuError> {
        let upload_start = Instant::now();
        let mut upload = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flagstat upload commands"),
            });
        encode_uploads(
            &mut upload,
            data_staging,
            spans_staging,
            params_staging,
            data_buffer,
            spans_buffer,
            params_buffer,
        );
        self.queue.submit([upload.finish()]);
        wait(&self.device)?;
        let h2d_ms = elapsed_ms(upload_start);

        let kernel_start = Instant::now();
        let mut compute = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flagstat compute commands"),
            });
        {
            let mut pass = compute.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flagstat compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(span_count, 1, 1);
        }
        self.queue.submit([compute.finish()]);
        wait(&self.device)?;
        let kernel_ms = elapsed_ms(kernel_start);

        let readback_start = Instant::now();
        let mut readback = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flagstat readback commands"),
            });
        encode_readbacks(
            &mut readback,
            results_buffer,
            statuses_buffer,
            result_readback,
            status_readback,
            result_bytes,
            status_bytes,
        );
        self.queue.submit([readback.finish()]);
        wait(&self.device)?;
        let d2h_ms = elapsed_ms(readback_start);
        Ok([h2d_ms, kernel_ms, d2h_ms])
    }
}

fn validate_input(data: &[u8], span_starts: &[u32]) -> Result<(), WgpuError> {
    if data.is_empty() || data.len() > u32::MAX as usize {
        return Err(WgpuError::new(
            "wgpu flagstat data must fit a nonempty u32 byte range",
        ));
    }
    if span_starts.is_empty()
        || span_starts[0] != 0
        || span_starts.windows(2).any(|pair| pair[0] >= pair[1])
        || *span_starts.last().unwrap() as usize >= data.len()
    {
        return Err(WgpuError::new(
            "wgpu flagstat span starts must begin at zero, increase strictly, and lie within data",
        ));
    }
    Ok(())
}

fn pack_bytes(data: &[u8]) -> Vec<u32> {
    data.chunks(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_uploads(
    encoder: &mut wgpu::CommandEncoder,
    data_staging: &wgpu::Buffer,
    spans_staging: &wgpu::Buffer,
    params_staging: &wgpu::Buffer,
    data_buffer: &wgpu::Buffer,
    spans_buffer: &wgpu::Buffer,
    params_buffer: &wgpu::Buffer,
) {
    encoder.copy_buffer_to_buffer(data_staging, 0, data_buffer, 0, data_buffer.size());
    encoder.copy_buffer_to_buffer(spans_staging, 0, spans_buffer, 0, spans_buffer.size());
    encoder.copy_buffer_to_buffer(params_staging, 0, params_buffer, 0, params_buffer.size());
}

#[allow(clippy::too_many_arguments)]
fn encode_readbacks(
    encoder: &mut wgpu::CommandEncoder,
    results_buffer: &wgpu::Buffer,
    statuses_buffer: &wgpu::Buffer,
    result_readback: &wgpu::Buffer,
    status_readback: &wgpu::Buffer,
    result_bytes: u64,
    status_bytes: u64,
) {
    encoder.copy_buffer_to_buffer(results_buffer, 0, result_readback, 0, result_bytes);
    encoder.copy_buffer_to_buffer(statuses_buffer, 0, status_readback, 0, status_bytes);
}

fn wait(device: &wgpu::Device) -> Result<(), WgpuError> {
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| WgpuError::new(format!("waiting for wgpu submission failed: {error}")))?;
    Ok(())
}

fn map_bytes(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<u8>, WgpuError> {
    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..size as u64)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    wait(device)?;
    receiver
        .recv()
        .map_err(|error| WgpuError::new(format!("wgpu map callback was lost: {error}")))?
        .map_err(|error| WgpuError::new(format!("wgpu readback mapping failed: {error}")))?;
    let view = buffer
        .slice(..size as u64)
        .get_mapped_range()
        .map_err(|error| WgpuError::new(format!("getting wgpu mapped range failed: {error}")))?;
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(bytes)
}

fn map_u32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<u32>, WgpuError> {
    let bytes = map_bytes(device, buffer, size)?;
    let (words, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    Ok(words.iter().map(|word| u32::from_le_bytes(*word)).collect())
}

fn map_u64_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<u64>, WgpuError> {
    let bytes = map_bytes(device, buffer, size)?;
    let (words, remainder) = bytes.as_chunks::<8>();
    debug_assert!(remainder.is_empty());
    Ok(words.iter().map(|word| u64::from_le_bytes(*word)).collect())
}

fn ticks_ms(start: u64, end: u64, period_ns: f64) -> f64 {
    end.saturating_sub(start) as f64 * period_ns / 1_000_000.0
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(flag: u16, mapq: u8, reference_id: i32, next_reference_id: i32) -> Vec<u8> {
        let mut value = vec![0u8; 36];
        value[..4].copy_from_slice(&32u32.to_le_bytes());
        value[4..8].copy_from_slice(&reference_id.to_le_bytes());
        value[13] = mapq;
        value[18..20].copy_from_slice(&flag.to_le_bytes());
        value[24..28].copy_from_slice(&next_reference_id.to_le_bytes());
        value
    }

    #[test]
    fn byte_packing_preserves_little_endian_offsets_and_padding() {
        assert_eq!(pack_bytes(&[1, 2, 3, 4, 5]), [0x0403_0201, 5]);
    }

    #[test]
    fn real_wgsl_classifier_matches_host_on_representative_flags() {
        let context = WgpuContext::create(0).expect("a hardware wgpu adapter is required");
        let mut data = record(0x100 | 0x800 | 0x400, 60, 0, 0);
        data.extend(record(0x001 | 0x040 | 0x002, 4, 0, 1));
        let second_span = data.len() as u32;
        data.extend(record(0x001 | 0x080, 5, 0, 1));
        data.extend(record(0x200 | 0x004 | 0x400, 0, -1, -1));

        let gpu = context.flagstat(&data, &[0, second_span]).unwrap();
        let mut total = FlagstatCounters::default();
        for partial in gpu.span_counts {
            total.add_assign(&partial);
        }
        let host = crate::bam::classify_records(&data).unwrap();
        assert_eq!(total, host);

        data.pop();
        let error = context.flagstat(&data, &[0, second_span]).unwrap_err();
        assert!(error.to_string().contains("record extends beyond span"));
    }
}
