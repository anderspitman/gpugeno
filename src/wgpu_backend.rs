//! Native `wgpu` implementation of the flagstat classifier.
//!
//! One WGSL workgroup owns each existing record-aligned span. The shader emits
//! bounded `u32` partial counters, which are widened to the shared host `u64`
//! representation after readback.

use crate::bam::FlagstatCounters;
use std::sync::mpsc;
use std::time::Instant;

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
    /// Legacy comparison metric for the removed full byte-to-word packing pass.
    /// Direct raw-byte upload leaves this at zero.
    pub packing_ms: f64,
    /// CPU time spent mapping and filling reusable upload buffers.
    pub staging_ms: f64,
    /// CPU time spent checking/growing slot resources and rebuilding bindings.
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
    flagstat: WgpuFlagstatState,
    slot: WgpuBufferSlot,
    adapter_info: WgpuAdapterInfo,
    limits: wgpu::Limits,
    timestamp_mode: bool,
}

/// Operation-neutral BAM bytes and physical work spans. This is deliberately
/// separate from flagstat's parameters and outputs so a later operation can
/// bind the same canonical raw-byte representation without inheriting the
/// flagstat result layout.
#[derive(Default)]
struct WgpuInputSlot {
    data: Option<UploadBuffer>,
    spans: Option<UploadBuffer>,
}

#[derive(Default)]
struct WgpuFlagstatSlot {
    parameters: Option<UploadBuffer>,
    results: Option<ReadbackBuffer>,
    statuses: Option<ReadbackBuffer>,
    timestamps: Option<TimestampBuffers>,
    bind_group: Option<wgpu::BindGroup>,
}

/// The current synchronized resource slot. A call completes all mapping and
/// GPU work before this slot can be reused. Keeping the slot explicit leaves
/// room for a future pool without adding overlap in this optimization.
#[derive(Default)]
struct WgpuBufferSlot {
    input: WgpuInputSlot,
    flagstat: WgpuFlagstatSlot,
}

struct WgpuFlagstatState {
    pipeline: wgpu::ComputePipeline,
}

struct UploadBuffer {
    device: wgpu::Buffer,
    upload: wgpu::Buffer,
    capacity: u64,
}

struct ReadbackBuffer {
    device: wgpu::Buffer,
    readback: wgpu::Buffer,
    capacity: u64,
}

struct TimestampBuffers {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct BatchBufferSizes {
    data: u64,
    spans: u64,
    results: u64,
    statuses: u64,
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
            flagstat: WgpuFlagstatState { pipeline },
            slot: WgpuBufferSlot::default(),
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
        &mut self,
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
        let padded_data_bytes = padded_data_size(data.len())
            .ok_or_else(|| WgpuError::new("wgpu padded data size overflow"))?;
        self.check_limits(padded_data_bytes, result_bytes, status_bytes, span_count)?;
        let sizes = BatchBufferSizes {
            data: padded_data_bytes as u64,
            spans: std::mem::size_of_val(span_starts) as u64,
            results: result_bytes as u64,
            statuses: status_bytes as u64,
        };

        let setup_start = Instant::now();
        self.ensure_slot(sizes)?;
        let setup_ms = elapsed_ms(setup_start);

        // IndexedBamBatch::data remains the canonical byte stream. Copy those
        // bytes directly into the mapped upload buffer and clear only the at
        // most three bytes needed to make the storage copy word-aligned.
        let parameters = [data.len() as u32, span_count as u32, 0, 0];
        let staging_start = Instant::now();
        write_upload_buffer(
            &self.device,
            self.slot.input.data.as_ref().unwrap(),
            data,
            sizes.data,
        )?;
        write_upload_buffer(
            &self.device,
            self.slot.input.spans.as_ref().unwrap(),
            bytemuck::cast_slice(span_starts),
            sizes.spans,
        )?;
        write_upload_buffer(
            &self.device,
            self.slot.flagstat.parameters.as_ref().unwrap(),
            bytemuck::cast_slice(&parameters),
            16,
        )?;
        let staging_ms = elapsed_ms(staging_start);

        let stage_times = if self.timestamp_mode {
            self.run_timestamped(span_count as u32, sizes)?
        } else {
            self.run_host_timed(span_count as u32, sizes)?
        };

        let result_words = map_u32_buffer(
            &self.device,
            &self.slot.flagstat.results.as_ref().unwrap().readback,
            result_bytes,
        )?;
        let statuses = map_u32_buffer(
            &self.device,
            &self.slot.flagstat.statuses.as_ref().unwrap().readback,
            status_bytes,
        )?;
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
                packing_ms: 0.0,
                staging_ms,
                setup_ms,
                h2d_ms: stage_times[0],
                kernel_ms: stage_times[1],
                d2h_ms: stage_times[2],
                source: self.timing_source(),
            },
        })
    }

    fn ensure_slot(&mut self, sizes: BatchBufferSizes) -> Result<(), WgpuError> {
        let storage_limit = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size);
        let data_capacity = growth_capacity(sizes.data, storage_limit);
        let span_capacity = growth_capacity(sizes.spans, storage_limit);
        let result_capacity = growth_capacity(sizes.results, storage_limit);
        let status_capacity = growth_capacity(sizes.statuses, storage_limit);
        let mut bindings_changed = false;
        bindings_changed |= ensure_upload_buffer(
            &self.device,
            &mut self.slot.input.data,
            "BAM data",
            "BAM upload staging",
            data_capacity,
            wgpu::BufferUsages::STORAGE,
        );
        bindings_changed |= ensure_upload_buffer(
            &self.device,
            &mut self.slot.input.spans,
            "span starts",
            "span upload staging",
            span_capacity,
            wgpu::BufferUsages::STORAGE,
        );
        bindings_changed |= ensure_upload_buffer(
            &self.device,
            &mut self.slot.flagstat.parameters,
            "flagstat parameters",
            "parameter upload staging",
            16,
            wgpu::BufferUsages::UNIFORM,
        );
        bindings_changed |= ensure_readback_buffer(
            &self.device,
            &mut self.slot.flagstat.results,
            "flagstat partial counters",
            "flagstat counter readback",
            result_capacity,
        );
        bindings_changed |= ensure_readback_buffer(
            &self.device,
            &mut self.slot.flagstat.statuses,
            "flagstat statuses",
            "flagstat status readback",
            status_capacity,
        );
        if self.timestamp_mode && self.slot.flagstat.timestamps.is_none() {
            self.slot.flagstat.timestamps = Some(create_timestamp_buffers(&self.device));
        }

        if bindings_changed || self.slot.flagstat.bind_group.is_none() {
            let input = &self.slot.input;
            let flagstat = &self.slot.flagstat;
            self.slot.flagstat.bind_group =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("flagstat bind group"),
                    layout: &self.flagstat.pipeline.get_bind_group_layout(0),
                    entries: &[
                        binding(0, &input.data.as_ref().unwrap().device),
                        binding(1, &input.spans.as_ref().unwrap().device),
                        binding(2, &flagstat.results.as_ref().unwrap().device),
                        binding(3, &flagstat.statuses.as_ref().unwrap().device),
                        binding(4, &flagstat.parameters.as_ref().unwrap().device),
                    ],
                }));
        }
        Ok(())
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

    fn run_timestamped(
        &self,
        span_count: u32,
        sizes: BatchBufferSizes,
    ) -> Result<[f64; 3], WgpuError> {
        let timestamps = self.slot.flagstat.timestamps.as_ref().unwrap();
        let bind_group = self.slot.flagstat.bind_group.as_ref().unwrap();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamped flagstat commands"),
            });
        encoder.write_timestamp(&timestamps.query_set, 0);
        encode_uploads(&mut encoder, &self.slot, sizes);
        encoder.write_timestamp(&timestamps.query_set, 1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flagstat compute"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &timestamps.query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
            });
            pass.set_pipeline(&self.flagstat.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(span_count, 1, 1);
        }
        encoder.write_timestamp(&timestamps.query_set, 4);
        encode_readbacks(&mut encoder, &self.slot, sizes);
        encoder.write_timestamp(&timestamps.query_set, 5);
        encoder.resolve_query_set(
            &timestamps.query_set,
            0..TIMESTAMP_COUNT,
            &timestamps.resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &timestamps.resolve,
            0,
            &timestamps.readback,
            0,
            u64::from(TIMESTAMP_COUNT) * 8,
        );
        self.queue.submit([encoder.finish()]);
        let words = map_u64_buffer(
            &self.device,
            &timestamps.readback,
            TIMESTAMP_COUNT as usize * 8,
        )?;
        let period_ns = f64::from(self.queue.get_timestamp_period());
        Ok([
            ticks_ms(words[0], words[1], period_ns),
            ticks_ms(words[2], words[3], period_ns),
            ticks_ms(words[4], words[5], period_ns),
        ])
    }

    fn run_host_timed(
        &self,
        span_count: u32,
        sizes: BatchBufferSizes,
    ) -> Result<[f64; 3], WgpuError> {
        let upload_start = Instant::now();
        let mut upload = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flagstat upload commands"),
            });
        encode_uploads(&mut upload, &self.slot, sizes);
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
            pass.set_pipeline(&self.flagstat.pipeline);
            pass.set_bind_group(0, self.slot.flagstat.bind_group.as_ref().unwrap(), &[]);
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
        encode_readbacks(&mut readback, &self.slot, sizes);
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

fn padded_data_size(byte_count: usize) -> Option<usize> {
    byte_count.checked_add(3).map(|count| count / 4 * 4)
}

fn growth_capacity(required: u64, maximum: u64) -> u64 {
    required
        .checked_next_power_of_two()
        .unwrap_or(required)
        .min(maximum)
}

fn ensure_upload_buffer(
    device: &wgpu::Device,
    current: &mut Option<UploadBuffer>,
    device_label: &'static str,
    upload_label: &'static str,
    required: u64,
    binding_usage: wgpu::BufferUsages,
) -> bool {
    if current
        .as_ref()
        .is_some_and(|buffers| buffers.capacity >= required)
    {
        return false;
    }
    *current = Some(UploadBuffer {
        device: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(device_label),
            size: required,
            usage: binding_usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        upload: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(upload_label),
            size: required,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        capacity: required,
    });
    true
}

fn ensure_readback_buffer(
    device: &wgpu::Device,
    current: &mut Option<ReadbackBuffer>,
    device_label: &'static str,
    readback_label: &'static str,
    required: u64,
) -> bool {
    if current
        .as_ref()
        .is_some_and(|buffers| buffers.capacity >= required)
    {
        return false;
    }
    *current = Some(ReadbackBuffer {
        device: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(device_label),
            size: required,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        readback: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(readback_label),
            size: required,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }),
        capacity: required,
    });
    true
}

fn create_timestamp_buffers(device: &wgpu::Device) -> TimestampBuffers {
    let bytes = u64::from(TIMESTAMP_COUNT) * 8;
    TimestampBuffers {
        query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flagstat stage timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: TIMESTAMP_COUNT,
        }),
        resolve: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        readback: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flagstat timestamp readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }),
    }
}

fn write_upload_buffer(
    device: &wgpu::Device,
    buffers: &UploadBuffer,
    contents: &[u8],
    copy_size: u64,
) -> Result<(), WgpuError> {
    if copy_size > buffers.capacity
        || !copy_size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
        || contents.len() as u64 > copy_size
        || copy_size - contents.len() as u64 > 3
    {
        return Err(WgpuError::new("internal invalid wgpu upload size"));
    }
    let slice = buffers.upload.slice(..copy_size);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Write, move |result| {
        let _ = sender.send(result);
    });
    wait(device)?;
    receiver
        .recv()
        .map_err(|error| WgpuError::new(format!("wgpu map callback was lost: {error}")))?
        .map_err(|error| WgpuError::new(format!("wgpu upload mapping failed: {error}")))?;
    let mut view = match slice.get_mapped_range_mut() {
        Ok(view) => view,
        Err(error) => {
            buffers.upload.unmap();
            return Err(WgpuError::new(format!(
                "getting wgpu upload range failed: {error}"
            )));
        }
    };
    let padding = view.len() - contents.len();
    view.slice(..contents.len()).copy_from_slice(contents);
    if padding != 0 {
        view.slice(contents.len()..)
            .copy_from_slice(&[0u8; 3][..padding]);
    }
    drop(view);
    buffers.upload.unmap();
    Ok(())
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn encode_uploads(
    encoder: &mut wgpu::CommandEncoder,
    slot: &WgpuBufferSlot,
    sizes: BatchBufferSizes,
) {
    let data = slot.input.data.as_ref().unwrap();
    let spans = slot.input.spans.as_ref().unwrap();
    let parameters = slot.flagstat.parameters.as_ref().unwrap();
    encoder.copy_buffer_to_buffer(&data.upload, 0, &data.device, 0, sizes.data);
    encoder.copy_buffer_to_buffer(&spans.upload, 0, &spans.device, 0, sizes.spans);
    encoder.copy_buffer_to_buffer(&parameters.upload, 0, &parameters.device, 0, 16);
}

fn encode_readbacks(
    encoder: &mut wgpu::CommandEncoder,
    slot: &WgpuBufferSlot,
    sizes: BatchBufferSizes,
) {
    let results = slot.flagstat.results.as_ref().unwrap();
    let statuses = slot.flagstat.statuses.as_ref().unwrap();
    encoder.copy_buffer_to_buffer(&results.device, 0, &results.readback, 0, sizes.results);
    encoder.copy_buffer_to_buffer(&statuses.device, 0, &statuses.readback, 0, sizes.statuses);
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
    fn grow_only_capacity_uses_bounded_power_of_two_classes() {
        assert_eq!(growth_capacity(5, 1024), 8);
        assert_eq!(growth_capacity(1024, 1024), 1024);
        assert_eq!(growth_capacity(999, 1000), 1000);
    }

    #[test]
    fn direct_upload_padding_is_at_most_one_partial_word() {
        assert_eq!(padded_data_size(1), Some(4));
        assert_eq!(padded_data_size(4), Some(4));
        assert_eq!(padded_data_size(5), Some(8));
        assert_eq!(padded_data_size(usize::MAX), None);
    }

    #[test]
    fn real_wgsl_classifier_matches_host_on_representative_flags() {
        let mut context = WgpuContext::create(0).expect("a hardware wgpu adapter is required");
        let mut first = record(0x100 | 0x800 | 0x400, 60, 0, 0);
        first[..4].copy_from_slice(&33u32.to_le_bytes());
        first.push(0x7f);
        let mut data = first;
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
