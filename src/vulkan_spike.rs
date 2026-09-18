//! Thin direct-Vulkan flagstat integration spike.
//!
//! This module intentionally owns only the resources needed to execute a small
//! synthetic dispatch. It is not wired into the public backend CLI and is not
//! a production resource-reuse abstraction.

use crate::bam::FlagstatCounters;
use ash::{vk, Entry};
use std::ffi::{CStr, CString};
use std::time::Instant;

const COUNTERS_PER_SPAN: usize = 32;
const STATUS_TRAILING_BLOCK_SIZE: u32 = 1;
const STATUS_SMALL_CORE: u32 = 2;
const STATUS_RECORD_OVERRUN: u32 = 3;

#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub index: u32,
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanTimingSource {
    GpuTimestamp,
    HostSynchronized,
}

impl std::fmt::Display for VulkanTimingSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuTimestamp => formatter.write_str("gpu-timestamp"),
            Self::HostSynchronized => formatter.write_str("host-synchronized"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VulkanTimings {
    pub kernel_ms: f64,
    pub source: VulkanTimingSource,
}

#[derive(Debug)]
pub struct VulkanFlagstatBatch {
    pub span_counts: Vec<FlagstatCounters>,
    pub statuses: Vec<u32>,
    pub timings: VulkanTimings,
}

#[derive(Debug)]
pub struct VulkanError(String);

impl VulkanError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for VulkanError {}

struct InstanceOwner {
    instance: ash::Instance,
    // Keep the loader library alive for every function pointer in `instance`.
    _entry: Entry,
}

impl InstanceOwner {
    fn create() -> Result<Self, VulkanError> {
        let entry = unsafe { Entry::load() }.map_err(|error| {
            VulkanError::new(format!("loading the Vulkan loader failed: {error}"))
        })?;
        let application_name = CString::new("gpugeno-vulkan-spike").unwrap();
        let application = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(1)
            .engine_name(&application_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|error| {
            VulkanError::new(format!(
                "vkCreateInstance for the Vulkan spike failed: {error:?}"
            ))
        })?;
        Ok(Self {
            instance,
            _entry: entry,
        })
    }

    fn physical_devices(&self) -> Result<Vec<vk::PhysicalDevice>, VulkanError> {
        unsafe { self.instance.enumerate_physical_devices() }.map_err(|error| {
            VulkanError::new(format!("vkEnumeratePhysicalDevices failed: {error:?}"))
        })
    }

    fn device_info(&self, index: usize, device: vk::PhysicalDevice) -> VulkanDeviceInfo {
        let properties = unsafe { self.instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        VulkanDeviceInfo {
            index: index as u32,
            name,
            device_type: properties.device_type,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            api_version: properties.api_version,
        }
    }
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

struct DeviceResources {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    query_pool: vk::QueryPool,
}

impl DeviceResources {
    fn new(device: ash::Device, queue: vk::Queue) -> Self {
        Self {
            device,
            queue,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            query_pool: vk::QueryPool::null(),
        }
    }
}

impl Drop for DeviceResources {
    fn drop(&mut self) {
        unsafe {
            // Every normal dispatch waits its fence. This best-effort wait also
            // makes early returns and future edits conservative at destruction.
            let _ = self.device.device_wait_idle();
            if self.query_pool != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.query_pool, None);
            }
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
        }
    }
}

struct OwnedBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    allocation_size: vk::DeviceSize,
    coherent: bool,
}

impl OwnedBuffer {
    fn write(&self, contents: &[u8], label: &str) -> Result<(), VulkanError> {
        if contents.len() as u64 > self.size {
            return Err(VulkanError::new(format!(
                "internal {label} write of {} bytes exceeds {}-byte buffer",
                contents.len(),
                self.size
            )));
        }
        let pointer = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.allocation_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|error| VulkanError::new(format!("vkMapMemory for {label} failed: {error:?}")))?;
        unsafe {
            std::ptr::copy_nonoverlapping(contents.as_ptr(), pointer.cast(), contents.len());
        }
        let flush_result = if self.coherent {
            Ok(())
        } else {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe { self.device.flush_mapped_memory_ranges(&[range]) }.map_err(|error| {
                VulkanError::new(format!(
                    "vkFlushMappedMemoryRanges for {label} failed: {error:?}"
                ))
            })
        };
        unsafe { self.device.unmap_memory(self.memory) };
        flush_result
    }

    fn read(&self, byte_count: usize, label: &str) -> Result<Vec<u8>, VulkanError> {
        if byte_count as u64 > self.size {
            return Err(VulkanError::new(format!(
                "internal {label} read of {byte_count} bytes exceeds {}-byte buffer",
                self.size
            )));
        }
        let pointer = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.allocation_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|error| VulkanError::new(format!("vkMapMemory for {label} failed: {error:?}")))?;
        let invalidate_result = if self.coherent {
            Ok(())
        } else {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) }.map_err(|error| {
                VulkanError::new(format!(
                    "vkInvalidateMappedMemoryRanges for {label} failed: {error:?}"
                ))
            })
        };
        let bytes = if invalidate_result.is_ok() {
            unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), byte_count) }.to_vec()
        } else {
            Vec::new()
        };
        unsafe { self.device.unmap_memory(self.memory) };
        invalidate_result.map(|()| bytes)
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct BatchResources {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    buffers: Vec<OwnedBuffer>,
}

impl BatchResources {
    fn new(device: &ash::Device) -> Self {
        Self {
            device: device.clone(),
            descriptor_pool: vk::DescriptorPool::null(),
            buffers: Vec::with_capacity(5),
        }
    }
}

impl Drop for BatchResources {
    fn drop(&mut self) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None)
            };
        }
        // Buffers are dropped after this method, while the logical device is
        // still alive in the parent context.
    }
}

pub struct VulkanContext {
    // Field order is intentional: device resources must drop before instance.
    resources: DeviceResources,
    _instance: InstanceOwner,
    _physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: vk::PhysicalDeviceProperties,
    _queue_family_index: u32,
    timestamp_valid_bits: u32,
    info: VulkanDeviceInfo,
}

impl VulkanContext {
    pub fn enumerate_devices() -> Result<Vec<VulkanDeviceInfo>, VulkanError> {
        let instance = InstanceOwner::create()?;
        Ok(instance
            .physical_devices()?
            .into_iter()
            .enumerate()
            .map(|(index, device)| instance.device_info(index, device))
            .collect())
    }

    /// Selects exactly the physical device at `device_index` in Vulkan's
    /// enumeration. CPU/software physical devices are rejected.
    pub fn create(device_index: u32) -> Result<Self, VulkanError> {
        let instance = InstanceOwner::create()?;
        let physical_devices = instance.physical_devices()?;
        let available = physical_devices
            .iter()
            .enumerate()
            .map(|(index, &device)| {
                let info = instance.device_info(index, device);
                format!("{index}: {} ({:?})", info.name, info.device_type)
            })
            .collect::<Vec<_>>()
            .join(", ");
        let physical_device = *physical_devices
            .get(device_index as usize)
            .ok_or_else(|| {
                VulkanError::new(format!(
                    "Vulkan physical device {device_index} is unavailable; enumerated devices: [{available}]"
                ))
            })?;
        let info = instance.device_info(device_index as usize, physical_device);
        if is_software_device(&info) {
            return Err(VulkanError::new(format!(
                "Vulkan physical device {device_index} ({}) is a CPU/software device; refusing GPU fallback",
                info.name
            )));
        }

        let queue_families = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };
        let (queue_family_index, queue_family) = queue_families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .min_by_key(|(_, family)| {
                u8::from(family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
            .ok_or_else(|| {
                VulkanError::new(format!(
                    "Vulkan physical device {device_index} ({}) has no compute queue family",
                    info.name
                ))
            })?;
        let queue_family_index = queue_family_index as u32;
        let timestamp_valid_bits = queue_family.timestamp_valid_bits;
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe {
            instance
                .instance
                .create_device(physical_device, &device_info, None)
        }
        .map_err(|error| {
            VulkanError::new(format!(
                "vkCreateDevice for physical device {device_index} ({}) failed: {error:?}",
                info.name
            ))
        })?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let mut resources = DeviceResources::new(device, queue);

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        resources.command_pool = unsafe {
            resources
                .device
                .create_command_pool(&command_pool_info, None)
        }
        .map_err(|error| VulkanError::new(format!("vkCreateCommandPool failed: {error:?}")))?;
        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(resources.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        resources.command_buffer =
            unsafe { resources.device.allocate_command_buffers(&command_info) }.map_err(
                |error| VulkanError::new(format!("vkAllocateCommandBuffers failed: {error:?}")),
            )?[0];
        resources.fence = unsafe {
            resources
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| VulkanError::new(format!("vkCreateFence failed: {error:?}")))?;

        let properties = unsafe {
            instance
                .instance
                .get_physical_device_properties(physical_device)
        };
        if properties.limits.timestamp_compute_and_graphics != 0 && timestamp_valid_bits != 0 {
            let query_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            resources.query_pool = unsafe { resources.device.create_query_pool(&query_info, None) }
                .map_err(|error| {
                    VulkanError::new(format!(
                        "vkCreateQueryPool for timestamps failed: {error:?}"
                    ))
                })?;
        }

        let bindings = [
            descriptor_binding(0, vk::DescriptorType::STORAGE_BUFFER),
            descriptor_binding(1, vk::DescriptorType::STORAGE_BUFFER),
            descriptor_binding(2, vk::DescriptorType::STORAGE_BUFFER),
            descriptor_binding(3, vk::DescriptorType::STORAGE_BUFFER),
            descriptor_binding(4, vk::DescriptorType::UNIFORM_BUFFER),
        ];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        resources.descriptor_set_layout = unsafe {
            resources
                .device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
        }
        .map_err(|error| {
            VulkanError::new(format!("vkCreateDescriptorSetLayout failed: {error:?}"))
        })?;
        let set_layouts = [resources.descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        resources.pipeline_layout = unsafe {
            resources
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|error| VulkanError::new(format!("vkCreatePipelineLayout failed: {error:?}")))?;

        let shader_words = embedded_shader_words()?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
        let shader = unsafe { resources.device.create_shader_module(&shader_info, None) }
            .map_err(|error| VulkanError::new(format!("vkCreateShaderModule failed: {error:?}")))?;
        let entry_name = CString::new("flagstat").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(resources.pipeline_layout)];
        let pipeline_result = unsafe {
            resources.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &pipeline_info,
                None,
            )
        };
        unsafe { resources.device.destroy_shader_module(shader, None) };
        resources.pipeline = match pipeline_result {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { resources.device.destroy_pipeline(pipeline, None) };
                }
                return Err(VulkanError::new(format!(
                    "vkCreateComputePipelines for flagstat failed: {error:?}"
                )));
            }
        };

        let memory_properties = unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(physical_device)
        };
        Ok(Self {
            resources,
            _instance: instance,
            _physical_device: physical_device,
            memory_properties,
            properties,
            _queue_family_index: queue_family_index,
            timestamp_valid_bits,
            info,
        })
    }

    pub fn device_info(&self) -> &VulkanDeviceInfo {
        &self.info
    }

    pub fn timing_source(&self) -> VulkanTimingSource {
        if self.resources.query_pool == vk::QueryPool::null() {
            VulkanTimingSource::HostSynchronized
        } else {
            VulkanTimingSource::GpuTimestamp
        }
    }

    pub fn flagstat(
        &mut self,
        data: &[u8],
        span_starts: &[u32],
    ) -> Result<VulkanFlagstatBatch, VulkanError> {
        validate_input(data, span_starts)?;

        let padded_data_size = data
            .len()
            .checked_add(3)
            .map(|size| size / 4 * 4)
            .ok_or_else(|| VulkanError::new("Vulkan padded BAM byte size overflow"))?;
        self.check_limits(padded_data_size, span_starts.len())?;
        let result_bytes = span_starts
            .len()
            .checked_mul(COUNTERS_PER_SPAN * 4)
            .ok_or_else(|| VulkanError::new("Vulkan counter buffer size overflow"))?;
        let status_bytes = span_starts
            .len()
            .checked_mul(4)
            .ok_or_else(|| VulkanError::new("Vulkan status buffer size overflow"))?;

        let mut batch = BatchResources::new(&self.resources.device);
        batch.buffers.push(self.create_buffer(
            padded_data_size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "BAM data",
        )?);
        batch.buffers.push(self.create_buffer(
            std::mem::size_of_val(span_starts) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "span starts",
        )?);
        batch.buffers.push(self.create_buffer(
            result_bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "partial counters",
        )?);
        batch.buffers.push(self.create_buffer(
            status_bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "statuses",
        )?);
        batch.buffers.push(self.create_buffer(
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "parameters",
        )?);

        let mut padded_data = vec![0u8; padded_data_size];
        padded_data[..data.len()].copy_from_slice(data);
        batch.buffers[0].write(&padded_data, "BAM data")?;
        batch.buffers[1].write(
            &span_starts
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            "span starts",
        )?;
        let parameters = [data.len() as u32, span_starts.len() as u32, 0, 0];
        batch.buffers[4].write(
            &parameters
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            "parameters",
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        batch.descriptor_pool = unsafe {
            self.resources
                .device
                .create_descriptor_pool(&pool_info, None)
        }
        .map_err(|error| VulkanError::new(format!("vkCreateDescriptorPool failed: {error:?}")))?;
        let set_layouts = [self.resources.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(batch.descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe {
            self.resources
                .device
                .allocate_descriptor_sets(&allocate_info)
        }
        .map_err(|error| VulkanError::new(format!("vkAllocateDescriptorSets failed: {error:?}")))?
            [0];
        let buffer_infos = batch
            .buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size)
            })
            .collect::<Vec<_>>();
        let writes = (0..5)
            .map(|binding| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(if binding == 4 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(std::slice::from_ref(&buffer_infos[binding]))
            })
            .collect::<Vec<_>>();
        unsafe { self.resources.device.update_descriptor_sets(&writes, &[]) };

        let timing = self.dispatch(descriptor_set, span_starts.len() as u32)?;
        let counter_bytes = batch.buffers[2].read(result_bytes, "partial counters")?;
        let status_bytes = batch.buffers[3].read(status_bytes, "statuses")?;
        let counter_words = bytes_to_u32(&counter_bytes);
        let statuses = bytes_to_u32(&status_bytes);
        let (chunks, remainder) = counter_words.as_chunks::<COUNTERS_PER_SPAN>();
        if !remainder.is_empty() || chunks.len() != span_starts.len() {
            return Err(VulkanError::new(
                "internal Vulkan counter readback shape mismatch",
            ));
        }
        let span_counts = chunks
            .iter()
            .map(|values| FlagstatCounters::from_u32_flat(values))
            .collect();
        Ok(VulkanFlagstatBatch {
            span_counts,
            statuses,
            timings: timing,
        })
    }

    fn check_limits(&self, data_bytes: usize, span_count: usize) -> Result<(), VulkanError> {
        let limits = self.properties.limits;
        if limits.max_compute_work_group_size[0] < 128
            || limits.max_compute_work_group_invocations < 128
        {
            return Err(VulkanError::new(format!(
                "selected Vulkan device cannot run the 128-lane shader (size limit {}, invocation limit {})",
                limits.max_compute_work_group_size[0], limits.max_compute_work_group_invocations
            )));
        }
        if span_count as u32 > limits.max_compute_work_group_count[0] {
            return Err(VulkanError::new(format!(
                "Vulkan dispatch needs {span_count} workgroups but device limit is {}",
                limits.max_compute_work_group_count[0]
            )));
        }
        let result_bytes = span_count.saturating_mul(COUNTERS_PER_SPAN * 4);
        let status_bytes = span_count.saturating_mul(4);
        for (label, size) in [
            ("BAM data", data_bytes),
            ("partial counters", result_bytes),
            ("statuses", status_bytes),
        ] {
            if size > limits.max_storage_buffer_range as usize {
                return Err(VulkanError::new(format!(
                    "{label} needs {size} bytes but maxStorageBufferRange is {}",
                    limits.max_storage_buffer_range
                )));
            }
        }
        Ok(())
    }

    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        label: &str,
    ) -> Result<OwnedBuffer, VulkanError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.resources.device.create_buffer(&create_info, None) }.map_err(
            |error| VulkanError::new(format!("vkCreateBuffer for {label} failed: {error:?}")),
        )?;
        let requirements = unsafe { self.resources.device.get_buffer_memory_requirements(buffer) };
        let memory_choice =
            select_host_memory(&self.memory_properties, requirements.memory_type_bits);
        let (memory_type_index, coherent) = match memory_choice {
            Some(choice) => choice,
            None => {
                unsafe { self.resources.device.destroy_buffer(buffer, None) };
                return Err(VulkanError::new(format!(
                    "no HOST_VISIBLE Vulkan memory type is compatible with {label}"
                )));
            }
        };
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe {
            self.resources
                .device
                .allocate_memory(&allocation_info, None)
        } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.resources.device.destroy_buffer(buffer, None) };
                return Err(VulkanError::new(format!(
                    "vkAllocateMemory for {label} ({} bytes) failed: {error:?}",
                    requirements.size
                )));
            }
        };
        if let Err(error) = unsafe { self.resources.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.resources.device.destroy_buffer(buffer, None);
                self.resources.device.free_memory(memory, None);
            }
            return Err(VulkanError::new(format!(
                "vkBindBufferMemory for {label} failed: {error:?}"
            )));
        }
        Ok(OwnedBuffer {
            device: self.resources.device.clone(),
            buffer,
            memory,
            size,
            allocation_size: requirements.size,
            coherent,
        })
    }

    fn dispatch(
        &mut self,
        descriptor_set: vk::DescriptorSet,
        span_count: u32,
    ) -> Result<VulkanTimings, VulkanError> {
        let device = &self.resources.device;
        unsafe {
            device
                .reset_command_pool(
                    self.resources.command_pool,
                    vk::CommandPoolResetFlags::empty(),
                )
                .map_err(|error| {
                    VulkanError::new(format!("vkResetCommandPool failed: {error:?}"))
                })?;
            device
                .reset_fences(&[self.resources.fence])
                .map_err(|error| VulkanError::new(format!("vkResetFences failed: {error:?}")))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.resources.command_buffer, &begin)
                .map_err(|error| {
                    VulkanError::new(format!("vkBeginCommandBuffer failed: {error:?}"))
                })?;
            if self.resources.query_pool != vk::QueryPool::null() {
                device.cmd_reset_query_pool(
                    self.resources.command_buffer,
                    self.resources.query_pool,
                    0,
                    2,
                );
                device.cmd_write_timestamp(
                    self.resources.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.resources.query_pool,
                    0,
                );
            }
            device.cmd_bind_pipeline(
                self.resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.resources.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.resources.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_dispatch(self.resources.command_buffer, span_count, 1, 1);
            if self.resources.query_pool != vk::QueryPool::null() {
                device.cmd_write_timestamp(
                    self.resources.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.resources.query_pool,
                    1,
                );
            }
            device
                .end_command_buffer(self.resources.command_buffer)
                .map_err(|error| {
                    VulkanError::new(format!("vkEndCommandBuffer failed: {error:?}"))
                })?;
        }

        let command_buffers = [self.resources.command_buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let host_start = Instant::now();
        unsafe { device.queue_submit(self.resources.queue, &submit, self.resources.fence) }
            .map_err(|error| VulkanError::new(format!("vkQueueSubmit failed: {error:?}")))?;
        if let Err(error) =
            unsafe { device.wait_for_fences(&[self.resources.fence], true, u64::MAX) }
        {
            // A submitted command must stop using per-call buffers before they
            // drop. Device loss is the only practical case where this may also
            // fail; Vulkan then guarantees no useful work can be recovered.
            let idle_result = unsafe { device.device_wait_idle() };
            return Err(VulkanError::new(format!(
                "vkWaitForFences failed after submission: {error:?}; vkDeviceWaitIdle fallback: {idle_result:?}"
            )));
        }
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;

        if self.resources.query_pool == vk::QueryPool::null() {
            return Ok(VulkanTimings {
                kernel_ms: host_ms,
                source: VulkanTimingSource::HostSynchronized,
            });
        }
        let mut ticks = [0u64; 2];
        unsafe {
            device.get_query_pool_results(
                self.resources.query_pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|error| {
            VulkanError::new(format!(
                "vkGetQueryPoolResults for timestamps failed: {error:?}"
            ))
        })?;
        let valid_mask = if self.timestamp_valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.timestamp_valid_bits) - 1
        };
        let elapsed_ticks = ticks[1].wrapping_sub(ticks[0]) & valid_mask;
        Ok(VulkanTimings {
            kernel_ms: elapsed_ticks as f64 * f64::from(self.properties.limits.timestamp_period)
                / 1_000_000.0,
            source: VulkanTimingSource::GpuTimestamp,
        })
    }
}

fn descriptor_binding(
    binding: u32,
    descriptor_type: vk::DescriptorType,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(descriptor_type)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn embedded_shader_words() -> Result<Vec<u32>, VulkanError> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/vulkan_flagstat.spv"));
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(VulkanError::new(
            "embedded Vulkan shader is not a whole number of SPIR-V words",
        ));
    }
    let words = words
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect::<Vec<_>>();
    if words.first().copied() != Some(0x0723_0203) {
        return Err(VulkanError::new(
            "embedded Vulkan shader has invalid SPIR-V magic",
        ));
    }
    Ok(words)
}

fn select_host_memory(
    properties: &vk::PhysicalDeviceMemoryProperties,
    compatible_bits: u32,
) -> Option<(u32, bool)> {
    let mut visible = None;
    for index in 0..properties.memory_type_count {
        if compatible_bits & (1 << index) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            continue;
        }
        let coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        if coherent {
            return Some((index, true));
        }
        visible = Some((index, false));
    }
    visible
}

fn is_software_device(info: &VulkanDeviceInfo) -> bool {
    if matches!(
        info.device_type,
        vk::PhysicalDeviceType::CPU | vk::PhysicalDeviceType::OTHER
    ) {
        return true;
    }
    let name = info.name.to_ascii_lowercase();
    ["llvmpipe", "lavapipe", "swiftshader", "software rasterizer"]
        .iter()
        .any(|marker| name.contains(marker))
}

fn validate_input(data: &[u8], span_starts: &[u32]) -> Result<(), VulkanError> {
    if data.is_empty() || data.len() > u32::MAX as usize {
        return Err(VulkanError::new(
            "Vulkan flagstat data must fit a nonempty u32 byte range",
        ));
    }
    if span_starts.is_empty()
        || span_starts[0] != 0
        || span_starts.windows(2).any(|pair| pair[0] >= pair[1])
        || *span_starts.last().unwrap() as usize >= data.len()
    {
        return Err(VulkanError::new(
            "Vulkan flagstat span starts must begin at zero, increase strictly, and lie within data",
        ));
    }
    Ok(())
}

fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    words.iter().map(|word| u32::from_ne_bytes(*word)).collect()
}

pub fn shader_status_description(status: u32) -> &'static str {
    match status {
        0 => "success",
        STATUS_TRAILING_BLOCK_SIZE => "trailing bytes shorter than block_size",
        STATUS_SMALL_CORE => "block_size smaller than BAM core",
        STATUS_RECORD_OVERRUN => "record extends beyond span",
        _ => "unknown shader status",
    }
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

    fn representative_data() -> (Vec<u8>, Vec<u32>) {
        let mut data = record(0x100 | 0x800 | 0x400, 60, 0, 0);
        data.extend(record(0x001 | 0x040 | 0x002, 4, 0, 1));
        let second_span = data.len() as u32;
        data.extend(record(0x001 | 0x080, 5, 0, 1));
        data.extend(record(0x200 | 0x004 | 0x400, 0, -1, -1));
        (data, vec![0, second_span])
    }

    #[test]
    fn direct_vulkan_classifier_matches_host_and_reports_malformed_status() {
        let _gpu_test_guard = crate::GPU_TEST_LOCK.lock().unwrap();
        let mut context = VulkanContext::create(0).expect("a hardware Vulkan device is required");
        let (data, spans) = representative_data();
        let gpu = context.flagstat(&data, &spans).unwrap();
        assert_eq!(gpu.statuses, [0, 0]);

        let mut reduced_gpu = FlagstatCounters::default();
        for (index, partial) in gpu.span_counts.iter().enumerate() {
            let begin = spans[index] as usize;
            let end = spans
                .get(index + 1)
                .copied()
                .map(|value| value as usize)
                .unwrap_or(data.len());
            let host = crate::bam::classify_records(&data[begin..end]).unwrap();
            assert_eq!(*partial, host, "per-span counters differ at span {index}");
            reduced_gpu.add_assign(partial);
        }
        assert_eq!(reduced_gpu, crate::bam::classify_records(&data).unwrap());

        let mut malformed = data.clone();
        malformed.pop();
        let rejected = context.flagstat(&malformed, &spans).unwrap();
        assert_eq!(rejected.statuses, [0, STATUS_RECORD_OVERRUN]);
        assert_eq!(
            shader_status_description(rejected.statuses[1]),
            "record extends beyond span"
        );

        // A second valid dispatch verifies command-pool/fence reuse and that
        // malformed-call batch resources were released safely.
        let repeated = context.flagstat(&data, &spans).unwrap();
        assert_eq!(repeated.statuses, [0, 0]);
    }

    #[test]
    fn direct_vulkan_device_selection_fails_without_fallback() {
        let _gpu_test_guard = crate::GPU_TEST_LOCK.lock().unwrap();
        let error = VulkanContext::create(u32::MAX).err().unwrap();
        assert!(error.to_string().contains("is unavailable"));

        let devices = VulkanContext::enumerate_devices().unwrap();
        if let Some(cpu) = devices
            .iter()
            .find(|device| device.device_type == vk::PhysicalDeviceType::CPU)
        {
            let error = VulkanContext::create(cpu.index).err().unwrap();
            assert!(error.to_string().contains("CPU/software device"));
        }
    }
}
