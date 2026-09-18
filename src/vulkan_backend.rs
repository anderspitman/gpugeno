//! Direct Vulkan flagstat backend.
//!
//! The backend owns one synchronized, grow-only Vulkan resource slot.  It
//! consumes the canonical bytes and record-aligned span starts produced by the
//! indexed BAM stream; it does not call `wgpu`, CUDA, or a host classifier.
//! The dedicated WGSL classifier is translated to SPIR-V by `build.rs` and is
//! embedded in this module.

use crate::bam::FlagstatCounters;
use ash::{vk, Entry};
use std::ffi::{CStr, CString};
use std::time::Instant;

const COUNTERS_PER_SPAN: usize = 32;
const PARAMETER_BYTES: u64 = 16;
const TIMESTAMP_COUNT: u32 = 6;
const WORKGROUP_LANES: u32 = 128;

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
    GpuTimestamps,
    HostSynchronized,
}

impl std::fmt::Display for VulkanTimingSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuTimestamps => formatter.write_str("gpu-timestamps"),
            Self::HostSynchronized => formatter.write_str("host-synchronized"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VulkanTimings {
    /// Host time spent mapping and filling the three transfer upload buffers.
    pub staging_ms: f64,
    /// Host time spent checking/growing resources and rebuilding descriptors.
    pub setup_ms: f64,
    /// Copy time from host-visible upload buffers into device-side buffers.
    pub h2d_ms: f64,
    /// Compute dispatch time, excluding the transfer stages.
    pub kernel_ms: f64,
    /// Copy time from device-side result buffers into host-visible readbacks.
    pub d2h_ms: f64,
    pub source: VulkanTimingSource,
}

#[derive(Debug)]
pub struct VulkanFlagstatBatch {
    pub span_counts: Vec<FlagstatCounters>,
    /// Successful calls always contain one zero status per span.  A nonzero
    /// status is converted to `VulkanError` before this result is returned.
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
        let application_name = CString::new("gpugeno-vulkan").expect("static application name");
        let application = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(1)
            .engine_name(&application_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|error| {
            VulkanError::new(format!(
                "vkCreateInstance for the Vulkan backend failed: {error:?}"
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
            index: u32::try_from(index).unwrap_or(u32::MAX),
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
    timestamp_valid_bits: u32,
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
            timestamp_valid_bits: 0,
        }
    }

    fn timestamp_enabled(&self) -> bool {
        self.query_pool != vk::QueryPool::null()
    }
}

impl Drop for DeviceResources {
    fn drop(&mut self) {
        unsafe {
            // All normal calls wait on the fence.  This is conservative for
            // construction failures and for a caller dropping after an error.
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

#[derive(Clone, Copy)]
enum AllocationKind {
    Device,
    Host,
}

struct OwnedBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Actual logical Vulkan buffer size/capacity used for descriptors/copies.
    capacity: vk::DeviceSize,
    /// Allocation size returned by vkGetBufferMemoryRequirements.  This is
    /// deliberately tracked separately from `capacity`.
    allocation_size: vk::DeviceSize,
    host_visible: bool,
    coherent: bool,
}

impl OwnedBuffer {
    fn create(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        allocation_kind: AllocationKind,
        label: &str,
    ) -> Result<Self, VulkanError> {
        if capacity == 0 {
            return Err(VulkanError::new(format!(
                "cannot create zero-sized Vulkan buffer for {label}"
            )));
        }
        let create_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&create_info, None) }.map_err(|error| {
            VulkanError::new(format!("vkCreateBuffer for {label} failed: {error:?}"))
        })?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_choice = match allocation_kind {
            AllocationKind::Device => {
                select_device_memory(memory_properties, requirements.memory_type_bits)
                    .map(|index| (index, false, false))
            }
            AllocationKind::Host => {
                select_host_memory(memory_properties, requirements.memory_type_bits)
                    .map(|(index, coherent)| (index, true, coherent))
            }
        };
        let (memory_type_index, host_visible, coherent) = match memory_choice {
            Some(choice) => choice,
            None => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(VulkanError::new(format!(
                    "no compatible {} Vulkan memory type exists for {label}",
                    match allocation_kind {
                        AllocationKind::Device => "device-local",
                        AllocationKind::Host => "HOST_VISIBLE",
                    }
                )));
            }
        };
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { device.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(VulkanError::new(format!(
                    "vkAllocateMemory for {label} ({} bytes) failed: {error:?}",
                    requirements.size
                )));
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(VulkanError::new(format!(
                "vkBindBufferMemory for {label} failed: {error:?}"
            )));
        }
        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            capacity,
            allocation_size: requirements.size,
            host_visible,
            coherent,
        })
    }

    fn write_bytes(
        &self,
        contents: &[u8],
        write_size: vk::DeviceSize,
        zero_tail: bool,
        label: &str,
    ) -> Result<(), VulkanError> {
        if !self.host_visible {
            return Err(VulkanError::new(format!(
                "internal {label} buffer is not HOST_VISIBLE"
            )));
        }
        let write_size_usize = usize::try_from(write_size)
            .map_err(|_| VulkanError::new(format!("{label} write size does not fit host usize")))?;
        if write_size == 0
            || write_size > self.capacity
            || contents.len() > write_size_usize
            || (!zero_tail && contents.len() != write_size_usize)
            || (zero_tail && write_size_usize - contents.len() > 3)
        {
            return Err(VulkanError::new(format!(
                "internal invalid {label} write of {} bytes into {}-byte buffer",
                contents.len(),
                self.capacity
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
            if !contents.is_empty() {
                std::ptr::copy_nonoverlapping(contents.as_ptr(), pointer.cast(), contents.len());
            }
            if zero_tail {
                std::ptr::write_bytes(
                    pointer.cast::<u8>().add(contents.len()),
                    0,
                    write_size_usize - contents.len(),
                );
            }
        }
        let flush_result = self.flush(label);
        unsafe { self.device.unmap_memory(self.memory) };
        flush_result
    }

    fn write_u32(&self, values: &[u32], label: &str) -> Result<(), VulkanError> {
        if !self.host_visible {
            return Err(VulkanError::new(format!(
                "internal {label} buffer is not HOST_VISIBLE"
            )));
        }
        let byte_count = values
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError::new(format!("{label} byte size overflow")))?;
        if byte_count == 0 || byte_count as u64 > self.capacity {
            return Err(VulkanError::new(format!(
                "internal {label} write of {byte_count} bytes exceeds {}-byte buffer",
                self.capacity
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
            for (index, value) in values.iter().enumerate() {
                let bytes = value.to_le_bytes();
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    pointer.cast::<u8>().add(index * std::mem::size_of::<u32>()),
                    bytes.len(),
                );
            }
        }
        let flush_result = self.flush(label);
        unsafe { self.device.unmap_memory(self.memory) };
        flush_result
    }

    fn read_u32(&self, word_count: usize, label: &str) -> Result<Vec<u32>, VulkanError> {
        if !self.host_visible {
            return Err(VulkanError::new(format!(
                "internal {label} buffer is not HOST_VISIBLE"
            )));
        }
        let byte_count = word_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError::new(format!("{label} byte size overflow")))?;
        if byte_count == 0 || byte_count as u64 > self.capacity {
            return Err(VulkanError::new(format!(
                "internal {label} read of {byte_count} bytes exceeds {}-byte buffer",
                self.capacity
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
        let invalidate_result = self.invalidate(label);
        let words = if invalidate_result.is_ok() {
            let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), byte_count) };
            let (chunks, remainder) = bytes.as_chunks::<4>();
            debug_assert!(remainder.is_empty());
            chunks
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect()
        } else {
            Vec::new()
        };
        unsafe { self.device.unmap_memory(self.memory) };
        invalidate_result.map(|()| words)
    }

    fn flush(&self, label: &str) -> Result<(), VulkanError> {
        if self.coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { self.device.flush_mapped_memory_ranges(&[range]) }.map_err(|error| {
            VulkanError::new(format!(
                "vkFlushMappedMemoryRanges for {label} failed: {error:?}"
            ))
        })
    }

    fn invalidate(&self, label: &str) -> Result<(), VulkanError> {
        if self.coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) }.map_err(|error| {
            VulkanError::new(format!(
                "vkInvalidateMappedMemoryRanges for {label} failed: {error:?}"
            ))
        })
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

struct DeviceHostPair {
    device: OwnedBuffer,
    host: OwnedBuffer,
}

struct ResourceSlot {
    device: ash::Device,
    data: Option<DeviceHostPair>,
    spans: Option<DeviceHostPair>,
    parameters: Option<DeviceHostPair>,
    results: Option<DeviceHostPair>,
    statuses: Option<DeviceHostPair>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl ResourceSlot {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            data: None,
            spans: None,
            parameters: None,
            results: None,
            statuses: None,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
        }
    }

    /// Descriptor sets are children of the pool and refer to device buffers.
    /// Clear them before replacing any bound buffer pair.
    fn clear_descriptor(&mut self) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None)
            };
        }
        self.descriptor_pool = vk::DescriptorPool::null();
        self.descriptor_set = vk::DescriptorSet::null();
    }
}

impl Drop for ResourceSlot {
    fn drop(&mut self) {
        self.clear_descriptor();
        // The five DeviceHostPair fields then drop, destroying buffers before
        // the parent DeviceResources owner destroys the logical device.
    }
}

#[derive(Clone, Copy)]
struct BatchBufferSizes {
    data: vk::DeviceSize,
    spans: vk::DeviceSize,
    parameters: vk::DeviceSize,
    results: vk::DeviceSize,
    statuses: vk::DeviceSize,
}

#[derive(Clone, Copy)]
struct DispatchHandles {
    data_device: vk::Buffer,
    data_capacity: vk::DeviceSize,
    data_upload: vk::Buffer,
    spans_device: vk::Buffer,
    spans_capacity: vk::DeviceSize,
    spans_upload: vk::Buffer,
    parameters_device: vk::Buffer,
    parameters_capacity: vk::DeviceSize,
    parameters_upload: vk::Buffer,
    results_device: vk::Buffer,
    results_capacity: vk::DeviceSize,
    results_readback: vk::Buffer,
    statuses_device: vk::Buffer,
    statuses_capacity: vk::DeviceSize,
    statuses_readback: vk::Buffer,
    descriptor_set: vk::DescriptorSet,
}

/// One synchronized public direct-Vulkan backend context.
///
/// Struct field order is intentional: the slot (including all buffers and
/// descriptor pool) drops before static device resources, and the instance
/// drops after the logical device.
pub struct VulkanContext {
    slot: ResourceSlot,
    resources: DeviceResources,
    _instance: InstanceOwner,
    _physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: vk::PhysicalDeviceProperties,
    _queue_family_index: u32,
    info: VulkanDeviceInfo,
    poisoned: bool,
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
    /// enumeration. CPU/OTHER and recognized software implementations are
    /// rejected rather than becoming a hidden software fallback.
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
        let device_position = usize::try_from(device_index).map_err(|_| {
            VulkanError::new(format!(
                "Vulkan physical device index {device_index} does not fit this host usize"
            ))
        })?;
        let physical_device = *physical_devices.get(device_position).ok_or_else(|| {
            VulkanError::new(format!(
                "Vulkan physical device {device_index} is unavailable; enumerated devices: [{available}]"
            ))
        })?;
        let info = instance.device_info(device_position, physical_device);
        if info.api_version < vk::API_VERSION_1_1 {
            return Err(VulkanError::new(format!(
                "Vulkan physical device {device_index} ({}) exposes API {}.{}.{}; SPIR-V 1.3 backend requires Vulkan 1.1",
                info.name,
                info.api_version >> 22,
                (info.api_version >> 12) & 0x3ff,
                info.api_version & 0xfff,
            )));
        }
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
        let (queue_family_position, queue_family) = queue_families
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
        let queue_family_index = u32::try_from(queue_family_position)
            .map_err(|_| VulkanError::new("Vulkan queue-family index does not fit u32"))?;
        let timestamp_valid_bits = queue_family.timestamp_valid_bits;
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe {
            instance
                .instance
                .create_device(physical_device, &device_create_info, None)
        }
        .map_err(|error| {
            VulkanError::new(format!(
                "vkCreateDevice for physical device {device_index} ({}) failed: {error:?}",
                info.name
            ))
        })?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let mut resources = DeviceResources::new(device, queue);
        resources.timestamp_valid_bits = timestamp_valid_bits;

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
        let command_buffers = unsafe { resources.device.allocate_command_buffers(&command_info) }
            .map_err(|error| {
            VulkanError::new(format!("vkAllocateCommandBuffers failed: {error:?}"))
        })?;
        resources.command_buffer = command_buffers
            .first()
            .copied()
            .ok_or_else(|| VulkanError::new("vkAllocateCommandBuffers returned no buffer"))?;
        resources.fence = unsafe {
            resources.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
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
                .query_count(TIMESTAMP_COUNT);
            resources.query_pool = unsafe { resources.device.create_query_pool(&query_info, None) }
                .map_err(|error| {
                    VulkanError::new(format!(
                        "vkCreateQueryPool for Vulkan stage timestamps failed: {error:?}"
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
        let entry_name = CString::new("flagstat").expect("static shader entry name");
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
            Ok(pipelines) => pipelines
                .into_iter()
                .next()
                .ok_or_else(|| VulkanError::new("vkCreateComputePipelines returned no pipeline"))?,
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
        let slot = ResourceSlot::new(resources.device.clone());
        Ok(Self {
            slot,
            resources,
            _instance: instance,
            _physical_device: physical_device,
            memory_properties,
            properties,
            _queue_family_index: queue_family_index,
            info,
            poisoned: false,
        })
    }

    pub fn device_info(&self) -> &VulkanDeviceInfo {
        &self.info
    }

    pub fn timing_source(&self) -> VulkanTimingSource {
        if self.resources.timestamp_enabled() {
            VulkanTimingSource::GpuTimestamps
        } else {
            VulkanTimingSource::HostSynchronized
        }
    }

    /// Classifies one canonical indexed BAM batch. The call is fully
    /// synchronized before returning, so the single slot can grow or be
    /// reused safely on the next call.
    pub fn flagstat(
        &mut self,
        data: &[u8],
        span_starts: &[u32],
    ) -> Result<VulkanFlagstatBatch, VulkanError> {
        if self.poisoned {
            return Err(VulkanError::new(
                "Vulkan context is unusable after a device synchronization failure",
            ));
        }
        validate_input(data, span_starts)?;
        let span_count = span_starts.len();
        let padded_data_size = padded_data_size(data.len())?;
        let spans_size = span_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError::new("Vulkan span-start buffer size overflow"))?;
        let results_size = span_count
            .checked_mul(COUNTERS_PER_SPAN)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| VulkanError::new("Vulkan counter buffer size overflow"))?;
        let statuses_size = span_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| VulkanError::new("Vulkan status buffer size overflow"))?;
        self.check_limits(
            padded_data_size,
            spans_size,
            results_size,
            statuses_size,
            span_count,
        )?;
        let sizes = BatchBufferSizes {
            data: u64::try_from(padded_data_size)
                .map_err(|_| VulkanError::new("Vulkan BAM size does not fit DeviceSize"))?,
            spans: u64::try_from(spans_size)
                .map_err(|_| VulkanError::new("Vulkan span size does not fit DeviceSize"))?,
            parameters: PARAMETER_BYTES,
            results: u64::try_from(results_size)
                .map_err(|_| VulkanError::new("Vulkan result size does not fit DeviceSize"))?,
            statuses: u64::try_from(statuses_size)
                .map_err(|_| VulkanError::new("Vulkan status size does not fit DeviceSize"))?,
        };

        let setup_start = Instant::now();
        self.ensure_slot(sizes)?;
        let setup_ms = elapsed_ms(setup_start);

        let staging_start = Instant::now();
        {
            let data_pair = self
                .slot
                .data
                .as_ref()
                .ok_or_else(|| VulkanError::new("Vulkan BAM resource slot is missing"))?;
            data_pair
                .host
                .write_bytes(data, sizes.data, true, "BAM upload")?;
            let spans_pair = self
                .slot
                .spans
                .as_ref()
                .ok_or_else(|| VulkanError::new("Vulkan span resource slot is missing"))?;
            spans_pair.host.write_u32(span_starts, "span upload")?;
            let parameters = [
                u32::try_from(data.len())
                    .map_err(|_| VulkanError::new("logical BAM byte count does not fit u32"))?,
                u32::try_from(span_count)
                    .map_err(|_| VulkanError::new("span count does not fit u32"))?,
                0,
                0,
            ];
            let parameter_pair = self
                .slot
                .parameters
                .as_ref()
                .ok_or_else(|| VulkanError::new("Vulkan parameter resource slot is missing"))?;
            parameter_pair
                .host
                .write_u32(&parameters, "parameter upload")?;
        }
        let staging_ms = elapsed_ms(staging_start);

        let handles = self.dispatch_handles()?;
        let stage_times = if self.resources.timestamp_enabled() {
            self.run_timestamped(
                handles,
                sizes,
                u32::try_from(span_count).map_err(|_| {
                    VulkanError::new("Vulkan span count does not fit dispatch count")
                })?,
            )?
        } else {
            self.run_host_timed(
                handles,
                sizes,
                u32::try_from(span_count).map_err(|_| {
                    VulkanError::new("Vulkan span count does not fit dispatch count")
                })?,
            )?
        };

        let status_words = self
            .slot
            .statuses
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan status resource slot is missing"))?
            .host
            .read_u32(span_count, "status readback")?;
        if let Some((span, status)) = status_words
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| *status != 0)
        {
            return Err(VulkanError::new(format!(
                "Vulkan shader rejected span {span} with status {status} ({})",
                shader_status_description(status)
            )));
        }

        let result_words = self
            .slot
            .results
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan result resource slot is missing"))?
            .host
            .read_u32(span_count * COUNTERS_PER_SPAN, "counter readback")?;
        let (chunks, remainder) = result_words.as_chunks::<COUNTERS_PER_SPAN>();
        if !remainder.is_empty() || chunks.len() != span_count {
            return Err(VulkanError::new(
                "internal Vulkan counter readback shape mismatch",
            ));
        }
        let span_counts = chunks
            .iter()
            .map(|values| FlagstatCounters::from_u32_flat(values))
            .collect::<Vec<_>>();
        Ok(VulkanFlagstatBatch {
            span_counts,
            statuses: status_words,
            timings: VulkanTimings {
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
        spans_bytes: usize,
        results_bytes: usize,
        statuses_bytes: usize,
        span_count: usize,
    ) -> Result<(), VulkanError> {
        let limits = self.properties.limits;
        if limits.max_compute_work_group_size[0] < WORKGROUP_LANES
            || limits.max_compute_work_group_invocations < WORKGROUP_LANES
        {
            return Err(VulkanError::new(format!(
                "selected Vulkan device cannot run the 128-lane shader (size limit {}, invocation limit {})",
                limits.max_compute_work_group_size[0], limits.max_compute_work_group_invocations
            )));
        }
        let span_count_u32 = u32::try_from(span_count)
            .map_err(|_| VulkanError::new("Vulkan span count does not fit u32"))?;
        if span_count_u32 > limits.max_compute_work_group_count[0] {
            return Err(VulkanError::new(format!(
                "Vulkan dispatch needs {span_count} workgroups but device limit is {}",
                limits.max_compute_work_group_count[0]
            )));
        }
        if limits.max_uniform_buffer_range < PARAMETER_BYTES as u32 {
            return Err(VulkanError::new(format!(
                "Vulkan maxUniformBufferRange {} is smaller than the 16-byte parameter block",
                limits.max_uniform_buffer_range
            )));
        }
        let max_storage = u64::from(limits.max_storage_buffer_range);
        for (label, size) in [
            ("BAM data", data_bytes),
            ("span starts", spans_bytes),
            ("partial counters", results_bytes),
            ("statuses", statuses_bytes),
        ] {
            let size = u64::try_from(size)
                .map_err(|_| VulkanError::new(format!("{label} size does not fit DeviceSize")))?;
            if size == 0 || size > max_storage {
                return Err(VulkanError::new(format!(
                    "{label} needs {size} bytes but maxStorageBufferRange is {}",
                    limits.max_storage_buffer_range
                )));
            }
        }
        Ok(())
    }

    fn ensure_slot(&mut self, sizes: BatchBufferSizes) -> Result<(), VulkanError> {
        let storage_limit = u64::from(self.properties.limits.max_storage_buffer_range);
        let data_capacity = capacity_class(sizes.data, storage_limit).ok_or_else(|| {
            VulkanError::new(format!(
                "BAM resource size {} cannot fit maxStorageBufferRange {}",
                sizes.data, storage_limit
            ))
        })?;
        let spans_capacity = capacity_class(sizes.spans, storage_limit).ok_or_else(|| {
            VulkanError::new(format!(
                "span resource size {} cannot fit maxStorageBufferRange {}",
                sizes.spans, storage_limit
            ))
        })?;
        let results_capacity = capacity_class(sizes.results, storage_limit).ok_or_else(|| {
            VulkanError::new(format!(
                "counter resource size {} cannot fit maxStorageBufferRange {}",
                sizes.results, storage_limit
            ))
        })?;
        let statuses_capacity = capacity_class(sizes.statuses, storage_limit).ok_or_else(|| {
            VulkanError::new(format!(
                "status resource size {} cannot fit maxStorageBufferRange {}",
                sizes.statuses, storage_limit
            ))
        })?;
        let parameters_missing = self.slot.parameters.is_none();
        let changes = parameters_missing
            || pair_needs_growth(&self.slot.data, sizes.data)
            || pair_needs_growth(&self.slot.spans, sizes.spans)
            || pair_needs_growth(&self.slot.results, sizes.results)
            || pair_needs_growth(&self.slot.statuses, sizes.statuses);
        if changes {
            self.slot.clear_descriptor();
        }

        let device = &self.resources.device;
        let memory_properties = &self.memory_properties;
        if pair_needs_capacity(&self.slot.data, data_capacity) {
            ensure_pair(
                device,
                memory_properties,
                &mut self.slot.data,
                PairSpec {
                    capacity: data_capacity,
                    device_usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    host_usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    device_label: "BAM device",
                    host_label: "BAM upload",
                },
            )?;
        }
        if pair_needs_capacity(&self.slot.spans, spans_capacity) {
            ensure_pair(
                device,
                memory_properties,
                &mut self.slot.spans,
                PairSpec {
                    capacity: spans_capacity,
                    device_usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    host_usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    device_label: "span device",
                    host_label: "span upload",
                },
            )?;
        }
        if self.slot.parameters.is_none() {
            ensure_pair(
                device,
                memory_properties,
                &mut self.slot.parameters,
                PairSpec {
                    capacity: PARAMETER_BYTES,
                    device_usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    host_usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    device_label: "parameter device",
                    host_label: "parameter upload",
                },
            )?;
        }
        if pair_needs_capacity(&self.slot.results, results_capacity) {
            ensure_pair(
                device,
                memory_properties,
                &mut self.slot.results,
                PairSpec {
                    capacity: results_capacity,
                    device_usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    host_usage: vk::BufferUsageFlags::TRANSFER_DST,
                    device_label: "counter device",
                    host_label: "counter readback",
                },
            )?;
        }
        if pair_needs_capacity(&self.slot.statuses, statuses_capacity) {
            ensure_pair(
                device,
                memory_properties,
                &mut self.slot.statuses,
                PairSpec {
                    capacity: statuses_capacity,
                    device_usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    host_usage: vk::BufferUsageFlags::TRANSFER_DST,
                    device_label: "status device",
                    host_label: "status readback",
                },
            )?;
        }
        if self.slot.descriptor_set == vk::DescriptorSet::null() {
            self.rebuild_descriptor_set()?;
        }
        Ok(())
    }

    fn rebuild_descriptor_set(&mut self) -> Result<(), VulkanError> {
        self.slot.clear_descriptor();
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
        let pool = unsafe {
            self.resources
                .device
                .create_descriptor_pool(&pool_info, None)
        }
        .map_err(|error| VulkanError::new(format!("vkCreateDescriptorPool failed: {error:?}")))?;
        self.slot.descriptor_pool = pool;
        let set_layouts = [self.resources.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        let descriptor_set = match unsafe {
            self.resources
                .device
                .allocate_descriptor_sets(&allocate_info)
        } {
            Ok(mut sets) => match sets.pop() {
                Some(set) => set,
                None => {
                    self.slot.clear_descriptor();
                    return Err(VulkanError::new(
                        "vkAllocateDescriptorSets returned no descriptor set",
                    ));
                }
            },
            Err(error) => {
                self.slot.clear_descriptor();
                return Err(VulkanError::new(format!(
                    "vkAllocateDescriptorSets failed: {error:?}"
                )));
            }
        };

        let data = self
            .slot
            .data
            .as_ref()
            .ok_or_else(|| VulkanError::new("cannot bind missing BAM resource"))?;
        let spans = self
            .slot
            .spans
            .as_ref()
            .ok_or_else(|| VulkanError::new("cannot bind missing span resource"))?;
        let results = self
            .slot
            .results
            .as_ref()
            .ok_or_else(|| VulkanError::new("cannot bind missing counter resource"))?;
        let statuses = self
            .slot
            .statuses
            .as_ref()
            .ok_or_else(|| VulkanError::new("cannot bind missing status resource"))?;
        let parameters = self
            .slot
            .parameters
            .as_ref()
            .ok_or_else(|| VulkanError::new("cannot bind missing parameter resource"))?;
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(data.device.buffer)
                .offset(0)
                .range(data.device.capacity),
            vk::DescriptorBufferInfo::default()
                .buffer(spans.device.buffer)
                .offset(0)
                .range(spans.device.capacity),
            vk::DescriptorBufferInfo::default()
                .buffer(results.device.buffer)
                .offset(0)
                .range(results.device.capacity),
            vk::DescriptorBufferInfo::default()
                .buffer(statuses.device.buffer)
                .offset(0)
                .range(statuses.device.capacity),
            vk::DescriptorBufferInfo::default()
                .buffer(parameters.device.buffer)
                .offset(0)
                .range(parameters.device.capacity),
        ];
        let writes = [
            descriptor_write(
                descriptor_set,
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                &buffer_infos[0],
            ),
            descriptor_write(
                descriptor_set,
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                &buffer_infos[1],
            ),
            descriptor_write(
                descriptor_set,
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                &buffer_infos[2],
            ),
            descriptor_write(
                descriptor_set,
                3,
                vk::DescriptorType::STORAGE_BUFFER,
                &buffer_infos[3],
            ),
            descriptor_write(
                descriptor_set,
                4,
                vk::DescriptorType::UNIFORM_BUFFER,
                &buffer_infos[4],
            ),
        ];
        unsafe { self.resources.device.update_descriptor_sets(&writes, &[]) };
        self.slot.descriptor_set = descriptor_set;
        Ok(())
    }

    fn dispatch_handles(&self) -> Result<DispatchHandles, VulkanError> {
        let data = self
            .slot
            .data
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan BAM resource slot is missing"))?;
        let spans = self
            .slot
            .spans
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan span resource slot is missing"))?;
        let parameters = self
            .slot
            .parameters
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan parameter resource slot is missing"))?;
        let results = self
            .slot
            .results
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan result resource slot is missing"))?;
        let statuses = self
            .slot
            .statuses
            .as_ref()
            .ok_or_else(|| VulkanError::new("Vulkan status resource slot is missing"))?;
        if self.slot.descriptor_set == vk::DescriptorSet::null() {
            return Err(VulkanError::new("Vulkan descriptor set is missing"));
        }
        Ok(DispatchHandles {
            data_device: data.device.buffer,
            data_capacity: data.device.capacity,
            data_upload: data.host.buffer,
            spans_device: spans.device.buffer,
            spans_capacity: spans.device.capacity,
            spans_upload: spans.host.buffer,
            parameters_device: parameters.device.buffer,
            parameters_capacity: parameters.device.capacity,
            parameters_upload: parameters.host.buffer,
            results_device: results.device.buffer,
            results_capacity: results.device.capacity,
            results_readback: results.host.buffer,
            statuses_device: statuses.device.buffer,
            statuses_capacity: statuses.device.capacity,
            statuses_readback: statuses.host.buffer,
            descriptor_set: self.slot.descriptor_set,
        })
    }

    fn begin_recording(&self) -> Result<(), VulkanError> {
        let device = &self.resources.device;
        unsafe {
            device
                .reset_command_pool(
                    self.resources.command_pool,
                    vk::CommandPoolResetFlags::empty(),
                )
                .map_err(|error| {
                    VulkanError::new(format!(
                        "vkResetCommandPool before dispatch failed: {error:?}"
                    ))
                })?;
            device
                .reset_fences(&[self.resources.fence])
                .map_err(|error| {
                    VulkanError::new(format!("vkResetFences before dispatch failed: {error:?}"))
                })?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.resources.command_buffer, &begin)
                .map_err(|error| {
                    VulkanError::new(format!("vkBeginCommandBuffer failed: {error:?}"))
                })?;
        }
        Ok(())
    }

    fn end_recording(&self) -> Result<(), VulkanError> {
        unsafe {
            self.resources
                .device
                .end_command_buffer(self.resources.command_buffer)
                .map_err(|error| VulkanError::new(format!("vkEndCommandBuffer failed: {error:?}")))
        }
    }

    fn cleanup_pre_submit(&self) {
        unsafe {
            let _ = self.resources.device.reset_command_pool(
                self.resources.command_pool,
                vk::CommandPoolResetFlags::empty(),
            );
        }
    }

    fn submit_and_wait(&mut self, label: &str) -> Result<(), VulkanError> {
        let command_buffers = [self.resources.command_buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        if let Err(error) = unsafe {
            self.resources
                .device
                .queue_submit(self.resources.queue, &submit, self.resources.fence)
        } {
            return Err(self.after_submit_error(label, error));
        }
        if let Err(error) = unsafe {
            self.resources
                .device
                .wait_for_fences(&[self.resources.fence], true, u64::MAX)
        } {
            return Err(self.after_submit_error(label, error));
        }
        Ok(())
    }

    fn after_submit_error(&mut self, label: &str, error: vk::Result) -> VulkanError {
        let idle_result = unsafe { self.resources.device.device_wait_idle() };
        if idle_result.is_err() {
            self.poisoned = true;
        }
        VulkanError::new(format!(
            "{label} failed after Vulkan submission: {error:?}; vkDeviceWaitIdle fallback: {idle_result:?}"
        ))
    }

    fn run_timestamped(
        &mut self,
        handles: DispatchHandles,
        sizes: BatchBufferSizes,
        span_count: u32,
    ) -> Result<[f64; 3], VulkanError> {
        if let Err(error) = self.begin_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        let command = self.resources.command_buffer;
        let device = &self.resources.device;
        unsafe {
            device.cmd_reset_query_pool(command, self.resources.query_pool, 0, TIMESTAMP_COUNT);
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::TRANSFER,
                self.resources.query_pool,
                0,
            );
            record_upload_copies(device, command, handles, sizes);
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::TRANSFER,
                self.resources.query_pool,
                1,
            );
            record_transfer_to_compute_barrier(device, command, handles);
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                self.resources.query_pool,
                2,
            );
            record_dispatch(
                device,
                command,
                self.resources.pipeline,
                self.resources.pipeline_layout,
                handles,
                span_count,
            );
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                self.resources.query_pool,
                3,
            );
            record_compute_to_transfer_barrier(device, command, handles);
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::TRANSFER,
                self.resources.query_pool,
                4,
            );
            record_readback_copies(device, command, handles, sizes);
            device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::TRANSFER,
                self.resources.query_pool,
                5,
            );
        }
        if let Err(error) = self.end_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        self.submit_and_wait("timestamped Vulkan dispatch")?;

        let mut ticks = [0u64; TIMESTAMP_COUNT as usize];
        let device = &self.resources.device;
        unsafe {
            device
                .get_query_pool_results(
                    self.resources.query_pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| {
                    VulkanError::new(format!(
                        "vkGetQueryPoolResults for Vulkan stage timestamps failed: {error:?}"
                    ))
                })?;
        }
        let period_ns = f64::from(self.properties.limits.timestamp_period);
        Ok([
            timestamp_ms(
                ticks[0],
                ticks[1],
                self.resources.timestamp_valid_bits,
                period_ns,
            ),
            timestamp_ms(
                ticks[2],
                ticks[3],
                self.resources.timestamp_valid_bits,
                period_ns,
            ),
            timestamp_ms(
                ticks[4],
                ticks[5],
                self.resources.timestamp_valid_bits,
                period_ns,
            ),
        ])
    }

    fn run_host_timed(
        &mut self,
        handles: DispatchHandles,
        sizes: BatchBufferSizes,
        span_count: u32,
    ) -> Result<[f64; 3], VulkanError> {
        let upload_start = Instant::now();
        if let Err(error) = self.begin_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        unsafe {
            record_upload_copies(
                &self.resources.device,
                self.resources.command_buffer,
                handles,
                sizes,
            );
            record_transfer_to_compute_barrier(
                &self.resources.device,
                self.resources.command_buffer,
                handles,
            );
        }
        if let Err(error) = self.end_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        self.submit_and_wait("Vulkan upload submission")?;
        let h2d_ms = elapsed_ms(upload_start);

        let kernel_start = Instant::now();
        if let Err(error) = self.begin_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        unsafe {
            record_dispatch(
                &self.resources.device,
                self.resources.command_buffer,
                self.resources.pipeline,
                self.resources.pipeline_layout,
                handles,
                span_count,
            );
            record_compute_to_transfer_barrier(
                &self.resources.device,
                self.resources.command_buffer,
                handles,
            );
        }
        if let Err(error) = self.end_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        self.submit_and_wait("Vulkan compute submission")?;
        let kernel_ms = elapsed_ms(kernel_start);

        let readback_start = Instant::now();
        if let Err(error) = self.begin_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        unsafe {
            record_readback_copies(
                &self.resources.device,
                self.resources.command_buffer,
                handles,
                sizes,
            );
        }
        if let Err(error) = self.end_recording() {
            self.cleanup_pre_submit();
            return Err(error);
        }
        self.submit_and_wait("Vulkan readback submission")?;
        let d2h_ms = elapsed_ms(readback_start);
        Ok([h2d_ms, kernel_ms, d2h_ms])
    }
}

struct PairSpec<'a> {
    capacity: vk::DeviceSize,
    device_usage: vk::BufferUsageFlags,
    host_usage: vk::BufferUsageFlags,
    device_label: &'a str,
    host_label: &'a str,
}

fn ensure_pair(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    current: &mut Option<DeviceHostPair>,
    spec: PairSpec<'_>,
) -> Result<(), VulkanError> {
    if current.as_ref().is_some_and(|pair| {
        pair.device.capacity >= spec.capacity && pair.host.capacity >= spec.capacity
    }) {
        return Ok(());
    }
    let device_buffer = OwnedBuffer::create(
        device,
        memory_properties,
        spec.capacity,
        spec.device_usage,
        AllocationKind::Device,
        spec.device_label,
    )?;
    let host_buffer = OwnedBuffer::create(
        device,
        memory_properties,
        spec.capacity,
        spec.host_usage,
        AllocationKind::Host,
        spec.host_label,
    )?;
    *current = Some(DeviceHostPair {
        device: device_buffer,
        host: host_buffer,
    });
    Ok(())
}

fn pair_needs_growth(pair: &Option<DeviceHostPair>, required: vk::DeviceSize) -> bool {
    pair.as_ref()
        .is_none_or(|pair| pair.device.capacity < required || pair.host.capacity < required)
}

fn pair_needs_capacity(pair: &Option<DeviceHostPair>, capacity: vk::DeviceSize) -> bool {
    pair.as_ref()
        .is_none_or(|pair| pair.device.capacity < capacity || pair.host.capacity < capacity)
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

fn descriptor_write<'a>(
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .buffer_info(std::slice::from_ref(buffer_info))
}

unsafe fn record_upload_copies(
    device: &ash::Device,
    command: vk::CommandBuffer,
    handles: DispatchHandles,
    sizes: BatchBufferSizes,
) {
    device.cmd_copy_buffer(
        command,
        handles.data_upload,
        handles.data_device,
        &[vk::BufferCopy::default().size(sizes.data)],
    );
    device.cmd_copy_buffer(
        command,
        handles.spans_upload,
        handles.spans_device,
        &[vk::BufferCopy::default().size(sizes.spans)],
    );
    device.cmd_copy_buffer(
        command,
        handles.parameters_upload,
        handles.parameters_device,
        &[vk::BufferCopy::default().size(sizes.parameters)],
    );
}

unsafe fn record_readback_copies(
    device: &ash::Device,
    command: vk::CommandBuffer,
    handles: DispatchHandles,
    sizes: BatchBufferSizes,
) {
    device.cmd_copy_buffer(
        command,
        handles.results_device,
        handles.results_readback,
        &[vk::BufferCopy::default().size(sizes.results)],
    );
    device.cmd_copy_buffer(
        command,
        handles.statuses_device,
        handles.statuses_readback,
        &[vk::BufferCopy::default().size(sizes.statuses)],
    );
}

unsafe fn record_transfer_to_compute_barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    handles: DispatchHandles,
) {
    let barriers = [
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(handles.data_device)
            .offset(0)
            .size(handles.data_capacity),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(handles.spans_device)
            .offset(0)
            .size(handles.spans_capacity),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::UNIFORM_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(handles.parameters_device)
            .offset(0)
            .size(handles.parameters_capacity),
    ];
    device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &barriers,
        &[],
    );
}

unsafe fn record_compute_to_transfer_barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    handles: DispatchHandles,
) {
    let barriers = [
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(handles.results_device)
            .offset(0)
            .size(handles.results_capacity),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(handles.statuses_device)
            .offset(0)
            .size(handles.statuses_capacity),
    ];
    device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &barriers,
        &[],
    );
}

unsafe fn record_dispatch(
    device: &ash::Device,
    command: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    handles: DispatchHandles,
    span_count: u32,
) {
    device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
    device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[handles.descriptor_set],
        &[],
    );
    device.cmd_dispatch(command, span_count, 1, 1);
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

fn select_device_memory(
    properties: &vk::PhysicalDeviceMemoryProperties,
    compatible_bits: u32,
) -> Option<u32> {
    let mut fallback = None;
    for index in 0..properties.memory_type_count {
        if compatible_bits & (1u32 << index) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if fallback.is_none() {
            fallback = Some(index);
        }
        if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return Some(index);
        }
    }
    fallback
}

fn select_host_memory(
    properties: &vk::PhysicalDeviceMemoryProperties,
    compatible_bits: u32,
) -> Option<(u32, bool)> {
    let mut visible = None;
    for index in 0..properties.memory_type_count {
        if compatible_bits & (1u32 << index) == 0 {
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
        || span_starts
            .last()
            .is_none_or(|value| *value as usize >= data.len())
    {
        return Err(VulkanError::new(
            "Vulkan flagstat span starts must begin at zero, increase strictly, and lie within data",
        ));
    }
    Ok(())
}

fn padded_data_size(byte_count: usize) -> Result<usize, VulkanError> {
    byte_count
        .checked_add(3)
        .map(|count| count / 4 * 4)
        .ok_or_else(|| VulkanError::new("Vulkan padded BAM byte size overflow"))
}

/// Returns a bounded next-power-of-two capacity.  When `maximum` is below
/// that power, the maximum is returned as the explicitly clamped class.
fn capacity_class(required: vk::DeviceSize, maximum: vk::DeviceSize) -> Option<vk::DeviceSize> {
    if required == 0 || required > maximum || maximum == 0 {
        return None;
    }
    Some(
        required
            .checked_next_power_of_two()
            .unwrap_or(required)
            .min(maximum),
    )
}

fn timestamp_mask(valid_bits: u32) -> u64 {
    if valid_bits == 0 {
        0
    } else if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    end.wrapping_sub(start) & timestamp_mask(valid_bits)
}

fn timestamp_ms(start: u64, end: u64, valid_bits: u32, period_ns: f64) -> f64 {
    timestamp_delta(start, end, valid_bits) as f64 * period_ns / 1_000_000.0
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
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
    fn grow_only_capacity_classes_are_bounded_and_clamped() {
        assert_eq!(capacity_class(1, 1024), Some(1));
        assert_eq!(capacity_class(5, 1024), Some(8));
        assert_eq!(capacity_class(1024, 1024), Some(1024));
        assert_eq!(capacity_class(999, 1000), Some(1000));
        assert_eq!(capacity_class(1001, 1000), None);
        assert_eq!(capacity_class(0, 1000), None);
    }

    #[test]
    fn timestamp_conversion_masks_valid_bits_and_wraps() {
        assert_eq!(timestamp_delta(10, 25, 64), 15);
        assert_eq!(timestamp_delta(0xffff_fffe, 1, 32), 3);
        assert_eq!(timestamp_delta(0xffff, 0, 16), 1);
        assert_eq!(timestamp_ms(0, 1_000, 64, 2.0), 0.002);
    }

    #[test]
    fn direct_vulkan_classifier_matches_host_reuses_and_reports_malformed() {
        let _gpu_test_guard = crate::GPU_TEST_LOCK.lock().unwrap();
        let mut context = VulkanContext::create(0).expect("a hardware Vulkan device is required");
        let one = record(0x001 | 0x040, 30, 0, 0);
        let one_counts = crate::bam::classify_records(&one).unwrap();
        let first = context.flagstat(&one, &[0]).unwrap();
        assert_eq!(first.statuses, [0]);
        assert_eq!(first.span_counts, [one_counts]);

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
        let rejected = context.flagstat(&malformed, &spans).unwrap_err();
        assert!(rejected.to_string().contains("span 1"));
        assert!(rejected.to_string().contains("status 3"));
        assert!(rejected.to_string().contains("record extends beyond span"));

        // The failed status call waited for its fence.  A valid smaller call
        // then exercises reuse without stale output from the larger batch.
        let repeated = context.flagstat(&one, &[0]).unwrap();
        assert_eq!(repeated.statuses, [0]);
        assert_eq!(repeated.span_counts, [one_counts]);
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
