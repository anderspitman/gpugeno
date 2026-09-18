//! Self-contained direct-Vulkan synthetic flagstat integration spike.

use gpugeno::bam::{classify_records, FlagstatCounters};
use gpugeno::vulkan_spike::VulkanContext;
use std::error::Error;

fn record(flag: u16, mapq: u8, reference_id: i32, next_reference_id: i32) -> Vec<u8> {
    let mut value = vec![0u8; 36];
    value[..4].copy_from_slice(&32u32.to_le_bytes());
    value[4..8].copy_from_slice(&reference_id.to_le_bytes());
    value[13] = mapq;
    value[18..20].copy_from_slice(&flag.to_le_bytes());
    value[24..28].copy_from_slice(&next_reference_id.to_le_bytes());
    value
}

fn parse_device() -> Result<u32, Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut device = 0;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device" => {
                device = arguments
                    .next()
                    .ok_or("--device requires a numeric index")?
                    .parse()
                    .map_err(|error| format!("invalid --device index: {error}"))?;
            }
            _ => return Err(format!("unknown argument {argument}; usage: --device N").into()),
        }
    }
    Ok(device)
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = parse_device()?;
    let mut context = VulkanContext::create(device)?;
    let info = context.device_info();
    eprintln!(
        "direct_vulkan_device={} name={:?} type={:?} vendor=0x{:04x} device=0x{:04x} api={}.{}.{} timing_source={}",
        info.index,
        info.name,
        info.device_type,
        info.vendor_id,
        info.device_id,
        vk_major(info.api_version),
        vk_minor(info.api_version),
        vk_patch(info.api_version),
        context.timing_source(),
    );

    let mut data = record(0x100 | 0x800 | 0x400, 60, 0, 0);
    data.extend(record(0x001 | 0x040 | 0x002, 4, 0, 1));
    let second_span = data.len() as u32;
    data.extend(record(0x001 | 0x080, 5, 0, 1));
    data.extend(record(0x200 | 0x004 | 0x400, 0, -1, -1));
    let spans = [0, second_span];

    let gpu = context.flagstat(&data, &spans)?;
    if gpu.statuses.iter().any(|&status| status != 0) {
        return Err(format!("shader returned statuses {:?}", gpu.statuses).into());
    }
    let mut reduced = FlagstatCounters::default();
    for (index, counters) in gpu.span_counts.iter().enumerate() {
        let begin = spans[index] as usize;
        let end = spans
            .get(index + 1)
            .copied()
            .map(|offset| offset as usize)
            .unwrap_or(data.len());
        let host = classify_records(&data[begin..end])?;
        if *counters != host {
            return Err(format!("GPU/host mismatch in span {index}").into());
        }
        reduced.add_assign(counters);
    }
    if reduced != classify_records(&data)? {
        return Err("reduced GPU counters differ from the independent host oracle".into());
    }
    println!(
        "validated {} records in {} spans; kernel={:.6} ms ({})",
        reduced.n_reads.iter().sum::<u64>(),
        spans.len(),
        gpu.timings.kernel_ms,
        gpu.timings.source,
    );
    Ok(())
}

fn vk_major(version: u32) -> u32 {
    version >> 22
}

fn vk_minor(version: u32) -> u32 {
    (version >> 12) & 0x3ff
}

fn vk_patch(version: u32) -> u32 {
    version & 0xfff
}
