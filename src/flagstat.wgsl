// One workgroup classifies one record-aligned BAM span. The host packs the
// byte stream into little-endian u32 words because portable WGSL storage
// buffers cannot expose an array of bytes.

struct Parameters {
    byte_count: u32,
    span_count: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> span_starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<storage, read_write> statuses: array<u32>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

var<workgroup> record_offsets: array<u32, 128>;
var<workgroup> next_offset: u32;
var<workgroup> cursor: u32;
var<workgroup> record_count: u32;
var<workgroup> error_code: u32;
var<workgroup> stats: array<atomic<u32>, 32>;

fn load_u8(offset: u32) -> u32 {
    let word = data[offset >> 2u];
    return (word >> ((offset & 3u) * 8u)) & 0xffu;
}

fn load_u16(offset: u32) -> u32 {
    return load_u8(offset) | (load_u8(offset + 1u) << 8u);
}

fn load_u32(offset: u32) -> u32 {
    return load_u8(offset) |
        (load_u8(offset + 1u) << 8u) |
        (load_u8(offset + 2u) << 16u) |
        (load_u8(offset + 3u) << 24u);
}

fn increment(field: u32, category: u32) {
    atomicAdd(&stats[field * 2u + category], 1u);
}

fn classify_record(record: u32) {
    let flag = load_u16(record + 18u);
    var category = 0u;
    if ((flag & 0x200u) != 0u) {
        category = 1u;
    }

    increment(0u, category); // total reads
    if ((flag & 0x100u) != 0u) {
        increment(11u, category); // secondary
    } else if ((flag & 0x800u) != 0u) {
        increment(12u, category); // supplementary
    } else {
        increment(13u, category); // primary
        if ((flag & 0x001u) != 0u) {
            let reference_id = bitcast<i32>(load_u32(record + 4u));
            let next_reference_id = bitcast<i32>(load_u32(record + 24u));
            let mapq = load_u8(record + 13u);
            increment(2u, category); // paired in sequencing
            if ((flag & 0x002u) != 0u && (flag & 0x004u) == 0u) {
                increment(4u, category); // properly paired
            }
            if ((flag & 0x040u) != 0u) {
                increment(6u, category); // read1
            }
            if ((flag & 0x080u) != 0u) {
                increment(7u, category); // read2
            }
            if ((flag & 0x008u) != 0u && (flag & 0x004u) == 0u) {
                increment(5u, category); // singleton
            }
            if ((flag & 0x004u) == 0u && (flag & 0x008u) == 0u) {
                increment(3u, category); // read and mate mapped
                if (reference_id != next_reference_id) {
                    increment(9u, category); // different chromosome
                    if (mapq >= 5u) {
                        increment(10u, category); // different chromosome, mapq >= 5
                    }
                }
            }
        }
        if ((flag & 0x004u) == 0u) {
            increment(14u, category); // primary mapped
        }
        if ((flag & 0x400u) != 0u) {
            increment(15u, category); // primary duplicate
        }
    }
    if ((flag & 0x004u) == 0u) {
        increment(1u, category); // mapped
    }
    if ((flag & 0x400u) != 0u) {
        increment(8u, category); // duplicate
    }
}

@compute @workgroup_size(128)
fn flagstat(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let span = workgroup_id.x;
    if (span >= parameters.span_count) {
        return;
    }
    let begin = span_starts[span];
    var end = parameters.byte_count;
    if (span + 1u < parameters.span_count) {
        end = span_starts[span + 1u];
    }
    let span_size = end - begin;

    if (local_index < 32u) {
        atomicStore(&stats[local_index], 0u);
    }
    if (local_index == 0u) {
        cursor = 0u;
        error_code = 0u;
    }
    workgroupBarrier();

    loop {
        if (local_index == 0u) {
            var offset = cursor;
            var count = 0u;
            loop {
                if (count >= 128u || offset >= span_size) {
                    break;
                }
                if (span_size - offset < 4u) {
                    error_code = 1u;
                    break;
                }
                let block_size = load_u32(begin + offset);
                if (block_size < 32u) {
                    error_code = 2u;
                    break;
                }
                if (block_size > span_size - offset - 4u) {
                    error_code = 3u;
                    break;
                }
                record_offsets[count] = offset;
                count += 1u;
                offset += 4u + block_size;
            }
            record_count = count;
            next_offset = offset;
        }
        workgroupBarrier();

        let failed = error_code;
        let count = record_count;
        if (failed != 0u || count == 0u) {
            break;
        }
        if (local_index < count) {
            classify_record(begin + record_offsets[local_index]);
        }
        workgroupBarrier();
        if (local_index == 0u) {
            cursor = next_offset;
        }
        workgroupBarrier();
    }

    if (local_index < 32u) {
        results[span * 32u + local_index] = atomicLoad(&stats[local_index]);
    }
    if (local_index == 0u) {
        statuses[span] = error_code;
    }
}
