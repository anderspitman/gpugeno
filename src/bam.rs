//! Minimal BAM metadata and flagstat semantics shared by the streaming host
//! path and the CUDA boundary.

use crate::bgzf::{read_block_at, BgzfError, VirtualOffset};
use libdeflater::Decompressor;
use std::fs::File;
use std::path::{Path, PathBuf};

const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const BAM_CORE_SIZE: usize = 32;

const FPAIRED: u16 = 0x001;
const FPROPER_PAIR: u16 = 0x002;
const FUNMAP: u16 = 0x004;
const FMUNMAP: u16 = 0x008;
const FREAD1: u16 = 0x040;
const FREAD2: u16 = 0x080;
const FSECONDARY: u16 = 0x100;
const FQCFAIL: u16 = 0x200;
const FDUP: u16 = 0x400;
const FSUPPLEMENTARY: u16 = 0x800;

#[derive(Debug)]
pub struct BamError(String);

impl BamError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for BamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BamError {}

impl From<BgzfError> for BamError {
    fn from(error: BgzfError) -> Self {
        Self(error.to_string())
    }
}

/// Metadata needed to connect a BAM index to the physical record stream.
#[derive(Debug, Clone, Copy)]
pub struct BamHeader {
    pub reference_count: u32,
    pub first_record: VirtualOffset,
}

/// Reads enough BGZF members to parse the BAM header and locate the first
/// alignment record. No alignment data is retained.
pub fn read_bam_header(path: &Path) -> Result<BamHeader, BamError> {
    let mut file = File::open(path)
        .map_err(|error| BamError::new(format!("failed to open {}: {error}", path.display())))?;
    let mut decompressor = Decompressor::new();
    let mut compressed_offset = 0u64;
    let mut data = Vec::new();
    let mut blocks = Vec::new();

    loop {
        let block = read_block_at(&mut file, &mut decompressor, path, compressed_offset)?
            .ok_or_else(|| {
                BamError::new(format!("{} ended before its BAM header", path.display()))
            })?;
        if block.is_eof {
            return Err(BamError::new(format!(
                "{} reached the BGZF EOF marker before its BAM header was complete",
                path.display()
            )));
        }
        let uncompressed_start = data.len();
        data.extend_from_slice(&block.data);
        blocks.push((
            block.compressed_offset,
            block.compressed_len,
            uncompressed_start,
            block.data.len(),
        ));
        compressed_offset = compressed_offset
            .checked_add(block.compressed_len as u64)
            .ok_or_else(|| BamError::new("compressed offset overflow while reading BAM header"))?;

        if let Some((reference_count, header_len)) = parse_header_prefix(&data)? {
            let first_record = virtual_offset_for_position(header_len, &blocks)?;
            return Ok(BamHeader {
                reference_count,
                first_record,
            });
        }
        if data.len() > MAX_HEADER_BYTES {
            return Err(BamError::new(format!(
                "BAM header exceeds the prototype limit of {MAX_HEADER_BYTES} bytes"
            )));
        }
    }
}

fn parse_header_prefix(data: &[u8]) -> Result<Option<(u32, usize)>, BamError> {
    if data.len() < 8 {
        return Ok(None);
    }
    if &data[..4] != b"BAM\x01" {
        return Err(BamError::new("invalid BAM magic (expected BAM\\x01)"));
    }
    let text_len = read_i32(data, 4)?;
    if text_len < 0 {
        return Err(BamError::new("negative BAM header text length"));
    }
    let mut position = 8usize
        .checked_add(text_len as usize)
        .ok_or_else(|| BamError::new("BAM header text length overflows usize"))?;
    if data.len() < position + 4 {
        return Ok(None);
    }
    let reference_count = read_i32(data, position)?;
    if reference_count < 0 {
        return Err(BamError::new("negative BAM reference count"));
    }
    position += 4;
    for _ in 0..reference_count {
        if data.len() < position + 4 {
            return Ok(None);
        }
        let name_len = read_i32(data, position)?;
        if name_len <= 0 {
            return Err(BamError::new("BAM reference name length must be positive"));
        }
        position = position
            .checked_add(4 + name_len as usize)
            .ok_or_else(|| BamError::new("BAM reference entry length overflows usize"))?;
        if data.len() < position + 4 {
            return Ok(None);
        }
        if data[position - 1] != 0 {
            return Err(BamError::new("BAM reference name is not NUL terminated"));
        }
        let reference_len = read_i32(data, position)?;
        if reference_len < 0 {
            return Err(BamError::new("negative BAM reference length"));
        }
        position += 4;
    }
    Ok(Some((reference_count as u32, position)))
}

fn virtual_offset_for_position(
    position: usize,
    blocks: &[(u64, usize, usize, usize)],
) -> Result<VirtualOffset, BamError> {
    for &(compressed, compressed_len, start, len) in blocks {
        let end = start + len;
        if position < end {
            let within = position - start;
            let within = u16::try_from(within)
                .map_err(|_| BamError::new("BAM header position exceeds BGZF virtual offset"))?;
            return VirtualOffset::new(compressed, within).map_err(Into::into);
        }
        if position == end {
            let next = compressed
                .checked_add(compressed_len as u64)
                .ok_or_else(|| BamError::new("compressed offset overflow after BAM header"))?;
            return VirtualOffset::new(next, 0).map_err(Into::into);
        }
    }
    Err(BamError::new(
        "internal error locating BAM header end in BGZF members",
    ))
}

fn read_i32(data: &[u8], position: usize) -> Result<i32, BamError> {
    let bytes = data
        .get(position..position + 4)
        .ok_or_else(|| BamError::new("truncated BAM integer"))?;
    Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
}

/// Counter layout matches the CUDA C structure: every field stores
/// `[QC-passed, QC-failed]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlagstatCounters {
    pub n_reads: [u64; 2],
    pub n_mapped: [u64; 2],
    pub n_pair_all: [u64; 2],
    pub n_pair_map: [u64; 2],
    pub n_pair_good: [u64; 2],
    pub n_sgltn: [u64; 2],
    pub n_read1: [u64; 2],
    pub n_read2: [u64; 2],
    pub n_dup: [u64; 2],
    pub n_diffchr: [u64; 2],
    pub n_diffhigh: [u64; 2],
    pub n_secondary: [u64; 2],
    pub n_supp: [u64; 2],
    pub n_primary: [u64; 2],
    pub n_pmapped: [u64; 2],
    pub n_pdup: [u64; 2],
}

impl FlagstatCounters {
    pub fn add_assign(&mut self, other: &Self) {
        for (destination, source) in self.as_mut_flat().iter_mut().zip(other.as_flat()) {
            *destination += source;
        }
    }

    fn as_flat(&self) -> &[u64; 32] {
        // The repr(C) structure consists solely of 16 adjacent [u64; 2]
        // fields. This assertion and cast centralize the layout dependency.
        const _: () = assert!(std::mem::size_of::<FlagstatCounters>() == 32 * 8);
        unsafe { &*(self as *const Self).cast::<[u64; 32]>() }
    }

    fn as_mut_flat(&mut self) -> &mut [u64; 32] {
        unsafe { &mut *(self as *mut Self).cast::<[u64; 32]>() }
    }
}

/// Classifies all complete BAM records in a byte span. The span must begin and
/// end on record boundaries; this is used as an independent validation oracle.
pub fn classify_records(data: &[u8]) -> Result<FlagstatCounters, BamError> {
    let mut counters = FlagstatCounters::default();
    let mut position = 0usize;
    while position < data.len() {
        if data.len() - position < 4 {
            return Err(BamError::new(format!(
                "record span ends with {} bytes, fewer than a block_size",
                data.len() - position
            )));
        }
        let block_size =
            u32::from_le_bytes(data[position..position + 4].try_into().unwrap()) as usize;
        if block_size < BAM_CORE_SIZE {
            return Err(BamError::new(format!(
                "BAM record at batch offset {position} has block_size {block_size}, smaller than the core"
            )));
        }
        let end = position
            .checked_add(4 + block_size)
            .ok_or_else(|| BamError::new("BAM record size overflows usize"))?;
        if end > data.len() {
            return Err(BamError::new(format!(
                "BAM record at batch offset {position} extends {} bytes beyond its span",
                end - data.len()
            )));
        }
        classify_one(&data[position..end], &mut counters);
        position = end;
    }
    Ok(counters)
}

fn classify_one(record: &[u8], counters: &mut FlagstatCounters) {
    let flag = u16::from_le_bytes([record[18], record[19]]);
    let category = usize::from(flag & FQCFAIL != 0);
    counters.n_reads[category] += 1;

    if flag & FSECONDARY != 0 {
        counters.n_secondary[category] += 1;
    } else if flag & FSUPPLEMENTARY != 0 {
        counters.n_supp[category] += 1;
    } else {
        counters.n_primary[category] += 1;
        if flag & FPAIRED != 0 {
            let reference_id = i32::from_le_bytes(record[4..8].try_into().unwrap());
            let next_reference_id = i32::from_le_bytes(record[24..28].try_into().unwrap());
            let mapq = record[13];
            counters.n_pair_all[category] += 1;
            if flag & FPROPER_PAIR != 0 && flag & FUNMAP == 0 {
                counters.n_pair_good[category] += 1;
            }
            if flag & FREAD1 != 0 {
                counters.n_read1[category] += 1;
            }
            if flag & FREAD2 != 0 {
                counters.n_read2[category] += 1;
            }
            if flag & FMUNMAP != 0 && flag & FUNMAP == 0 {
                counters.n_sgltn[category] += 1;
            }
            if flag & FUNMAP == 0 && flag & FMUNMAP == 0 {
                counters.n_pair_map[category] += 1;
                if reference_id != next_reference_id {
                    counters.n_diffchr[category] += 1;
                    if mapq >= 5 {
                        counters.n_diffhigh[category] += 1;
                    }
                }
            }
        }
        if flag & FUNMAP == 0 {
            counters.n_pmapped[category] += 1;
        }
        if flag & FDUP != 0 {
            counters.n_pdup[category] += 1;
        }
    }
    if flag & FUNMAP == 0 {
        counters.n_mapped[category] += 1;
    }
    if flag & FDUP != 0 {
        counters.n_dup[category] += 1;
    }
}

pub fn format_flagstat(counters: &FlagstatCounters) -> String {
    let p = 0;
    let f = 1;
    let percent = |numerator: u64, denominator: u64| {
        if denominator == 0 {
            "N/A".to_string()
        } else {
            format!("{:.2}%", 100.0 * numerator as f64 / denominator as f64)
        }
    };
    let mut lines = Vec::with_capacity(16);
    lines.push(format!(
        "{} + {} in total (QC-passed reads + QC-failed reads)",
        counters.n_reads[p], counters.n_reads[f]
    ));
    lines.push(format!(
        "{} + {} primary",
        counters.n_primary[p], counters.n_primary[f]
    ));
    lines.push(format!(
        "{} + {} secondary",
        counters.n_secondary[p], counters.n_secondary[f]
    ));
    lines.push(format!(
        "{} + {} supplementary",
        counters.n_supp[p], counters.n_supp[f]
    ));
    lines.push(format!(
        "{} + {} duplicates",
        counters.n_dup[p], counters.n_dup[f]
    ));
    lines.push(format!(
        "{} + {} primary duplicates",
        counters.n_pdup[p], counters.n_pdup[f]
    ));
    lines.push(format!(
        "{} + {} mapped ({} : {})",
        counters.n_mapped[p],
        counters.n_mapped[f],
        percent(counters.n_mapped[p], counters.n_reads[p]),
        percent(counters.n_mapped[f], counters.n_reads[f])
    ));
    lines.push(format!(
        "{} + {} primary mapped ({} : {})",
        counters.n_pmapped[p],
        counters.n_pmapped[f],
        percent(counters.n_pmapped[p], counters.n_primary[p]),
        percent(counters.n_pmapped[f], counters.n_primary[f])
    ));
    lines.push(format!(
        "{} + {} paired in sequencing",
        counters.n_pair_all[p], counters.n_pair_all[f]
    ));
    lines.push(format!(
        "{} + {} read1",
        counters.n_read1[p], counters.n_read1[f]
    ));
    lines.push(format!(
        "{} + {} read2",
        counters.n_read2[p], counters.n_read2[f]
    ));
    lines.push(format!(
        "{} + {} properly paired ({} : {})",
        counters.n_pair_good[p],
        counters.n_pair_good[f],
        percent(counters.n_pair_good[p], counters.n_pair_all[p]),
        percent(counters.n_pair_good[f], counters.n_pair_all[f])
    ));
    lines.push(format!(
        "{} + {} with itself and mate mapped",
        counters.n_pair_map[p], counters.n_pair_map[f]
    ));
    lines.push(format!(
        "{} + {} singletons ({} : {})",
        counters.n_sgltn[p],
        counters.n_sgltn[f],
        percent(counters.n_sgltn[p], counters.n_pair_all[p]),
        percent(counters.n_sgltn[f], counters.n_pair_all[f])
    ));
    lines.push(format!(
        "{} + {} with mate mapped to a different chr",
        counters.n_diffchr[p], counters.n_diffchr[f]
    ));
    lines.push(format!(
        "{} + {} with mate mapped to a different chr (mapQ>=5)",
        counters.n_diffhigh[p], counters.n_diffhigh[f]
    ));
    lines.join("\n") + "\n"
}

pub fn default_bai_path(bam_path: &Path) -> PathBuf {
    let mut value = bam_path.as_os_str().to_owned();
    value.push(".bai");
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(flag: u16, mapq: u8, reference_id: i32, next_reference_id: i32) -> Vec<u8> {
        let mut value = vec![0u8; 4 + BAM_CORE_SIZE];
        value[..4].copy_from_slice(&(BAM_CORE_SIZE as u32).to_le_bytes());
        value[4..8].copy_from_slice(&reference_id.to_le_bytes());
        value[13] = mapq;
        value[18..20].copy_from_slice(&flag.to_le_bytes());
        value[24..28].copy_from_slice(&next_reference_id.to_le_bytes());
        value
    }

    #[test]
    fn classifier_matches_reference_precedence_and_mapq_boundary() {
        let mut data = record(FSECONDARY | FSUPPLEMENTARY | FDUP, 60, 0, 0);
        data.extend(record(FPAIRED | FREAD1 | FPROPER_PAIR, 4, 0, 1));
        data.extend(record(FPAIRED | FREAD2, 5, 0, 1));
        data.extend(record(FQCFAIL | FUNMAP | FDUP, 0, -1, -1));

        let result = classify_records(&data).unwrap();
        assert_eq!(result.n_reads, [3, 1]);
        assert_eq!(result.n_secondary, [1, 0]);
        assert_eq!(result.n_supp, [0, 0]);
        assert_eq!(result.n_primary, [2, 1]);
        assert_eq!(result.n_diffchr, [2, 0]);
        assert_eq!(result.n_diffhigh, [1, 0]);
        assert_eq!(result.n_dup, [1, 1]);
        assert_eq!(result.n_pdup, [0, 1]);
        assert_eq!(result.n_mapped, [3, 0]);
    }

    #[test]
    fn classifier_rejects_partial_record() {
        let mut value = record(0, 0, 0, 0);
        value.pop();
        assert!(classify_records(&value)
            .unwrap_err()
            .to_string()
            .contains("extends"));
    }

    #[test]
    fn formatting_has_sixteen_lines_and_na_for_zero_denominator() {
        let output = format_flagstat(&FlagstatCounters::default());
        assert_eq!(output.lines().count(), 16);
        assert!(output.contains("mapped (N/A : N/A)"));
    }
}
