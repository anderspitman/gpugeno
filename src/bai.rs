//! Minimal BAI reader. Linear entries are retained as coordinate-bearing work
//! items, including repeated virtual offsets, because pileup and flagstat need
//! different operation-specific views of the same index.

use crate::bgzf::VirtualOffset;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const LINEAR_WINDOW_BASES: u64 = 16_384;
const MAX_INDEX_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct BaiError(String);

impl BaiError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for BaiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BaiError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearWorkItem {
    /// `(reference_id << 32) | zero-based genomic window start`.
    pub coordinate: u64,
    pub virtual_offset: VirtualOffset,
}

#[derive(Debug)]
pub struct BaiIndex {
    pub reference_count: u32,
    /// Nonzero linear entries in index order. Repeated offsets are intentional.
    pub linear_work_items: Vec<LinearWorkItem>,
}

pub fn read_bai(path: &Path) -> Result<BaiIndex, BaiError> {
    let mut file = File::open(path)
        .map_err(|error| BaiError::new(format!("failed to open {}: {error}", path.display())))?;
    let length = file
        .metadata()
        .map_err(|error| BaiError::new(format!("failed to stat {}: {error}", path.display())))?
        .len();
    if length > MAX_INDEX_BYTES {
        return Err(BaiError::new(format!(
            "{} is {length} bytes, exceeding the BAI prototype limit of {MAX_INDEX_BYTES}",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| BaiError::new(format!("failed to read {}: {error}", path.display())))?;
    parse_bai(&bytes).map_err(|error| BaiError::new(format!("{}: {error}", path.display())))
}

fn parse_bai(bytes: &[u8]) -> Result<BaiIndex, BaiError> {
    let mut reader = SliceReader::new(bytes);
    if reader.take(4)? != b"BAI\x01" {
        return Err(BaiError::new("invalid BAI magic (expected BAI\\x01)"));
    }
    let reference_count = reader.nonnegative_i32("reference count")? as u32;
    let mut linear_work_items = Vec::new();

    for reference_id in 0..reference_count {
        let bin_count = reader.nonnegative_i32("bin count")? as usize;
        for _ in 0..bin_count {
            reader.u32("bin number")?;
            let chunk_count = reader.nonnegative_i32("chunk count")? as usize;
            let chunk_bytes = chunk_count
                .checked_mul(16)
                .ok_or_else(|| BaiError::new("BAI chunk byte count overflows usize"))?;
            reader.take(chunk_bytes)?;
        }

        let interval_count = reader.nonnegative_i32("linear interval count")? as usize;
        linear_work_items
            .try_reserve(interval_count)
            .map_err(|error| BaiError::new(format!("could not reserve BAI work items: {error}")))?;
        for interval in 0..interval_count {
            let raw = reader.u64("linear virtual offset")?;
            if raw != 0 {
                let position = (interval as u64)
                    .checked_mul(LINEAR_WINDOW_BASES)
                    .ok_or_else(|| BaiError::new("BAI linear coordinate overflows u64"))?;
                if position > u32::MAX as u64 {
                    return Err(BaiError::new(format!(
                        "BAI linear coordinate {position} exceeds the 32-bit coordinate representation"
                    )));
                }
                linear_work_items.push(LinearWorkItem {
                    coordinate: (u64::from(reference_id) << 32) | position,
                    virtual_offset: VirtualOffset::from_raw(raw),
                });
            }
        }
    }

    // BAI optionally ends with one u64 n_no_coor value.
    if reader.remaining() != 0 && reader.remaining() != 8 {
        return Err(BaiError::new(format!(
            "unexpected {} trailing bytes after BAI references",
            reader.remaining()
        )));
    }
    if reader.remaining() == 8 {
        reader.u64("unplaced-unmapped count")?;
    }

    Ok(BaiIndex {
        reference_count,
        linear_work_items,
    })
}

/// Builds exact-once physical anchors for flagstat while leaving the index's
/// original repeated, coordinate-bearing work items untouched for pileup.
pub fn flagstat_anchors(
    first_record: VirtualOffset,
    data_end: VirtualOffset,
    work_items: &[LinearWorkItem],
) -> Result<Vec<VirtualOffset>, BaiError> {
    if first_record >= data_end {
        return Err(BaiError::new(format!(
            "first BAM record virtual offset {} is not before data endpoint {}",
            first_record.raw(),
            data_end.raw()
        )));
    }
    let mut anchors = Vec::with_capacity(work_items.len() + 2);
    anchors.push(first_record);
    anchors.extend(work_items.iter().filter_map(|item| {
        (item.virtual_offset > first_record && item.virtual_offset < data_end)
            .then_some(item.virtual_offset)
    }));
    anchors.push(data_end);
    anchors.sort_unstable();
    anchors.dedup();
    if anchors.first() != Some(&first_record) || anchors.last() != Some(&data_end) {
        return Err(BaiError::new(
            "internal error constructing flagstat anchors",
        ));
    }
    Ok(anchors)
}

struct SliceReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> SliceReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], BaiError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| BaiError::new("BAI offset overflows usize"))?;
        let value = self.bytes.get(self.position..end).ok_or_else(|| {
            BaiError::new(format!(
                "truncated BAI at byte {} while reading {length} bytes",
                self.position
            ))
        })?;
        self.position = end;
        Ok(value)
    }

    fn u32(&mut self, what: &str) -> Result<u32, BaiError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().unwrap();
        let value = u32::from_le_bytes(bytes);
        let _ = what;
        Ok(value)
    }

    fn nonnegative_i32(&mut self, what: &str) -> Result<i32, BaiError> {
        let value = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
        if value < 0 {
            return Err(BaiError::new(format!("negative BAI {what}: {value}")));
        }
        Ok(value)
    }

    fn u64(&mut self, _what: &str) -> Result<u64, BaiError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_bai() -> Vec<u8> {
        let mut bytes = b"BAI\x01".to_vec();
        bytes.extend_from_slice(&1i32.to_le_bytes()); // n_ref
        bytes.extend_from_slice(&1i32.to_le_bytes()); // n_bin
        bytes.extend_from_slice(&4681u32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes()); // n_chunk
        bytes.extend_from_slice(&10u64.to_le_bytes());
        bytes.extend_from_slice(&20u64.to_le_bytes());
        bytes.extend_from_slice(&3i32.to_le_bytes()); // n_intv
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0x0001_0002u64.to_le_bytes());
        bytes.extend_from_slice(&0x0001_0002u64.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes()); // n_no_coor
        bytes
    }

    #[test]
    fn preserves_repeated_linear_work_items_and_coordinates() {
        let index = parse_bai(&tiny_bai()).unwrap();
        assert_eq!(index.reference_count, 1);
        assert_eq!(index.linear_work_items.len(), 2);
        assert_eq!(index.linear_work_items[0].coordinate, 16_384);
        assert_eq!(index.linear_work_items[1].coordinate, 32_768);
        assert_eq!(
            index.linear_work_items[0].virtual_offset,
            index.linear_work_items[1].virtual_offset
        );
    }

    #[test]
    fn flagstat_view_deduplicates_and_adds_physical_endpoints() {
        let work_items = vec![
            LinearWorkItem {
                coordinate: 0,
                virtual_offset: VirtualOffset::from_raw(20),
            },
            LinearWorkItem {
                coordinate: 1,
                virtual_offset: VirtualOffset::from_raw(20),
            },
            LinearWorkItem {
                coordinate: 2,
                virtual_offset: VirtualOffset::from_raw(30),
            },
        ];
        let result = flagstat_anchors(
            VirtualOffset::from_raw(10),
            VirtualOffset::from_raw(40),
            &work_items,
        )
        .unwrap();
        assert_eq!(
            result.iter().map(|value| value.raw()).collect::<Vec<_>>(),
            vec![10, 20, 30, 40]
        );
    }
}
