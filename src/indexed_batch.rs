//! Bounded indexed BAM batches. The shared input representation is physical
//! and virtual-offset based; operation-specific planners decide whether starts
//! are disjoint (flagstat) or repeated/overlapping genomic windows (pileup).

use crate::bgzf::{
    read_compressed_block_at, BgzfBlock, BgzfError, BgzfWorkerPool, CompletedBlock, VirtualOffset,
};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct BatchError(String);

impl BatchError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BatchError {}

impl From<BgzfError> for BatchError {
    fn from(error: BgzfError) -> Self {
        Self(error.to_string())
    }
}

/// Mapping retained for future operation planners that need to translate
/// additional virtual offsets into this decompressed envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMap {
    pub compressed_offset: u64,
    pub source_uncompressed_start: u32,
    pub source_uncompressed_end: u32,
    pub batch_start: u32,
    pub batch_end: u32,
}

/// One bounded exact-once physical batch for flagstat.
#[derive(Debug)]
pub struct IndexedBamBatch {
    pub data: Vec<u8>,
    /// Batch-relative starts. The end of the final span is `data.len()`.
    pub span_starts: Vec<u32>,
    pub virtual_start: VirtualOffset,
    pub virtual_end: VirtualOffset,
    pub block_map: Vec<BlockMap>,
    pub blocks_decompressed: usize,
    pub compressed_bytes_read: u64,
    pub build_time: Duration,
}

impl IndexedBamBatch {
    pub fn span_count(&self) -> usize {
        self.span_starts.len()
    }

    /// Translates a virtual offset retained by this batch. This is not needed
    /// by flagstat after planning, but is the primitive future pileup work
    /// items use to preserve coordinate-bearing repeated starts.
    pub fn translate_virtual_offset(&self, offset: VirtualOffset) -> Option<u32> {
        self.block_map.iter().find_map(|mapping| {
            if mapping.compressed_offset != offset.compressed() {
                return None;
            }
            let source = u32::from(offset.uncompressed());
            if source < mapping.source_uncompressed_start
                || source > mapping.source_uncompressed_end
            {
                return None;
            }
            Some(mapping.batch_start + source - mapping.source_uncompressed_start)
        })
    }
}

/// Streams sorted, unique physical anchors as bounded decompressed batches.
/// Boundary BGZF members may be decompressed twice when an anchor lies inside
/// a member; memory remains bounded and every emitted record byte appears in
/// exactly one logical batch.
pub struct DisjointBamStream {
    path: PathBuf,
    file: File,
    workers: BgzfWorkerPool,
    anchors: Vec<VirtualOffset>,
    next_anchor: usize,
    max_uncompressed_bytes: usize,
}

struct PendingBatch {
    data: Vec<u8>,
    planned_len: usize,
    outputs: Vec<Option<BgzfBlock>>,
    next_output: usize,
    mapped: Vec<(usize, usize)>,
    candidate: Option<(usize, usize, usize)>,
    block_map: Vec<BlockMap>,
    blocks_decompressed: usize,
    compressed_bytes_read: u64,
}

impl PendingBatch {
    fn new(start_index: usize) -> Self {
        Self {
            data: Vec::new(),
            planned_len: 0,
            outputs: Vec::new(),
            next_output: 0,
            mapped: vec![(start_index, 0)],
            candidate: None,
            block_map: Vec::new(),
            blocks_decompressed: 0,
            compressed_bytes_read: 0,
        }
    }

    fn map_anchor(&mut self, anchor_index: usize, relative: usize, cap: usize) -> bool {
        self.mapped.push((anchor_index, relative));
        if relative <= cap {
            self.candidate = Some((anchor_index, self.mapped.len() - 1, relative));
            true
        } else {
            false
        }
    }
}

impl DisjointBamStream {
    pub fn open(
        path: &Path,
        anchors: Vec<VirtualOffset>,
        max_uncompressed_bytes: usize,
        thread_count: usize,
    ) -> Result<Self, BatchError> {
        if max_uncompressed_bytes == 0 || max_uncompressed_bytes > u32::MAX as usize {
            return Err(BatchError::new(format!(
                "batch size must be between 1 and {} bytes",
                u32::MAX
            )));
        }
        if anchors.len() < 2 {
            return Err(BatchError::new(
                "at least two BAM stream anchors are required",
            ));
        }
        if anchors.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(BatchError::new(
                "BAM stream anchors must be strictly increasing",
            ));
        }
        if thread_count == 0 {
            return Err(BatchError::new(
                "decompression thread count must be positive",
            ));
        }
        let file = File::open(path).map_err(|error| {
            BatchError::new(format!("failed to open {}: {error}", path.display()))
        })?;
        let workers = BgzfWorkerPool::new(thread_count, path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            workers,
            anchors,
            next_anchor: 0,
            max_uncompressed_bytes,
        })
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn next_batch(&mut self) -> Result<Option<IndexedBamBatch>, BatchError> {
        if self.next_anchor + 1 >= self.anchors.len() {
            return Ok(None);
        }
        let build_start = Instant::now();
        let start_index = self.next_anchor;
        let start = self.anchors[start_index];
        let start_compressed = start.compressed();
        let start_uncompressed = usize::from(start.uncompressed());
        let mut compressed_offset = start_compressed;
        let mut anchor_cursor = start_index + 1;
        let mut pending = PendingBatch::new(start_index);

        loop {
            if anchor_cursor < self.anchors.len()
                && self.anchors[anchor_cursor].compressed() < compressed_offset
            {
                return Err(BatchError::new(format!(
                    "BAI virtual offset {} does not point to a BGZF member start",
                    self.anchors[anchor_cursor].raw()
                )));
            }

            // A zero in the next compressed member denotes the boundary after
            // the member already retained. It can be mapped without reading
            // the next block, including at physical EOF.
            while anchor_cursor < self.anchors.len()
                && self.anchors[anchor_cursor].compressed() == compressed_offset
                && self.anchors[anchor_cursor].uncompressed() == 0
            {
                let within_cap = pending.map_anchor(
                    anchor_cursor,
                    pending.planned_len,
                    self.max_uncompressed_bytes,
                );
                if !within_cap {
                    return self.finish_or_oversized(start_index, start, pending, build_start);
                }
                anchor_cursor += 1;
                if anchor_cursor == self.anchors.len() {
                    return self.finish_candidate(start_index, start, pending, build_start);
                }
            }

            if self.workers.at_capacity() {
                let completed = self.workers.receive()?;
                Self::store_completed(&mut pending, completed)?;
            }
            let member = read_compressed_block_at(&mut self.file, &self.path, compressed_offset)?
                .ok_or_else(|| {
                BatchError::new(format!(
                    "{} ended before virtual endpoint {}",
                    self.path.display(),
                    self.anchors.last().unwrap().raw()
                ))
            })?;
            if member.is_eof {
                return Err(BatchError::new(format!(
                    "encountered the BGZF EOF marker before virtual endpoint {}",
                    self.anchors.last().unwrap().raw()
                )));
            }

            let source_start = if compressed_offset == start_compressed {
                start_uncompressed
            } else {
                0
            };
            if source_start > member.uncompressed_len {
                return Err(BatchError::new(format!(
                    "virtual offset {} has uncompressed component {} beyond the BGZF member's {} bytes",
                    start.raw(), source_start, member.uncompressed_len
                )));
            }
            let batch_start = pending.planned_len;
            pending.planned_len = pending
                .planned_len
                .checked_add(member.uncompressed_len - source_start)
                .ok_or_else(|| BatchError::new("decompressed batch size overflows usize"))?;
            pending.block_map.push(BlockMap {
                compressed_offset,
                source_uncompressed_start: source_start as u32,
                source_uncompressed_end: member.uncompressed_len as u32,
                batch_start: u32::try_from(batch_start)
                    .map_err(|_| BatchError::new("batch offset exceeds u32"))?,
                batch_end: u32::try_from(pending.planned_len)
                    .map_err(|_| BatchError::new("batch end exceeds u32"))?,
            });
            let compressed_len = member.bytes.len();
            let sequence = pending.outputs.len();
            pending.outputs.push(None);
            self.workers.submit(sequence, member)?;
            pending.blocks_decompressed += 1;
            pending.compressed_bytes_read = pending
                .compressed_bytes_read
                .checked_add(compressed_len as u64)
                .ok_or_else(|| BatchError::new("compressed batch byte count overflows u64"))?;

            while anchor_cursor < self.anchors.len()
                && self.anchors[anchor_cursor].compressed() == compressed_offset
            {
                let source_offset = usize::from(self.anchors[anchor_cursor].uncompressed());
                if source_offset < source_start
                    || source_offset
                        > pending.block_map.last().unwrap().source_uncompressed_end as usize
                {
                    return Err(BatchError::new(format!(
                        "virtual offset {} is outside the retained part of its BGZF member",
                        self.anchors[anchor_cursor].raw()
                    )));
                }
                let relative = batch_start + source_offset - source_start;
                let within_cap =
                    pending.map_anchor(anchor_cursor, relative, self.max_uncompressed_bytes);
                if !within_cap {
                    return self.finish_or_oversized(start_index, start, pending, build_start);
                }
                anchor_cursor += 1;
                if anchor_cursor == self.anchors.len() {
                    return self.finish_candidate(start_index, start, pending, build_start);
                }
            }

            if pending.planned_len > self.max_uncompressed_bytes {
                return self.finish_or_oversized(start_index, start, pending, build_start);
            }
            compressed_offset = compressed_offset
                .checked_add(compressed_len as u64)
                .ok_or_else(|| BatchError::new("compressed BGZF offset overflows u64"))?;
        }
    }

    fn finish_or_oversized(
        &mut self,
        start_index: usize,
        start: VirtualOffset,
        mut pending: PendingBatch,
        build_start: Instant,
    ) -> Result<Option<IndexedBamBatch>, BatchError> {
        if pending.candidate.is_some() {
            self.finish_candidate(start_index, start, pending, build_start)
        } else {
            self.complete_pending(&mut pending)?;
            Err(BatchError::new(format!(
                "the BAM span beginning at virtual offset {} exceeds the configured {}-byte batch cap before the next BAI anchor",
                start.raw(), self.max_uncompressed_bytes
            )))
        }
    }

    fn finish_candidate(
        &mut self,
        start_index: usize,
        start: VirtualOffset,
        mut pending: PendingBatch,
        build_start: Instant,
    ) -> Result<Option<IndexedBamBatch>, BatchError> {
        self.complete_pending(&mut pending)?;
        let (end_index, mapped_index, end_relative) = pending
            .candidate
            .ok_or_else(|| BatchError::new("internal missing terminal candidate for BAM batch"))?;
        if end_index <= start_index || end_relative == 0 {
            return Err(BatchError::new(format!(
                "empty physical BAM span between virtual offsets {} and {}",
                start.raw(),
                self.anchors[end_index].raw()
            )));
        }
        pending.data.truncate(end_relative);
        let end_relative_u32 =
            u32::try_from(end_relative).map_err(|_| BatchError::new("batch end exceeds u32"))?;
        pending
            .block_map
            .retain(|mapping| mapping.batch_start < end_relative_u32);
        if let Some(last) = pending.block_map.last_mut() {
            if last.batch_end > end_relative_u32 {
                let retained = end_relative_u32 - last.batch_start;
                last.batch_end = end_relative_u32;
                last.source_uncompressed_end = last.source_uncompressed_start + retained;
            }
        }

        let mut span_starts = Vec::with_capacity(end_index - start_index);
        for &(anchor_index, relative) in &pending.mapped[..mapped_index] {
            let expected = start_index + span_starts.len();
            if anchor_index != expected {
                return Err(BatchError::new("internal noncontiguous anchor mapping"));
            }
            span_starts.push(
                u32::try_from(relative)
                    .map_err(|_| BatchError::new("batch-relative span offset exceeds u32"))?,
            );
        }
        if span_starts.first() != Some(&0) || span_starts.len() != end_index - start_index {
            return Err(BatchError::new("internal invalid span-start table"));
        }
        self.next_anchor = end_index;
        Ok(Some(IndexedBamBatch {
            data: pending.data,
            span_starts,
            virtual_start: start,
            virtual_end: self.anchors[end_index],
            block_map: pending.block_map,
            blocks_decompressed: pending.blocks_decompressed,
            compressed_bytes_read: pending.compressed_bytes_read,
            build_time: build_start.elapsed(),
        }))
    }

    fn store_completed(
        pending: &mut PendingBatch,
        completed: CompletedBlock,
    ) -> Result<(), BatchError> {
        let slot = pending.outputs.get_mut(completed.sequence).ok_or_else(|| {
            BatchError::new("BGZF worker returned an invalid batch sequence number")
        })?;
        if slot.is_some() {
            return Err(BatchError::new(
                "BGZF worker returned a duplicate batch sequence number",
            ));
        }
        *slot = Some(completed.block?);
        while pending.next_output < pending.outputs.len()
            && pending.outputs[pending.next_output].is_some()
        {
            let sequence = pending.next_output;
            let block = pending.outputs[sequence]
                .take()
                .expect("checked completed BGZF output");
            let mapping = pending.block_map.get(sequence).ok_or_else(|| {
                BatchError::new("missing block map for completed BGZF worker result")
            })?;
            if block.compressed_offset != mapping.compressed_offset
                || block.data.len() != mapping.source_uncompressed_end as usize
            {
                return Err(BatchError::new(format!(
                    "BGZF worker result {sequence} does not match its planned member"
                )));
            }
            let source = &block.data[mapping.source_uncompressed_start as usize..];
            pending.data.try_reserve(source.len()).map_err(|error| {
                BatchError::new(format!("could not reserve output batch memory: {error}"))
            })?;
            pending.data.extend_from_slice(source);
            pending.next_output += 1;
        }
        Ok(())
    }

    fn complete_pending(&mut self, pending: &mut PendingBatch) -> Result<(), BatchError> {
        while self.workers.has_in_flight() {
            let completed = self.workers.receive()?;
            Self::store_completed(pending, completed)?;
        }
        if pending.next_output != pending.outputs.len() {
            return Err(BatchError::new("missing ordered BGZF worker output"));
        }
        if pending.data.len() != pending.planned_len {
            return Err(BatchError::new(
                "assembled BGZF worker output length does not match the batch plan",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdeflater::{CompressionLvl, Compressor};
    use std::fs;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn bgzf_member(compressor: &mut Compressor, input: &[u8]) -> Vec<u8> {
        let mut gzip = vec![0u8; compressor.gzip_compress_bound(input.len())];
        let compressed_len = compressor.gzip_compress(input, &mut gzip).unwrap();
        gzip.truncate(compressed_len);
        let mut bgzf = Vec::with_capacity(gzip.len() + 8);
        bgzf.extend_from_slice(&gzip[..3]);
        bgzf.push(gzip[3] | 0x04);
        bgzf.extend_from_slice(&gzip[4..10]);
        bgzf.extend_from_slice(&6u16.to_le_bytes());
        bgzf.extend_from_slice(b"BC");
        bgzf.extend_from_slice(&2u16.to_le_bytes());
        bgzf.extend_from_slice(&[0, 0]);
        bgzf.extend_from_slice(&gzip[10..]);
        let bsize = u16::try_from(bgzf.len() - 1).unwrap();
        bgzf[16..18].copy_from_slice(&bsize.to_le_bytes());
        bgzf
    }

    fn fixture() -> (PathBuf, u64) {
        let mut compressor = Compressor::new(CompressionLvl::default());
        let first = bgzf_member(&mut compressor, b"abcdefghij");
        let second_offset = first.len() as u64;
        let second = bgzf_member(&mut compressor, b"klmnopqrst");
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gpugeno-indexed-batch-{}-{id}.bam",
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(&first).unwrap();
        file.write_all(&second).unwrap();
        (path, second_offset)
    }

    #[test]
    fn bounded_batches_share_a_boundary_member_without_losing_bytes() {
        let (path, second) = fixture();
        let anchors = vec![
            VirtualOffset::new(0, 2).unwrap(),
            VirtualOffset::new(0, 6).unwrap(),
            VirtualOffset::new(second, 3).unwrap(),
            VirtualOffset::new(second, 8).unwrap(),
        ];
        let mut baseline = DisjointBamStream::open(&path, anchors.clone(), 12, 1).unwrap();
        let baseline_first = baseline.next_batch().unwrap().unwrap();
        let baseline_second = baseline.next_batch().unwrap().unwrap();
        assert!(baseline.next_batch().unwrap().is_none());
        drop(baseline);

        let mut stream = DisjointBamStream::open(&path, anchors, 12, 4).unwrap();
        let first = stream.next_batch().unwrap().unwrap();
        let second_batch = stream.next_batch().unwrap().unwrap();
        assert!(stream.next_batch().unwrap().is_none());

        assert_eq!(first.data, baseline_first.data);
        assert_eq!(first.span_starts, baseline_first.span_starts);
        assert_eq!(first.virtual_start, baseline_first.virtual_start);
        assert_eq!(first.virtual_end, baseline_first.virtual_end);
        assert_eq!(first.block_map, baseline_first.block_map);
        assert_eq!(
            first.blocks_decompressed,
            baseline_first.blocks_decompressed
        );
        assert_eq!(
            first.compressed_bytes_read,
            baseline_first.compressed_bytes_read
        );
        assert_eq!(second_batch.data, baseline_second.data);
        assert_eq!(second_batch.span_starts, baseline_second.span_starts);
        assert_eq!(second_batch.virtual_start, baseline_second.virtual_start);
        assert_eq!(second_batch.virtual_end, baseline_second.virtual_end);
        assert_eq!(second_batch.block_map, baseline_second.block_map);
        assert_eq!(
            second_batch.blocks_decompressed,
            baseline_second.blocks_decompressed
        );
        assert_eq!(
            second_batch.compressed_bytes_read,
            baseline_second.compressed_bytes_read
        );

        assert_eq!(first.data, b"cdefghijklm");
        assert_eq!(first.span_starts, [0, 4]);
        assert_eq!(second_batch.data, b"nopqr");
        assert_eq!(second_batch.span_starts, [0]);
        assert_eq!(
            first.translate_virtual_offset(VirtualOffset::new(second, 3).unwrap()),
            Some(11)
        );
        assert_eq!(
            second_batch.translate_virtual_offset(VirtualOffset::new(second, 3).unwrap()),
            Some(0)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn worker_decompression_error_is_reported_without_hanging() {
        let mut compressor = Compressor::new(CompressionLvl::default());
        let mut first = bgzf_member(&mut compressor, b"abcdefghij");
        let crc_position = first.len() - 8;
        first[crc_position] ^= 0xff;
        let second_offset = first.len() as u64;
        let second = bgzf_member(&mut compressor, b"klmnopqrst");
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gpugeno-indexed-batch-corrupt-{}-{id}.bam",
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(&first).unwrap();
        file.write_all(&second).unwrap();
        drop(file);

        let anchors = vec![
            VirtualOffset::new(0, 0).unwrap(),
            VirtualOffset::new(second_offset, 0).unwrap(),
        ];
        let mut stream = DisjointBamStream::open(&path, anchors, 1024, 2).unwrap();
        let error = stream.next_batch().unwrap_err();
        assert!(error
            .to_string()
            .contains("libdeflate gzip decompression failed"));
        drop(stream);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_zero_decompression_threads() {
        let (path, second) = fixture();
        let anchors = vec![
            VirtualOffset::new(0, 0).unwrap(),
            VirtualOffset::new(second, 0).unwrap(),
        ];
        let error = DisjointBamStream::open(&path, anchors, 1024, 0)
            .err()
            .expect("zero workers must fail");
        assert!(error.to_string().contains("must be positive"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_an_indivisible_span_larger_than_the_cap() {
        let (path, second) = fixture();
        let anchors = vec![
            VirtualOffset::new(0, 2).unwrap(),
            VirtualOffset::new(0, 6).unwrap(),
            VirtualOffset::new(second, 8).unwrap(),
        ];
        let mut stream = DisjointBamStream::open(&path, anchors, 3, 2).unwrap();
        let error = stream.next_batch().unwrap_err();
        assert!(error.to_string().contains("exceeds the configured"));
        fs::remove_file(path).unwrap();
    }
}
