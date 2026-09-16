//! Minimal BGZF framing and bounded prefix decompression for the upload spike.
//!
//! This module deliberately builds one batch. It is not a general streaming
//! abstraction and does not interpret BAM records.

use libdeflater::Decompressor;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::{Duration, Instant};

const GZIP_FIXED_HEADER_LEN: usize = 10;
const GZIP_TRAILER_LEN: usize = 8;
const BGZF_MAX_MEMBER_LEN: usize = 65_536;
const BGZF_MAX_ISIZE: usize = 65_536;
const FLAG_FHCRC: u8 = 0x02;
const FLAG_FEXTRA: u8 = 0x04;
const FLAG_FNAME: u8 = 0x08;
const FLAG_FCOMMENT: u8 = 0x10;
const FLAG_RESERVED: u8 = 0xe0;

/// The canonical 28-byte empty BGZF end-of-file marker.
const BGZF_EOF: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// One bounded, contiguous decompressed BGZF prefix.
#[derive(Debug)]
pub struct BgzfBatch {
    /// Member outputs concatenated in file order.
    pub data: Vec<u8>,
    /// Number of included BGZF members (the canonical EOF marker is excluded).
    pub blocks: usize,
    /// Sum of complete compressed-member lengths included in this batch.
    pub compressed_bytes: u64,
    /// Wall time spent specifically inside `Decompressor::gzip_decompress`.
    pub inflate_time: Duration,
}

/// An input, framing, allocation, or decompression error with path and offset
/// context where applicable.
#[derive(Debug)]
pub struct BgzfError {
    message: String,
}

impl BgzfError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn at(path: &Path, offset: u64, message: impl std::fmt::Display) -> Self {
        Self::new(format!(
            "{} at compressed offset {offset}: {message}",
            path.display()
        ))
    }
}

impl std::fmt::Display for BgzfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BgzfError {}

/// Reads and decompresses complete BGZF members from the start of `path`.
///
/// A member is included only when all of its uncompressed output fits within
/// `max_uncompressed_bytes`. The function stops before the first member that
/// would exceed that cap. One libdeflate decompressor is reused for the whole
/// batch, and every included member is passed to its gzip decoder as a complete
/// gzip member so framing and CRC validation remain enabled.
pub fn read_bgzf_prefix(
    path: &Path,
    max_uncompressed_bytes: usize,
) -> Result<BgzfBatch, BgzfError> {
    if max_uncompressed_bytes == 0 {
        return Err(BgzfError::new(
            "max_uncompressed_bytes must be greater than zero",
        ));
    }

    let file = File::open(path)
        .map_err(|error| BgzfError::new(format!("failed to open {}: {error}", path.display())))?;
    read_bgzf_prefix_from(BufReader::new(file), path, max_uncompressed_bytes)
}

fn read_bgzf_prefix_from<R: Read>(
    mut reader: R,
    path: &Path,
    max_uncompressed_bytes: usize,
) -> Result<BgzfBatch, BgzfError> {
    let mut decompressor = Decompressor::new();
    let mut data = Vec::new();
    let mut blocks = 0usize;
    let mut compressed_bytes = 0u64;
    let mut inflate_time = Duration::ZERO;
    let mut offset = 0u64;

    while let Some(member) = read_member(&mut reader, path, offset)? {
        // The canonical marker terminates a BAM even if bytes happen to follow
        // it. It is a sentinel, not part of the uploaded data batch.
        if member.bytes == BGZF_EOF {
            break;
        }

        let next_len = data
            .len()
            .checked_add(member.uncompressed_len)
            .ok_or_else(|| BgzfError::at(path, offset, "cumulative output size overflows usize"))?;
        if next_len > max_uncompressed_bytes {
            if blocks == 0 {
                return Err(BgzfError::at(
                    path,
                    offset,
                    format!(
                        "first BGZF member expands to {} bytes, exceeding the configured cap of {} bytes",
                        member.uncompressed_len, max_uncompressed_bytes
                    ),
                ));
            }
            break;
        }

        let mut output = vec![0u8; member.uncompressed_len];
        let inflate_start = Instant::now();
        let written = decompressor
            .gzip_decompress(&member.bytes, &mut output)
            .map_err(|error| {
                BgzfError::at(
                    path,
                    offset,
                    format!("libdeflate gzip decompression failed: {error}"),
                )
            })?;
        inflate_time += inflate_start.elapsed();
        if written != member.uncompressed_len {
            return Err(BgzfError::at(
                path,
                offset,
                format!(
                    "libdeflate returned {written} bytes but the gzip trailer ISIZE is {}",
                    member.uncompressed_len
                ),
            ));
        }

        data.try_reserve_exact(output.len()).map_err(|error| {
            BgzfError::at(
                path,
                offset,
                format!("could not reserve output batch memory: {error}"),
            )
        })?;
        data.append(&mut output);
        blocks = blocks.checked_add(1).ok_or_else(|| {
            BgzfError::at(path, offset, "included BGZF member count overflows usize")
        })?;
        let member_len_u64 = u64::try_from(member.bytes.len()).map_err(|_| {
            BgzfError::at(path, offset, "compressed member length does not fit u64")
        })?;
        compressed_bytes = compressed_bytes
            .checked_add(member_len_u64)
            .ok_or_else(|| {
                BgzfError::at(path, offset, "cumulative compressed size overflows u64")
            })?;
        offset = offset
            .checked_add(member_len_u64)
            .ok_or_else(|| BgzfError::at(path, offset, "compressed file offset overflows u64"))?;
    }

    if blocks == 0 {
        return Err(BgzfError::new(format!(
            "{} contains no non-EOF BGZF member for the batch",
            path.display()
        )));
    }

    Ok(BgzfBatch {
        data,
        blocks,
        compressed_bytes,
        inflate_time,
    })
}

struct Member {
    bytes: Vec<u8>,
    uncompressed_len: usize,
}

fn read_member<R: Read>(
    reader: &mut R,
    path: &Path,
    offset: u64,
) -> Result<Option<Member>, BgzfError> {
    let mut fixed = [0u8; GZIP_FIXED_HEADER_LEN];
    if !read_first_byte(reader, &mut fixed[0], path, offset)? {
        return Ok(None);
    }
    read_exact_at(reader, &mut fixed[1..], path, offset, "fixed gzip header")?;

    if fixed[0..2] != [0x1f, 0x8b] {
        return Err(BgzfError::at(path, offset, "invalid gzip magic bytes"));
    }
    if fixed[2] != 8 {
        return Err(BgzfError::at(
            path,
            offset,
            format!(
                "unsupported gzip compression method {} (expected 8)",
                fixed[2]
            ),
        ));
    }
    let flags = fixed[3];
    if flags & FLAG_RESERVED != 0 {
        return Err(BgzfError::at(
            path,
            offset,
            format!("gzip reserved flag bits are set (FLG=0x{flags:02x})"),
        ));
    }
    if flags & FLAG_FEXTRA == 0 {
        return Err(BgzfError::at(
            path,
            offset,
            "gzip FEXTRA flag is not set; input is not BGZF",
        ));
    }

    let mut xlen_bytes = [0u8; 2];
    read_exact_at(reader, &mut xlen_bytes, path, offset, "gzip XLEN")?;
    let xlen = usize::from(u16::from_le_bytes(xlen_bytes));
    let mut extra = vec![0u8; xlen];
    read_exact_at(reader, &mut extra, path, offset, "gzip extra fields")?;
    let bsize = parse_bsize(&extra, path, offset)?;
    let member_len = usize::from(bsize)
        .checked_add(1)
        .ok_or_else(|| BgzfError::at(path, offset, "BGZF BSIZE overflows"))?;
    if member_len > BGZF_MAX_MEMBER_LEN {
        return Err(BgzfError::at(
            path,
            offset,
            format!("BGZF member length {member_len} exceeds 65536 bytes"),
        ));
    }

    let prefix_len = GZIP_FIXED_HEADER_LEN
        .checked_add(2)
        .and_then(|length| length.checked_add(xlen))
        .ok_or_else(|| BgzfError::at(path, offset, "gzip header length overflows usize"))?;
    let minimum_len = prefix_len
        .checked_add(GZIP_TRAILER_LEN)
        .ok_or_else(|| BgzfError::at(path, offset, "minimum member length overflows usize"))?;
    if member_len < minimum_len {
        return Err(BgzfError::at(
            path,
            offset,
            format!(
                "BGZF member length {member_len} is too small for its {prefix_len}-byte header and gzip trailer"
            ),
        ));
    }

    let mut bytes = Vec::with_capacity(member_len);
    bytes.extend_from_slice(&fixed);
    bytes.extend_from_slice(&xlen_bytes);
    bytes.extend_from_slice(&extra);
    let remaining = member_len - bytes.len();
    let current_len = bytes.len();
    bytes.resize(member_len, 0);
    read_exact_at(
        reader,
        &mut bytes[current_len..current_len + remaining],
        path,
        offset,
        "complete BGZF member",
    )?;

    validate_optional_header(flags, &bytes, prefix_len, path, offset)?;
    let trailer_start = member_len - GZIP_TRAILER_LEN;
    let isize_start = trailer_start + 4;
    let uncompressed_len = u32::from_le_bytes([
        bytes[isize_start],
        bytes[isize_start + 1],
        bytes[isize_start + 2],
        bytes[isize_start + 3],
    ]) as usize;
    if uncompressed_len > BGZF_MAX_ISIZE {
        return Err(BgzfError::at(
            path,
            offset,
            format!("gzip trailer ISIZE {uncompressed_len} exceeds the BGZF limit of 65536"),
        ));
    }

    Ok(Some(Member {
        bytes,
        uncompressed_len,
    }))
}

fn parse_bsize(extra: &[u8], path: &Path, offset: u64) -> Result<u16, BgzfError> {
    let mut position = 0usize;
    let mut bsize = None;
    while position < extra.len() {
        let header_end = position.checked_add(4).ok_or_else(|| {
            BgzfError::at(path, offset, "gzip extra-subfield header offset overflows")
        })?;
        if header_end > extra.len() {
            return Err(BgzfError::at(
                path,
                offset,
                "truncated gzip extra-subfield header",
            ));
        }
        let subfield_len = usize::from(u16::from_le_bytes([
            extra[position + 2],
            extra[position + 3],
        ]));
        let data_end = header_end
            .checked_add(subfield_len)
            .ok_or_else(|| BgzfError::at(path, offset, "gzip extra-subfield length overflows"))?;
        if data_end > extra.len() {
            return Err(BgzfError::at(
                path,
                offset,
                "gzip extra subfield extends beyond XLEN",
            ));
        }

        if extra[position] == b'B' && extra[position + 1] == b'C' {
            if subfield_len != 2 {
                return Err(BgzfError::at(
                    path,
                    offset,
                    format!("BGZF BC subfield has SLEN {subfield_len}, expected 2"),
                ));
            }
            if bsize.is_some() {
                return Err(BgzfError::at(
                    path,
                    offset,
                    "duplicate BGZF BC extra subfield",
                ));
            }
            bsize = Some(u16::from_le_bytes([
                extra[header_end],
                extra[header_end + 1],
            ]));
        }
        position = data_end;
    }

    bsize.ok_or_else(|| BgzfError::at(path, offset, "missing BGZF BC extra subfield"))
}

fn validate_optional_header(
    flags: u8,
    bytes: &[u8],
    mut position: usize,
    path: &Path,
    offset: u64,
) -> Result<(), BgzfError> {
    let trailer_start = bytes.len() - GZIP_TRAILER_LEN;
    for (flag, name) in [(FLAG_FNAME, "FNAME"), (FLAG_FCOMMENT, "FCOMMENT")] {
        if flags & flag != 0 {
            let relative_end = bytes[position..trailer_start]
                .iter()
                .position(|&byte| byte == 0)
                .ok_or_else(|| {
                    BgzfError::at(path, offset, format!("unterminated gzip {name} field"))
                })?;
            position = position
                .checked_add(relative_end + 1)
                .ok_or_else(|| BgzfError::at(path, offset, "gzip header offset overflows"))?;
        }
    }
    if flags & FLAG_FHCRC != 0 {
        position = position
            .checked_add(2)
            .ok_or_else(|| BgzfError::at(path, offset, "gzip FHCRC offset overflows"))?;
        if position > trailer_start {
            return Err(BgzfError::at(
                path,
                offset,
                "gzip FHCRC extends into the trailer",
            ));
        }
    }
    if position > trailer_start {
        return Err(BgzfError::at(
            path,
            offset,
            "parsed gzip header extends into the trailer",
        ));
    }
    Ok(())
}

fn read_first_byte<R: Read>(
    reader: &mut R,
    byte: &mut u8,
    path: &Path,
    offset: u64,
) -> Result<bool, BgzfError> {
    loop {
        match reader.read(std::slice::from_mut(byte)) {
            Ok(0) => return Ok(false),
            Ok(1) => return Ok(true),
            Ok(_) => unreachable!("a one-byte read returned more than one byte"),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(BgzfError::at(
                    path,
                    offset,
                    format!("failed to read gzip header: {error}"),
                ));
            }
        }
    }
}

fn read_exact_at<R: Read>(
    reader: &mut R,
    buffer: &mut [u8],
    path: &Path,
    offset: u64,
    what: &str,
) -> Result<(), BgzfError> {
    reader.read_exact(buffer).map_err(|error| {
        let detail = if error.kind() == io::ErrorKind::UnexpectedEof {
            format!("truncated while reading {what}")
        } else {
            format!("failed while reading {what}: {error}")
        };
        BgzfError::at(path, offset, detail)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdeflater::{CompressionLvl, Compressor};
    use std::io::Cursor;

    fn test_path() -> &'static Path {
        Path::new("synthetic.bam")
    }

    fn bgzf_member(compressor: &mut Compressor, input: &[u8]) -> Vec<u8> {
        let mut gzip = vec![0u8; compressor.gzip_compress_bound(input.len())];
        let compressed_len = compressor.gzip_compress(input, &mut gzip).unwrap();
        gzip.truncate(compressed_len);

        let mut bgzf = Vec::with_capacity(gzip.len() + 8);
        bgzf.extend_from_slice(&gzip[..3]);
        bgzf.push(gzip[3] | FLAG_FEXTRA);
        bgzf.extend_from_slice(&gzip[4..GZIP_FIXED_HEADER_LEN]);
        bgzf.extend_from_slice(&6u16.to_le_bytes());
        bgzf.extend_from_slice(b"BC");
        bgzf.extend_from_slice(&2u16.to_le_bytes());
        bgzf.extend_from_slice(&[0, 0]);
        bgzf.extend_from_slice(&gzip[GZIP_FIXED_HEADER_LEN..]);
        assert!(bgzf.len() <= BGZF_MAX_MEMBER_LEN);
        let bsize = u16::try_from(bgzf.len() - 1).unwrap();
        bgzf[16..18].copy_from_slice(&bsize.to_le_bytes());
        bgzf
    }

    fn compressor() -> Compressor {
        Compressor::new(CompressionLvl::default())
    }

    #[test]
    fn concatenates_multiple_members_in_order() {
        let mut compressor = compressor();
        let first = bgzf_member(&mut compressor, b"BAM-prefix-");
        let second = bgzf_member(&mut compressor, b"then-more-data");
        let expected_compressed = (first.len() + second.len()) as u64;
        let mut input = first;
        input.extend_from_slice(&second);

        let batch = read_bgzf_prefix_from(Cursor::new(input), test_path(), 1024).unwrap();

        assert_eq!(batch.data, b"BAM-prefix-then-more-data");
        assert_eq!(batch.blocks, 2);
        assert_eq!(batch.compressed_bytes, expected_compressed);
    }

    #[test]
    fn cap_stops_before_member_that_would_exceed_it() {
        let mut compressor = compressor();
        let first = bgzf_member(&mut compressor, b"12345");
        let second = bgzf_member(&mut compressor, b"6789");
        let first_compressed = first.len() as u64;
        let mut input = first;
        input.extend_from_slice(&second);

        let batch = read_bgzf_prefix_from(Cursor::new(input), test_path(), 8).unwrap();

        assert_eq!(batch.data, b"12345");
        assert_eq!(batch.blocks, 1);
        assert_eq!(batch.compressed_bytes, first_compressed);
    }

    #[test]
    fn errors_when_first_member_cannot_fit() {
        let mut compressor = compressor();
        let input = bgzf_member(&mut compressor, b"too large");

        let error = read_bgzf_prefix_from(Cursor::new(input), test_path(), 3).unwrap_err();

        assert!(error.to_string().contains("first BGZF member"));
    }

    #[test]
    fn supports_a_65536_byte_member() {
        let mut compressor = compressor();
        let expected = vec![0x5a; BGZF_MAX_ISIZE];
        let input = bgzf_member(&mut compressor, &expected);

        let batch = read_bgzf_prefix_from(Cursor::new(input), test_path(), BGZF_MAX_ISIZE).unwrap();

        assert_eq!(batch.data, expected);
        assert_eq!(batch.blocks, 1);
    }

    #[test]
    fn missing_bc_is_an_error_not_a_panic() {
        let mut compressor = compressor();
        let mut input = bgzf_member(&mut compressor, b"data");
        input[12] = b'X';
        input[13] = b'Y';

        let error = read_bgzf_prefix_from(Cursor::new(input), test_path(), 1024).unwrap_err();

        assert!(error.to_string().contains("missing BGZF BC"));
    }

    #[test]
    fn truncated_member_is_an_error_not_a_panic() {
        let mut compressor = compressor();
        let mut input = bgzf_member(&mut compressor, b"data");
        input.pop();

        let error = read_bgzf_prefix_from(Cursor::new(input), test_path(), 1024).unwrap_err();

        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn canonical_eof_stops_without_reading_following_garbage() {
        let mut compressor = compressor();
        let first = bgzf_member(&mut compressor, b"data");
        let mut input = first;
        input.extend_from_slice(&BGZF_EOF);
        input.extend_from_slice(b"not another member");

        let batch = read_bgzf_prefix_from(Cursor::new(input), test_path(), 1024).unwrap();

        assert_eq!(batch.data, b"data");
        assert_eq!(batch.blocks, 1);
    }
}
