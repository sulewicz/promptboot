use promptboot_core::sha256::Sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const CANONICAL_SIZE: u64 = 426_762_944;
const CEILING: u64 = 469_762_048;
const SOURCE_BYTES: u64 = 428_730_208;
const SOURCE_SHA256: [u8; 32] = [
    0x76, 0x71, 0xc0, 0xc3, 0x04, 0xe6, 0xce, 0x5a, 0x7f, 0xc5, 0x77, 0xbc, 0xb1, 0x2a, 0xba, 0x01,
    0xe2, 0xc1, 0x55, 0xcc, 0x2e, 0xfd, 0x29, 0xb2, 0x21, 0x3c, 0x95, 0xb1, 0x8e, 0xda, 0xf6, 0xed,
];
const CANONICAL_OFFSETS: [u64; 7] = [
    704, 608_512, 1_981_312, 2_133_248, 3_949_952, 3_952_512, 3_980_480,
];
const CANONICAL_HASHES: [&str; 7] = [
    "6dba0abed184bf48dc2eabb85df22fb1abbe48cd43a1a7e4ed43f9825795922b",
    "ff42d75c5cfcc56b3eb4784e6f7b024ef4c8ed784d592bc19e4d77866dade74d",
    "d25702a9d41b2475a2cc04388091d0af2351b1c0e8c709546e7e9658243ca27f",
    "774198a3f5cbc298c6446fd4e7133e55a3676ac95ef21fcb9e792cb31e6c4c31",
    "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4",
    "8f45ffc679189fec8b4f5ddc89f876400c4cd80f352ed76d401b490d92b45698",
    "61710d62f0c5e616798c8d1a70f9140c3933e4f13a6b230341eb74517d8ba6b6",
];

#[derive(Clone)]
struct SourceTensor {
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    length: u64,
}

struct SourceReader<'a> {
    file: &'a mut File,
    position: u64,
    size: u64,
}

impl<'a> SourceReader<'a> {
    fn read(&mut self, amount: usize) -> Result<Vec<u8>, String> {
        let end = self
            .position
            .checked_add(amount as u64)
            .ok_or_else(|| "source read overflow".to_owned())?;
        if end > self.size {
            return Err("source GGUF is truncated".to_owned());
        }
        let mut bytes = vec![0; amount];
        self.file
            .read_exact(&mut bytes)
            .map_err(|_| "source GGUF short read".to_owned())?;
        self.position = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self.u64()?;
        if length > 16 * 1024 * 1024 {
            return Err("source string exceeds inspector bound".to_owned());
        }
        String::from_utf8(self.read(length as usize)?)
            .map_err(|_| "source contains invalid UTF-8".to_owned())
    }

    fn skip_value(&mut self, dtype: u32) -> Result<(), String> {
        let scalar_size = |kind| match kind {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4 | 5 | 6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        match dtype {
            8 => {
                self.string()?;
            }
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if count > 200_000 {
                    return Err("source array exceeds inspector bound".to_owned());
                }
                for _ in 0..count {
                    if element == 8 {
                        self.string()?;
                    } else if let Some(size) = scalar_size(element) {
                        self.read(size)?;
                    } else {
                        return Err(format!("unsupported source array type {element}"));
                    }
                }
            }
            _ => {
                let size = scalar_size(dtype)
                    .ok_or_else(|| format!("unsupported source value type {dtype}"))?;
                self.read(size)?;
            }
        }
        Ok(())
    }
}

fn source_tensors(file: &mut File, size: u64) -> Result<BTreeMap<String, SourceTensor>, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("cannot seek source: {error}"))?;
    let mut reader = SourceReader {
        file,
        position: 0,
        size,
    };
    if reader.read(4)?.as_slice() != b"GGUF" || reader.u32()? != 3 {
        return Err("source is not little-endian GGUF v3".to_owned());
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count != 291 || metadata_count != 26 {
        return Err("source GGUF counts do not match frozen identity".to_owned());
    }
    let mut keys = BTreeSet::new();
    for _ in 0..metadata_count {
        let key = reader.string()?;
        if !keys.insert(key) {
            return Err("duplicate source metadata key".to_owned());
        }
        let dtype = reader.u32()?;
        reader.skip_value(dtype)?;
    }
    let mut infos = Vec::with_capacity(tensor_count as usize);
    let mut names = BTreeSet::new();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        if !names.insert(name.clone()) {
            return Err("duplicate source tensor name".to_owned());
        }
        let rank = reader.u32()?;
        if !(1..=4).contains(&rank) {
            return Err("bad source tensor rank".to_owned());
        }
        let mut dims = Vec::with_capacity(rank as usize);
        let mut count = 1u64;
        for _ in 0..rank {
            let dim = reader.u64()?;
            if dim == 0 {
                return Err("zero source tensor dimension".to_owned());
            }
            count = count
                .checked_mul(dim)
                .ok_or_else(|| "source tensor dimension overflow".to_owned())?;
            dims.push(dim);
        }
        let dtype = reader.u32()?;
        let relative = reader.u64()?;
        let length = match dtype {
            0 => count.checked_mul(4),
            2 if count % 32 == 0 => count.checked_div(32).and_then(|v| v.checked_mul(18)),
            8 if count % 32 == 0 => count.checked_div(32).and_then(|v| v.checked_mul(34)),
            _ => None,
        }
        .ok_or_else(|| "bad source tensor dtype or length".to_owned())?;
        infos.push((name, dims, dtype, relative, length));
    }
    if reader.position != 5_947_741 {
        return Err("source tensor-info end mismatch".to_owned());
    }
    let data_start = (reader.position + 31) & !31;
    let mut result = BTreeMap::new();
    let mut intervals = Vec::with_capacity(infos.len());
    for (name, dims, dtype, relative, length) in infos {
        let offset = data_start
            .checked_add(relative)
            .ok_or_else(|| "source tensor offset overflow".to_owned())?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| "source tensor end overflow".to_owned())?;
        if end > size {
            return Err("source tensor exceeds source file".to_owned());
        }
        intervals.push((offset, end));
        result.insert(
            name,
            SourceTensor {
                dims,
                dtype,
                offset,
                length,
            },
        );
    }
    intervals.sort();
    if intervals.first().map(|row| row.0) != Some(data_start)
        || intervals.last().map(|row| row.1) != Some(size)
        || intervals.windows(2).any(|pair| pair[0].1 != pair[1].0)
    {
        return Err("source tensors do not span source data exactly".to_owned());
    }
    Ok(result)
}

fn read_at(file: &mut File, offset: u64, length: usize, size: u64) -> Result<Vec<u8>, String> {
    let end = offset
        .checked_add(length as u64)
        .ok_or_else(|| "model read overflow".to_owned())?;
    if end > size {
        return Err(format!("truncated region at {offset}"));
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("cannot seek model: {error}"))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("cannot read model: {error}"))?;
    Ok(bytes)
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn hash_region(file: &mut File, offset: u64, length: u64) -> Result<[u8; 32], String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("cannot seek hash region: {error}"))?;
    let mut remaining = length;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let count = file
            .read(&mut buffer[..wanted])
            .map_err(|error| format!("cannot hash region: {error}"))?;
        if count == 0 {
            return Err(format!(
                "truncated region at {}",
                offset + length - remaining
            ));
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(digest.finish())
}

fn require_zero(file: &mut File, offset: u64, length: u64) -> Result<(), String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("cannot seek padding: {error}"))?;
    let mut remaining = length;
    let mut buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..wanted])
            .map_err(|_| "truncated padding".to_owned())?;
        if buffer[..wanted].iter().any(|byte| *byte != 0) {
            return Err("nonzero byte in padding".to_owned());
        }
        remaining -= wanted as u64;
    }
    Ok(())
}

fn same_region(
    left: &mut File,
    left_offset: u64,
    right: &mut File,
    right_offset: u64,
    length: u64,
) -> Result<bool, String> {
    left.seek(SeekFrom::Start(left_offset))
        .map_err(|error| format!("cannot seek model tensor: {error}"))?;
    right
        .seek(SeekFrom::Start(right_offset))
        .map_err(|error| format!("cannot seek source tensor: {error}"))?;
    let mut remaining = length;
    let mut left_buffer = vec![0; 1024 * 1024];
    let mut right_buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(left_buffer.len() as u64) as usize;
        left.read_exact(&mut left_buffer[..wanted])
            .map_err(|_| "truncated model tensor".to_owned())?;
        right
            .read_exact(&mut right_buffer[..wanted])
            .map_err(|_| "truncated source tensor".to_owned())?;
        if left_buffer[..wanted] != right_buffer[..wanted] {
            return Ok(false);
        }
        remaining -= wanted as u64;
    }
    Ok(true)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

#[cfg(test)]
fn decode_hex(value: &str) -> [u8; 32] {
    let mut result = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => unreachable!(),
        };
        result[index] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    result
}

fn inspector_tensor_rows() -> Vec<(String, Vec<u64>, u32, u16, u16)> {
    let mut rows = vec![(
        String::from("token_embd.weight"),
        vec![896, 151936],
        2,
        u16::MAX,
        1,
    )];
    let roles: [(&str, &[u64], u32, u16); 12] = [
        ("attn_norm.weight", &[896], 0, 10),
        ("ffn_down.weight", &[4864, 896], 2, 11),
        ("ffn_gate.weight", &[896, 4864], 2, 12),
        ("ffn_up.weight", &[896, 4864], 2, 13),
        ("ffn_norm.weight", &[896], 0, 14),
        ("attn_k.bias", &[128], 0, 15),
        ("attn_k.weight", &[896, 128], 2, 16),
        ("attn_output.weight", &[896, 896], 2, 17),
        ("attn_q.bias", &[896], 0, 18),
        ("attn_q.weight", &[896, 896], 2, 19),
        ("attn_v.bias", &[128], 0, 20),
        ("attn_v.weight", &[896, 128], 2, 21),
    ];
    for layer in 0..24u16 {
        for &(suffix, dims, dtype, role) in &roles {
            rows.push((
                format!("blk.{layer}.{suffix}"),
                dims.to_vec(),
                dtype,
                layer,
                role,
            ));
        }
    }
    rows.push((
        String::from("output_norm.weight"),
        vec![896],
        0,
        u16::MAX,
        2,
    ));
    rows.push((
        String::from("output.weight"),
        vec![896, 151936],
        8,
        u16::MAX,
        3,
    ));
    rows
}

fn validate_header(header: &[u8], model_size: u64, content_hash: &[u8; 32]) -> Result<(), String> {
    if header.len() != 256 {
        return Err("truncated header".to_owned());
    }
    if &header[..8] != b"PBTQW25\0" {
        return Err("bad PBTQW25 magic".to_owned());
    }
    for (offset, expected) in [
        (0x08, 1),
        (0x0c, 256),
        (0x10, 0x01020304),
        (0x14, 7),
        (0x18, 291),
        (0x1c, 151936),
        (0x20, 151387),
        (0x24, promptboot_core::CONTEXT_LIMIT),
        (0x28, 24),
        (0x2c, 896),
        (0x30, 4864),
        (0x34, 14),
        (0x38, 2),
        (0x3c, 64),
        (0x40, 151645),
        (0x44, 151643),
        (0x48, 151643),
        (0x4c, 0),
    ] {
        if u32_at(header, offset) != expected {
            return Err(format!("header u32 mismatch at 0x{offset:02x}"));
        }
    }
    for (offset, expected) in [
        (0x50, SOURCE_BYTES),
        (0x58, model_size),
        (0x60, 256),
        (0x68, 448),
        (0x70, 5_947_744),
        (0x78, 422_782_464),
    ] {
        if u64_at(header, offset) != expected {
            return Err(format!("header u64 mismatch at 0x{offset:02x}"));
        }
    }
    if header[0x80..0xa0] != SOURCE_SHA256 {
        return Err("embedded source digest mismatch".to_owned());
    }
    if header[0xa0..0xc0] != *content_hash {
        return Err("content digest mismatch".to_owned());
    }
    if header[0xc0..0x100].iter().any(|byte| *byte != 0) {
        return Err("header reserved bytes are nonzero".to_owned());
    }
    Ok(())
}

#[derive(Clone)]
struct SectionFields {
    id: u32,
    dtype: u32,
    count: u64,
    offset: u64,
    length: u64,
    digest: [u8; 32],
}

fn section_fields(row: &[u8]) -> SectionFields {
    SectionFields {
        id: u32_at(row, 0),
        dtype: u32_at(row, 4),
        count: u64_at(row, 8),
        offset: u64_at(row, 16),
        length: u64_at(row, 24),
        digest: row[32..64].try_into().unwrap(),
    }
}

fn validate_section_layout(
    index: usize,
    fields: &SectionFields,
    previous_end: u64,
    model_size: u64,
) -> Result<u64, String> {
    let expected = [
        (1, 1, 151937, 151937 * 4),
        (2, 2, 1372758, 1372758),
        (3, 2, 151936, 151936),
        (4, 3, 151387, 151387 * 12),
        (5, 2, 2509, 2509),
        (6, 4, 291, 291 * 96),
    ];
    if index < 6 {
        if (fields.id, fields.dtype, fields.count, fields.length) != expected[index] {
            return Err(format!("section {} schema mismatch", index + 1));
        }
    } else if fields.id != 7 || fields.dtype != 2 || fields.count != fields.length {
        return Err("tensor-data section schema mismatch".to_owned());
    }
    let end = fields
        .offset
        .checked_add(fields.length)
        .ok_or_else(|| format!("section {} end overflow", fields.id))?;
    if fields.offset % 64 != 0 || fields.offset < previous_end || end > model_size {
        return Err(format!(
            "section {} offset/alignment/overlap invalid",
            fields.id
        ));
    }
    if fields.offset != CANONICAL_OFFSETS[index] {
        return Err(format!("section {} canonical offset mismatch", fields.id));
    }
    Ok(end)
}

fn validate_section_evidence(
    index: usize,
    fields: &SectionFields,
    actual_hash: &[u8; 32],
    padding_zero: bool,
) -> Result<(), String> {
    if !padding_zero {
        return Err(format!("nonzero padding before section {}", fields.id));
    }
    if fields.digest != *actual_hash {
        return Err(format!("section {} SHA-256 mismatch", fields.id));
    }
    if hex(actual_hash) != CANONICAL_HASHES[index] {
        return Err(format!("section {} canonical SHA-256 mismatch", fields.id));
    }
    Ok(())
}

#[derive(Clone)]
struct TensorFields {
    id: u32,
    layer: u16,
    role: u16,
    dtype: u32,
    rank: u32,
    dims: [u32; 4],
    offset: u64,
    length: u64,
    digest: [u8; 32],
    reserved_zero: bool,
}

fn tensor_fields(row: &[u8]) -> TensorFields {
    TensorFields {
        id: u32_at(row, 0),
        layer: u16_at(row, 4),
        role: u16_at(row, 6),
        dtype: u32_at(row, 8),
        rank: u32_at(row, 12),
        dims: [
            u32_at(row, 16),
            u32_at(row, 20),
            u32_at(row, 24),
            u32_at(row, 28),
        ],
        offset: u64_at(row, 32),
        length: u64_at(row, 40),
        digest: row[48..80].try_into().unwrap(),
        reserved_zero: row[80..96].iter().all(|byte| *byte == 0),
    }
}

fn validate_tensor_layout(
    tensor_id: usize,
    fields: &TensorFields,
    expected_dims: &[u64],
    source_dtype: u32,
    expected_layer: u16,
    expected_role: u16,
    source_tensor: &SourceTensor,
    previous_end: u64,
) -> Result<(), String> {
    let mapped_dtype = match source_dtype {
        0 => 1,
        2 => 2,
        8 => 3,
        _ => return Err("inspector dtype table error".to_owned()),
    };
    let mut padded = [1u32; 4];
    for (slot, dimension) in padded.iter_mut().zip(expected_dims) {
        *slot = u32::try_from(*dimension)
            .map_err(|_| format!("expected tensor dimension exceeds u32 id={tensor_id}"))?;
    }
    if (
        fields.id,
        fields.layer,
        fields.role,
        fields.dtype,
        fields.rank,
        fields.dims,
    ) != (
        tensor_id as u32,
        expected_layer,
        expected_role,
        mapped_dtype,
        expected_dims.len() as u32,
        padded,
    ) {
        return Err(format!("tensor entry schema mismatch id={tensor_id}"));
    }
    if source_tensor.dims != expected_dims || source_tensor.dtype != source_dtype {
        return Err(format!("source tensor schema mismatch id={tensor_id}"));
    }
    if fields.length != source_tensor.length
        || fields.offset % 64 != 0
        || fields.offset < previous_end
    {
        return Err(format!("tensor layout mismatch id={tensor_id}"));
    }
    if !fields.reserved_zero {
        return Err(format!("tensor reserved bytes nonzero id={tensor_id}"));
    }
    Ok(())
}

fn validate_tensor_evidence(
    tensor_id: usize,
    fields: &TensorFields,
    actual_hash: &[u8; 32],
    padding_zero: bool,
    source_equal: bool,
) -> Result<(), String> {
    if !padding_zero {
        return Err(format!("nonzero padding before tensor {tensor_id}"));
    }
    if fields.digest != *actual_hash {
        return Err(format!("tensor digest mismatch id={tensor_id}"));
    }
    if !source_equal {
        return Err(format!("tensor payload differs from source id={tensor_id}"));
    }
    Ok(())
}

#[derive(Clone)]
struct SectionReport {
    count: u64,
    length: u64,
    name: &'static str,
    offset: u64,
    sha256: String,
}

#[derive(Clone)]
struct TensorReport {
    id: usize,
    name: String,
    sha256: String,
}

pub struct InspectionReport {
    pub file_sha256: String,
    pub file_size: u64,
    pub tensor_count: usize,
    pub tensor_hash_manifest_sha256: String,
    content_sha256: String,
    source_sha256: String,
    sections: Vec<SectionReport>,
    tensors: Vec<TensorReport>,
    section_payloads: u64,
    inter_section_padding: u64,
}

fn json_string(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 2);
    result.push('"');
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            character if character < ' ' => {
                result.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => result.push(character),
        }
    }
    result.push('"');
    result
}

impl InspectionReport {
    pub fn json(&self) -> String {
        let mut result = format!(
            concat!(
                "{{\"accounting\":{{\"header\":256,\"inter_section_padding\":{},",
                "\"section_directory\":448,\"section_payloads\":{}}},",
                "\"content_sha256\":\"{}\",\"file_sha256\":\"{}\",\"file_size\":{},",
                "\"result\":\"PASS\",\"sections\":["
            ),
            self.inter_section_padding,
            self.section_payloads,
            self.content_sha256,
            self.file_sha256,
            self.file_size
        );
        for (index, section) in self.sections.iter().enumerate() {
            if index != 0 {
                result.push(',');
            }
            result.push_str(&format!(
                "{{\"count\":{},\"length\":{},\"name\":{},\"offset\":{},\"sha256\":\"{}\"}}",
                section.count,
                section.length,
                json_string(section.name),
                section.offset,
                section.sha256
            ));
        }
        result.push_str(&format!(
            "],\"source_sha256\":\"{}\",\"tensor_count\":{},\"tensor_hash_manifest_sha256\":\"{}\",\"tensor_sha256\":[",
            self.source_sha256, self.tensor_count, self.tensor_hash_manifest_sha256
        ));
        for (index, tensor) in self.tensors.iter().enumerate() {
            if index != 0 {
                result.push(',');
            }
            result.push_str(&format!(
                "{{\"id\":{},\"name\":{},\"sha256\":\"{}\"}}",
                tensor.id,
                json_string(&tensor.name),
                tensor.sha256
            ));
        }
        result.push_str("]}\n");
        result
    }
}

pub fn inspect(model_path: &Path, source_path: &Path) -> Result<InspectionReport, String> {
    let model_size = model_path
        .metadata()
        .map_err(|error| format!("cannot stat model: {error}"))?
        .len();
    let source_size = source_path
        .metadata()
        .map_err(|error| format!("cannot stat source: {error}"))?
        .len();
    if model_size > CEILING {
        return Err(format!("container size {model_size} exceeds {CEILING}"));
    }
    if source_size != SOURCE_BYTES {
        return Err("source size mismatch".to_owned());
    }
    let mut model =
        File::open(model_path).map_err(|error| format!("cannot open model: {error}"))?;
    let mut source =
        File::open(source_path).map_err(|error| format!("cannot open source: {error}"))?;
    if hash_region(&mut source, 0, source_size)? != SOURCE_SHA256 {
        return Err("source SHA-256 mismatch".to_owned());
    }
    let source_map = source_tensors(&mut source, source_size)?;
    let header = read_at(&mut model, 0, 256, model_size)?;
    let content_hash = hash_region(&mut model, 256, model_size.saturating_sub(256))?;
    validate_header(&header, model_size, &content_hash)?;
    let directory = read_at(&mut model, 256, 448, model_size)?;
    let names = [
        "token_offsets",
        "token_bytes",
        "token_types",
        "merge_rules",
        "chat_template",
        "tensor_directory",
        "tensor_data",
    ];
    let mut sections = Vec::with_capacity(7);
    let mut previous_end = 704u64;
    for index in 0..7 {
        let row = &directory[index * 64..(index + 1) * 64];
        let fields = section_fields(row);
        let end = validate_section_layout(index, &fields, previous_end, model_size)?;
        let padding_zero =
            require_zero(&mut model, previous_end, fields.offset - previous_end).is_ok();
        let actual = hash_region(&mut model, fields.offset, fields.length)?;
        validate_section_evidence(index, &fields, &actual, padding_zero)?;
        sections.push(SectionReport {
            count: fields.count,
            length: fields.length,
            name: names[index],
            offset: fields.offset,
            sha256: hex(&actual),
        });
        previous_end = end;
    }
    if previous_end != model_size {
        return Err("trailing bytes after tensor-data section".to_owned());
    }

    let tensor_directory = &sections[5];
    let tensor_bytes = read_at(
        &mut model,
        tensor_directory.offset,
        tensor_directory.length as usize,
        model_size,
    )?;
    let expected_rows = inspector_tensor_rows();
    let mut previous_tensor_end = sections[6].offset;
    let mut tensor_reports = Vec::with_capacity(291);
    let mut manifest_digest = Sha256::new();
    for (tensor_id, (name, expected_dims, source_dtype, expected_layer, expected_role)) in
        expected_rows.into_iter().enumerate()
    {
        let row = &tensor_bytes[tensor_id * 96..(tensor_id + 1) * 96];
        let fields = tensor_fields(row);
        let source_tensor = source_map
            .get(&name)
            .ok_or_else(|| format!("source tensor missing: {name}"))?;
        validate_tensor_layout(
            tensor_id,
            &fields,
            &expected_dims,
            source_dtype,
            expected_layer,
            expected_role,
            source_tensor,
            previous_tensor_end,
        )?;
        let padding_zero = require_zero(
            &mut model,
            previous_tensor_end,
            fields.offset - previous_tensor_end,
        )
        .is_ok();
        let actual_hash = hash_region(&mut model, fields.offset, fields.length)?;
        let source_equal = same_region(
            &mut model,
            fields.offset,
            &mut source,
            source_tensor.offset,
            fields.length,
        )?;
        validate_tensor_evidence(tensor_id, &fields, &actual_hash, padding_zero, source_equal)?;
        manifest_digest.update(&actual_hash);
        tensor_reports.push(TensorReport {
            id: tensor_id,
            name,
            sha256: hex(&actual_hash),
        });
        previous_tensor_end = fields.offset + fields.length;
    }
    if previous_tensor_end != model_size {
        return Err("tensor directory does not account for final file byte".to_owned());
    }
    if model_size != CANONICAL_SIZE {
        return Err(format!("canonical container size mismatch: {model_size}"));
    }
    let section_payloads: u64 = sections.iter().map(|section| section.length).sum();
    let inter_section_padding: u64 = sections
        .iter()
        .enumerate()
        .map(|(index, section)| {
            section.offset
                - if index == 0 {
                    704
                } else {
                    sections[index - 1].offset + sections[index - 1].length
                }
        })
        .sum();
    if 256 + 448 + section_payloads + inter_section_padding != model_size {
        return Err("size accounting does not sum to file size".to_owned());
    }
    Ok(InspectionReport {
        file_sha256: hex(&hash_region(&mut model, 0, model_size)?),
        file_size: model_size,
        tensor_count: tensor_reports.len(),
        tensor_hash_manifest_sha256: hex(&manifest_digest.finish()),
        content_sha256: hex(&content_hash),
        source_sha256: hex(&SOURCE_SHA256),
        sections,
        tensors: tensor_reports,
        section_payloads,
        inter_section_padding,
    })
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create report directory: {error}"))?;
    }
    let name = path
        .file_name()
        .ok_or_else(|| "report output must name a file".to_owned())?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{name}.tmp"));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)
            .map_err(|error| format!("cannot create report: {error}"))?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("cannot write report: {error}"))?;
        fs::rename(&temporary, path).map_err(|error| format!("cannot publish report: {error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn valid_header(content_hash: [u8; 32]) -> Vec<u8> {
        let mut header = vec![0; 256];
        header[..8].copy_from_slice(b"PBTQW25\0");
        for (offset, value) in [
            (0x08, 1),
            (0x0c, 256),
            (0x10, 0x01020304),
            (0x14, 7),
            (0x18, 291),
            (0x1c, 151936),
            (0x20, 151387),
            (0x24, promptboot_core::CONTEXT_LIMIT),
            (0x28, 24),
            (0x2c, 896),
            (0x30, 4864),
            (0x34, 14),
            (0x38, 2),
            (0x3c, 64),
            (0x40, 151645),
            (0x44, 151643),
            (0x48, 151643),
            (0x4c, 0),
        ] {
            put_u32(&mut header, offset, value);
        }
        for (offset, value) in [
            (0x50, SOURCE_BYTES),
            (0x58, CANONICAL_SIZE),
            (0x60, 256),
            (0x68, 448),
            (0x70, 5_947_744),
            (0x78, 422_782_464),
        ] {
            put_u64(&mut header, offset, value);
        }
        header[0x80..0xa0].copy_from_slice(&SOURCE_SHA256);
        header[0xa0..0xc0].copy_from_slice(&content_hash);
        header
    }

    #[test]
    fn mos_header_mutation_matrix_covers_identity_and_hash_contracts() {
        let content_hash = [0x5a; 32];
        let valid = valid_header(content_hash);
        validate_header(&valid, CANONICAL_SIZE, &content_hash).unwrap();
        let mut cases = Vec::new();
        let mut magic = valid.clone();
        magic[0] ^= 1;
        cases.push(("magic", magic, "magic"));
        let mut version = valid.clone();
        put_u32(&mut version, 0x08, 2);
        cases.push(("version", version, "u32"));
        let mut size = valid.clone();
        put_u64(&mut size, 0x58, CANONICAL_SIZE - 1);
        cases.push(("size", size, "u64"));
        let mut source = valid.clone();
        source[0x80] ^= 1;
        cases.push(("source identity", source, "source digest"));
        let mut content = valid.clone();
        content[0xa0] ^= 1;
        cases.push(("content hash", content, "content digest"));
        let mut reserved = valid.clone();
        reserved[0xc0] = 1;
        cases.push(("reserved", reserved, "reserved"));
        for (name, header, expected) in cases {
            assert!(
                validate_header(&header, CANONICAL_SIZE, &content_hash)
                    .unwrap_err()
                    .contains(expected),
                "{name}"
            );
        }
    }

    #[test]
    fn mos_section_mutation_matrix_covers_schema_layout_and_hashes() {
        let canonical_hash = decode_hex(CANONICAL_HASHES[0]);
        let valid = SectionFields {
            id: 1,
            dtype: 1,
            count: 151937,
            offset: 704,
            length: 151937 * 4,
            digest: canonical_hash,
        };
        validate_section_layout(0, &valid, 704, CANONICAL_SIZE).unwrap();
        validate_section_evidence(0, &valid, &canonical_hash, true).unwrap();

        let mut layout_cases = Vec::new();
        let mut id = valid.clone();
        id.id = 2;
        layout_cases.push(("id", id, "schema"));
        let mut count = valid.clone();
        count.count -= 1;
        layout_cases.push(("count", count, "schema"));
        let mut alignment = valid.clone();
        alignment.offset += 1;
        layout_cases.push(("alignment", alignment, "alignment"));
        let mut overlap = valid.clone();
        overlap.offset = 640;
        layout_cases.push(("overlap", overlap, "overlap"));
        for (name, fields, expected) in layout_cases {
            assert!(
                validate_section_layout(0, &fields, 704, CANONICAL_SIZE)
                    .unwrap_err()
                    .contains(expected),
                "{name}"
            );
        }

        let mut digest = valid.clone();
        digest.digest[0] ^= 1;
        let noncanonical = [0; 32];
        let evidence_cases = [
            (
                "directory hash",
                validate_section_evidence(0, &digest, &canonical_hash, true),
                "SHA-256 mismatch",
            ),
            (
                "payload hash",
                validate_section_evidence(0, &valid, &noncanonical, true),
                "SHA-256 mismatch",
            ),
            (
                "padding",
                validate_section_evidence(0, &valid, &canonical_hash, false),
                "padding",
            ),
        ];
        for (name, result, expected) in evidence_cases {
            assert!(result.unwrap_err().contains(expected), "{name}");
        }
    }

    #[test]
    fn mos_tensor_mutation_matrix_covers_schema_hash_and_source_equality() {
        let length = 896 * 151936 / 32 * 18;
        let source = SourceTensor {
            dims: vec![896, 151936],
            dtype: 2,
            offset: 5_947_744,
            length,
        };
        let valid = TensorFields {
            id: 0,
            layer: u16::MAX,
            role: 1,
            dtype: 2,
            rank: 2,
            dims: [896, 151936, 1, 1],
            offset: 3_980_480,
            length,
            digest: [0x33; 32],
            reserved_zero: true,
        };
        validate_tensor_layout(
            0,
            &valid,
            &[896, 151936],
            2,
            u16::MAX,
            1,
            &source,
            3_980_480,
        )
        .unwrap();
        validate_tensor_evidence(0, &valid, &[0x33; 32], true, true).unwrap();

        let mut layout_cases = Vec::new();
        let mut id = valid.clone();
        id.id = 1;
        layout_cases.push(("id", id, source.clone(), "schema"));
        let mut dims = valid.clone();
        dims.dims[0] += 1;
        layout_cases.push(("dimensions", dims, source.clone(), "schema"));
        let mut dtype = valid.clone();
        dtype.dtype = 1;
        layout_cases.push(("dtype", dtype, source.clone(), "schema"));
        let mut offset = valid.clone();
        offset.offset += 1;
        layout_cases.push(("alignment", offset, source.clone(), "layout"));
        let mut reserved = valid.clone();
        reserved.reserved_zero = false;
        layout_cases.push(("reserved", reserved, source.clone(), "reserved"));
        let mut source_schema = source.clone();
        source_schema.dtype = 0;
        layout_cases.push(("source schema", valid.clone(), source_schema, "source"));
        for (name, fields, source, expected) in layout_cases {
            assert!(
                validate_tensor_layout(
                    0,
                    &fields,
                    &[896, 151936],
                    2,
                    u16::MAX,
                    1,
                    &source,
                    3_980_480,
                )
                .unwrap_err()
                .contains(expected),
                "{name}"
            );
        }

        let mut digest = valid.clone();
        digest.digest[0] ^= 1;
        let evidence_cases = [
            (
                "directory hash",
                validate_tensor_evidence(0, &digest, &[0x33; 32], true, true),
                "digest",
            ),
            (
                "padding",
                validate_tensor_evidence(0, &valid, &[0x33; 32], false, true),
                "padding",
            ),
            (
                "source payload",
                validate_tensor_evidence(0, &valid, &[0x33; 32], true, false),
                "source",
            ),
        ];
        for (name, result, expected) in evidence_cases {
            assert!(result.unwrap_err().contains(expected), "{name}");
        }
    }

    #[test]
    fn inspector_owns_its_frozen_tensor_schema() {
        let rows = inspector_tensor_rows();
        assert_eq!(rows.len(), 291);
        assert_eq!(rows.first().unwrap().0, "token_embd.weight");
        assert_eq!(rows.last().unwrap().0, "output.weight");
    }

    #[test]
    fn json_strings_escape_control_and_delimiter_bytes() {
        assert_eq!(json_string("a\"\\\n"), "\"a\\\"\\\\\\n\"");
    }

    #[test]
    fn padding_check_rejects_nonzero_and_truncation() {
        let path = std::env::temp_dir().join(format!(
            "promptboot-tools-padding-test-{}",
            std::process::id()
        ));
        fs::write(&path, [0, 0, 1, 0]).unwrap();
        let mut file = File::open(&path).unwrap();
        assert!(require_zero(&mut file, 0, 2).is_ok());
        assert!(require_zero(&mut file, 0, 4)
            .unwrap_err()
            .contains("nonzero"));
        assert!(require_zero(&mut file, 3, 2)
            .unwrap_err()
            .contains("truncated"));
        fs::remove_file(path).unwrap();
    }
}
