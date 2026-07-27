use promptboot_core::sha256::Sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const SOURCE_BYTES: u64 = 428_730_208;
pub const SOURCE_SHA256: [u8; 32] = [
    0x76, 0x71, 0xc0, 0xc3, 0x04, 0xe6, 0xce, 0x5a, 0x7f, 0xc5, 0x77, 0xbc, 0xb1, 0x2a, 0xba, 0x01,
    0xe2, 0xc1, 0x55, 0xcc, 0x2e, 0xfd, 0x29, 0xb2, 0x21, 0x3c, 0x95, 0xb1, 0x8e, 0xda, 0xf6, 0xed,
];

#[derive(Clone, Debug)]
pub enum Value {
    ScalarU64(u32, u64),
    ScalarString(String),
    ArrayString(Vec<String>),
    ArrayI32(Vec<i32>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, Tensor>,
}

struct Reader<R> {
    inner: R,
    position: u64,
    size: u64,
}

impl<R: Read> Reader<R> {
    fn new(inner: R, size: u64) -> Self {
        Self {
            inner,
            position: 0,
            size,
        }
    }

    fn read(&mut self, length: usize) -> Result<Vec<u8>, String> {
        let end = self
            .position
            .checked_add(length as u64)
            .ok_or_else(|| "GGUF read offset overflow".to_owned())?;
        if end > self.size {
            return Err(format!(
                "read of {length} bytes at {} exceeds bounded input {}",
                self.position, self.size
            ));
        }
        let mut bytes = vec![0; length];
        self.inner
            .read_exact(&mut bytes)
            .map_err(|_| format!("short read at {}", self.position))?;
        self.position = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.read(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }

    fn scalar_bits(&mut self, dtype: u32) -> Result<u64, String> {
        match dtype {
            0 => Ok(self.u8()? as u64),
            1 => Ok(self.u8()? as i8 as i64 as u64),
            2 => Ok(u16::from_le_bytes(self.read(2)?.try_into().unwrap()) as u64),
            3 => Ok(i16::from_le_bytes(self.read(2)?.try_into().unwrap()) as i64 as u64),
            4 => Ok(self.u32()? as u64),
            5 => Ok(self.i32()? as i64 as u64),
            6 => Ok(self.u32()? as u64),
            7 => Ok(self.u8()? as u64),
            10 => self.u64(),
            11 => Ok(i64::from_le_bytes(self.read(8)?.try_into().unwrap()) as u64),
            12 => self.u64(),
            _ => Err(format!("unknown or nonscalar GGUF value type {dtype}")),
        }
    }

    fn string(&mut self, maximum: u64) -> Result<String, String> {
        let length = self.u64()?;
        if length > maximum {
            return Err(format!("string length {length} exceeds {maximum}"));
        }
        String::from_utf8(self.read(length as usize)?).map_err(|error| {
            format!(
                "invalid UTF-8 string at {}",
                error.utf8_error().valid_up_to()
            )
        })
    }
}

fn tensor_length(dims: &[u64], dtype: u32) -> Result<u64, String> {
    let mut elements = 1u64;
    for &dim in dims {
        if dim == 0 {
            return Err(format!("invalid zero tensor dimension {dims:?}"));
        }
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| format!("tensor dimension product overflow {dims:?}"))?;
    }
    match dtype {
        0 => elements
            .checked_mul(4)
            .ok_or_else(|| "F32 tensor length overflow".to_owned()),
        2 if elements % 32 == 0 => elements
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(18))
            .ok_or_else(|| "Q4_0 tensor length overflow".to_owned()),
        8 if elements % 32 == 0 => elements
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(34))
            .ok_or_else(|| "Q8_0 tensor length overflow".to_owned()),
        2 | 8 => Err(format!("quantized tensor is not block divisible: {dims:?}")),
        _ => Err(format!("unsupported tensor type {dtype}")),
    }
}

fn read_value<R: Read>(reader: &mut Reader<R>, dtype: u32) -> Result<Value, String> {
    match dtype {
        8 => Ok(Value::ScalarString(reader.string(16 * 1024 * 1024)?)),
        9 => {
            let element = reader.u32()?;
            let count = reader.u64()?;
            if count > 200_000 {
                return Err(format!("array count {count} exceeds 200000"));
            }
            match element {
                8 => {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        values.push(reader.string(4096)?);
                    }
                    Ok(Value::ArrayString(values))
                }
                5 => {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        values.push(reader.i32()?);
                    }
                    Ok(Value::ArrayI32(values))
                }
                _ => {
                    for _ in 0..count {
                        reader.scalar_bits(element)?;
                    }
                    Err(format!(
                        "unsupported frozen GGUF array element type {element}"
                    ))
                }
            }
        }
        _ => Ok(Value::ScalarU64(dtype, reader.scalar_bits(dtype)?)),
    }
}

pub fn parse_path(path: &Path) -> Result<Gguf, String> {
    let size = path
        .metadata()
        .map_err(|error| format!("cannot stat source: {error}"))?
        .len();
    let file = File::open(path).map_err(|error| format!("cannot open source: {error}"))?;
    parse_reader(file, size)
}

fn parse_reader<R: Read>(input: R, size: u64) -> Result<Gguf, String> {
    let mut reader = Reader::new(input, size);
    if reader.read(4)?.as_slice() != b"GGUF" {
        return Err("bad GGUF magic".to_owned());
    }
    let version = reader.u32()?;
    if version != 3 {
        return Err(format!("expected GGUF version 3, got {version}"));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > 4096 || metadata_count > 1024 {
        return Err("GGUF header count exceeds parser bounds".to_owned());
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string(1024)?;
        let dtype = reader.u32()?;
        let value = read_value(&mut reader, dtype)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate metadata key {key}"));
        }
    }
    let mut infos = Vec::with_capacity(tensor_count as usize);
    let mut names = BTreeSet::new();
    for _ in 0..tensor_count {
        let name = reader.string(1024)?;
        if !names.insert(name.clone()) {
            return Err(format!("duplicate tensor name {name}"));
        }
        let rank = reader.u32()?;
        if !(1..=4).contains(&rank) {
            return Err(format!("invalid rank {rank} for {name}"));
        }
        let mut dims = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dims.push(reader.u64()?);
        }
        let dtype = reader.u32()?;
        let relative = reader.u64()?;
        let length = tensor_length(&dims, dtype)?;
        infos.push((name, dims, dtype, relative, length));
    }
    let info_end = reader.position;
    let data_start = info_end
        .checked_add(31)
        .ok_or_else(|| "GGUF data alignment overflow".to_owned())?
        & !31;
    if data_start > size {
        return Err("aligned tensor data starts beyond input".to_owned());
    }
    let mut tensors = BTreeMap::new();
    let mut intervals = Vec::with_capacity(infos.len());
    for (name, dims, dtype, relative, length) in infos {
        let offset = data_start
            .checked_add(relative)
            .ok_or_else(|| format!("tensor offset overflow {name}"))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| format!("tensor end overflow {name}"))?;
        if end > size {
            return Err(format!("tensor {name} exceeds input"));
        }
        intervals.push((offset, end, name.clone()));
        tensors.insert(
            name,
            Tensor {
                dims,
                dtype,
                offset,
                length,
            },
        );
    }
    intervals.sort();
    let mut previous = data_start;
    for (start, end, name) in &intervals {
        if *start < previous {
            return Err(format!("tensor {name} overlaps previous tensor"));
        }
        previous = *end;
    }
    validate_exact(&metadata, &tensors, info_end, data_start, size, &intervals)?;
    Ok(Gguf { metadata, tensors })
}

fn scalar_matches(value: &Value, dtype: u32, expected: u64) -> bool {
    matches!(value, Value::ScalarU64(actual_type, actual) if *actual_type == dtype && *actual == expected)
}

fn string_matches(value: &Value, expected: &str) -> bool {
    matches!(value, Value::ScalarString(actual) if actual == expected)
}

fn canonical_strings(values: &[String]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(&(value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.finish()
}

fn canonical_i32(values: &[i32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(&(*value as i64 as u64).to_le_bytes());
    }
    digest.finish()
}

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

fn validate_string_array(
    value: Option<&Value>,
    key: &str,
    count: usize,
    expected_hash: &str,
) -> Result<(), String> {
    let Some(Value::ArrayString(values)) = value else {
        return Err(format!("{key} array schema mismatch"));
    };
    if values.len() != count {
        return Err(format!("{key} array count mismatch"));
    }
    if canonical_strings(values) != decode_hex(expected_hash) {
        return Err(format!("{key} canonical SHA-256 mismatch"));
    }
    Ok(())
}

fn validate_i32_array(
    value: Option<&Value>,
    key: &str,
    count: usize,
    expected_hash: &str,
) -> Result<(), String> {
    let Some(Value::ArrayI32(values)) = value else {
        return Err(format!("{key} array schema mismatch"));
    };
    if values.len() != count {
        return Err(format!("{key} array count mismatch"));
    }
    if canonical_i32(values) != decode_hex(expected_hash) {
        return Err(format!("{key} canonical SHA-256 mismatch"));
    }
    Ok(())
}

fn validate_template(value: Option<&Value>) -> Result<(), String> {
    let Some(Value::ScalarString(template)) = value else {
        return Err("tokenizer.chat_template type mismatch".to_owned());
    };
    if template.len() != 2509
        || promptboot_core::sha256::digest(template.as_bytes())
            != decode_hex("d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4")
    {
        return Err("tokenizer.chat_template identity mismatch".to_owned());
    }
    Ok(())
}

fn validate_metadata(metadata: &BTreeMap<String, Value>) -> Result<(), String> {
    let strings = [
        ("general.architecture", "qwen2"),
        ("general.type", "model"),
        ("general.name", "qwen2.5-0.5b-instruct"),
        ("general.version", "v0.1"),
        ("general.finetune", "qwen2.5-0.5b-instruct"),
        ("general.size_label", "630M"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen2"),
    ];
    let scalars = [
        ("qwen2.block_count", 4, 24),
        ("qwen2.context_length", 4, 32768),
        ("qwen2.embedding_length", 4, 896),
        ("qwen2.feed_forward_length", 4, 4864),
        ("qwen2.attention.head_count", 4, 14),
        ("qwen2.attention.head_count_kv", 4, 2),
        ("qwen2.rope.freq_base", 6, 0x49742400),
        ("qwen2.attention.layer_norm_rms_epsilon", 6, 0x358637bd),
        ("general.file_type", 4, 2),
        ("tokenizer.ggml.eos_token_id", 4, 151645),
        ("tokenizer.ggml.padding_token_id", 4, 151643),
        ("tokenizer.ggml.bos_token_id", 4, 151643),
        ("tokenizer.ggml.add_bos_token", 7, 0),
        ("general.quantization_version", 4, 2),
    ];
    let expected_keys: BTreeSet<&str> = strings
        .iter()
        .map(|row| row.0)
        .chain(scalars.iter().map(|row| row.0))
        .chain([
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.chat_template",
        ])
        .collect();
    if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_keys {
        return Err("metadata key set mismatch".to_owned());
    }
    for (key, expected) in strings {
        if !metadata
            .get(key)
            .is_some_and(|value| string_matches(value, expected))
        {
            return Err(format!("{key} value mismatch"));
        }
    }
    for (key, dtype, expected) in scalars {
        if !metadata
            .get(key)
            .is_some_and(|value| scalar_matches(value, dtype, expected))
        {
            return Err(format!("{key} value mismatch"));
        }
    }
    validate_string_array(
        metadata.get("tokenizer.ggml.tokens"),
        "tokenizer.ggml.tokens",
        151936,
        "8656e079473b857729cf444f772368cd6428dcd64513d46aac3f694b2f695282",
    )?;
    validate_i32_array(
        metadata.get("tokenizer.ggml.token_type"),
        "tokenizer.ggml.token_type",
        151936,
        "f3760e7fbfc96d388f5b8d7cd67c5cd46535c8682ff84ebc9817cda61bc98f4d",
    )?;
    validate_string_array(
        metadata.get("tokenizer.ggml.merges"),
        "tokenizer.ggml.merges",
        151387,
        "907981a313a6e78ef2223229042674a39cd4995ee4c4ed40a31b58754e82c26c",
    )?;
    validate_template(metadata.get("tokenizer.chat_template"))?;
    Ok(())
}

fn validate_tensors(
    tensors: &BTreeMap<String, Tensor>,
    data_start: u64,
    size: u64,
) -> Result<(), String> {
    let expected = pack_tensor_rows();
    if tensors.len() != expected.len()
        || tensors
            .keys()
            .any(|name| !expected.iter().any(|row| row.0.as_str() == name))
    {
        return Err("tensor name set is not the frozen 291-tensor schema".to_owned());
    }
    let mut counts = [0usize; 9];
    for (name, dims, dtype, _, _) in &expected {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if tensor.offset % 32 != 0 || tensor.dims != *dims || tensor.dtype != *dtype {
            return Err(format!("tensor schema mismatch for {name}"));
        }
        counts[*dtype as usize] += 1;
    }
    if counts[0] != 121 || counts[2] != 169 || counts[8] != 1 {
        return Err("tensor dtype counts mismatch".to_owned());
    }
    let mut intervals: Vec<(u64, u64)> = tensors
        .values()
        .map(|tensor| {
            tensor
                .offset
                .checked_add(tensor.length)
                .map(|end| (tensor.offset, end))
                .ok_or_else(|| "source tensor interval overflow".to_owned())
        })
        .collect::<Result<_, _>>()?;
    intervals.sort();
    if intervals.first().map(|row| row.0) != Some(data_start)
        || intervals.last().map(|row| row.1) != Some(size)
        || intervals.windows(2).any(|pair| pair[0].1 != pair[1].0)
        || size - data_start != 422_782_464
    {
        return Err("source tensor payload layout mismatch".to_owned());
    }
    Ok(())
}

fn validate_exact(
    metadata: &BTreeMap<String, Value>,
    tensors: &BTreeMap<String, Tensor>,
    info_end: u64,
    data_start: u64,
    size: u64,
    _intervals: &[(u64, u64, String)],
) -> Result<(), String> {
    if size != SOURCE_BYTES {
        return Err(format!("unexpected source size {size}"));
    }
    if info_end != 5_947_741 || data_start != 5_947_744 {
        return Err(format!(
            "unexpected tensor-info/data offsets {info_end}/{data_start}"
        ));
    }
    validate_metadata(metadata)?;
    validate_tensors(tensors, data_start, size)
}

pub type TensorRow = (String, Vec<u64>, u32, u16, u16);

pub fn pack_tensor_rows() -> Vec<TensorRow> {
    let mut result = vec![(
        "token_embd.weight".to_owned(),
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
        for (suffix, dims, dtype, role) in roles {
            result.push((
                format!("blk.{layer}.{suffix}"),
                dims.to_vec(),
                dtype,
                layer,
                role,
            ));
        }
    }
    result.push(("output_norm.weight".to_owned(), vec![896], 0, u16::MAX, 2));
    result.push((
        "output.weight".to_owned(),
        vec![896, 151936],
        8,
        u16::MAX,
        3,
    ));
    result
}

pub fn verify_identity(path: &Path) -> Result<(), String> {
    let metadata = path
        .metadata()
        .map_err(|error| format!("cannot stat source: {error}"))?;
    if metadata.len() != SOURCE_BYTES {
        return Err(format!(
            "asset qwen_gguf size mismatch: expected {}, got {}",
            SOURCE_BYTES,
            metadata.len()
        ));
    }
    let mut file = File::open(path).map_err(|error| format!("cannot open source: {error}"))?;
    let mut state = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash source: {error}"))?;
        if count == 0 {
            break;
        }
        state.update(&buffer[..count]);
    }
    if state.finish() != SOURCE_SHA256 {
        return Err("asset qwen_gguf SHA-256 mismatch".to_owned());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("cannot rewind source: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn header(version: u32, tensors: u64, metadata: u64) -> Vec<u8> {
        let mut result = b"GGUF".to_vec();
        result.extend(version.to_le_bytes());
        result.extend(tensors.to_le_bytes());
        result.extend(metadata.to_le_bytes());
        result
    }

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }

    fn scalar_metadata(bytes: &mut Vec<u8>, key: &str, dtype: u32, value: &[u8]) {
        string(bytes, key);
        bytes.extend(dtype.to_le_bytes());
        bytes.extend(value);
    }

    fn tensor(bytes: &mut Vec<u8>, name: &str, rank: u32, dims: &[u64], dtype: u32, offset: u64) {
        string(bytes, name);
        bytes.extend(rank.to_le_bytes());
        for dimension in dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(dtype.to_le_bytes());
        bytes.extend(offset.to_le_bytes());
    }

    fn parse_error(bytes: Vec<u8>) -> String {
        parse_reader(Cursor::new(&bytes), bytes.len() as u64).unwrap_err()
    }

    #[test]
    fn bounded_parser_rejects_malformed_families() {
        let mut truncated = header(3, 0, 0);
        truncated.pop();
        let mut bad_magic = header(3, 0, 0);
        bad_magic[0] = b'X';

        let mut duplicate_metadata = header(3, 0, 2);
        for _ in 0..2 {
            scalar_metadata(&mut duplicate_metadata, "x", 4, &1u32.to_le_bytes());
        }
        let mut long_string = header(3, 0, 1);
        long_string.extend(1025u64.to_le_bytes());
        let mut long_array = header(3, 0, 1);
        string(&mut long_array, "x");
        long_array.extend(9u32.to_le_bytes());
        long_array.extend(4u32.to_le_bytes());
        long_array.extend(200_001u64.to_le_bytes());
        let mut unsupported_scalar = header(3, 0, 1);
        string(&mut unsupported_scalar, "x");
        unsupported_scalar.extend(13u32.to_le_bytes());
        let mut unsupported_array = header(3, 0, 1);
        string(&mut unsupported_array, "x");
        unsupported_array.extend(9u32.to_le_bytes());
        unsupported_array.extend(13u32.to_le_bytes());
        unsupported_array.extend(0u64.to_le_bytes());

        let mut duplicate_tensor = header(3, 2, 0);
        tensor(&mut duplicate_tensor, "x", 1, &[1], 0, 0);
        tensor(&mut duplicate_tensor, "x", 1, &[1], 0, 4);
        let mut rank = header(3, 1, 0);
        tensor(&mut rank, "x", 0, &[], 0, 0);
        let mut dimension = header(3, 1, 0);
        tensor(&mut dimension, "x", 1, &[0], 0, 0);
        let mut product = header(3, 1, 0);
        tensor(&mut product, "x", 2, &[u64::MAX, 2], 0, 0);
        let mut block = header(3, 1, 0);
        tensor(&mut block, "x", 1, &[1], 2, 0);
        let mut tensor_type = header(3, 1, 0);
        tensor(&mut tensor_type, "x", 1, &[1], 1, 0);
        let mut overlap = header(3, 2, 0);
        tensor(&mut overlap, "x", 1, &[1], 0, 0);
        tensor(&mut overlap, "y", 1, &[1], 0, 0);
        let aligned = (overlap.len() + 31) & !31;
        overlap.resize(aligned + 4, 0);

        let cases = vec![
            ("truncation", truncated, "read"),
            ("magic", bad_magic, "magic"),
            ("version", header(2, 0, 0), "version 3"),
            ("header count", header(3, 4097, 0), "count"),
            (
                "duplicate metadata",
                duplicate_metadata,
                "duplicate metadata",
            ),
            ("string overflow", long_string, "string length"),
            ("array overflow", long_array, "array count"),
            ("scalar type", unsupported_scalar, "nonscalar"),
            ("array type", unsupported_array, "array element type"),
            ("duplicate tensor", duplicate_tensor, "duplicate tensor"),
            ("rank", rank, "rank"),
            ("zero dimension", dimension, "zero tensor dimension"),
            ("dimension product", product, "product overflow"),
            ("block divisibility", block, "block divisible"),
            ("tensor type", tensor_type, "tensor type"),
            ("tensor overlap", overlap, "overlaps"),
        ];
        for (name, bytes, expected) in cases {
            assert!(
                parse_error(bytes).contains(expected),
                "{name} did not report {expected}"
            );
        }
    }

    fn metadata_shape() -> BTreeMap<String, Value> {
        let mut metadata = BTreeMap::new();
        for (key, value) in [
            ("general.architecture", "qwen2"),
            ("general.type", "model"),
            ("general.name", "qwen2.5-0.5b-instruct"),
            ("general.version", "v0.1"),
            ("general.finetune", "qwen2.5-0.5b-instruct"),
            ("general.size_label", "630M"),
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "qwen2"),
        ] {
            metadata.insert(key.to_owned(), Value::ScalarString(value.to_owned()));
        }
        for (key, dtype, value) in [
            ("qwen2.block_count", 4, 24),
            ("qwen2.context_length", 4, 32768),
            ("qwen2.embedding_length", 4, 896),
            ("qwen2.feed_forward_length", 4, 4864),
            ("qwen2.attention.head_count", 4, 14),
            ("qwen2.attention.head_count_kv", 4, 2),
            ("qwen2.rope.freq_base", 6, 0x49742400),
            ("qwen2.attention.layer_norm_rms_epsilon", 6, 0x358637bd),
            ("general.file_type", 4, 2),
            ("tokenizer.ggml.eos_token_id", 4, 151645),
            ("tokenizer.ggml.padding_token_id", 4, 151643),
            ("tokenizer.ggml.bos_token_id", 4, 151643),
            ("tokenizer.ggml.add_bos_token", 7, 0),
            ("general.quantization_version", 4, 2),
        ] {
            metadata.insert(key.to_owned(), Value::ScalarU64(dtype, value));
        }
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            Value::ArrayString(Vec::new()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            Value::ArrayI32(Vec::new()),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            Value::ArrayString(Vec::new()),
        );
        metadata.insert(
            "tokenizer.chat_template".to_owned(),
            Value::ScalarString(String::new()),
        );
        metadata
    }

    #[test]
    fn frozen_metadata_token_and_merge_schemas_reject_mutations() {
        let mut metadata = metadata_shape();
        assert!(validate_metadata(&BTreeMap::new())
            .unwrap_err()
            .contains("key set"));
        metadata.insert(
            "general.architecture".to_owned(),
            Value::ScalarString("other".to_owned()),
        );
        assert!(validate_metadata(&metadata)
            .unwrap_err()
            .contains("general.architecture"));

        let string_cases = [
            (
                "token type",
                Value::ArrayI32(vec![]),
                1,
                "00".repeat(32),
                "schema",
            ),
            (
                "token count",
                Value::ArrayString(vec![]),
                1,
                "00".repeat(32),
                "count",
            ),
            (
                "token identity",
                Value::ArrayString(vec!["x".to_owned()]),
                1,
                "00".repeat(32),
                "SHA-256",
            ),
            (
                "merge schema",
                Value::ScalarString(String::new()),
                1,
                "00".repeat(32),
                "schema",
            ),
        ];
        for (name, value, count, hash, expected) in string_cases {
            assert!(
                validate_string_array(Some(&value), name, count, &hash)
                    .unwrap_err()
                    .contains(expected),
                "{name}"
            );
        }
        let type_cases = [
            (Value::ArrayString(vec![]), 1, "schema"),
            (Value::ArrayI32(vec![]), 1, "count"),
            (Value::ArrayI32(vec![1]), 1, "SHA-256"),
        ];
        for (value, count, expected) in type_cases {
            assert!(validate_i32_array(
                Some(&value),
                "tokenizer.ggml.token_type",
                count,
                &"00".repeat(32)
            )
            .unwrap_err()
            .contains(expected));
        }
        assert!(validate_template(Some(&Value::ArrayString(vec![])))
            .unwrap_err()
            .contains("type"));
        assert!(
            validate_template(Some(&Value::ScalarString("x".to_owned())))
                .unwrap_err()
                .contains("identity")
        );
    }

    fn exact_tensors() -> (BTreeMap<String, Tensor>, u64) {
        let mut tensors = BTreeMap::new();
        let mut offset = 5_947_744;
        for (name, dims, dtype, _, _) in pack_tensor_rows() {
            let length = tensor_length(&dims, dtype).unwrap();
            tensors.insert(
                name,
                Tensor {
                    dims,
                    dtype,
                    offset,
                    length,
                },
            );
            offset += length;
        }
        (tensors, offset)
    }

    #[test]
    fn frozen_tensor_schema_and_payload_layout_reject_mutations() {
        let (tensors, end) = exact_tensors();
        assert_eq!(end, SOURCE_BYTES);
        validate_tensors(&tensors, 5_947_744, SOURCE_BYTES).unwrap();

        let mut cases = Vec::new();
        let mut missing = tensors.clone();
        missing.remove("output.weight");
        cases.push(("name set", missing, "291-tensor"));
        let mut dims = tensors.clone();
        dims.get_mut("output.weight").unwrap().dims[0] += 1;
        cases.push(("dimensions", dims, "schema mismatch"));
        let mut dtype = tensors.clone();
        dtype.get_mut("output.weight").unwrap().dtype = 0;
        cases.push(("dtype", dtype, "schema mismatch"));
        let mut alignment = tensors.clone();
        alignment.get_mut("output.weight").unwrap().offset += 1;
        cases.push(("alignment", alignment, "schema mismatch"));
        let mut layout = tensors.clone();
        layout.get_mut("output.weight").unwrap().length -= 1;
        cases.push(("payload layout", layout, "payload layout"));

        for (name, changed, expected) in cases {
            assert!(
                validate_tensors(&changed, 5_947_744, SOURCE_BYTES)
                    .unwrap_err()
                    .contains(expected),
                "{name}"
            );
        }
    }

    #[test]
    fn frozen_pack_schema_has_expected_count_and_types() {
        let rows = pack_tensor_rows();
        assert_eq!(rows.len(), 291);
        assert_eq!(rows.iter().filter(|row| row.2 == 0).count(), 121);
        assert_eq!(rows.iter().filter(|row| row.2 == 2).count(), 169);
        assert_eq!(rows.iter().filter(|row| row.2 == 8).count(), 1);
    }
}
