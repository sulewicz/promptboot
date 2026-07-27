use core::mem::{align_of, size_of};

use crate::sha256::Sha256;

pub const MODEL_BYTES: usize = 426_762_944;
pub const VOCAB_COUNT: u32 = 151_936;
pub const MERGE_COUNT: u32 = 151_387;
pub const TENSOR_COUNT: u32 = 291;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ModelStatus {
    OK = 0,
    LENGTH = 1,
    FULL_HASH = 2,
    MAGIC = 3,
    VERSION = 4,
    ENDIAN = 5,
    IDENTITY = 6,
    RESERVED = 7,
    ARITHMETIC_OVERFLOW = 8,
    ALIGNMENT = 9,
    ORDER = 10,
    OVERLAP = 11,
    PADDING = 12,
    SECTION_HASH = 13,
    TENSOR_HASH = 14,
    CONTENT_HASH = 15,
    SECTION_SCHEMA = 16,
    TENSOR_SCHEMA = 17,
    TOKENIZER_SCHEMA = 18,
    UTF8 = 19,
    TOKEN_ID = 20,
    INPUT = 21,
    OUTPUT_CAPACITY = 22,
    SCRATCH_CAPACITY = 23,
    INDEX_CAPACITY = 24,
    INDEX_ALIGNMENT = 25,
    INDEX_CORRUPT = 26,
    STATE = 27,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ErrorDomain {
    NONE = 0,
    HEADER = 1,
    SECTION = 2,
    TENSOR = 3,
    TOKEN = 4,
    MERGE = 5,
    CHAT = 6,
    INDEX = 7,
    USER = 8,
    RENDERED_OUTPUT = 9,
    TOKEN_OUTPUT = 10,
    PIECE_OUTPUT = 11,
    SCRATCH = 12,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum FieldKind {
    NONE = 0,
    FILE_BYTES = 1,
    BASE_ALIGNMENT = 2,
    DIGEST_BYTE = 3,
    MAGIC_BYTE = 4,
    VERSION = 5,
    ENDIAN = 6,
    HEADER_FIELD = 7,
    RESERVED_BYTE = 8,
    SECTION_ID = 9,
    ELEMENT_TYPE = 10,
    RANGE_START = 11,
    RANGE_LENGTH = 12,
    ORDER = 13,
    GAP_BYTE = 14,
    TENSOR_ID = 15,
    LAYER = 16,
    ROLE = 17,
    DTYPE = 18,
    RANK = 19,
    DIMENSION = 20,
    BLOCK_LENGTH = 21,
    TOKEN_OFFSET = 22,
    TOKEN_TYPE = 23,
    UTF8_BYTE = 24,
    MERGE_LEFT = 25,
    MERGE_RIGHT = 26,
    MERGE_RESULT = 27,
    MERGE_DUPLICATE = 28,
    CONCAT_BYTE = 29,
    TEMPLATE_BYTE = 30,
    INDEX_HASH_BYTE = 31,
    INDEX_MAX_DISPLACEMENT = 32,
    INDEX_MAX_CLUSTER = 33,
    INDEX_HIT_BOUND = 34,
    INDEX_MISS_BOUND = 35,
    USER_BYTE = 36,
    TOKEN_ID_FIELD = 37,
    CAPACITY = 38,
    STATE_FIELD = 39,
    OP_ADD = 40,
    OP_MUL = 41,
    OP_USIZE = 42,
}

pub(crate) const fn detail(kind: FieldKind, sub: u32) -> u64 {
    ((kind as u64) << 32) | sub as u64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ModelError {
    pub status: u32,
    pub domain: u32,
    pub index: u32,
    pub reserved: u32,
    pub offset: u64,
    pub needed: u64,
    pub available: u64,
    pub detail: u64,
}

impl ModelError {
    pub(crate) const fn new(
        status: ModelStatus,
        domain: ErrorDomain,
        index: u32,
        offset: u64,
        needed: u64,
        available: u64,
        kind: FieldKind,
        sub: u32,
    ) -> Self {
        Self {
            status: status as u32,
            domain: domain as u32,
            index,
            reserved: 0,
            offset,
            needed,
            available,
            detail: detail(kind, sub),
        }
    }

    pub fn full_hash_mismatch(expected: &[u8; 32], actual: &[u8; 32]) -> Option<Self> {
        digest_mismatch(
            ModelStatus::FULL_HASH,
            ErrorDomain::HEADER,
            0,
            0,
            expected,
            actual,
            FieldKind::DIGEST_BYTE,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ModelConfig {
    pub version: u32,
    pub context_limit: u32,
    pub block_count: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab_count: u32,
    pub merge_count: u32,
    pub tensor_count: u32,
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub add_bos: u32,
    pub reserved: u32,
    pub model_bytes: u64,
    pub tensor_data_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TensorMeta {
    pub id: u32,
    pub layer: u16,
    pub role: u16,
    pub dtype: u32,
    pub rank: u32,
    pub dims: [u32; 4],
    pub data_offset: u64,
    pub data_length: u64,
    pub elements: u64,
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    pub result: u32,
    pub rank: u32,
}

#[derive(Clone, Copy)]
struct Section {
    offset: u64,
    length: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorView<'bytes> {
    meta: TensorMeta,
    data: &'bytes [u8],
}

impl<'bytes> TensorView<'bytes> {
    pub fn meta(&self) -> TensorMeta {
        self.meta
    }

    pub fn data(&self) -> &'bytes [u8] {
        self.data
    }
}

pub struct ModelView<'bytes> {
    bytes: &'bytes [u8],
    sections: [Section; 7],
    config: ModelConfig,
}

const _: [(); 48] = [(); size_of::<ModelError>()];
const _: [(); 8] = [(); align_of::<ModelError>()];
const _: [(); 80] = [(); size_of::<ModelConfig>()];
const _: [(); 8] = [(); align_of::<ModelConfig>()];
const _: [(); 64] = [(); size_of::<TensorMeta>()];
const _: [(); 8] = [(); align_of::<TensorMeta>()];
const _: [(); 16] = [(); size_of::<Merge>()];

const CONFIG: ModelConfig = ModelConfig {
    version: 1,
    context_limit: crate::inference::CONTEXT_LIMIT,
    block_count: 24,
    embedding: 896,
    feed_forward: 4864,
    query_heads: 14,
    kv_heads: 2,
    head_dim: 64,
    vocab_count: VOCAB_COUNT,
    merge_count: MERGE_COUNT,
    tensor_count: TENSOR_COUNT,
    bos: 151_643,
    eos: 151_645,
    pad: 151_643,
    add_bos: 0,
    reserved: 0,
    model_bytes: MODEL_BYTES as u64,
    tensor_data_bytes: 422_782_464,
};

const SECTION_OFFSETS: [u64; 7] = [
    704, 608_512, 1_981_312, 2_133_248, 3_949_952, 3_952_512, 3_980_480,
];
const SECTION_LENGTHS: [u64; 7] = [
    607_748,
    1_372_758,
    151_936,
    1_816_644,
    2_509,
    27_936,
    422_782_464,
];
const SECTION_COUNTS: [u64; 7] = [
    151_937,
    1_372_758,
    151_936,
    151_387,
    2_509,
    291,
    422_782_464,
];
const SECTION_TYPES: [u32; 7] = [1, 2, 2, 3, 2, 4, 2];
const SOURCE_HASH: [u8; 32] =
    hex32(*b"7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed");
const TOKEN_DIGEST: [u8; 32] =
    hex32(*b"8656e079473b857729cf444f772368cd6428dcd64513d46aac3f694b2f695282");
const TYPE_DIGEST: [u8; 32] =
    hex32(*b"f3760e7fbfc96d388f5b8d7cd67c5cd46535c8682ff84ebc9817cda61bc98f4d");
const MERGE_DIGEST: [u8; 32] =
    hex32(*b"907981a313a6e78ef2223229042674a39cd4995ee4c4ed40a31b58754e82c26c");

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

const fn hex32(input: [u8; 64]) -> [u8; 32] {
    let mut output = [0u8; 32];
    let mut index = 0;
    while index < 32 {
        output[index] = (hex_nibble(input[index * 2]) << 4) | hex_nibble(input[index * 2 + 1]);
        index += 1;
    }
    output
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0; 4]))
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap_or([0; 8]))
}

fn digest_mismatch(
    status: ModelStatus,
    domain: ErrorDomain,
    index: u32,
    offset: u64,
    expected: &[u8; 32],
    actual: &[u8; 32],
    kind: FieldKind,
) -> Option<ModelError> {
    for at in 0..32 {
        if expected[at] != actual[at] {
            return Some(ModelError::new(
                status,
                domain,
                index,
                offset,
                expected[at] as u64,
                actual[at] as u64,
                kind,
                at as u32,
            ));
        }
    }
    None
}

fn scalar_error(
    status: ModelStatus,
    offset: usize,
    expected: u64,
    actual: u64,
    kind: FieldKind,
) -> ModelError {
    ModelError::new(
        status,
        ErrorDomain::HEADER,
        offset as u32,
        offset as u64,
        expected,
        actual,
        kind,
        0,
    )
}

impl<'bytes> ModelView<'bytes> {
    pub fn open_authenticated(bytes: &'bytes [u8]) -> Result<Self, ModelError> {
        if bytes.len() != MODEL_BYTES {
            return Err(ModelError::new(
                ModelStatus::LENGTH,
                ErrorDomain::HEADER,
                0,
                0,
                MODEL_BYTES as u64,
                bytes.len() as u64,
                FieldKind::FILE_BYTES,
                0,
            ));
        }
        let misalignment = (bytes.as_ptr() as usize & 63) as u64;
        if misalignment != 0 {
            return Err(ModelError::new(
                ModelStatus::ALIGNMENT,
                ErrorDomain::HEADER,
                0,
                0,
                64,
                misalignment,
                FieldKind::BASE_ALIGNMENT,
                0,
            ));
        }
        for at in 0..8 {
            let expected = b"PBTQW25\0"[at];
            if bytes[at] != expected {
                return Err(ModelError::new(
                    ModelStatus::MAGIC,
                    ErrorDomain::HEADER,
                    at as u32,
                    at as u64,
                    expected as u64,
                    bytes[at] as u64,
                    FieldKind::MAGIC_BYTE,
                    at as u32,
                ));
            }
        }
        let expected32 = [
            (8, 1, FieldKind::VERSION, ModelStatus::VERSION),
            (12, 256, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (16, 0x0102_0304, FieldKind::ENDIAN, ModelStatus::ENDIAN),
            (20, 7, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (24, 291, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (28, 151_936, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (32, 151_387, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (
                36,
                crate::inference::CONTEXT_LIMIT,
                FieldKind::HEADER_FIELD,
                ModelStatus::IDENTITY,
            ),
            (40, 24, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (44, 896, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (48, 4864, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (52, 14, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (56, 2, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (60, 64, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (64, 151_645, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (68, 151_643, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (72, 151_643, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
            (76, 0, FieldKind::HEADER_FIELD, ModelStatus::IDENTITY),
        ];
        for &(offset, expected, kind, status) in &expected32 {
            let actual = u32_at(bytes, offset);
            if actual != expected {
                return Err(scalar_error(
                    status,
                    offset,
                    expected as u64,
                    actual as u64,
                    kind,
                ));
            }
        }
        let expected64 = [
            (80, 428_730_208),
            (88, MODEL_BYTES as u64),
            (96, 256),
            (104, 448),
            (112, 5_947_744),
            (120, 422_782_464),
        ];
        for &(offset, expected) in &expected64 {
            let actual = u64_at(bytes, offset);
            if actual != expected {
                return Err(scalar_error(
                    ModelStatus::IDENTITY,
                    offset,
                    expected,
                    actual,
                    FieldKind::HEADER_FIELD,
                ));
            }
        }
        for at in 0..32 {
            if bytes[128 + at] != SOURCE_HASH[at] {
                return Err(ModelError::new(
                    ModelStatus::IDENTITY,
                    ErrorDomain::HEADER,
                    (128 + at) as u32,
                    (128 + at) as u64,
                    SOURCE_HASH[at] as u64,
                    bytes[128 + at] as u64,
                    FieldKind::DIGEST_BYTE,
                    at as u32,
                ));
            }
        }
        for at in 192..256 {
            if bytes[at] != 0 {
                return Err(ModelError::new(
                    ModelStatus::RESERVED,
                    ErrorDomain::HEADER,
                    at as u32,
                    at as u64,
                    0,
                    bytes[at] as u64,
                    FieldKind::RESERVED_BYTE,
                    (at - 192) as u32,
                ));
            }
        }

        let mut sections = [Section {
            offset: 0,
            length: 0,
        }; 7];
        let mut previous_end = 704u64;
        for ordinal in 0..7usize {
            let base = 256 + ordinal * 64;
            let actuals = [
                u32_at(bytes, base) as u64,
                u32_at(bytes, base + 4) as u64,
                u64_at(bytes, base + 8),
                u64_at(bytes, base + 16),
                u64_at(bytes, base + 24),
            ];
            let expected = [
                (ordinal + 1) as u64,
                SECTION_TYPES[ordinal] as u64,
                SECTION_COUNTS[ordinal],
                SECTION_OFFSETS[ordinal],
                SECTION_LENGTHS[ordinal],
            ];
            let kinds = [
                FieldKind::SECTION_ID,
                FieldKind::ELEMENT_TYPE,
                FieldKind::RANGE_LENGTH,
                FieldKind::RANGE_START,
                FieldKind::RANGE_LENGTH,
            ];
            let field_offsets = [0u32, 4, 8, 16, 24];
            for field in [0usize, 1, 2, 4] {
                if actuals[field] != expected[field] {
                    return Err(ModelError::new(
                        ModelStatus::SECTION_SCHEMA,
                        ErrorDomain::SECTION,
                        ordinal as u32,
                        (base + field_offsets[field] as usize) as u64,
                        expected[field],
                        actuals[field],
                        kinds[field],
                        field_offsets[field],
                    ));
                }
            }
            if actuals[3] & 63 != 0 {
                return Err(ModelError::new(
                    ModelStatus::ALIGNMENT,
                    ErrorDomain::SECTION,
                    ordinal as u32,
                    (base + 16) as u64,
                    64,
                    actuals[3] & 63,
                    FieldKind::RANGE_START,
                    16,
                ));
            }
            if actuals[3] < previous_end {
                return Err(ModelError::new(
                    ModelStatus::OVERLAP,
                    ErrorDomain::SECTION,
                    ordinal as u32,
                    (base + 16) as u64,
                    previous_end,
                    actuals[3],
                    FieldKind::RANGE_START,
                    16,
                ));
            }
            if actuals[3] != expected[3] {
                return Err(ModelError::new(
                    ModelStatus::SECTION_SCHEMA,
                    ErrorDomain::SECTION,
                    ordinal as u32,
                    (base + 16) as u64,
                    expected[3],
                    actuals[3],
                    FieldKind::RANGE_START,
                    16,
                ));
            }
            for at in previous_end as usize..actuals[3] as usize {
                if bytes[at] != 0 {
                    return Err(ModelError::new(
                        ModelStatus::PADDING,
                        ErrorDomain::SECTION,
                        ordinal as u32,
                        at as u64,
                        0,
                        bytes[at] as u64,
                        FieldKind::GAP_BYTE,
                        (at - previous_end as usize) as u32,
                    ));
                }
            }
            sections[ordinal] = Section {
                offset: actuals[3],
                length: actuals[4],
            };
            previous_end = actuals[3].checked_add(actuals[4]).ok_or_else(|| {
                ModelError::new(
                    ModelStatus::ARITHMETIC_OVERFLOW,
                    ErrorDomain::SECTION,
                    ordinal as u32,
                    (base + 24) as u64,
                    u64::MAX - actuals[3],
                    actuals[4],
                    FieldKind::OP_ADD,
                    0,
                )
            })?;
        }
        if previous_end != MODEL_BYTES as u64 {
            return Err(ModelError::new(
                ModelStatus::SECTION_SCHEMA,
                ErrorDomain::SECTION,
                6,
                previous_end,
                MODEL_BYTES as u64,
                previous_end,
                FieldKind::RANGE_LENGTH,
                0,
            ));
        }

        Self::validate_tensors(bytes)?;
        Self::validate_tokenizer(bytes, &sections)?;

        Ok(Self {
            bytes,
            sections,
            config: CONFIG,
        })
    }

    pub fn config(&self) -> ModelConfig {
        self.config
    }

    pub fn section(&self, id: u32) -> Result<&'bytes [u8], ModelError> {
        if !(1..=7).contains(&id) {
            return Err(ModelError::new(
                ModelStatus::SECTION_SCHEMA,
                ErrorDomain::SECTION,
                id,
                0,
                (1u64 << 32) | 7,
                id as u64,
                FieldKind::SECTION_ID,
                0,
            ));
        }
        let section = self.sections[(id - 1) as usize];
        Ok(&self.bytes[section.offset as usize..(section.offset + section.length) as usize])
    }

    pub fn tensor(&self, id: u32) -> Result<TensorView<'bytes>, ModelError> {
        if id >= TENSOR_COUNT {
            return Err(ModelError::new(
                ModelStatus::TENSOR_SCHEMA,
                ErrorDomain::TENSOR,
                id,
                0,
                TENSOR_COUNT as u64,
                id as u64,
                FieldKind::TENSOR_ID,
                0,
            ));
        }
        let meta = Self::tensor_meta_at(self.bytes, id);
        Ok(TensorView {
            meta,
            data: &self.bytes
                [meta.data_offset as usize..(meta.data_offset + meta.data_length) as usize],
        })
    }

    pub fn tensor_for(&self, layer: u16, role: u16) -> Result<TensorView<'bytes>, ModelError> {
        let packed = ((layer as u32) << 16) | role as u32;
        let id = if layer == 0xffff {
            match role {
                1 => 0,
                2 => 289,
                3 => 290,
                _ => {
                    return Err(ModelError::new(
                        ModelStatus::TENSOR_SCHEMA,
                        ErrorDomain::TENSOR,
                        packed,
                        0,
                        (1u64 << 32) | 3,
                        role as u64,
                        FieldKind::ROLE,
                        1,
                    ))
                }
            }
        } else if layer < 24 {
            if !(10..=21).contains(&role) {
                return Err(ModelError::new(
                    ModelStatus::TENSOR_SCHEMA,
                    ErrorDomain::TENSOR,
                    packed,
                    0,
                    (10u64 << 32) | 21,
                    role as u64,
                    FieldKind::ROLE,
                    0,
                ));
            }
            1 + layer as u32 * 12 + (role as u32 - 10)
        } else {
            return Err(ModelError::new(
                ModelStatus::TENSOR_SCHEMA,
                ErrorDomain::TENSOR,
                packed,
                0,
                24,
                layer as u64,
                FieldKind::LAYER,
                0,
            ));
        };
        self.tensor(id)
    }

    pub fn token_bytes(&self, id: u32) -> Result<&'bytes [u8], ModelError> {
        if id >= VOCAB_COUNT {
            return Err(ModelError::new(
                ModelStatus::TOKEN_ID,
                ErrorDomain::TOKEN,
                id,
                0,
                VOCAB_COUNT as u64,
                id as u64,
                FieldKind::TOKEN_ID_FIELD,
                0,
            ));
        }
        let offsets = self.sections[0].offset as usize;
        let start = u32_at(self.bytes, offsets + id as usize * 4) as usize;
        let end = u32_at(self.bytes, offsets + (id as usize + 1) * 4) as usize;
        let base = self.sections[1].offset as usize;
        Ok(&self.bytes[base + start..base + end])
    }

    pub fn token_type(&self, id: u32) -> Result<u8, ModelError> {
        if id >= VOCAB_COUNT {
            return Err(ModelError::new(
                ModelStatus::TOKEN_ID,
                ErrorDomain::TOKEN,
                id,
                0,
                VOCAB_COUNT as u64,
                id as u64,
                FieldKind::TOKEN_ID_FIELD,
                0,
            ));
        }
        Ok(self.bytes[self.sections[2].offset as usize + id as usize])
    }

    pub fn merge(&self, rank: u32) -> Result<Merge, ModelError> {
        if rank >= MERGE_COUNT {
            return Err(ModelError::new(
                ModelStatus::TOKENIZER_SCHEMA,
                ErrorDomain::MERGE,
                rank,
                0,
                MERGE_COUNT as u64,
                rank as u64,
                FieldKind::MERGE_RESULT,
                0,
            ));
        }
        let at = self.sections[3].offset as usize + rank as usize * 12;
        Ok(Merge {
            left: u32_at(self.bytes, at),
            right: u32_at(self.bytes, at + 4),
            result: u32_at(self.bytes, at + 8),
            rank,
        })
    }

    fn tensor_meta_at(bytes: &[u8], id: u32) -> TensorMeta {
        let at = SECTION_OFFSETS[5] as usize + id as usize * 96;
        let dims = [
            u32_at(bytes, at + 16),
            u32_at(bytes, at + 20),
            u32_at(bytes, at + 24),
            u32_at(bytes, at + 28),
        ];
        // ModelView exists only after every dimension has matched the frozen
        // schema. Derive this public scalar from that trusted schema rather
        // than multiplying untrusted directory values.
        let expected_dims = Self::expected_tensor(id).4;
        let elements = expected_dims[0] as u64
            * expected_dims[1] as u64
            * expected_dims[2] as u64
            * expected_dims[3] as u64;
        TensorMeta {
            id: u32_at(bytes, at),
            layer: u16_at(bytes, at + 4),
            role: u16_at(bytes, at + 6),
            dtype: u32_at(bytes, at + 8),
            rank: u32_at(bytes, at + 12),
            dims,
            data_offset: u64_at(bytes, at + 32),
            data_length: u64_at(bytes, at + 40),
            elements,
            reserved: 0,
        }
    }

    fn expected_tensor(id: u32) -> (u16, u16, u32, u32, [u32; 4]) {
        if id == 0 {
            return (0xffff, 1, 2, 2, [896, 151_936, 1, 1]);
        }
        if id == 289 {
            return (0xffff, 2, 1, 1, [896, 1, 1, 1]);
        }
        if id == 290 {
            return (0xffff, 3, 3, 2, [896, 151_936, 1, 1]);
        }
        let layer = ((id - 1) / 12) as u16;
        let slot = (id - 1) % 12;
        let role = 10 + slot as u16;
        match slot {
            0 | 4 => (layer, role, 1, 1, [896, 1, 1, 1]),
            1 => (layer, role, 2, 2, [4864, 896, 1, 1]),
            2 | 3 => (layer, role, 2, 2, [896, 4864, 1, 1]),
            5 | 10 => (layer, role, 1, 1, [128, 1, 1, 1]),
            6 | 11 => (layer, role, 2, 2, [896, 128, 1, 1]),
            7 | 9 => (layer, role, 2, 2, [896, 896, 1, 1]),
            8 => (layer, role, 1, 1, [896, 1, 1, 1]),
            _ => (0, 0, 0, 0, [0; 4]),
        }
    }

    fn validate_tensors(bytes: &[u8]) -> Result<(), ModelError> {
        let mut previous_end = SECTION_OFFSETS[6];
        for id in 0..TENSOR_COUNT {
            let at = SECTION_OFFSETS[5] as usize + id as usize * 96;
            let meta = Self::tensor_meta_at(bytes, id);
            let (layer, role, dtype, rank, dims) = Self::expected_tensor(id);
            let fields = [
                (meta.id as u64, id as u64, FieldKind::TENSOR_ID, 0u32),
                (meta.layer as u64, layer as u64, FieldKind::LAYER, 0),
                (meta.role as u64, role as u64, FieldKind::ROLE, 0),
                (meta.dtype as u64, dtype as u64, FieldKind::DTYPE, 0),
                (meta.rank as u64, rank as u64, FieldKind::RANK, 0),
            ];
            let offsets = [0, 4, 6, 8, 12];
            for field in 0..fields.len() {
                if fields[field].0 != fields[field].1 {
                    return Err(ModelError::new(
                        ModelStatus::TENSOR_SCHEMA,
                        ErrorDomain::TENSOR,
                        id,
                        (at + offsets[field]) as u64,
                        fields[field].1,
                        fields[field].0,
                        fields[field].2,
                        fields[field].3,
                    ));
                }
            }
            for dim in 0..4 {
                if meta.dims[dim] != dims[dim] {
                    return Err(ModelError::new(
                        ModelStatus::TENSOR_SCHEMA,
                        ErrorDomain::TENSOR,
                        id,
                        (at + 16 + dim * 4) as u64,
                        dims[dim] as u64,
                        meta.dims[dim] as u64,
                        FieldKind::DIMENSION,
                        dim as u32,
                    ));
                }
            }
            let length = match dtype {
                1 => meta.elements.checked_mul(4),
                2 => {
                    if meta.elements % 32 == 0 {
                        meta.elements
                            .checked_div(32)
                            .and_then(|v| v.checked_mul(18))
                    } else {
                        None
                    }
                }
                3 => {
                    if meta.elements % 32 == 0 {
                        meta.elements
                            .checked_div(32)
                            .and_then(|v| v.checked_mul(34))
                    } else {
                        None
                    }
                }
                _ => None,
            }
            .unwrap_or(0);
            if meta.data_length != length {
                return Err(ModelError::new(
                    ModelStatus::TENSOR_SCHEMA,
                    ErrorDomain::TENSOR,
                    id,
                    (at + 40) as u64,
                    length,
                    meta.data_length,
                    FieldKind::BLOCK_LENGTH,
                    dtype,
                ));
            }
            if meta.data_offset != previous_end {
                return Err(ModelError::new(
                    if meta.data_offset < previous_end {
                        ModelStatus::OVERLAP
                    } else {
                        ModelStatus::ORDER
                    },
                    ErrorDomain::TENSOR,
                    id,
                    (at + 32) as u64,
                    previous_end,
                    meta.data_offset,
                    FieldKind::RANGE_START,
                    0,
                ));
            }
            if meta.data_offset & 63 != 0 {
                return Err(ModelError::new(
                    ModelStatus::ALIGNMENT,
                    ErrorDomain::TENSOR,
                    id,
                    (at + 32) as u64,
                    64,
                    meta.data_offset & 63,
                    FieldKind::RANGE_START,
                    0,
                ));
            }
            for reserved in 80..96 {
                if bytes[at + reserved] != 0 {
                    return Err(ModelError::new(
                        ModelStatus::RESERVED,
                        ErrorDomain::TENSOR,
                        id,
                        (at + reserved) as u64,
                        0,
                        bytes[at + reserved] as u64,
                        FieldKind::RESERVED_BYTE,
                        (reserved - 80) as u32,
                    ));
                }
            }
            previous_end = meta
                .data_offset
                .checked_add(meta.data_length)
                .ok_or_else(|| {
                    ModelError::new(
                        ModelStatus::ARITHMETIC_OVERFLOW,
                        ErrorDomain::TENSOR,
                        id,
                        (at + 40) as u64,
                        u64::MAX - meta.data_offset,
                        meta.data_length,
                        FieldKind::OP_ADD,
                        0,
                    )
                })?;
        }
        if previous_end != MODEL_BYTES as u64 {
            return Err(ModelError::new(
                ModelStatus::TENSOR_SCHEMA,
                ErrorDomain::TENSOR,
                290,
                previous_end,
                MODEL_BYTES as u64,
                previous_end,
                FieldKind::RANGE_LENGTH,
                0,
            ));
        }
        Ok(())
    }

    fn validate_tokenizer(bytes: &[u8], sections: &[Section; 7]) -> Result<(), ModelError> {
        let offsets = sections[0].offset as usize;
        if u32_at(bytes, offsets) != 0 {
            return Err(ModelError::new(
                ModelStatus::TOKENIZER_SCHEMA,
                ErrorDomain::TOKEN,
                0,
                offsets as u64,
                0,
                u32_at(bytes, offsets) as u64,
                FieldKind::TOKEN_OFFSET,
                0,
            ));
        }
        let token_base = sections[1].offset as usize;
        let type_base = sections[2].offset as usize;
        let mut prior = 0u32;
        let mut counts = [0u64; 4];
        let mut token_canonical = Sha256::new();
        let mut type_canonical = Sha256::new();
        for id in 0..VOCAB_COUNT {
            let current = u32_at(bytes, offsets + (id as usize + 1) * 4);
            if current <= prior {
                return Err(ModelError::new(
                    ModelStatus::TOKENIZER_SCHEMA,
                    ErrorDomain::TOKEN,
                    id,
                    (offsets + (id as usize + 1) * 4) as u64,
                    prior as u64 + 1,
                    current as u64,
                    FieldKind::TOKEN_OFFSET,
                    1,
                ));
            }
            if current > 1_372_758 {
                return Err(ModelError::new(
                    ModelStatus::TOKENIZER_SCHEMA,
                    ErrorDomain::TOKEN,
                    id,
                    (offsets + (id as usize + 1) * 4) as u64,
                    1_372_758,
                    current as u64,
                    FieldKind::RANGE_LENGTH,
                    0,
                ));
            }
            let token = &bytes[token_base + prior as usize..token_base + current as usize];
            if let Err(error) = core::str::from_utf8(token) {
                let at = error.valid_up_to();
                return Err(ModelError::new(
                    ModelStatus::UTF8,
                    ErrorDomain::TOKEN,
                    id,
                    (token_base + prior as usize + at) as u64,
                    0,
                    token[at] as u64,
                    FieldKind::UTF8_BYTE,
                    at as u32,
                ));
            }
            token_canonical.update(&(token.len() as u64).to_le_bytes());
            token_canonical.update(token);
            let token_type = bytes[type_base + id as usize];
            let count_index = match token_type {
                1 => 0,
                3 => 1,
                4 => 2,
                5 => 3,
                _ => {
                    return Err(ModelError::new(
                        ModelStatus::TOKENIZER_SCHEMA,
                        ErrorDomain::TOKEN,
                        id,
                        (type_base + id as usize) as u64,
                        58,
                        token_type as u64,
                        FieldKind::TOKEN_TYPE,
                        0,
                    ))
                }
            };
            counts[count_index] += 1;
            type_canonical.update(&(token_type as u64).to_le_bytes());
            prior = current;
        }
        if prior != 1_372_758 {
            return Err(ModelError::new(
                ModelStatus::TOKENIZER_SCHEMA,
                ErrorDomain::TOKEN,
                VOCAB_COUNT,
                (offsets + VOCAB_COUNT as usize * 4) as u64,
                1_372_758,
                prior as u64,
                FieldKind::TOKEN_OFFSET,
                2,
            ));
        }
        let expected_counts = [151_643u64, 20, 2, 271];
        for at in 0..4 {
            if counts[at] != expected_counts[at] {
                return Err(ModelError::new(
                    ModelStatus::TOKENIZER_SCHEMA,
                    ErrorDomain::TOKEN,
                    0,
                    type_base as u64,
                    expected_counts[at],
                    counts[at],
                    FieldKind::TOKEN_TYPE,
                    (at + 1) as u32,
                ));
            }
        }
        let token_digest = token_canonical.finish();
        if let Some(error) = digest_mismatch(
            ModelStatus::TOKENIZER_SCHEMA,
            ErrorDomain::TOKEN,
            0,
            token_base as u64,
            &TOKEN_DIGEST,
            &token_digest,
            FieldKind::DIGEST_BYTE,
        ) {
            return Err(error);
        }
        let type_digest = type_canonical.finish();
        if let Some(error) = digest_mismatch(
            ModelStatus::TOKENIZER_SCHEMA,
            ErrorDomain::TOKEN,
            0,
            type_base as u64,
            &TYPE_DIGEST,
            &type_digest,
            FieldKind::DIGEST_BYTE,
        ) {
            return Err(error);
        }

        let merge_base = sections[3].offset as usize;
        let mut merge_canonical = Sha256::new();
        for rank in 0..MERGE_COUNT {
            let at = merge_base + rank as usize * 12;
            let left = u32_at(bytes, at);
            let right = u32_at(bytes, at + 4);
            let result = u32_at(bytes, at + 8);
            for (value, kind, off) in [
                (left, FieldKind::MERGE_LEFT, at),
                (right, FieldKind::MERGE_RIGHT, at + 4),
                (result, FieldKind::MERGE_RESULT, at + 8),
            ] {
                if value >= VOCAB_COUNT {
                    return Err(ModelError::new(
                        ModelStatus::TOKENIZER_SCHEMA,
                        ErrorDomain::MERGE,
                        rank,
                        off as u64,
                        VOCAB_COUNT as u64,
                        value as u64,
                        kind,
                        0,
                    ));
                }
            }
            if result >= 151_643 {
                return Err(ModelError::new(
                    ModelStatus::TOKENIZER_SCHEMA,
                    ErrorDomain::MERGE,
                    rank,
                    (at + 8) as u64,
                    151_643,
                    result as u64,
                    FieldKind::MERGE_RESULT,
                    1,
                ));
            }
            let left_start = u32_at(bytes, offsets + left as usize * 4) as usize;
            let left_end = u32_at(bytes, offsets + (left as usize + 1) * 4) as usize;
            let right_start = u32_at(bytes, offsets + right as usize * 4) as usize;
            let right_end = u32_at(bytes, offsets + (right as usize + 1) * 4) as usize;
            let result_start = u32_at(bytes, offsets + result as usize * 4) as usize;
            let result_end = u32_at(bytes, offsets + (result as usize + 1) * 4) as usize;
            let expected_len = left_end - left_start + right_end - right_start;
            if result_end - result_start != expected_len {
                if let Some(first_rank) =
                    Self::prior_duplicate_merge(bytes, merge_base, rank, left, right)
                {
                    return Err(Self::duplicate_merge_error(rank, at, first_rank));
                }
                return Err(ModelError::new(
                    ModelStatus::TOKENIZER_SCHEMA,
                    ErrorDomain::MERGE,
                    rank,
                    (token_base + result_start) as u64,
                    expected_len as u64,
                    (result_end - result_start) as u64,
                    FieldKind::RANGE_LENGTH,
                    0,
                ));
            }
            let mut concat = 0usize;
            for byte in bytes[token_base + left_start..token_base + left_end]
                .iter()
                .chain(bytes[token_base + right_start..token_base + right_end].iter())
            {
                if bytes[token_base + result_start + concat] != *byte {
                    if let Some(first_rank) =
                        Self::prior_duplicate_merge(bytes, merge_base, rank, left, right)
                    {
                        return Err(Self::duplicate_merge_error(rank, at, first_rank));
                    }
                    return Err(ModelError::new(
                        ModelStatus::TOKENIZER_SCHEMA,
                        ErrorDomain::MERGE,
                        rank,
                        (token_base + result_start + concat) as u64,
                        *byte as u64,
                        bytes[token_base + result_start + concat] as u64,
                        FieldKind::CONCAT_BYTE,
                        concat as u32,
                    ));
                }
                concat += 1;
            }
            let left_token = &bytes[token_base + left_start..token_base + left_end];
            let right_token = &bytes[token_base + right_start..token_base + right_end];
            merge_canonical
                .update(&((left_token.len() + 1 + right_token.len()) as u64).to_le_bytes());
            merge_canonical.update(left_token);
            merge_canonical.update(b" ");
            merge_canonical.update(right_token);
        }
        let merge_digest = merge_canonical.finish();
        if let Some(error) = digest_mismatch(
            ModelStatus::TOKENIZER_SCHEMA,
            ErrorDomain::MERGE,
            0,
            merge_base as u64,
            &MERGE_DIGEST,
            &merge_digest,
            FieldKind::DIGEST_BYTE,
        ) {
            // The canonical path remains O(n). This bounded O(n^2) scan is used
            // only to classify an already-invalid merge table deterministically.
            for rank in 1..MERGE_COUNT {
                let at = merge_base + rank as usize * 12;
                let left = u32_at(bytes, at);
                let right = u32_at(bytes, at + 4);
                if let Some(first_rank) =
                    Self::prior_duplicate_merge(bytes, merge_base, rank, left, right)
                {
                    return Err(Self::duplicate_merge_error(rank, at, first_rank));
                }
            }
            return Err(error);
        }
        let template =
            &bytes[sections[4].offset as usize..(sections[4].offset + sections[4].length) as usize];
        if template[0] != b'{' {
            return Err(ModelError::new(
                ModelStatus::TOKENIZER_SCHEMA,
                ErrorDomain::CHAT,
                0,
                sections[4].offset,
                b'{' as u64,
                template[0] as u64,
                FieldKind::TEMPLATE_BYTE,
                0,
            ));
        }
        if let Err(error) = core::str::from_utf8(template) {
            let at = error.valid_up_to();
            return Err(ModelError::new(
                ModelStatus::UTF8,
                ErrorDomain::CHAT,
                0,
                (sections[4].offset as usize + at) as u64,
                0,
                template[at] as u64,
                FieldKind::UTF8_BYTE,
                1,
            ));
        }
        Ok(())
    }

    fn prior_duplicate_merge(
        bytes: &[u8],
        merge_base: usize,
        rank: u32,
        left: u32,
        right: u32,
    ) -> Option<u32> {
        for first_rank in 0..rank {
            let first = merge_base + first_rank as usize * 12;
            if u32_at(bytes, first) == left && u32_at(bytes, first + 4) == right {
                return Some(first_rank);
            }
        }
        None
    }

    fn duplicate_merge_error(rank: u32, at: usize, first_rank: u32) -> ModelError {
        ModelError::new(
            ModelStatus::TOKENIZER_SCHEMA,
            ErrorDomain::MERGE,
            rank,
            at as u64,
            first_rank as u64,
            rank as u64,
            FieldKind::MERGE_DUPLICATE,
            0,
        )
    }
}
