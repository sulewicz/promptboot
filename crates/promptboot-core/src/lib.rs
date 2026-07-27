#![no_std]
#![allow(non_camel_case_types)]

mod arena;
mod fp32_sse2;
mod inference;
mod model;
pub mod sha256;
mod tokenizer;

pub use arena::{Arena, ArenaError, ArenaUsage, Region};
pub use inference::{
    greedy_token, sample_token, sample_token_with_repetition, top_logits_8, InferenceDomain,
    InferenceEngine, InferenceError, InferenceFieldKind, InferenceState, InferenceStatus,
    InferenceStep, InferenceUsage, SamplingState, TopLogit, CONTEXT_LIMIT, KV_BYTES, LOGIT_WORDS,
    NO_LAYER, NO_TENSOR, SAMPLING_POLICY, SAMPLING_REPETITION_PENALTY_MILLI,
    SAMPLING_TEMPERATURE_MILLI, SAMPLING_TOP_K, SAMPLING_TOP_P_MILLI, SCRATCH_BYTES,
};
pub use model::{
    ErrorDomain, FieldKind, Merge, ModelConfig, ModelError, ModelStatus, ModelView, TensorMeta,
    TensorView,
};
pub use tokenizer::{
    ConversationUsage, FrozenTokenizer, PieceKind, PieceUsage, PromptUsage, TokenizerUsage,
    INDEX_BYTES, TOKENIZER_INDEX_SHA256_HEX,
};

use core::mem::{align_of, size_of};
use core::ptr;

pub const ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PrimitiveStatus {
    OK = 0,
    NULL = 1,
    ALIGNMENT = 2,
    LENGTH = 3,
    OVERLAP = 4,
    ARITHMETIC_OVERFLOW = 5,
    ARENA_CAPACITY = 6,
    ARENA_SEALED = 7,
    BLOCK_ENCODING = 8,
    DIMENSION = 9,
    INDEX = 10,
    STATE = 11,
    NONFINITE_INPUT = 12,
    NONFINITE_OUTPUT = 13,
    UNSUPPORTED_OPERATION = 14,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PrimitiveOp {
    BIAS_RESIDUAL = 1,
    Q4 = 2,
    Q8 = 3,
    RMSNORM = 4,
    ROPE = 5,
    SOFTMAX = 6,
    GQA_ATTENTION = 7,
    SILU_SWIGLU = 8,
    ARGMAX = 9,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ArenaKind {
    NONE = 0,
    PRIMITIVE_SCRATCH = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrimitiveRequest {
    pub abi_version: u32,
    pub operation: u32,
    pub input: *const u8,
    pub input_bytes: u64,
    pub aux: *const u8,
    pub aux_bytes: u64,
    pub output: *mut u32,
    pub output_capacity_words: u64,
    pub scratch: *mut u8,
    pub scratch_bytes: u64,
    pub dim0: u32,
    pub dim1: u32,
    pub dim2: u32,
    pub dim3: u32,
    pub position: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PrimitiveResult {
    pub status: u32,
    pub output_words: u32,
    pub error_operation: u32,
    pub error_arena: u32,
    pub error_index: u32,
    pub reserved: u32,
    pub needed_bytes: u64,
    pub available_bytes: u64,
    pub arena_capacity: u64,
    pub arena_requested: u64,
    pub arena_committed: u64,
    pub arena_current: u64,
    pub arena_high_water: u64,
}

// The preceding field declaration order is chosen to preserve the exact Flux
// offsets: needed/available begin at +24/+32 and the five arena counters at
// +40 through +72. Keep these compile-time checks next to the ABI definition.
const _: [(); 96] = [(); size_of::<PrimitiveRequest>()];
const _: [(); 8] = [(); align_of::<PrimitiveRequest>()];
const _: [(); 80] = [(); size_of::<PrimitiveResult>()];
const _: [(); 8] = [(); align_of::<PrimitiveResult>()];

impl PrimitiveResult {
    const fn failure_safe() -> Self {
        Self {
            status: PrimitiveStatus::STATE as u32,
            output_words: 0,
            error_operation: 0,
            error_arena: ArenaKind::NONE as u32,
            error_index: 0,
            reserved: 0,
            needed_bytes: 0,
            available_bytes: 0,
            arena_capacity: 0,
            arena_requested: 0,
            arena_committed: 0,
            arena_current: 0,
            arena_high_water: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct AddressRange {
    start: usize,
    end: usize,
}

impl AddressRange {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy)]
struct Schema {
    input_bytes: u64,
    aux_bytes: u64,
    output_words: u64,
    input_f32: bool,
    aux_f32: bool,
}

fn checked_mul(left: u64, right: u64) -> Result<u64, PrimitiveStatus> {
    left.checked_mul(right)
        .ok_or(PrimitiveStatus::ARITHMETIC_OVERFLOW)
}

fn checked_add(left: u64, right: u64) -> Result<u64, PrimitiveStatus> {
    left.checked_add(right)
        .ok_or(PrimitiveStatus::ARITHMETIC_OVERFLOW)
}

fn f32_bytes(words: u64) -> Result<u64, PrimitiveStatus> {
    checked_mul(words, 4)
}

fn is_divisible(numerator: u64, denominator: u64) -> bool {
    // Callers establish denominator != 0. A bitwise remainder keeps the
    // freestanding rlib free of compiler-generated divide-by-zero panic paths.
    let mut remainder = 0u64;
    let mut bit = u64::BITS;
    while bit != 0 {
        bit -= 1;
        remainder = (remainder << 1) | ((numerator >> bit) & 1);
        if remainder >= denominator {
            remainder -= denominator;
        }
    }
    remainder == 0
}

fn schema(request: &PrimitiveRequest) -> Result<Schema, PrimitiveStatus> {
    let d0 = u64::from(request.dim0);
    let d1 = u64::from(request.dim1);
    let d2 = u64::from(request.dim2);
    let d3 = u64::from(request.dim3);
    let unused_zero = |values: &[u32]| values.iter().all(|value| *value == 0);
    match request.operation {
        1 => {
            if d0 == 0
                || !unused_zero(&[request.dim1, request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            Ok(Schema {
                input_bytes: f32_bytes(checked_mul(d0, 3)?)?,
                aux_bytes: 0,
                output_words: d0,
                input_f32: true,
                aux_f32: false,
            })
        }
        2 | 3 => {
            if d0 == 0
                || d1 == 0
                || d1 % 32 != 0
                || !unused_zero(&[request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            let blocks = checked_mul(d0, d1 / 32)?;
            let block_bytes = if request.operation == 2 { 18 } else { 34 };
            let dequant = checked_mul(d0, d1)?;
            Ok(Schema {
                input_bytes: checked_mul(blocks, block_bytes)?,
                aux_bytes: f32_bytes(d1)?,
                output_words: checked_add(checked_add(dequant, 1)?, d0)?,
                input_f32: false,
                aux_f32: true,
            })
        }
        4 => {
            if d0 == 0
                || !unused_zero(&[request.dim1, request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            Ok(Schema {
                input_bytes: f32_bytes(d0)?,
                aux_bytes: f32_bytes(d0)?,
                output_words: d0,
                input_f32: true,
                aux_f32: true,
            })
        }
        5 => {
            if !(request.dim0 == 4 || request.dim0 == 64)
                || d1 == 0
                || request.position >= CONTEXT_LIMIT
                || !unused_zero(&[request.dim2, request.dim3])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            let words = checked_mul(d0, d1)?;
            Ok(Schema {
                input_bytes: f32_bytes(words)?,
                aux_bytes: 0,
                output_words: words,
                input_f32: true,
                aux_f32: false,
            })
        }
        6 => {
            if d0 == 0
                || !unused_zero(&[request.dim1, request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            Ok(Schema {
                input_bytes: f32_bytes(d0)?,
                aux_bytes: 0,
                output_words: d0,
                input_f32: true,
                aux_f32: false,
            })
        }
        7 => {
            if d0 == 0
                || d1 == 0
                || !is_divisible(d0, d1)
                || !(request.dim3 == 2 || request.dim3 == 64)
                || d2 == 0
                || d2 > CONTEXT_LIMIT as u64
                || request.position != 0
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            let queries = checked_mul(d0, d3)?;
            let kv_words = checked_mul(checked_mul(d2, d1)?, d3)?;
            let input_words = checked_add(queries, checked_mul(kv_words, 2)?)?;
            let per_head = checked_add(d3, checked_mul(d2, 2)?)?;
            Ok(Schema {
                input_bytes: f32_bytes(input_words)?,
                aux_bytes: 0,
                output_words: checked_mul(d0, per_head)?,
                input_f32: true,
                aux_f32: false,
            })
        }
        8 => {
            if d0 == 0
                || !unused_zero(&[request.dim1, request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            Ok(Schema {
                input_bytes: f32_bytes(d0)?,
                aux_bytes: f32_bytes(d0)?,
                output_words: checked_mul(d0, 2)?,
                input_f32: true,
                aux_f32: true,
            })
        }
        9 => {
            if d0 == 0
                || !unused_zero(&[request.dim1, request.dim2, request.dim3, request.position])
            {
                return Err(PrimitiveStatus::DIMENSION);
            }
            Ok(Schema {
                input_bytes: f32_bytes(d0)?,
                aux_bytes: 0,
                output_words: 1,
                input_f32: true,
                aux_f32: false,
            })
        }
        _ => Err(PrimitiveStatus::UNSUPPORTED_OPERATION),
    }
}

fn fixed_range(address: usize, bytes: usize) -> Result<AddressRange, PrimitiveStatus> {
    let end = address
        .checked_add(bytes)
        .ok_or(PrimitiveStatus::ARITHMETIC_OVERFLOW)?;
    Ok(AddressRange {
        start: address,
        end,
    })
}

fn declared_range(pointer: *const u8, bytes: u64) -> Result<Option<AddressRange>, PrimitiveStatus> {
    if pointer.is_null() || bytes == 0 {
        return Ok(None);
    }
    let length = usize::try_from(bytes).map_err(|_| PrimitiveStatus::ARITHMETIC_OVERFLOW)?;
    fixed_range(pointer as usize, length).map(Some)
}

fn write_failure(
    result: *mut PrimitiveResult,
    status: PrimitiveStatus,
    operation: u32,
    arena: ArenaKind,
    index: u32,
    needed: u64,
    available: u64,
    usage: Option<ArenaUsage>,
) -> u32 {
    unsafe {
        (*result).status = status as u32;
        (*result).output_words = 0;
        (*result).error_operation = operation;
        (*result).error_arena = arena as u32;
        (*result).error_index = index;
        (*result).needed_bytes = needed;
        (*result).available_bytes = available;
        if let Some(value) = usage {
            (*result).arena_capacity = value.capacity;
            (*result).arena_requested = value.requested;
            (*result).arena_committed = value.committed;
            (*result).arena_current = value.current;
            (*result).arena_high_water = value.high_water;
        }
    }
    status as u32
}

fn invalid_empty(pointer: *const u8, bytes: u64) -> Option<PrimitiveStatus> {
    if pointer.is_null() && bytes != 0 {
        Some(PrimitiveStatus::NULL)
    } else if !pointer.is_null() && bytes == 0 {
        Some(PrimitiveStatus::LENGTH)
    } else {
        None
    }
}

/// Execute one deterministic primitive through the integer/pointer-only ABI.
///
/// # Safety
/// Every non-null declared caller range must be accessible for its declared
/// length. The function validates alignment, arithmetic and the complete alias
/// matrix before dereferencing those ranges.
#[no_mangle]
pub unsafe extern "C" fn promptboot_run_primitive(
    request: *const PrimitiveRequest,
    result: *mut PrimitiveResult,
) -> u32 {
    if result.is_null() {
        return PrimitiveStatus::NULL as u32;
    }
    if (result as usize) & (align_of::<PrimitiveResult>() - 1) != 0 {
        return PrimitiveStatus::ALIGNMENT as u32;
    }
    let result_range = match fixed_range(result as usize, size_of::<PrimitiveResult>()) {
        Ok(value) => value,
        Err(status) => return status as u32,
    };
    if request.is_null() {
        return PrimitiveStatus::NULL as u32;
    }
    if (request as usize) & (align_of::<PrimitiveRequest>() - 1) != 0 {
        return PrimitiveStatus::ALIGNMENT as u32;
    }
    let request_range = match fixed_range(request as usize, size_of::<PrimitiveRequest>()) {
        Ok(value) => value,
        Err(status) => return status as u32,
    };
    if request_range.overlaps(result_range) {
        return PrimitiveStatus::OVERLAP as u32;
    }

    let local = ptr::read(request);
    let output_bytes_declared = match local.output_capacity_words.checked_mul(4) {
        Some(value) => value,
        None => return PrimitiveStatus::ARITHMETIC_OVERFLOW as u32,
    };
    let ranges = [
        declared_range(local.input, local.input_bytes),
        declared_range(local.aux, local.aux_bytes),
        declared_range(local.output.cast(), output_bytes_declared),
        declared_range(local.scratch, local.scratch_bytes),
    ];
    let mut checked = [None; 4];
    for (index, value) in ranges.into_iter().enumerate() {
        match value {
            Ok(range) => checked[index] = range,
            Err(status) => return status as u32,
        }
    }
    for range in checked.into_iter().flatten() {
        if result_range.overlaps(range) {
            return PrimitiveStatus::OVERLAP as u32;
        }
    }

    ptr::write(result, PrimitiveResult::failure_safe());
    let input_range = checked[0];
    let aux_range = checked[1];
    let output_range = checked[2];
    let scratch_range = checked[3];
    let forbidden = [
        (output_range, scratch_range),
        (output_range, Some(request_range)),
        (output_range, input_range),
        (output_range, aux_range),
        (scratch_range, Some(request_range)),
        (scratch_range, input_range),
        (scratch_range, aux_range),
    ];
    for (left, right) in forbidden {
        if matches!((left, right), (Some(a), Some(b)) if a.overlaps(b)) {
            return write_failure(
                result,
                PrimitiveStatus::OVERLAP,
                local.operation,
                ArenaKind::NONE,
                0,
                0,
                0,
                None,
            );
        }
    }

    if local.abi_version != ABI_VERSION || local.flags != 0 {
        return write_failure(
            result,
            PrimitiveStatus::STATE,
            local.operation,
            ArenaKind::NONE,
            0,
            ABI_VERSION as u64,
            local.abi_version as u64,
            None,
        );
    }
    if local.operation == PrimitiveOp::ROPE as u32 && local.position >= CONTEXT_LIMIT {
        return write_failure(
            result,
            PrimitiveStatus::INDEX,
            local.operation,
            ArenaKind::NONE,
            local.position,
            CONTEXT_LIMIT as u64,
            u64::from(local.position),
            None,
        );
    }
    let schema = match schema(&local) {
        Ok(value) => value,
        Err(status) => {
            return write_failure(
                result,
                status,
                local.operation,
                ArenaKind::NONE,
                0,
                0,
                0,
                None,
            )
        }
    };
    if schema.output_words > u64::from(u32::MAX) {
        return write_failure(
            result,
            PrimitiveStatus::ARITHMETIC_OVERFLOW,
            local.operation,
            ArenaKind::NONE,
            0,
            schema.output_words,
            u64::from(u32::MAX),
            None,
        );
    }
    let output_bytes = match schema
        .output_words
        .checked_mul(4)
        .and_then(|value| usize::try_from(value).ok().map(|_| value))
    {
        Some(value) => value,
        None => {
            return write_failure(
                result,
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                local.operation,
                ArenaKind::NONE,
                0,
                schema.output_words,
                u64::from(u32::MAX),
                None,
            )
        }
    };

    for (pointer, bytes) in [
        (local.input, local.input_bytes),
        (local.aux, local.aux_bytes),
        (local.output.cast_const().cast(), output_bytes_declared),
        (local.scratch.cast_const(), local.scratch_bytes),
    ] {
        if let Some(status) = invalid_empty(pointer, bytes) {
            return write_failure(
                result,
                status,
                local.operation,
                ArenaKind::NONE,
                0,
                0,
                0,
                None,
            );
        }
    }
    if local.input_bytes != schema.input_bytes {
        return write_failure(
            result,
            PrimitiveStatus::LENGTH,
            local.operation,
            ArenaKind::NONE,
            0,
            schema.input_bytes,
            local.input_bytes,
            None,
        );
    }
    if local.aux_bytes != schema.aux_bytes {
        return write_failure(
            result,
            PrimitiveStatus::LENGTH,
            local.operation,
            ArenaKind::NONE,
            0,
            schema.aux_bytes,
            local.aux_bytes,
            None,
        );
    }
    if local.output.is_null() {
        return write_failure(
            result,
            PrimitiveStatus::LENGTH,
            local.operation,
            ArenaKind::NONE,
            0,
            schema.output_words,
            0,
            None,
        );
    }
    if local.output_capacity_words < schema.output_words {
        return write_failure(
            result,
            PrimitiveStatus::LENGTH,
            local.operation,
            ArenaKind::NONE,
            0,
            schema.output_words,
            local.output_capacity_words,
            None,
        );
    }
    if local.scratch.is_null() {
        return write_failure(
            result,
            PrimitiveStatus::ARENA_CAPACITY,
            local.operation,
            ArenaKind::PRIMITIVE_SCRATCH,
            0,
            output_bytes,
            0,
            None,
        );
    }
    if local.scratch_bytes < output_bytes {
        return write_failure(
            result,
            PrimitiveStatus::ARENA_CAPACITY,
            local.operation,
            ArenaKind::PRIMITIVE_SCRATCH,
            0,
            output_bytes,
            local.scratch_bytes,
            None,
        );
    }
    if schema.input_f32 && (local.input as usize) & 3 != 0 {
        return write_failure(
            result,
            PrimitiveStatus::ALIGNMENT,
            local.operation,
            ArenaKind::NONE,
            0,
            4,
            0,
            None,
        );
    }
    if schema.aux_f32 && (local.aux as usize) & 3 != 0 {
        return write_failure(
            result,
            PrimitiveStatus::ALIGNMENT,
            local.operation,
            ArenaKind::NONE,
            0,
            4,
            0,
            None,
        );
    }
    if (local.output as usize) & 3 != 0 {
        return write_failure(
            result,
            PrimitiveStatus::ALIGNMENT,
            local.operation,
            ArenaKind::NONE,
            0,
            4,
            0,
            None,
        );
    }
    if (local.scratch as usize) & 63 != 0 {
        return write_failure(
            result,
            PrimitiveStatus::ALIGNMENT,
            local.operation,
            ArenaKind::PRIMITIVE_SCRATCH,
            0,
            64,
            0,
            None,
        );
    }

    let scratch_len = match usize::try_from(local.scratch_bytes) {
        Ok(value) => value,
        Err(_) => {
            return write_failure(
                result,
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                local.operation,
                ArenaKind::PRIMITIVE_SCRATCH,
                0,
                local.scratch_bytes,
                usize::MAX as u64,
                None,
            )
        }
    };
    let scratch = core::slice::from_raw_parts_mut(local.scratch, scratch_len);
    let mut arena = match Arena::new(scratch) {
        Ok(value) => value,
        Err(error) => {
            return write_failure(
                result,
                error.status,
                local.operation,
                ArenaKind::PRIMITIVE_SCRATCH,
                0,
                error.needed_bytes,
                error.available_bytes,
                None,
            )
        }
    };
    let region = match arena.allocate(output_bytes, 64) {
        Ok(value) => value,
        Err(error) => {
            let usage = arena.usage();
            return write_failure(
                result,
                error.status,
                local.operation,
                ArenaKind::PRIMITIVE_SCRATCH,
                0,
                error.needed_bytes,
                error.available_bytes,
                Some(usage),
            );
        }
    };
    if let Err(error) = arena.seal() {
        let usage = arena.usage();
        return write_failure(
            result,
            error.status,
            local.operation,
            ArenaKind::PRIMITIVE_SCRATCH,
            0,
            error.needed_bytes,
            error.available_bytes,
            Some(usage),
        );
    }
    let stage_pointer = {
        let stage = match arena.region_mut(region) {
            Ok(value) => value,
            Err(error) => {
                let usage = arena.usage();
                return write_failure(
                    result,
                    error.status,
                    local.operation,
                    ArenaKind::PRIMITIVE_SCRATCH,
                    0,
                    error.needed_bytes,
                    error.available_bytes,
                    Some(usage),
                );
            }
        };
        stage.as_mut_ptr()
    };
    let outcome = fp32_sse2::execute(
        local.operation,
        local.input,
        local.aux,
        stage_pointer,
        local.dim0,
        local.dim1,
        local.dim2,
        local.dim3,
        local.position,
    );
    let usage = arena.usage();
    if outcome.status != PrimitiveStatus::OK {
        return write_failure(
            result,
            outcome.status,
            local.operation,
            ArenaKind::PRIMITIVE_SCRATCH,
            outcome.index,
            0,
            0,
            Some(usage),
        );
    }
    ptr::copy_nonoverlapping(stage_pointer, local.output.cast(), output_bytes as usize);
    (*result).arena_capacity = usage.capacity;
    (*result).arena_requested = usage.requested;
    (*result).arena_committed = usage.committed;
    (*result).arena_current = usage.current;
    (*result).arena_high_water = usage.high_water;
    (*result).output_words = schema.output_words as u32;
    (*result).status = PrimitiveStatus::OK as u32;
    PrimitiveStatus::OK as u32
}

#[cfg(test)]
mod tests;
