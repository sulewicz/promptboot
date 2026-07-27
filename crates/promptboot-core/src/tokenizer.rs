use core::mem::{align_of, size_of};

use crate::model::{
    ErrorDomain, FieldKind, ModelError, ModelStatus, ModelView, MERGE_COUNT, VOCAB_COUNT,
};
use crate::sha256::digest;
use crate::CONTEXT_LIMIT;

#[cfg(test)]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(test)]
static TEST_INDEX_IDENTITY_FAULT: AtomicU8 = AtomicU8::new(0);
#[cfg(test)]
static TEST_PROMPT_AFTER_WRITE_FAULT: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
pub(crate) fn set_index_identity_fault_for_test(value: u8) {
    TEST_INDEX_IDENTITY_FAULT.store(value, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn set_prompt_after_write_fault_for_test(value: bool) {
    TEST_PROMPT_AFTER_WRITE_FAULT.store(u8::from(value), Ordering::SeqCst);
}

pub const INDEX_BYTES: usize = 4_195_328;
const TABLE_BYTES: usize = 4_194_304;
const TABLE_ENTRIES: usize = 262_144;
const BYTE_MAP_OFFSET: usize = TABLE_BYTES;
const SCRATCH_BYTES: usize = 5_120;
const STAGED_TOKENS: usize = 0;
const CURRENT_SEGMENT: usize = 2_396;
const STAGED_RENDERED: usize = 4_444;
pub(crate) const MAX_TOKENS: usize = 599;
const MAX_SEGMENT: usize = 512;
const IM_START: u32 = 151_644;
pub(crate) const IM_END: u32 = 151_645;
const END_OF_TEXT: u32 = 151_643;
const TOOL_CALL: &[u8] = b"<tool_call>";
const TOOL_END: &[u8] = b"</tool_call>";

pub const TOKENIZER_INDEX_SHA256_HEX: &[u8; 64] =
    b"393d697ecc53a12cb4ea6940f08bfbe8ba4f56bc0809a68fb28af3cbc87cc32a";
const INDEX_HASH: [u8; 32] = hex32(*TOKENIZER_INDEX_SHA256_HEX);
const BYTE_MAP_HASH: [u8; 32] =
    hex32(*b"686aa5b5be47b58e8b4eee7a5632994d9e3dd851c3f69b7ddfabc39e19aa755a");

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}
const fn hex32(input: [u8; 64]) -> [u8; 32] {
    let mut output = [0u8; 32];
    let mut at = 0;
    while at < 32 {
        output[at] = hex_nibble(input[at * 2]) << 4 | hex_nibble(input[at * 2 + 1]);
        at += 1;
    }
    output
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TokenizerUsage {
    pub capacity: u64,
    pub requested: u64,
    pub committed: u64,
    pub current: u64,
    pub high_water: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PromptUsage {
    pub rendered_bytes: u32,
    pub token_count: u32,
    pub prefix_tokens: u32,
    pub im_start_count: u32,
    pub im_end_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ConversationUsage {
    pub rendered_bytes: u32,
    pub user_tokens: u32,
    pub fresh_prompt_tokens: u32,
    pub history_tokens: u32,
    pub prompt_tokens: u32,
    pub prefix_tokens: u32,
}

impl ConversationUsage {
    pub const ZERO: Self = Self {
        rendered_bytes: 0,
        user_tokens: 0,
        fresh_prompt_tokens: 0,
        history_tokens: 0,
        prompt_tokens: 0,
        prefix_tokens: 0,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PieceKind {
    TEXT = 1,
    EOS = 2,
    SUPPRESSED = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PieceUsage {
    pub kind: u32,
    pub bytes: u32,
}

const _: [(); 40] = [(); size_of::<TokenizerUsage>()];
const _: [(); 8] = [(); align_of::<TokenizerUsage>()];
const _: [(); 20] = [(); size_of::<PromptUsage>()];
const _: [(); 24] = [(); size_of::<ConversationUsage>()];
const _: [(); 8] = [(); size_of::<PieceUsage>()];

struct PromptComputation {
    rendered_bytes: usize,
    token_count: usize,
    prefix_tokens: usize,
    user_tokens: usize,
}

pub struct FrozenTokenizer<'model, 'bytes, 'index> {
    model: &'model ModelView<'bytes>,
    index: &'index [u8],
    usage: TokenizerUsage,
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}
fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn hash_slot(left: u32, right: u32) -> usize {
    let key = ((left as u64) << 32) | right as u64;
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 46) as usize
}

fn index_error(kind: FieldKind, expected: u64, actual: u64) -> ModelError {
    ModelError::new(
        ModelStatus::INDEX_CORRUPT,
        ErrorDomain::INDEX,
        0,
        0,
        expected,
        actual,
        kind,
        0,
    )
}

fn first_hash_error(expected: &[u8; 32], actual: &[u8; 32], offset: u64) -> Option<ModelError> {
    for at in 0..32 {
        if expected[at] != actual[at] {
            return Some(ModelError::new(
                ModelStatus::INDEX_CORRUPT,
                ErrorDomain::INDEX,
                0,
                offset,
                expected[at] as u64,
                actual[at] as u64,
                FieldKind::INDEX_HASH_BYTE,
                at as u32,
            ));
        }
    }
    None
}

fn gpt2_codepoint(byte: u8) -> u32 {
    if (0x21..=0x7e).contains(&byte)
        || (0xa1..=0xac).contains(&byte)
        || (0xae..=0xff).contains(&byte)
    {
        return byte as u32;
    }
    let mut ordinal = 0u32;
    let mut candidate = 0u16;
    while candidate < byte as u16 {
        let value = candidate as u8;
        if !((0x21..=0x7e).contains(&value)
            || (0xa1..=0xac).contains(&value)
            || (0xae..=0xff).contains(&value))
        {
            ordinal += 1;
        }
        candidate += 1;
    }
    256 + ordinal
}

fn encoded_byte(byte: u8, output: &mut [u8; 2]) -> usize {
    let point = gpt2_codepoint(byte);
    if point < 0x80 {
        output[0] = point as u8;
        1
    } else {
        output[0] = 0xc0 | (point >> 6) as u8;
        output[1] = 0x80 | (point & 63) as u8;
        2
    }
}

fn validate_prompt_request(
    user: &[u8],
    rendered_len: usize,
    token_len: usize,
    scratch: &[u8],
) -> Result<(), ModelError> {
    if user.len() > 512 {
        return Err(ModelError::new(
            ModelStatus::LENGTH,
            ErrorDomain::USER,
            0,
            0,
            512,
            user.len() as u64,
            FieldKind::CAPACITY,
            0,
        ));
    }
    for (at, byte) in user.iter().copied().enumerate() {
        if !(0x20..=0x7e).contains(&byte) {
            return Err(ModelError::new(
                ModelStatus::INPUT,
                ErrorDomain::USER,
                at as u32,
                at as u64,
                0x20_007e,
                byte as u64,
                FieldKind::USER_BYTE,
                0,
            ));
        }
    }
    let rendered_needed = 148 + user.len();
    if rendered_len < rendered_needed {
        return Err(ModelError::new(
            ModelStatus::OUTPUT_CAPACITY,
            ErrorDomain::RENDERED_OUTPUT,
            0,
            0,
            rendered_needed as u64,
            rendered_len as u64,
            FieldKind::CAPACITY,
            0,
        ));
    }
    if token_len < MAX_TOKENS {
        return Err(ModelError::new(
            ModelStatus::OUTPUT_CAPACITY,
            ErrorDomain::TOKEN_OUTPUT,
            0,
            0,
            MAX_TOKENS as u64,
            token_len as u64,
            FieldKind::CAPACITY,
            0,
        ));
    }
    if scratch.len() < SCRATCH_BYTES {
        return Err(ModelError::new(
            ModelStatus::SCRATCH_CAPACITY,
            ErrorDomain::SCRATCH,
            0,
            0,
            SCRATCH_BYTES as u64,
            scratch.len() as u64,
            FieldKind::CAPACITY,
            0,
        ));
    }
    let scratch_alignment = scratch.as_ptr() as usize & 63;
    if scratch_alignment != 0 {
        return Err(ModelError::new(
            ModelStatus::ALIGNMENT,
            ErrorDomain::SCRATCH,
            0,
            0,
            64,
            scratch_alignment as u64,
            FieldKind::BASE_ALIGNMENT,
            0,
        ));
    }
    Ok(())
}

impl<'model, 'bytes, 'index> FrozenTokenizer<'model, 'bytes, 'index> {
    pub fn validate_conversation_history(&self, history: &[u32]) -> Result<(), ModelError> {
        if history.is_empty()
            || history.len() > CONTEXT_LIMIT as usize
            || history.last().copied() != Some(IM_END)
        {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::TOKEN_OUTPUT,
                history.len().saturating_sub(1) as u32,
                0,
                IM_END as u64,
                history.last().copied().unwrap_or(0) as u64,
                FieldKind::STATE_FIELD,
                11,
            ));
        }
        Ok(())
    }

    pub fn build(
        model: &'model ModelView<'bytes>,
        index_storage: &'index mut [u8],
    ) -> Result<Self, ModelError> {
        if index_storage.len() < INDEX_BYTES {
            return Err(ModelError::new(
                ModelStatus::INDEX_CAPACITY,
                ErrorDomain::INDEX,
                0,
                0,
                INDEX_BYTES as u64,
                index_storage.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        let misalignment = index_storage.as_ptr() as usize & 63;
        if misalignment != 0 {
            return Err(ModelError::new(
                ModelStatus::INDEX_ALIGNMENT,
                ErrorDomain::INDEX,
                0,
                0,
                64,
                misalignment as u64,
                FieldKind::BASE_ALIGNMENT,
                0,
            ));
        }
        if model.config().vocab_count != VOCAB_COUNT || model.config().merge_count != MERGE_COUNT {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::INDEX,
                0,
                0,
                1,
                0,
                FieldKind::STATE_FIELD,
                0,
            ));
        }
        for (at, byte) in index_storage[..INDEX_BYTES].iter().copied().enumerate() {
            if byte != 0 {
                return Err(ModelError::new(
                    ModelStatus::STATE,
                    ErrorDomain::INDEX,
                    0,
                    at as u64,
                    0,
                    byte as u64,
                    FieldKind::STATE_FIELD,
                    1,
                ));
            }
        }

        let prefix = &mut index_storage[..INDEX_BYTES];
        prefix[..TABLE_BYTES].fill(0xff);
        let construction = Self::construct(model, prefix);
        if let Err(error) = construction {
            prefix.fill(0);
            return Err(error);
        }
        let index: &'index [u8] = &*index_storage;
        Ok(Self {
            model,
            index: &index[..INDEX_BYTES],
            usage: TokenizerUsage {
                capacity: index_storage.len() as u64,
                requested: INDEX_BYTES as u64,
                committed: INDEX_BYTES as u64,
                current: INDEX_BYTES as u64,
                high_water: INDEX_BYTES as u64,
            },
        })
    }

    fn construct(model: &ModelView<'_>, prefix: &mut [u8]) -> Result<(), ModelError> {
        let mut byte_ids = [u32::MAX; 256];
        for token in 0..VOCAB_COUNT {
            let piece = model.token_bytes(token)?;
            if piece.len() <= 2 {
                for byte in 0..=255u16 {
                    if byte_ids[byte as usize] != u32::MAX {
                        continue;
                    }
                    let mut encoded = [0u8; 2];
                    let length = encoded_byte(byte as u8, &mut encoded);
                    if piece == &encoded[..length] {
                        byte_ids[byte as usize] = token;
                    }
                }
            }
        }
        for (byte, token) in byte_ids.iter().copied().enumerate() {
            if token == u32::MAX {
                return Err(ModelError::new(
                    ModelStatus::STATE,
                    ErrorDomain::INDEX,
                    byte as u32,
                    0,
                    1,
                    0,
                    FieldKind::STATE_FIELD,
                    0,
                ));
            }
            put_u32(prefix, BYTE_MAP_OFFSET + byte * 4, token);
        }

        let mut max_displacement = 0usize;
        for rank in 0..MERGE_COUNT {
            let merge = model.merge(rank)?;
            let initial = hash_slot(merge.left, merge.right);
            let mut displacement = 0usize;
            loop {
                if displacement >= TABLE_ENTRIES {
                    return Err(index_error(
                        FieldKind::INDEX_MISS_BOUND,
                        51,
                        displacement as u64,
                    ));
                }
                let slot = (initial + displacement) & (TABLE_ENTRIES - 1);
                let at = slot * 16;
                let left = u32_at(prefix, at);
                if left == u32::MAX {
                    put_u32(prefix, at, merge.left);
                    put_u32(prefix, at + 4, merge.right);
                    put_u32(prefix, at + 8, merge.result);
                    put_u32(prefix, at + 12, rank);
                    if displacement > max_displacement {
                        max_displacement = displacement;
                    }
                    break;
                }
                if left == merge.left && u32_at(prefix, at + 4) == merge.right {
                    return Err(ModelError::new(
                        ModelStatus::TOKENIZER_SCHEMA,
                        ErrorDomain::MERGE,
                        rank,
                        0,
                        u32_at(prefix, at + 12) as u64,
                        rank as u64,
                        FieldKind::MERGE_DUPLICATE,
                        0,
                    ));
                }
                displacement += 1;
            }
        }
        #[cfg(test)]
        let max_displacement = if TEST_INDEX_IDENTITY_FAULT.load(Ordering::SeqCst) == 1 {
            41
        } else {
            max_displacement
        };
        if max_displacement != 42 {
            return Err(index_error(
                FieldKind::INDEX_MAX_DISPLACEMENT,
                42,
                max_displacement as u64,
            ));
        }

        let mut max_cluster = 0usize;
        let mut current = 0usize;
        for slot in 0..TABLE_ENTRIES * 2 {
            if u32_at(prefix, (slot & (TABLE_ENTRIES - 1)) * 16) != u32::MAX {
                current += 1;
                if current > max_cluster {
                    max_cluster = current;
                }
            } else {
                current = 0;
            }
            if slot >= TABLE_ENTRIES && current == 0 {
                break;
            }
        }
        #[cfg(test)]
        let max_cluster = if TEST_INDEX_IDENTITY_FAULT.load(Ordering::SeqCst) == 2 {
            49
        } else {
            max_cluster
        };
        if max_cluster != 50 {
            return Err(index_error(
                FieldKind::INDEX_MAX_CLUSTER,
                50,
                max_cluster as u64,
            ));
        }
        let mut max_hit = 0usize;
        for rank in 0..MERGE_COUNT {
            let merge = model.merge(rank)?;
            let (_, probes) = lookup(prefix, merge.left, merge.right);
            if probes > max_hit {
                max_hit = probes;
            }
        }
        #[cfg(test)]
        let max_hit = if TEST_INDEX_IDENTITY_FAULT.load(Ordering::SeqCst) == 3 {
            42
        } else {
            max_hit
        };
        if max_hit != 43 {
            return Err(index_error(FieldKind::INDEX_HIT_BOUND, 43, max_hit as u64));
        }
        let max_miss = max_cluster + 1;
        #[cfg(test)]
        let max_miss = if TEST_INDEX_IDENTITY_FAULT.load(Ordering::SeqCst) == 4 {
            50
        } else {
            max_miss
        };
        if max_miss != 51 {
            return Err(index_error(
                FieldKind::INDEX_MISS_BOUND,
                51,
                max_miss as u64,
            ));
        }
        let map_hash = digest(&prefix[BYTE_MAP_OFFSET..INDEX_BYTES]);
        if let Some(error) = first_hash_error(&BYTE_MAP_HASH, &map_hash, BYTE_MAP_OFFSET as u64) {
            return Err(error);
        }
        #[allow(unused_mut)]
        let mut whole_hash = digest(prefix);
        #[cfg(test)]
        if TEST_INDEX_IDENTITY_FAULT.load(Ordering::SeqCst) == 5 {
            whole_hash[0] ^= 1;
        }
        if let Some(error) = first_hash_error(&INDEX_HASH, &whole_hash, 0) {
            return Err(error);
        }
        Ok(())
    }

    pub fn usage(&self) -> TokenizerUsage {
        self.usage
    }

    /// All mutable buffers must be disjoint. Rust rejects aliases before this
    /// safe API can run:
    ///
    /// ```compile_fail
    /// # use promptboot_core::FrozenTokenizer;
    /// fn alias(tok: &FrozenTokenizer<'_, '_, '_>, bytes: &mut [u8], words: &mut [u32]) {
    ///     tok.render_and_tokenize(b"x", bytes, words, bytes).unwrap();
    /// }
    /// ```
    pub fn render_and_tokenize(
        &self,
        user: &[u8],
        rendered: &mut [u8],
        tokens: &mut [u32],
        scratch: &mut [u8],
    ) -> Result<PromptUsage, ModelError> {
        if user.len() > 512 {
            return Err(ModelError::new(
                ModelStatus::LENGTH,
                ErrorDomain::USER,
                0,
                0,
                512,
                user.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        for (at, byte) in user.iter().copied().enumerate() {
            if !(0x20..=0x7e).contains(&byte) {
                return Err(ModelError::new(
                    ModelStatus::INPUT,
                    ErrorDomain::USER,
                    at as u32,
                    at as u64,
                    0x20_007e,
                    byte as u64,
                    FieldKind::USER_BYTE,
                    0,
                ));
            }
        }
        let rendered_needed = 148 + user.len();
        if rendered.len() < rendered_needed {
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::RENDERED_OUTPUT,
                0,
                0,
                rendered_needed as u64,
                rendered.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        if tokens.len() < MAX_TOKENS {
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                MAX_TOKENS as u64,
                tokens.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        if scratch.len() < SCRATCH_BYTES {
            return Err(ModelError::new(
                ModelStatus::SCRATCH_CAPACITY,
                ErrorDomain::SCRATCH,
                0,
                0,
                SCRATCH_BYTES as u64,
                scratch.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        let scratch_alignment = scratch.as_ptr() as usize & 63;
        if scratch_alignment != 0 {
            return Err(ModelError::new(
                ModelStatus::ALIGNMENT,
                ErrorDomain::SCRATCH,
                0,
                0,
                64,
                scratch_alignment as u64,
                FieldKind::BASE_ALIGNMENT,
                0,
            ));
        }

        let computation = self.compute_prompt(user, &mut scratch[..SCRATCH_BYTES]);
        match computation {
            Ok(computation) => {
                rendered[..computation.rendered_bytes].copy_from_slice(
                    &scratch[STAGED_RENDERED..STAGED_RENDERED + computation.rendered_bytes],
                );
                for (at, output) in tokens[..computation.token_count].iter_mut().enumerate() {
                    *output = u32_at(scratch, STAGED_TOKENS + at * 4);
                }
                scratch[..SCRATCH_BYTES].fill(0);
                Ok(PromptUsage {
                    rendered_bytes: computation.rendered_bytes as u32,
                    token_count: computation.token_count as u32,
                    prefix_tokens: computation.prefix_tokens as u32,
                    im_start_count: 3,
                    im_end_count: 2,
                })
            }
            Err(error) => {
                scratch[..SCRATCH_BYTES].fill(0);
                Err(error)
            }
        }
    }

    /// Render and tokenize one prompt against an optional, already committed
    /// whole-turn token history.  `staging_tokens` is caller-owned transient
    /// storage (the target uses the logits arena before inference); `tokens`
    /// is published only after the complete prospective prompt is valid.
    pub fn render_conversation_and_tokenize(
        &self,
        history: &[u32],
        user: &[u8],
        rendered: &mut [u8],
        staging_tokens: &mut [u32],
        tokens: &mut [u32],
        scratch: &mut [u8],
        outcome: &mut ConversationUsage,
    ) -> Result<ConversationUsage, ModelError> {
        if history.len() > CONTEXT_LIMIT as usize {
            return Err(ModelError::new(
                ModelStatus::LENGTH,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                CONTEXT_LIMIT as u64,
                history.len() as u64,
                FieldKind::CAPACITY,
                10,
            ));
        }
        if !history.is_empty() {
            self.validate_conversation_history(history)?;
        }
        if staging_tokens.len() < MAX_TOKENS {
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                MAX_TOKENS as u64,
                staging_tokens.len() as u64,
                FieldKind::CAPACITY,
                12,
            ));
        }
        if tokens.len() < CONTEXT_LIMIT as usize {
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                CONTEXT_LIMIT as u64,
                tokens.len() as u64,
                FieldKind::CAPACITY,
                13,
            ));
        }

        if let Err(error) =
            validate_prompt_request(user, rendered.len(), staging_tokens.len(), scratch)
        {
            return Err(error);
        }
        let computation = match self.compute_prompt(user, &mut scratch[..SCRATCH_BYTES]) {
            Ok(value) => value,
            Err(error) => {
                scratch[..SCRATCH_BYTES].fill(0);
                return Err(error);
            }
        };
        let rendered_count = computation.rendered_bytes;
        let fresh_count = computation.token_count;
        *outcome = ConversationUsage {
            rendered_bytes: rendered_count as u32,
            user_tokens: computation.user_tokens as u32,
            fresh_prompt_tokens: fresh_count as u32,
            history_tokens: history.len() as u32,
            prompt_tokens: 0,
            prefix_tokens: computation.prefix_tokens as u32,
        };
        let suffix_start = if history.is_empty() {
            0
        } else {
            let mut seen = 0usize;
            let mut second = None;
            for at in 0..fresh_count {
                let token = u32_at(scratch, STAGED_TOKENS + at * 4);
                if token == IM_START {
                    seen += 1;
                    if seen == 2 {
                        second = Some(at);
                        break;
                    }
                }
            }
            match second.and_then(|at| at.checked_sub(1)) {
                Some(at) if u32_at(scratch, STAGED_TOKENS + at * 4) == 198 => at,
                _ => {
                    scratch[..SCRATCH_BYTES].fill(0);
                    return Err(ModelError::new(
                        ModelStatus::STATE,
                        ErrorDomain::TOKEN_OUTPUT,
                        0,
                        0,
                        1,
                        0,
                        FieldKind::STATE_FIELD,
                        14,
                    ));
                }
            }
        };
        let suffix_count = fresh_count - suffix_start;
        let prospective = match history.len().checked_add(suffix_count) {
            Some(value) => value,
            None => {
                scratch[..SCRATCH_BYTES].fill(0);
                return Err(ModelError::new(
                    ModelStatus::LENGTH,
                    ErrorDomain::TOKEN_OUTPUT,
                    0,
                    0,
                    CONTEXT_LIMIT as u64,
                    u64::MAX,
                    FieldKind::OP_ADD,
                    15,
                ));
            }
        };
        outcome.prompt_tokens = u32::try_from(prospective).unwrap_or(u32::MAX);
        if prospective > CONTEXT_LIMIT as usize {
            scratch[..SCRATCH_BYTES].fill(0);
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                CONTEXT_LIMIT as u64,
                prospective as u64,
                FieldKind::CAPACITY,
                16,
            ));
        }

        // Every fallible check is complete before any caller-visible output is
        // published. The scratch arena is the transaction staging area, so the
        // freestanding call chain needs no additional stack array.
        rendered[..rendered_count]
            .copy_from_slice(&scratch[STAGED_RENDERED..STAGED_RENDERED + rendered_count]);
        for (at, output) in staging_tokens[..fresh_count].iter_mut().enumerate() {
            *output = u32_at(scratch, STAGED_TOKENS + at * 4);
        }
        tokens[..history.len()].copy_from_slice(history);
        tokens[history.len()..prospective]
            .copy_from_slice(&staging_tokens[suffix_start..fresh_count]);
        staging_tokens[..MAX_TOKENS].fill(0);
        scratch[..SCRATCH_BYTES].fill(0);
        Ok(*outcome)
    }

    fn compute_prompt(
        &self,
        user: &[u8],
        scratch: &mut [u8],
    ) -> Result<PromptComputation, ModelError> {
        let before = b"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        let after = b"<|im_end|>\n<|im_start|>assistant\n";
        let rendered_count = before.len() + user.len() + after.len();
        scratch[STAGED_RENDERED..STAGED_RENDERED + before.len()].copy_from_slice(before);
        scratch[STAGED_RENDERED + before.len()..STAGED_RENDERED + before.len() + user.len()]
            .copy_from_slice(user);
        scratch[STAGED_RENDERED + before.len() + user.len()..STAGED_RENDERED + rendered_count]
            .copy_from_slice(after);
        #[cfg(test)]
        if TEST_PROMPT_AFTER_WRITE_FAULT.load(Ordering::SeqCst) != 0 {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::SCRATCH,
                0,
                0,
                1,
                0,
                FieldKind::STATE_FIELD,
                0,
            ));
        }
        let mut count = 0usize;
        self.control(scratch, &mut count, IM_START)?;
        self.ordinary(scratch, &mut count, b"system")?;
        self.ordinary(scratch, &mut count, b"\n")?;
        self.ordinary(
            scratch,
            &mut count,
            b"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        )?;
        self.control(scratch, &mut count, IM_END)?;
        self.ordinary(scratch, &mut count, b"\n")?;
        self.control(scratch, &mut count, IM_START)?;
        self.ordinary(scratch, &mut count, b"user")?;
        self.ordinary(scratch, &mut count, b"\n")?;
        let prefix_tokens = count;
        self.ordinary(scratch, &mut count, user)?;
        let user_tokens = count - prefix_tokens;
        self.control(scratch, &mut count, IM_END)?;
        self.ordinary(scratch, &mut count, b"\n")?;
        self.control(scratch, &mut count, IM_START)?;
        self.ordinary(scratch, &mut count, b"assistant")?;
        self.ordinary(scratch, &mut count, b"\n")?;
        Ok(PromptComputation {
            rendered_bytes: rendered_count,
            token_count: count,
            prefix_tokens,
            user_tokens,
        })
    }

    fn control(&self, scratch: &mut [u8], count: &mut usize, token: u32) -> Result<(), ModelError> {
        if *count >= MAX_TOKENS {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                MAX_TOKENS as u64,
                (*count + 1) as u64,
                FieldKind::STATE_FIELD,
                0,
            ));
        }
        put_u32(scratch, STAGED_TOKENS + *count * 4, token);
        *count += 1;
        Ok(())
    }

    fn ordinary(
        &self,
        scratch: &mut [u8],
        count: &mut usize,
        text: &[u8],
    ) -> Result<(), ModelError> {
        let mut cursor = 0usize;
        while cursor < text.len() {
            let (special_at, special, special_len) = next_user_defined(&text[cursor..]);
            if special_at != 0 {
                self.regex_fragments(scratch, count, &text[cursor..cursor + special_at])?;
            }
            cursor += special_at;
            if special != 0 {
                self.control(scratch, count, special)?;
                cursor += special_len;
            } else {
                if cursor < text.len() {
                    self.regex_fragments(scratch, count, &text[cursor..])?;
                }
                break;
            }
        }
        Ok(())
    }

    fn regex_fragments(
        &self,
        scratch: &mut [u8],
        count: &mut usize,
        text: &[u8],
    ) -> Result<(), ModelError> {
        let mut at = 0usize;
        while at < text.len() {
            let length = pretoken_length(&text[at..]);
            self.bpe(scratch, count, &text[at..at + length])?;
            at += length;
        }
        Ok(())
    }

    fn bpe(
        &self,
        scratch: &mut [u8],
        output_count: &mut usize,
        text: &[u8],
    ) -> Result<(), ModelError> {
        if text.len() > MAX_SEGMENT {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::SCRATCH,
                0,
                0,
                MAX_SEGMENT as u64,
                text.len() as u64,
                FieldKind::STATE_FIELD,
                0,
            ));
        }
        let mut count = text.len();
        for (at, byte) in text.iter().copied().enumerate() {
            put_u32(
                scratch,
                CURRENT_SEGMENT + at * 4,
                u32_at(self.index, BYTE_MAP_OFFSET + byte as usize * 4),
            );
        }
        loop {
            let mut best_rank = u32::MAX;
            let mut best_at = 0usize;
            let mut best_result = 0u32;
            if count >= 2 {
                for at in 0..count - 1 {
                    let left = u32_at(scratch, CURRENT_SEGMENT + at * 4);
                    let right = u32_at(scratch, CURRENT_SEGMENT + (at + 1) * 4);
                    let (entry, _) = lookup(self.index, left, right);
                    if let Some((result, rank)) = entry {
                        if rank < best_rank {
                            best_rank = rank;
                            best_at = at;
                            best_result = result;
                        }
                    }
                }
            }
            if best_rank == u32::MAX {
                break;
            }
            put_u32(scratch, CURRENT_SEGMENT + best_at * 4, best_result);
            for at in best_at + 1..count - 1 {
                let value = u32_at(scratch, CURRENT_SEGMENT + (at + 1) * 4);
                put_u32(scratch, CURRENT_SEGMENT + at * 4, value);
            }
            count -= 1;
        }
        if *output_count + count > MAX_TOKENS {
            return Err(ModelError::new(
                ModelStatus::STATE,
                ErrorDomain::TOKEN_OUTPUT,
                0,
                0,
                MAX_TOKENS as u64,
                (*output_count + count) as u64,
                FieldKind::STATE_FIELD,
                0,
            ));
        }
        for at in 0..count {
            let value = u32_at(scratch, CURRENT_SEGMENT + at * 4);
            put_u32(scratch, STAGED_TOKENS + (*output_count + at) * 4, value);
        }
        *output_count += count;
        Ok(())
    }

    pub fn decode_piece(&self, token: u32, output: &mut [u8]) -> Result<PieceUsage, ModelError> {
        if token >= VOCAB_COUNT {
            return Err(ModelError::new(
                ModelStatus::TOKEN_ID,
                ErrorDomain::TOKEN,
                token,
                0,
                VOCAB_COUNT as u64,
                token as u64,
                FieldKind::TOKEN_ID_FIELD,
                0,
            ));
        }
        let token_type = self.model.token_type(token)?;
        let kind = match token_type {
            1 | 4 => PieceKind::TEXT,
            3 if token == IM_END || token == END_OF_TEXT => PieceKind::EOS,
            3 | 5 => PieceKind::SUPPRESSED,
            _ => {
                return Err(ModelError::new(
                    ModelStatus::STATE,
                    ErrorDomain::TOKEN,
                    token,
                    0,
                    1,
                    0,
                    FieldKind::STATE_FIELD,
                    0,
                ))
            }
        };
        let mut staged = [0u8; 128];
        let mut count = 0usize;
        if kind == PieceKind::TEXT {
            let piece = self.model.token_bytes(token)?;
            if token_type == 4 {
                if piece.len() > staged.len() {
                    return Err(ModelError::new(
                        ModelStatus::STATE,
                        ErrorDomain::PIECE_OUTPUT,
                        token,
                        0,
                        128,
                        piece.len() as u64,
                        FieldKind::STATE_FIELD,
                        0,
                    ));
                }
                staged[..piece.len()].copy_from_slice(piece);
                count = piece.len();
            } else {
                let string = core::str::from_utf8(piece).map_err(|_| {
                    ModelError::new(
                        ModelStatus::STATE,
                        ErrorDomain::TOKEN,
                        token,
                        0,
                        1,
                        0,
                        FieldKind::STATE_FIELD,
                        0,
                    )
                })?;
                for character in string.chars() {
                    let point = character as u32;
                    let mut found = None;
                    for byte in 0..=255u16 {
                        if gpt2_codepoint(byte as u8) == point {
                            found = Some(byte as u8);
                            break;
                        }
                    }
                    let byte = found.ok_or(ModelError::new(
                        ModelStatus::STATE,
                        ErrorDomain::TOKEN,
                        token,
                        0,
                        1,
                        point as u64,
                        FieldKind::STATE_FIELD,
                        0,
                    ))?;
                    if count == staged.len() {
                        return Err(ModelError::new(
                            ModelStatus::STATE,
                            ErrorDomain::PIECE_OUTPUT,
                            token,
                            0,
                            128,
                            (count + 1) as u64,
                            FieldKind::STATE_FIELD,
                            0,
                        ));
                    }
                    staged[count] = byte;
                    count += 1;
                }
            }
        }
        if output.len() < count {
            return Err(ModelError::new(
                ModelStatus::OUTPUT_CAPACITY,
                ErrorDomain::PIECE_OUTPUT,
                0,
                0,
                count as u64,
                output.len() as u64,
                FieldKind::CAPACITY,
                0,
            ));
        }
        output[..count].copy_from_slice(&staged[..count]);
        Ok(PieceUsage {
            kind: kind as u32,
            bytes: count as u32,
        })
    }
}

fn lookup(index: &[u8], left: u32, right: u32) -> (Option<(u32, u32)>, usize) {
    let initial = hash_slot(left, right);
    for displacement in 0..51usize {
        let at = ((initial + displacement) & (TABLE_ENTRIES - 1)) * 16;
        let found_left = u32_at(index, at);
        if found_left == u32::MAX {
            return (None, displacement + 1);
        }
        if found_left == left && u32_at(index, at + 4) == right {
            return (
                Some((u32_at(index, at + 8), u32_at(index, at + 12))),
                displacement + 1,
            );
        }
    }
    (None, 51)
}

fn next_user_defined(text: &[u8]) -> (usize, u32, usize) {
    for at in 0..text.len() {
        if text[at..].starts_with(TOOL_CALL) {
            return (at, 151_657, TOOL_CALL.len());
        }
        if text[at..].starts_with(TOOL_END) {
            return (at, 151_658, TOOL_END.len());
        }
    }
    (text.len(), 0, 0)
}

fn is_letter(byte: u8) -> bool {
    byte.is_ascii_alphabetic()
}
fn is_digit(byte: u8) -> bool {
    byte.is_ascii_digit()
}
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}
fn is_punctuation(byte: u8) -> bool {
    !is_space(byte) && !is_letter(byte) && !is_digit(byte)
}

fn pretoken_length(text: &[u8]) -> usize {
    let contraction_lengths = [2usize, 2, 3, 3, 2, 3, 2];
    for &length in &contraction_lengths {
        if text.len() >= length && contraction_matches(&text[..length]) {
            return length;
        }
    }
    let optional = if !matches!(text[0], b'\r' | b'\n') && !is_letter(text[0]) && !is_digit(text[0])
    {
        1
    } else {
        0
    };
    if optional < text.len() && is_letter(text[optional]) {
        let mut end = optional + 1;
        while end < text.len() && is_letter(text[end]) {
            end += 1;
        }
        return end;
    }
    if is_digit(text[0]) {
        return 1;
    }
    let optional_space = usize::from(text[0] == b' ');
    if optional_space < text.len() && is_punctuation(text[optional_space]) {
        let mut end = optional_space + 1;
        while end < text.len() && is_punctuation(text[end]) {
            end += 1;
        }
        while end < text.len() && matches!(text[end], b'\r' | b'\n') {
            end += 1;
        }
        return end;
    }
    if is_space(text[0]) {
        let mut run = 1;
        while run < text.len() && is_space(text[run]) {
            run += 1;
        }
        let mut last_newline_end = 0;
        for at in 0..run {
            if matches!(text[at], b'\r' | b'\n') {
                last_newline_end = at + 1;
            }
        }
        if last_newline_end != 0 {
            return last_newline_end;
        }
        if run == text.len() {
            return run;
        }
        if run > 1 {
            return run - 1;
        }
        return run;
    }
    1
}

fn contraction_matches(text: &[u8]) -> bool {
    match text {
        [b'\'', b's' | b'S']
        | [b'\'', b't' | b'T']
        | [b'\'', b'm' | b'M']
        | [b'\'', b'd' | b'D'] => true,
        [b'\'', b'r' | b'R', b'e' | b'E']
        | [b'\'', b'v' | b'V', b'e' | b'E']
        | [b'\'', b'l' | b'L', b'l' | b'L'] => true,
        _ => false,
    }
}
