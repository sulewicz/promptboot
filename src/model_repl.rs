//! Resident persistent-UEFI model REPL over the shared loaded model/session.

use core::mem::transmute;
use core::slice;

use promptboot::console_contract::{
    is_generation_interrupt, read_result, reset_result, validate_bindings, wait_result,
};
use promptboot::editor::{Editor, Flow, Output};
use promptboot::repl_contract::{
    commit_eos, decide_history, is_event_toggle_command, is_help_command, is_new_command,
    render_console_text, selected_token_failure, HistoryDecision, Utf8Decoder, Utf8Error,
    CONTEXT_TOKENS, GENERATION_RESERVE, IM_END, PROMPT_LIMIT, REPETITION_BITMAP_BYTES,
    SESSION_BYTES,
};
use promptboot_core::{
    sample_token_with_repetition, FrozenTokenizer, InferenceEngine, InferenceError, ModelError,
    ModelView, PieceKind, SamplingState, INDEX_BYTES, KV_BYTES, LOGIT_WORDS, SAMPLING_POLICY,
    SCRATCH_BYTES, TOKENIZER_INDEX_SHA256_HEX,
};

use super::model_target::{timing_end, timing_start, Failure, Timing, EXPECTED_SHA_HEX};
use super::{
    emit_record_checked, establish_fp_state, reset_console_history, restore_fp_state,
    scroll_console_page, toggle_event_display, write_conout_utf16_checked, BootServices, EfiHandle,
    EfiInputKey, EfiKeyData, EfiKeyState, EfiStatus, InputReset, ReadKeyStroke, ReadKeyStrokeEx,
    Record, SimpleTextInput, SimpleTextInputEx, SystemTable, WaitForEvent, EFI_DEVICE_ERROR,
    EFI_OPEN_PROTOCOL_GET_PROTOCOL, EFI_SUCCESS, SAVED_FP_STATE, SIMPLE_TEXT_INPUT_EX_GUID,
};

const SCAN_PAGE_UP: u16 = 0x0009;
const SCAN_PAGE_DOWN: u16 = 0x000a;
const FALLBACK_SAMPLING_SEED: u64 = 0x6d6f_6465_6c2d_6f73;

#[derive(Clone, Copy)]
struct TraceEntry {
    id: u32,
    kind: u32,
    piece_bytes: u32,
    utf16_units: u32,
    infer_start: u64,
    infer_end: u64,
    output_start: u64,
    output_end: u64,
}

impl TraceEntry {
    const ZERO: Self = Self {
        id: 0,
        kind: 0,
        piece_bytes: 0,
        utf16_units: 0,
        infer_start: 0,
        infer_end: 0,
        output_start: 0,
        output_end: 0,
    };
}

struct GenerationStack {
    piece: [u8; 128],
    decoded: [u16; 132],
    rendered: [u16; 264],
}

impl GenerationStack {
    const ZERO: Self = Self {
        piece: [0; 128],
        decoded: [0; 132],
        rendered: [0; 264],
    };
}

const GENERATION_STACK_STATIC_BYTES: usize = core::mem::size_of::<GenerationStack>()
    + core::mem::size_of::<Utf8Decoder>()
    + core::mem::size_of::<TurnError>();
const _: [(); 48] = [(); core::mem::size_of::<TraceEntry>()];
const _: [(); 1] = [(); (GENERATION_STACK_STATIC_BYTES <= 4_096) as usize];

struct TurnError {
    code: &'static [u8],
    phase: &'static [u8],
    recoverable: bool,
    generated: u32,
    partial: bool,
    model: Option<ModelError>,
    inference: Option<InferenceError>,
    efi: Option<EfiStatus>,
    reported: bool,
}

impl TurnError {
    fn model(code: &'static [u8], phase: &'static [u8], error: ModelError) -> Self {
        Self {
            code,
            phase,
            recoverable: true,
            generated: 0,
            partial: false,
            model: Some(error),
            inference: None,
            efi: None,
            reported: false,
        }
    }

    fn inference(code: &'static [u8], phase: &'static [u8], error: InferenceError) -> Self {
        Self {
            code,
            phase,
            recoverable: true,
            generated: 0,
            partial: false,
            model: None,
            inference: Some(error),
            efi: None,
            reported: false,
        }
    }

    fn utf(code: &'static [u8]) -> Self {
        Self {
            code,
            phase: b"utf8",
            recoverable: true,
            generated: 0,
            partial: false,
            model: None,
            inference: None,
            efi: None,
            reported: false,
        }
    }

    fn efi(code: &'static [u8], phase: &'static [u8], status: EfiStatus) -> Self {
        Self {
            code,
            phase,
            recoverable: false,
            generated: 0,
            partial: false,
            model: None,
            inference: None,
            efi: Some(status),
            reported: false,
        }
    }

    fn simple(code: &'static [u8], phase: &'static [u8], recoverable: bool) -> Self {
        Self {
            code,
            phase,
            recoverable,
            generated: 0,
            partial: false,
            model: None,
            inference: None,
            efi: None,
            reported: false,
        }
    }

    fn after_selected(mut self, completed_tokens: u32, visible_tokens: u32) -> Self {
        let accounting = selected_token_failure(completed_tokens, visible_tokens);
        self.generated = accounting.generated;
        self.partial = accounting.partial;
        self
    }
}

pub(super) unsafe fn run(
    image: EfiHandle,
    system_table: *mut SystemTable,
    boot: *mut BootServices,
    model: &ModelView<'_>,
    tokenizer: &FrozenTokenizer<'_, '_, '_>,
    addresses: [u64; 6],
    timing: Timing,
) -> Result<(), Failure> {
    let con_in = (*system_table).con_in;
    let (reset_entry, read_entry, wait_for_key) = if con_in.is_null() {
        (0, 0, core::ptr::null_mut())
    } else {
        (
            (*con_in).reset,
            (*con_in).read_key_stroke,
            (*con_in).wait_for_key,
        )
    };
    validate_bindings(
        con_in as usize,
        reset_entry,
        read_entry,
        wait_for_key as usize,
        (*boot).wait_for_event,
    )
    .map_err(|error| Failure::simple(error.code(), b"repl_input"))?;
    let reset: InputReset = transmute(reset_entry);
    let read_key: ReadKeyStroke = transmute(read_entry);
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    reset_result(reset(con_in, false))
        .map_err(|error| Failure::simple(error.code(), b"repl_input"))?;
    let input_ex = open_input_ex(image, system_table, boot);

    let session = slice::from_raw_parts_mut(addresses[5] as *mut u8, SESSION_BYTES);
    session.fill(0);
    let tokens = slice::from_raw_parts_mut(session.as_mut_ptr().cast::<u32>(), CONTEXT_TOKENS * 2);
    let (committed, working) = tokens.split_at_mut(CONTEXT_TOKENS);
    let committed: &mut [u32; CONTEXT_TOKENS] = committed
        .try_into()
        .map_err(|_| Failure::simple(b"HISTORY_STATE", b"history"))?;
    let working: &mut [u32; CONTEXT_TOKENS] = working
        .try_into()
        .map_err(|_| Failure::simple(b"HISTORY_STATE", b"history"))?;
    let traces = slice::from_raw_parts_mut(
        session
            .as_mut_ptr()
            .add(CONTEXT_TOKENS * 2 * core::mem::size_of::<u32>())
            .cast::<TraceEntry>(),
        GENERATION_RESERVE,
    );
    let traces: &mut [TraceEntry; GENERATION_RESERVE] = traces
        .try_into()
        .map_err(|_| Failure::simple(b"HISTORY_STATE", b"history"))?;
    let repetition_offset = CONTEXT_TOKENS * 2 * core::mem::size_of::<u32>()
        + GENERATION_RESERVE * core::mem::size_of::<TraceEntry>();
    let repetition_bitmap = slice::from_raw_parts_mut(
        session.as_mut_ptr().add(repetition_offset),
        REPETITION_BITMAP_BYTES,
    );
    let repetition_bitmap: &mut [u8; REPETITION_BITMAP_BYTES] = repetition_bitmap
        .try_into()
        .map_err(|_| Failure::simple(b"HISTORY_STATE", b"history"))?;
    let kv = slice::from_raw_parts_mut(addresses[2] as *mut u8, KV_BYTES);
    let inference_scratch = slice::from_raw_parts_mut(addresses[3] as *mut u8, SCRATCH_BYTES);
    establish_fp_state();
    let engine = InferenceEngine::build(model, kv, inference_scratch);
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let engine = engine.map_err(|error| Failure::inference(b"ENGINE_BUILD", b"engine", error))?;
    let (seed, seed_source) = sampling_seed();
    let sampling = SamplingState::new(seed);
    let mut controller = ModelReplController {
        tokenizer,
        engine,
        timing,
        addresses,
        con_in,
        read_key,
        input_ex,
        committed,
        working,
        traces,
        repetition_bitmap,
        committed_len: 0,
        cached_len: 0,
        history_turns: 0,
        sampling_seed: sampling.state(),
        sampling_seed_source: seed_source,
        sampling,
        terminal: None,
    };
    controller.emit_repl_ready()?;
    write_ascii_checked(b"Ctrl-C stops generation; /help lists commands.\r\n")
        .map_err(|status| Failure::efi(b"CONOUT_WRITE", b"output", status))?;

    let wait: WaitForEvent = transmute((*boot).wait_for_event);
    let event = [wait_for_key];
    let mut editor = Editor::new();
    controller.prompt(editor.prompt_index());
    if let Some(failure) = controller.terminal.take() {
        return Err(failure);
    }
    loop {
        let mut index = usize::MAX;
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        wait_result(wait(1, event.as_ptr(), &mut index), index)
            .map_err(|error| Failure::simple(error.code(), b"repl_input"))?;
        loop {
            let mut key = EfiInputKey {
                scan_code: 0,
                unicode_char: 0,
            };
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            match read_result(read_key(con_in, &mut key)) {
                Ok(true) => {
                    if key.unicode_char == 0
                        && (key.scan_code == SCAN_PAGE_UP || key.scan_code == SCAN_PAGE_DOWN)
                    {
                        scroll_console_page(key.scan_code == SCAN_PAGE_UP)
                            .map_err(|_| Failure::simple(b"CONOUT_SCROLL", b"output"))?;
                        continue;
                    }
                    if editor.process_key(key.unicode_char, &mut controller)
                        == Flow::PromptIndexOverflow
                    {
                        return Err(Failure::simple(b"PROMPT_INDEX", b"repl_input"));
                    }
                    if let Some(failure) = controller.terminal.take() {
                        return Err(failure);
                    }
                }
                Ok(false) => break,
                Err(error) => return Err(Failure::simple(error.code(), b"repl_input")),
            }
        }
    }
}

struct ModelReplController<'a, 'bytes, 'index, 'kv, 'scratch> {
    tokenizer: &'a FrozenTokenizer<'a, 'bytes, 'index>,
    engine: InferenceEngine<'a, 'bytes, 'kv, 'scratch>,
    timing: Timing,
    addresses: [u64; 6],
    con_in: *mut SimpleTextInput,
    read_key: ReadKeyStroke,
    input_ex: Option<(*mut SimpleTextInputEx, ReadKeyStrokeEx)>,
    committed: &'a mut [u32; CONTEXT_TOKENS],
    working: &'a mut [u32; CONTEXT_TOKENS],
    traces: &'a mut [TraceEntry; GENERATION_RESERVE],
    repetition_bitmap: &'a mut [u8; REPETITION_BITMAP_BYTES],
    committed_len: usize,
    cached_len: usize,
    history_turns: u32,
    sampling_seed: u64,
    sampling_seed_source: &'static [u8],
    sampling: SamplingState,
    terminal: Option<Failure>,
}

const CONTROLLER_STACK_STATIC_BYTES: usize = core::mem::size_of::<
    ModelReplController<'static, 'static, 'static, 'static, 'static>,
>() + core::mem::size_of::<Editor>();
const _: [(); 1] = [(); (CONTROLLER_STACK_STATIC_BYTES <= 4_096) as usize];

impl ModelReplController<'_, '_, '_, '_, '_> {
    unsafe fn emit_repl_ready(&mut self) -> Result<(), Failure> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_REPL_READY mode=model_repl model_sha256=");
        record.push(EXPECTED_SHA_HEX.as_bytes());
        record.push(b" index_sha256=");
        record.push(TOKENIZER_INDEX_SHA256_HEX);
        record.push(b" context=");
        record.push_decimal(CONTEXT_TOKENS as u64);
        record.push(b" max_new_tokens=");
        record.push_decimal(GENERATION_RESERVE as u64);
        record.push(b" prompt_bytes=");
        record.push_decimal(SESSION_BYTES as u64);
        record.push(b" history=whole_turn timing_class=");
        record.push(if self.timing.timed {
            b"timed_invariant_tsc"
        } else {
            b"untimed_noninvariant_tsc"
        });
        record.push(b" sampling=");
        record.push(SAMPLING_POLICY);
        record.push(b" sampling_seed=");
        push_hex(&mut record, self.sampling_seed, 16);
        record.push(b" sampling_seed_source=");
        record.push(self.sampling_seed_source);
        record.push(b" interrupt_input=");
        record.push(if self.input_ex.is_some() {
            b"uefi_simple_text_input_ex"
        } else {
            b"uefi_simple_text_input"
        });
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| Failure::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_prompt_ready(&mut self, prompt_index: u64) -> Result<(), Failure> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=PROMPT_READY prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" input_limit=512 history_turns=");
        record.push_decimal(self.history_turns as u64);
        record.push(b" history_tokens=");
        record.push_decimal(self.committed_len as u64);
        record.push(b" reserve=");
        record.push_decimal(GENERATION_RESERVE as u64);
        record.push(b" sampling_draws=");
        record.push_decimal(self.sampling.draws());
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| Failure::efi(b"EVIDENCE_RECORD", b"record", status))?;
        write_ascii_checked(b"promptboot> ")
            .map_err(|status| Failure::efi(b"CONOUT_WRITE", b"output", status))?;
        Ok(())
    }

    unsafe fn accepted_turn(&mut self, prompt_index: u64, line: &[u8]) -> Result<(), TurnError> {
        write_conout_utf16_checked(&[13, 10])
            .map_err(|status| TurnError::efi(b"CONOUT_WRITE", b"output", status))?;
        let accepted_tsc = timing_start(self.timing);
        let mut accepted = Record::new();
        accepted.push(b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=");
        accepted.push_decimal(prompt_index);
        accepted.push(b" bytes=");
        accepted.push_decimal(line.len() as u64);
        accepted.push(b" accepted_tsc=");
        push_hex(&mut accepted, accepted_tsc, 16);
        accepted.push(b"\r\n");
        emit_record_checked(accepted.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))?;

        let logits_bytes = slice::from_raw_parts_mut(
            self.addresses[4] as *mut u8,
            LOGIT_WORDS * core::mem::size_of::<u32>(),
        );
        let (token_bytes, rendered_tail) = logits_bytes.split_at_mut(2_432);
        let staging = slice::from_raw_parts_mut(token_bytes.as_mut_ptr().cast::<u32>(), 599);
        let (rendered_storage, tokenizer_tail) = rendered_tail.split_at_mut(704);
        let rendered = &mut rendered_storage[..660];
        let tokenizer_scratch = &mut tokenizer_tail[..5_120];
        self.working.fill(0);
        let had_history = self.committed_len != 0;
        let history = &self.committed[..self.committed_len];
        let first = self.tokenizer.render_conversation_and_tokenize(
            history,
            line,
            rendered,
            staging,
            self.working,
            tokenizer_scratch,
        );
        let overflow = |error: &ModelError| {
            error.status == promptboot_core::ModelStatus::OUTPUT_CAPACITY as u32
                && error.domain == promptboot_core::ErrorDomain::TOKEN_OUTPUT as u32
                && error.available > CONTEXT_TOKENS as u64
        };
        let history_error = |error: &ModelError| {
            error.status == promptboot_core::ModelStatus::STATE as u32
                && error.domain == promptboot_core::ErrorDomain::TOKEN_OUTPUT as u32
                && error.detail as u32 == 11
        };
        let prospective_tokens = match &first {
            Ok(usage) => usage.prompt_tokens as usize,
            Err(error) if overflow(error) => usize::try_from(error.available).unwrap_or(usize::MAX),
            Err(error) => {
                return Err(TurnError::model(
                    if history_error(error) {
                        b"HISTORY_STATE"
                    } else {
                        b"TOKENIZE"
                    },
                    if history_error(error) {
                        b"history"
                    } else {
                        b"tokenize"
                    },
                    *error,
                ))
            }
        };
        let decision = decide_history(
            had_history,
            prospective_tokens,
            if had_history {
                None
            } else {
                Some(prospective_tokens)
            },
        );
        let (usage, prompt_count, reset) = match decision {
            HistoryDecision::Use { prompt_tokens } => match first {
                Ok(usage) => (usage, prompt_tokens, false),
                Err(_) => {
                    self.working.fill(0);
                    return Err(TurnError::utf(b"HISTORY_STATE"));
                }
            },
            HistoryDecision::RetryFresh => {
                self.working.fill(0);
                let fresh = self.tokenizer.render_conversation_and_tokenize(
                    &[],
                    line,
                    rendered,
                    staging,
                    self.working,
                    tokenizer_scratch,
                );
                let fresh_prompt_tokens = match &fresh {
                    Ok(usage) => usage.prompt_tokens as usize,
                    Err(error) if overflow(error) => {
                        usize::try_from(error.available).unwrap_or(usize::MAX)
                    }
                    Err(error) => {
                        self.working.fill(0);
                        return Err(TurnError::model(
                            if history_error(error) {
                                b"HISTORY_STATE"
                            } else {
                                b"TOKENIZE"
                            },
                            if history_error(error) {
                                b"history"
                            } else {
                                b"tokenize"
                            },
                            *error,
                        ));
                    }
                };
                match decide_history(true, prospective_tokens, Some(fresh_prompt_tokens)) {
                    HistoryDecision::Reset { prompt_tokens } => {
                        let usage = match fresh {
                            Ok(usage) => usage,
                            Err(_) => {
                                self.working.fill(0);
                                return Err(TurnError::utf(b"HISTORY_STATE"));
                            }
                        };
                        self.emit_history_reset(prompt_index)?;
                        self.reset_engine()?;
                        self.committed.fill(0);
                        self.committed_len = 0;
                        self.history_turns = 0;
                        (usage, prompt_tokens, true)
                    }
                    HistoryDecision::Reject {
                        fresh_prompt_tokens,
                    } => {
                        let user_tokens = match &fresh {
                            Ok(usage) => usage.user_tokens,
                            Err(error) => error.available.saturating_sub(29) as u32,
                        };
                        self.emit_context_rejected(
                            prompt_index,
                            user_tokens,
                            u32::try_from(fresh_prompt_tokens).unwrap_or(u32::MAX),
                        )?;
                        self.working.fill(0);
                        return Ok(());
                    }
                    HistoryDecision::Use { .. } | HistoryDecision::RetryFresh => {
                        self.working.fill(0);
                        return Err(TurnError::utf(b"HISTORY_STATE"));
                    }
                }
            }
            HistoryDecision::Reject {
                fresh_prompt_tokens,
            } => {
                let user_tokens = match &first {
                    Ok(usage) => usage.user_tokens,
                    Err(error) => error.available.saturating_sub(29) as u32,
                };
                self.emit_context_rejected(
                    prompt_index,
                    user_tokens,
                    u32::try_from(fresh_prompt_tokens).unwrap_or(u32::MAX),
                )?;
                self.working.fill(0);
                return Ok(());
            }
            HistoryDecision::Reset { .. } => {
                self.working.fill(0);
                return Err(TurnError::utf(b"HISTORY_STATE"));
            }
        };
        self.emit_prompt_tokenized(prompt_index, usage.user_tokens, prompt_count, reset)?;
        self.generate(prompt_index, accepted_tsc, prompt_count)
    }

    unsafe fn generate(
        &mut self,
        prompt_index: u64,
        accepted_tsc: u64,
        prompt_count: usize,
    ) -> Result<(), TurnError> {
        let logits = slice::from_raw_parts_mut(self.addresses[4] as *mut u32, LOGIT_WORDS);
        let append_start = if self.cached_len == 0 {
            0
        } else if self.cached_len + 1 == self.committed_len
            && self.cached_len < prompt_count
            && self.engine.position() as usize == self.cached_len
            && self.working[..self.cached_len] == self.committed[..self.cached_len]
        {
            self.cached_len
        } else {
            self.reset_engine()?;
            return Err(TurnError::utf(b"HISTORY_STATE"));
        };
        let generation_limit = core::cmp::min(GENERATION_RESERVE, CONTEXT_TOKENS - prompt_count);
        let generation_start = timing_start(self.timing);
        if let Err(error) = self.emit_generation_started(
            prompt_index,
            accepted_tsc,
            prompt_count,
            generation_limit,
            append_start,
            generation_start,
        ) {
            let _ = self.reset_engine();
            self.working.fill(0);
            self.traces.fill(TraceEntry::ZERO);
            return Err(error);
        }

        let mut stack = GenerationStack::ZERO;
        self.traces.fill(TraceEntry::ZERO);
        let mut decoder = Utf8Decoder::new();
        let mut previous_was_cr = false;
        let mut generated = 0usize;
        let mut visible_tokens = 0u32;
        let mut visible_units = 0u32;
        let mut committed = false;
        let mut reason = if generation_limit == GENERATION_RESERVE {
            b"MAX_NEW_TOKENS" as &'static [u8]
        } else {
            b"CONTEXT_LIMIT" as &'static [u8]
        };
        let mut previous = 0u32;
        let mut turn_error = None;
        let mut interrupted = false;

        while generated < generation_limit {
            match self.poll_generation_interrupt() {
                Ok(true) => {
                    reason = b"INTERRUPTED";
                    interrupted = true;
                    break;
                }
                Ok(false) => {}
                Err(error) => {
                    turn_error = Some(error);
                    break;
                }
            }
            establish_fp_state();
            let infer_start = timing_start(self.timing);
            let result = if generated == 0 {
                if append_start == 0 {
                    self.engine.prefill(
                        &self.working[..prompt_count],
                        generation_limit as u32,
                        logits,
                    )
                } else {
                    self.engine.append_prefill(
                        &self.working[append_start..prompt_count],
                        generation_limit as u32,
                        logits,
                    )
                }
            } else {
                self.engine.decode_selected(previous, logits)
            };
            let selected = match result {
                Ok(_) => sample_token_with_repetition(
                    logits,
                    &self.working[..prompt_count + generated],
                    self.repetition_bitmap,
                    &mut self.sampling,
                )
                .map_err(|error| TurnError::inference(b"INFERENCE_SAMPLE", b"sample", error)),
                Err(error) => Err(TurnError::inference(
                    if generated == 0 {
                        b"INFERENCE_PREFILL"
                    } else {
                        b"INFERENCE_DECODE"
                    },
                    if generated == 0 {
                        b"prefill"
                    } else {
                        b"decode"
                    },
                    error,
                )),
            };
            let infer_end = match timing_end(self.timing) {
                Ok(value) => value,
                Err(failure) => {
                    let _ = self.reset_engine();
                    self.working.fill(0);
                    self.traces.fill(TraceEntry::ZERO);
                    return Err(TurnError {
                        code: failure.code,
                        phase: failure.phase,
                        recoverable: false,
                        generated: generated as u32,
                        partial: visible_tokens != 0,
                        model: None,
                        inference: failure.inference,
                        efi: failure.efi,
                        reported: false,
                    });
                }
            };
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            previous = match selected {
                Ok(value) => value,
                Err(error) => {
                    turn_error = Some(error);
                    break;
                }
            };
            match self.poll_generation_interrupt() {
                Ok(true) => {
                    reason = b"INTERRUPTED";
                    interrupted = true;
                    break;
                }
                Ok(false) => {}
                Err(error) => {
                    turn_error = Some(error);
                    break;
                }
            }
            stack.piece.fill(0);
            let piece_usage = match self.tokenizer.decode_piece(previous, &mut stack.piece) {
                Ok(value) => value,
                Err(error) => {
                    turn_error = Some(
                        TurnError::model(b"PIECE_DECODE", b"piece", error)
                            .after_selected(generated as u32, visible_tokens),
                    );
                    break;
                }
            };
            stack.decoded.fill(0);
            let decoded_units = if piece_usage.kind == PieceKind::TEXT as u32 {
                match decoder.push(
                    &stack.piece[..piece_usage.bytes as usize],
                    &mut stack.decoded,
                ) {
                    Ok(value) => value,
                    Err(Utf8Error::Incomplete) => 0,
                    Err(_) => {
                        turn_error = Some(
                            TurnError::utf(b"UTF8_INVALID")
                                .after_selected(generated as u32, visible_tokens),
                        );
                        break;
                    }
                }
            } else {
                0
            };
            stack.rendered.fill(0);
            let units = match render_console_text(
                &stack.decoded[..decoded_units],
                &mut previous_was_cr,
                &mut stack.rendered,
            ) {
                Ok(value) => value,
                Err(_) => {
                    turn_error = Some(
                        TurnError::simple(b"UTF16_CAPACITY", b"output", true)
                            .after_selected(generated as u32, visible_tokens),
                    );
                    break;
                }
            };
            let output_start = timing_start(self.timing);
            if units != 0 {
                if let Err(status) = write_conout_utf16_checked(&stack.rendered[..units]) {
                    turn_error = Some(
                        TurnError::efi(b"CONOUT_WRITE", b"output", status)
                            .after_selected(generated as u32, visible_tokens),
                    );
                    break;
                }
                visible_tokens += 1;
                visible_units += units as u32;
            }
            let output_end = match timing_end(self.timing) {
                Ok(value) => value,
                Err(failure) => {
                    turn_error = Some(
                        TurnError {
                            code: failure.code,
                            phase: failure.phase,
                            recoverable: false,
                            generated: 0,
                            partial: false,
                            model: failure.model,
                            inference: failure.inference,
                            efi: failure.efi,
                            reported: false,
                        }
                        .after_selected(generated as u32, visible_tokens),
                    );
                    break;
                }
            };
            self.working[prompt_count + generated] = if piece_usage.kind == PieceKind::EOS as u32 {
                IM_END
            } else {
                previous
            };
            self.traces[generated] = TraceEntry {
                id: previous,
                kind: piece_usage.kind,
                piece_bytes: piece_usage.bytes,
                utf16_units: units as u32,
                infer_start,
                infer_end,
                output_start,
                output_end,
            };
            generated += 1;
            if piece_usage.kind == PieceKind::EOS as u32 {
                if decoder.finish().is_err() {
                    turn_error = Some(
                        TurnError::utf(b"UTF8_INCOMPLETE")
                            .after_selected(generated.saturating_sub(1) as u32, visible_tokens),
                    );
                } else {
                    let working: &[u32; CONTEXT_TOKENS] = &*self.working;
                    if commit_eos(self.committed, working, prompt_count + generated).is_err() {
                        let error = match self
                            .tokenizer
                            .validate_conversation_history(&working[..prompt_count + generated])
                        {
                            Err(error) => error,
                            Ok(()) => match self.tokenizer.validate_conversation_history(&[0]) {
                                Err(error) => error,
                                Ok(()) => {
                                    turn_error = Some(TurnError::utf(b"HISTORY_STATE"));
                                    break;
                                }
                            },
                        };
                        turn_error = Some(TurnError::model(b"HISTORY_STATE", b"history", error));
                    }
                    if turn_error.is_none() {
                        self.committed_len = prompt_count + generated;
                        self.history_turns = self.history_turns.saturating_add(1);
                        committed = true;
                        reason = b"EOS";
                    }
                }
                break;
            }
        }
        if turn_error.is_none() && !committed && !interrupted && decoder.finish().is_err() {
            turn_error = Some(
                TurnError::utf(b"UTF8_INCOMPLETE")
                    .after_selected(generated.saturating_sub(1) as u32, visible_tokens),
            );
        }
        if committed && turn_error.is_none() {
            let expected = self.committed_len.saturating_sub(1);
            if self.engine.position() as usize == expected {
                self.cached_len = expected;
            } else {
                turn_error = Some(TurnError::utf(b"HISTORY_STATE"));
            }
        }
        if !committed || turn_error.is_some() {
            if let Err(mut error) = self.reset_engine() {
                if turn_error.is_none() {
                    error.generated = generated as u32;
                    error.partial = visible_tokens != 0;
                    turn_error = Some(error);
                }
            }
        }

        if interrupted {
            if let Err(status) = write_ascii_checked(b"^C") {
                if turn_error.is_none() {
                    let mut error = TurnError::efi(b"CONOUT_WRITE", b"output", status);
                    error.generated = generated as u32;
                    error.partial = visible_tokens != 0;
                    turn_error = Some(error);
                }
            }
        }
        if let Err(status) = write_conout_utf16_checked(&[13, 10]) {
            if turn_error.is_none() {
                let mut error = TurnError::efi(b"CONOUT_WRITE", b"output", status);
                error.generated = generated as u32;
                error.partial = visible_tokens != 0;
                turn_error = Some(error);
            }
        }
        for index in 0..generated {
            let trace = self.traces[index];
            if let Err(record_error) = self.emit_token(prompt_index, index, &trace) {
                self.working.fill(0);
                self.traces.fill(TraceEntry::ZERO);
                return Err(match turn_error {
                    Some(error) => error,
                    None => record_error,
                });
            }
        }
        if let Some(mut error) = turn_error {
            error.generated = core::cmp::max(error.generated, generated as u32);
            error.partial |= visible_tokens != 0;
            self.working.fill(0);
            self.traces.fill(TraceEntry::ZERO);
            if self.emit_generation_failed(prompt_index, &error).is_err() {
                return Err(error);
            }
            error.reported = true;
            return if error.recoverable {
                Ok(())
            } else {
                Err(error)
            };
        }

        let end_tsc = timing_start(self.timing);
        self.emit_generation_complete(
            prompt_index,
            reason,
            generated,
            visible_tokens,
            visible_units,
            committed,
            generation_start,
            end_tsc,
        )?;
        self.working.fill(0);
        self.traces.fill(TraceEntry::ZERO);
        Ok(())
    }

    unsafe fn reset_engine(&mut self) -> Result<(), TurnError> {
        establish_fp_state();
        let result = self.engine.reset();
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        self.cached_len = 0;
        result.map_err(|error| TurnError::inference(b"ENGINE_RESET", b"reset", error))
    }

    unsafe fn poll_generation_interrupt(&self) -> Result<bool, TurnError> {
        if let Some((input, read)) = self.input_ex {
            loop {
                let mut data = EfiKeyData {
                    key: EfiInputKey {
                        scan_code: 0,
                        unicode_char: 0,
                    },
                    key_state: EfiKeyState {
                        key_shift_state: 0,
                        key_toggle_state: 0,
                    },
                };
                restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                match read_result(read(input, &mut data)) {
                    Ok(true)
                        if is_generation_interrupt(
                            data.key.unicode_char,
                            data.key_state.key_shift_state,
                        ) =>
                    {
                        return Ok(true);
                    }
                    Ok(true) => {}
                    Ok(false) => return Ok(false),
                    Err(error) => {
                        return Err(TurnError::simple(error.code(), b"repl_input", false));
                    }
                }
            }
        }
        loop {
            let mut key = EfiInputKey {
                scan_code: 0,
                unicode_char: 0,
            };
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            match read_result((self.read_key)(self.con_in, &mut key)) {
                Ok(true) if key.unicode_char == 0x0003 => return Ok(true),
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(error) => {
                    return Err(TurnError::simple(error.code(), b"repl_input", false));
                }
            }
        }
    }

    unsafe fn emit_history_reset(&self, prompt_index: u64) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=HISTORY_RESET prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" prior_turns=");
        record.push_decimal(self.history_turns as u64);
        record.push(b" prior_tokens=");
        record.push_decimal(self.committed_len as u64);
        record.push(b" reason=prospective_context\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_context_rejected(
        &self,
        prompt_index: u64,
        user_tokens: u32,
        fresh_tokens: u32,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=CONTEXT_REJECTED prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" user_tokens=");
        record.push_decimal(user_tokens as u64);
        record.push(b" fresh_prompt_tokens=");
        record.push_decimal(fresh_tokens as u64);
        record.push(b" limit=");
        record.push_decimal(PROMPT_LIMIT as u64);
        record.push(b" reserve=");
        record.push_decimal(GENERATION_RESERVE as u64);
        record.push(b" code=CONTEXT_LIMIT\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_prompt_tokenized(
        &self,
        prompt_index: u64,
        user_tokens: u32,
        prompt_tokens: usize,
        reset: bool,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=PROMPT_TOKENIZED prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" user_tokens=");
        record.push_decimal(user_tokens as u64);
        record.push(b" history_tokens=");
        record.push_decimal(self.committed_len as u64);
        record.push(b" prompt_tokens=");
        record.push_decimal(prompt_tokens as u64);
        record.push(b" reserve=");
        record.push_decimal(GENERATION_RESERVE as u64);
        record.push(b" reset=");
        record.push(if reset { b"1" } else { b"0" });
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_generation_started(
        &self,
        prompt_index: u64,
        accepted_tsc: u64,
        prompt_tokens: usize,
        generation_limit: usize,
        cached_tokens: usize,
        start_tsc: u64,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=GENERATION_STARTED prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" accepted_tsc=");
        push_hex(&mut record, accepted_tsc, 16);
        record.push(b" prompt_tokens=");
        record.push_decimal(prompt_tokens as u64);
        record.push(b" limit=");
        record.push_decimal(generation_limit as u64);
        record.push(b" cached_tokens=");
        record.push_decimal(cached_tokens as u64);
        record.push(b" start_tsc=");
        push_hex(&mut record, start_tsc, 16);
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_token(
        &self,
        prompt_index: u64,
        token_index: usize,
        trace: &TraceEntry,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=TOKEN prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" token_index=");
        record.push_decimal(token_index as u64);
        record.push(b" id=");
        record.push_decimal(trace.id as u64);
        record.push(b" kind=");
        record.push(match trace.kind {
            value if value == PieceKind::TEXT as u32 => b"TEXT",
            value if value == PieceKind::EOS as u32 => b"EOS",
            _ => b"SUPPRESSED",
        });
        record.push(b" piece_bytes=");
        record.push_decimal(trace.piece_bytes as u64);
        record.push(b" utf16_units=");
        record.push_decimal(trace.utf16_units as u64);
        for (name, value) in [
            (b" infer_start_tsc=" as &[u8], trace.infer_start),
            (b" infer_end_tsc=", trace.infer_end),
            (b" output_start_tsc=", trace.output_start),
            (b" output_end_tsc=", trace.output_end),
        ] {
            record.push(name);
            push_hex(&mut record, value, 16);
        }
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn emit_generation_complete(
        &self,
        prompt_index: u64,
        reason: &[u8],
        generated: usize,
        visible_tokens: u32,
        visible_units: u32,
        committed: bool,
        start_tsc: u64,
        end_tsc: u64,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=GENERATION_COMPLETE prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" reason=");
        record.push(reason);
        record.push(b" generated=");
        record.push_decimal(generated as u64);
        record.push(b" visible_tokens=");
        record.push_decimal(visible_tokens as u64);
        record.push(b" visible_utf16_units=");
        record.push_decimal(visible_units as u64);
        record.push(b" committed=");
        record.push(if committed { b"1" } else { b"0" });
        record.push(b" history_turns=");
        record.push_decimal(self.history_turns as u64);
        record.push(b" history_tokens=");
        record.push_decimal(self.committed_len as u64);
        record.push(b" start_tsc=");
        push_hex(&mut record, start_tsc, 16);
        record.push(b" end_tsc=");
        push_hex(&mut record, end_tsc, 16);
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))
    }

    unsafe fn emit_generation_failed(
        &self,
        prompt_index: u64,
        error: &TurnError,
    ) -> Result<(), TurnError> {
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=GENERATION_FAILED prompt_index=");
        record.push_decimal(prompt_index);
        record.push(b" code=");
        record.push(error.code);
        record.push(b" phase=");
        record.push(error.phase);
        record.push(b" partial=");
        record.push(if error.partial { b"1" } else { b"0" });
        record.push(b" recoverable=");
        record.push(if error.recoverable { b"1" } else { b"0" });
        record.push(b" generated=");
        record.push_decimal(error.generated as u64);
        record.push(b" model_status=");
        if let Some(value) = error.model {
            push_hex(&mut record, value.status as u64, 8);
        } else {
            record.push(b"none");
        }
        record.push(b" inference_status=");
        if let Some(value) = error.inference {
            push_hex(&mut record, value.status as u64, 8);
        } else {
            record.push(b"none");
        }
        record.push(b" efi_status=");
        if let Some(value) = error.efi {
            push_hex(&mut record, value as u64, 16);
        } else {
            record.push(b"none");
        }
        record.push(b"\r\n");
        emit_record_checked(record.as_bytes())
            .map_err(|status| TurnError::efi(b"EVIDENCE_RECORD", b"record", status))?;
        write_ascii_checked(b"promptboot: generation failed ")
            .map_err(|status| TurnError::efi(b"CONOUT_WRITE", b"output", status))?;
        write_ascii_checked(error.code)
            .map_err(|status| TurnError::efi(b"CONOUT_WRITE", b"output", status))?;
        write_ascii_checked(b"\r\n")
            .map_err(|status| TurnError::efi(b"CONOUT_WRITE", b"output", status))
    }
}

unsafe fn open_input_ex(
    image: EfiHandle,
    system_table: *mut SystemTable,
    boot: *mut BootServices,
) -> Option<(*mut SimpleTextInputEx, ReadKeyStrokeEx)> {
    let mut interface = core::ptr::null_mut();
    let status = ((*boot).open_protocol)(
        (*system_table).console_in_handle,
        &SIMPLE_TEXT_INPUT_EX_GUID,
        &mut interface,
        image,
        core::ptr::null_mut(),
        EFI_OPEN_PROTOCOL_GET_PROTOCOL,
    );
    if status != EFI_SUCCESS || interface.is_null() {
        return None;
    }
    let input = interface.cast::<SimpleTextInputEx>();
    if (*input).read_key_stroke_ex == 0 {
        return None;
    }
    Some((input, transmute((*input).read_key_stroke_ex)))
}

impl Output for ModelReplController<'_, '_, '_, '_, '_> {
    fn live_ascii(&mut self, byte: u8) {
        if self.terminal.is_none() {
            if let Err(status) = unsafe { write_ascii_checked(&[byte]) } {
                self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
            }
        }
    }

    fn erase_last(&mut self) {
        if self.terminal.is_none() {
            if let Err(status) = unsafe { write_ascii_checked(b"\x08 \x08") } {
                self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
            }
        }
    }

    fn accepted(&mut self, prompt_index: u64, line: &[u8]) {
        if self.terminal.is_some() {
            return;
        }
        if is_event_toggle_command(line) {
            unsafe {
                if let Err(status) = write_ascii_checked(b"\r\n") {
                    self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                    return;
                }
                let enabled = toggle_event_display();
                let status = if enabled {
                    write_ascii_checked(b"events: on\r\n")
                } else {
                    write_ascii_checked(b"events: off\r\n")
                };
                if let Err(status) = status {
                    self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                }
            }
            return;
        }
        if is_help_command(line) {
            unsafe {
                if let Err(status) = write_ascii_checked(b"\r\n") {
                    self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                    return;
                }
                for line in [
                    b"commands:\r\n" as &[u8],
                    b"/events - toggle structured event display\r\n",
                    b"/help - show this help\r\n",
                    b"/new - clear the session and scrollback\r\n",
                    b"Ctrl-C - stop generation\r\n",
                    b"Page Up/Page Down - scroll output\r\n",
                ] {
                    if let Err(status) = write_ascii_checked(line) {
                        self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                        return;
                    }
                }
            }
            return;
        }
        if is_new_command(line) {
            if let Err(error) = unsafe { self.reset_engine() } {
                self.terminal = Some(match error.inference {
                    Some(inference) => Failure::inference(error.code, error.phase, inference),
                    None => Failure::simple(error.code, error.phase),
                });
                return;
            }
            self.committed.fill(0);
            self.working.fill(0);
            self.traces.fill(TraceEntry::ZERO);
            self.committed_len = 0;
            self.history_turns = 0;
            unsafe {
                if let Err(status) = reset_console_history() {
                    self.terminal = Some(Failure::efi(b"CONOUT_CLEAR", b"output", status));
                    return;
                }
                if let Err(status) = write_ascii_checked(b"new session\r\n") {
                    self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                }
            }
            return;
        }
        match unsafe { self.accepted_turn(prompt_index, line) } {
            Ok(()) => {}
            Err(error) if error.recoverable => unsafe {
                self.working.fill(0);
                if !error.reported && self.emit_generation_failed(prompt_index, &error).is_err() {
                    self.terminal = Some(Failure::simple(b"EVIDENCE_RECORD", b"record"));
                }
            },
            Err(mut error) => {
                self.working.fill(0);
                if !error.reported
                    && error.code != b"EVIDENCE_RECORD"
                    && error.code != b"MODEL_TIMER"
                {
                    if unsafe { self.emit_generation_failed(prompt_index, &error) }.is_err() {
                        error = TurnError::efi(b"EVIDENCE_RECORD", b"record", EFI_DEVICE_ERROR);
                    } else {
                        error.reported = true;
                    }
                }
                self.terminal = Some(if let Some(status) = error.efi {
                    Failure::efi(error.code, error.phase, status)
                } else if let Some(model) = error.model {
                    Failure::model(error.code, error.phase, model)
                } else if let Some(inference) = error.inference {
                    Failure::inference(error.code, error.phase, inference)
                } else {
                    Failure::simple(error.code, error.phase)
                });
            }
        }
    }

    fn rejected(&mut self, prompt_index: u64) {
        if self.terminal.is_some() {
            return;
        }
        unsafe {
            if let Err(status) = write_ascii_checked(b"\r\n") {
                self.terminal = Some(Failure::efi(b"CONOUT_WRITE", b"output", status));
                return;
            }
            let mut record = Record::new();
            record.push(b"PROMPTBOOT_EVENT v=1 event=INPUT_REJECTED prompt_index=");
            record.push_decimal(prompt_index);
            record.push(b" code=TOO_LONG limit=512\r\n");
            if let Err(status) = emit_record_checked(record.as_bytes()) {
                self.terminal = Some(Failure::efi(b"EVIDENCE_RECORD", b"record", status));
            }
        }
    }

    fn prompt(&mut self, prompt_index: u64) {
        if self.terminal.is_none() {
            if let Err(failure) = unsafe { self.emit_prompt_ready(prompt_index) } {
                self.terminal = Some(failure);
            }
        }
    }
}

unsafe fn sampling_seed() -> (u64, &'static [u8]) {
    if core::arch::x86_64::__cpuid(1).ecx & (1 << 30) != 0 {
        for _ in 0..8 {
            let value: u64;
            let valid: u8;
            core::arch::asm!(
                "rdrand {value}",
                "setc {valid}",
                value = out(reg) value,
                valid = out(reg_byte) valid,
                options(nomem, nostack),
            );
            if valid != 0 {
                return (value, b"rdrand");
            }
        }
    }
    (FALLBACK_SAMPLING_SEED, b"fixed_fallback")
}

fn push_hex(record: &mut Record, value: u64, width: usize) {
    let digits = b"0123456789abcdef";
    let mut at = width;
    while at != 0 {
        at -= 1;
        record.push(&[digits[((value >> (at * 4)) & 15) as usize]]);
    }
}

unsafe fn write_ascii_checked(bytes: &[u8]) -> Result<(), EfiStatus> {
    if bytes.len() > 132 || bytes.iter().any(|byte| !byte.is_ascii() || *byte == 0) {
        return Err(EFI_DEVICE_ERROR);
    }
    let mut units = [0u16; 132];
    for (at, byte) in bytes.iter().copied().enumerate() {
        units[at] = byte as u16;
    }
    write_conout_utf16_checked(&units[..bytes.len()])
}

const _: usize = INDEX_BYTES;
const _: EfiStatus = EFI_SUCCESS;
const _: EfiStatus = EFI_DEVICE_ERROR;
const _: usize = core::mem::size_of::<SimpleTextInput>();
const _: [(); 8] = [(); core::mem::size_of::<EfiKeyState>()];
const _: [(); 12] = [(); core::mem::size_of::<EfiKeyData>()];
