use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::marker::{PhantomData, PhantomPinned};
use core::mem::{align_of, size_of};
use core::ptr;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::arena::ArenaUsage;
use crate::fp32_sse2;
use crate::model::{ModelView, TensorView, MODEL_BYTES, VOCAB_COUNT};
use crate::tokenizer::MAX_TOKENS;

pub const CONTEXT_LIMIT: u32 = 32_768;
pub const KV_BYTES: usize = 805_306_368;
pub const SCRATCH_BYTES: usize = 262_144;
pub const LOGIT_WORDS: usize = 151_936;
pub const NO_LAYER: u32 = u32::MAX;
pub const NO_TENSOR: u32 = u32::MAX;
pub const SAMPLING_TEMPERATURE_MILLI: u32 = 700;
pub const SAMPLING_TOP_K: u32 = 20;
pub const SAMPLING_TOP_P_MILLI: u32 = 800;
pub const SAMPLING_REPETITION_PENALTY_MILLI: u32 = 1_100;
pub const SAMPLING_POLICY: &[u8] = b"temperature_0p7_top_k_20_top_p_0p8_repetition_penalty_1p1";

pub fn inference_avx2_available() -> bool {
    fp32_sse2::inference_avx2_available()
}

const HIDDEN_WORDS: usize = 896;
const KV_HEADS: usize = 2;
const QUERY_HEADS: usize = 14;
const HEAD_WORDS: usize = 64;
const FFN_WORDS: usize = 4_864;
pub(crate) const LAYERS: usize = 24;
const KV_WORDS_PER_POSITION: usize = LAYERS * 2 * KV_HEADS * HEAD_WORDS;
pub(crate) const KV_TILE: usize = 4;
pub(crate) const KV_VALUE_WORDS: usize = KV_HEADS * CONTEXT_LIMIT as usize * HEAD_WORDS;
pub(crate) const KV_LAYER_WORDS: usize = 2 * KV_VALUE_WORDS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct ScratchRegion {
    pub offset: u32,
    pub bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct InferenceScratchLayout {
    pub hidden: ScratchRegion,
    pub residual: ScratchRegion,
    pub norm: ScratchRegion,
    pub q: ScratchRegion,
    pub k: ScratchRegion,
    pub v: ScratchRegion,
    pub attention: ScratchRegion,
    pub scores: ScratchRegion,
    pub gate: ScratchRegion,
    pub up: ScratchRegion,
    pub product: ScratchRegion,
    pub q8_staging: ScratchRegion,
    pub reserved: ScratchRegion,
}

pub(crate) const INFERENCE_SCRATCH_LAYOUT: InferenceScratchLayout = InferenceScratchLayout {
    hidden: ScratchRegion {
        offset: 0,
        bytes: 3_584,
    },
    residual: ScratchRegion {
        offset: 3_584,
        bytes: 3_584,
    },
    norm: ScratchRegion {
        offset: 7_168,
        bytes: 3_584,
    },
    q: ScratchRegion {
        offset: 10_752,
        bytes: 3_584,
    },
    k: ScratchRegion {
        offset: 14_336,
        bytes: 512,
    },
    v: ScratchRegion {
        offset: 14_848,
        bytes: 512,
    },
    attention: ScratchRegion {
        offset: 15_360,
        bytes: 3_584,
    },
    scores: ScratchRegion {
        offset: 18_944,
        bytes: 131_072,
    },
    gate: ScratchRegion {
        offset: 150_016,
        bytes: 19_456,
    },
    up: ScratchRegion {
        offset: 169_472,
        bytes: 19_456,
    },
    product: ScratchRegion {
        offset: 188_928,
        bytes: 19_456,
    },
    q8_staging: ScratchRegion {
        offset: 208_384,
        bytes: 5_184,
    },
    reserved: ScratchRegion {
        offset: 213_568,
        bytes: 48_576,
    },
};

const HIDDEN: usize = INFERENCE_SCRATCH_LAYOUT.hidden.offset as usize;
const RESIDUAL: usize = INFERENCE_SCRATCH_LAYOUT.residual.offset as usize;
const NORM: usize = INFERENCE_SCRATCH_LAYOUT.norm.offset as usize;
const Q: usize = INFERENCE_SCRATCH_LAYOUT.q.offset as usize;
const K: usize = INFERENCE_SCRATCH_LAYOUT.k.offset as usize;
const V: usize = INFERENCE_SCRATCH_LAYOUT.v.offset as usize;
const ATTENTION: usize = INFERENCE_SCRATCH_LAYOUT.attention.offset as usize;
const SCORES: usize = INFERENCE_SCRATCH_LAYOUT.scores.offset as usize;
const GATE: usize = INFERENCE_SCRATCH_LAYOUT.gate.offset as usize;
const UP: usize = INFERENCE_SCRATCH_LAYOUT.up.offset as usize;
const PRODUCT: usize = INFERENCE_SCRATCH_LAYOUT.product.offset as usize;
const Q8_STAGING: usize = INFERENCE_SCRATCH_LAYOUT.q8_staging.offset as usize;
const USED_SCRATCH: usize = INFERENCE_SCRATCH_LAYOUT.reserved.offset as usize;

#[cfg(test)]
pub(crate) fn active_scratch_regions_for_test() -> [(usize, usize); 12] {
    [
        (HIDDEN, INFERENCE_SCRATCH_LAYOUT.hidden.bytes as usize),
        (RESIDUAL, INFERENCE_SCRATCH_LAYOUT.residual.bytes as usize),
        (NORM, INFERENCE_SCRATCH_LAYOUT.norm.bytes as usize),
        (Q, INFERENCE_SCRATCH_LAYOUT.q.bytes as usize),
        (K, INFERENCE_SCRATCH_LAYOUT.k.bytes as usize),
        (V, INFERENCE_SCRATCH_LAYOUT.v.bytes as usize),
        (ATTENTION, INFERENCE_SCRATCH_LAYOUT.attention.bytes as usize),
        (SCORES, INFERENCE_SCRATCH_LAYOUT.scores.bytes as usize),
        (GATE, INFERENCE_SCRATCH_LAYOUT.gate.bytes as usize),
        (UP, INFERENCE_SCRATCH_LAYOUT.up.bytes as usize),
        (PRODUCT, INFERENCE_SCRATCH_LAYOUT.product.bytes as usize),
        (
            Q8_STAGING,
            INFERENCE_SCRATCH_LAYOUT.q8_staging.bytes as usize,
        ),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum InferenceStatus {
    OK = 0,
    STATE = 1,
    CONTEXT = 2,
    TOKEN_ID = 3,
    CAPACITY = 4,
    ALIGNMENT = 5,
    NONFINITE_INPUT = 6,
    NONFINITE_OUTPUT = 7,
    MODEL = 8,
    ARITHMETIC = 9,
    FAULTED = 10,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum InferenceDomain {
    NONE = 0,
    INIT = 1,
    PROMPT = 2,
    EMBEDDING = 3,
    ATTN_NORM = 4,
    Q = 5,
    K = 6,
    V = 7,
    ROPE = 8,
    KV = 9,
    ATTENTION = 10,
    ATTN_OUTPUT = 11,
    FFN_NORM = 12,
    GATE = 13,
    UP = 14,
    SWIGLU = 15,
    DOWN = 16,
    OUTPUT_NORM = 17,
    LOGITS = 18,
    ARGMAX = 19,
    RESET = 20,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum InferenceFieldKind {
    NONE = 0,
    STATE = 1,
    TOKEN = 2,
    CONTEXT = 3,
    BYTES = 4,
    ALIGNMENT = 5,
    TENSOR = 6,
    FINITE = 7,
    ADD = 8,
    MUL = 9,
    USIZE = 10,
    RESERVE = 11,
    LOGITS = 12,
    FAULT = 13,
    SELECTED = 14,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum InferenceState {
    RESET = 0,
    READY = 1,
    EOS = 2,
    FAULTED = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct InferenceError {
    pub status: u32,
    pub domain: u32,
    pub layer: u32,
    pub position: u32,
    pub tensor_id: u32,
    pub reserved: u32,
    pub needed: u64,
    pub available: u64,
    pub detail: u64,
}

impl InferenceError {
    const fn new(
        status: InferenceStatus,
        domain: InferenceDomain,
        layer: u32,
        position: u32,
        tensor_id: u32,
        needed: u64,
        available: u64,
        kind: InferenceFieldKind,
        sub: u32,
    ) -> Self {
        Self {
            status: status as u32,
            domain: domain as u32,
            layer,
            position,
            tensor_id,
            reserved: 0,
            needed,
            available,
            detail: ((kind as u64) << 32) | sub as u64,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct InferenceStep {
    pub position: u32,
    pub selected_token: u32,
    pub selected_logit_bits: u32,
    pub eos: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct InferenceUsage {
    pub weights: ArenaUsage,
    pub kv: ArenaUsage,
    pub scratch: ArenaUsage,
    pub position: u32,
    pub context_limit: u32,
    pub generation_reserve: u32,
    pub state: u32,
}

const _: [(); 48] = [(); size_of::<InferenceError>()];
const _: [(); 8] = [(); align_of::<InferenceError>()];
const _: [(); 16] = [(); size_of::<InferenceStep>()];
const _: [(); 136] = [(); size_of::<InferenceUsage>()];
const _: [(); 8] = [(); align_of::<InferenceUsage>()];

pub struct InferenceEngine<'model, 'bytes, 'kv, 'scratch> {
    model: &'model ModelView<'bytes>,
    kv: &'kv mut [u8],
    scratch: &'scratch mut [u8],
    position: u32,
    generation_reserve: u32,
    predictions_emitted: u32,
    last_selected: u32,
    state: InferenceState,
    backend: fp32_sse2::InferenceBackend,
    dispatcher: Option<InferenceDispatcher<'model>>,
    kv_high_water: u64,
    scratch_high_water: u64,
    retained_prefix: RetainedPrefix,
    #[cfg(test)]
    q8_preparations: u32,
}

pub struct InferenceDispatcher<'context> {
    context: *mut c_void,
    dispatch: unsafe extern "efiapi" fn(*mut c_void, *mut InferenceWorkerJob),
    _context: PhantomData<&'context mut c_void>,
}

impl<'context> InferenceDispatcher<'context> {
    /// The callback must not return unless every worker has returned and the
    /// job protocol is valid. Firmware dispatch failures are fatal and must
    /// not return to core because the job memory can no longer be inspected.
    pub unsafe fn new<T>(
        context: &'context mut T,
        dispatch: unsafe extern "efiapi" fn(*mut c_void, *mut InferenceWorkerJob),
    ) -> Self {
        Self {
            context: (context as *mut T).cast(),
            dispatch,
            _context: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
struct MatvecJob {
    weights: *const u8,
    output: *mut u8,
    prepared: *mut u8,
    rows: usize,
    columns: usize,
    q8: bool,
}

impl MatvecJob {
    const EMPTY: Self = Self {
        weights: ptr::null(),
        output: ptr::null_mut(),
        prepared: ptr::null_mut(),
        rows: 0,
        columns: 0,
        q8: false,
    };
}

struct CompletionBarrier {
    arrivals: AtomicUsize,
    generation: AtomicUsize,
}

impl CompletionBarrier {
    const fn new() -> Self {
        Self {
            arrivals: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    fn wait(&self, workers: usize) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrivals.fetch_add(1, Ordering::AcqRel) + 1 == workers {
            self.arrivals.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == generation {
                core::hint::spin_loop();
            }
        }
    }
}

/// Opaque, pinned per-token state passed through the firmware adapter.
pub struct InferenceWorkerJob {
    engine: *mut c_void,
    coordinate: unsafe fn(*mut c_void, *mut InferenceWorkerJob),
    logits: *mut u32,
    logits_len: usize,
    token: u32,
    produce_logits: bool,
    workers: AtomicUsize,
    backend: AtomicU32,
    rejection: AtomicU32,
    job_epoch: AtomicUsize,
    terminal: AtomicU32,
    row_hits: AtomicUsize,
    result: UnsafeCell<InferenceError>,
    descriptor: UnsafeCell<MatvecJob>,
    completion: CompletionBarrier,
    expected_jobs: usize,
    _pinned: PhantomPinned,
}

unsafe impl Sync for InferenceWorkerJob {}

impl InferenceWorkerJob {
    fn new(
        engine: *mut c_void,
        coordinate: unsafe fn(*mut c_void, *mut InferenceWorkerJob),
        token: u32,
        logits: &mut [u32],
        produce_logits: bool,
    ) -> Self {
        Self {
            engine,
            coordinate,
            logits: logits.as_mut_ptr(),
            logits_len: logits.len(),
            token,
            produce_logits,
            workers: AtomicUsize::new(0),
            backend: AtomicU32::new(u32::MAX),
            rejection: AtomicU32::new(0),
            job_epoch: AtomicUsize::new(0),
            terminal: AtomicU32::new(0),
            row_hits: AtomicUsize::new(0),
            result: UnsafeCell::new(InferenceError::new(
                InferenceStatus::STATE,
                InferenceDomain::PROMPT,
                NO_LAYER,
                0,
                NO_TENSOR,
                0,
                0,
                InferenceFieldKind::STATE,
                5,
            )),
            descriptor: UnsafeCell::new(MatvecJob::EMPTY),
            completion: CompletionBarrier::new(),
            expected_jobs: if produce_logits { 169 } else { 168 },
            _pinned: PhantomPinned,
        }
    }

    /// Returns zero only for a complete, internally consistent token dispatch.
    pub fn protocol_status(&self) -> u32 {
        let rejection = self.rejection.load(Ordering::Acquire);
        if rejection != 0 {
            return rejection;
        }
        let terminal = self.terminal.load(Ordering::Acquire);
        if terminal == 0 {
            return 10;
        }
        let job_epoch = self.job_epoch.load(Ordering::Acquire);
        if (terminal == 1 && job_epoch != self.expected_jobs)
            || (terminal == 2 && job_epoch > self.expected_jobs)
        {
            return 11;
        }
        if self.completion.arrivals.load(Ordering::Acquire) != 0
            || self.completion.generation.load(Ordering::Acquire) != job_epoch
        {
            return 12;
        }
        0
    }

    fn publish(&self, descriptor: MatvecJob, backend: fp32_sse2::InferenceBackend) {
        let epoch = self.job_epoch.load(Ordering::Relaxed) + 1;
        unsafe { ptr::write(self.descriptor.get(), descriptor) };
        self.row_hits.store(0, Ordering::Relaxed);
        self.job_epoch.store(epoch, Ordering::Release);
        let end = descriptor.rows / self.worker_count();
        unsafe { run_matvec_rows(descriptor, backend, 0, end) };
        self.row_hits.fetch_add(end, Ordering::AcqRel);
        self.completion.wait(self.worker_count());
        if self.row_hits.load(Ordering::Acquire) != descriptor.rows {
            self.rejection.store(4, Ordering::Release);
        }
    }

    fn worker_count(&self) -> usize {
        self.workers.load(Ordering::Acquire)
    }

    unsafe fn run(&self, slot: usize, workers: usize, backend: u32) -> u32 {
        if workers < 2 || slot >= workers || backend > 1 {
            self.rejection.store(1, Ordering::Release);
            return 1;
        }
        let stored_workers = self.workers.compare_exchange(
            0,
            workers,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if !match stored_workers {
            Ok(_) => true,
            Err(value) => value == workers,
        } {
            self.rejection.store(2, Ordering::Release);
            return 2;
        }
        let stored_backend = self.backend.compare_exchange(
            u32::MAX,
            backend,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if !match stored_backend {
            Ok(_) => true,
            Err(value) => value == backend,
        } {
            self.rejection.store(3, Ordering::Release);
            return 3;
        }
        if slot == 0 {
            (self.coordinate)(self.engine, (self as *const Self).cast_mut());
            return self.protocol_status();
        }

        let selected = if backend == 0 {
            fp32_sse2::InferenceBackend::Sse2
        } else {
            fp32_sse2::InferenceBackend::Avx2
        };
        let mut observed = 0;
        loop {
            let epoch = self.job_epoch.load(Ordering::Acquire);
            if epoch != observed {
                let descriptor = ptr::read(self.descriptor.get());
                let first = descriptor.rows * slot / workers;
                let end = descriptor.rows * (slot + 1) / workers;
                run_matvec_rows(descriptor, selected, first, end);
                self.row_hits.fetch_add(end - first, Ordering::AcqRel);
                self.completion.wait(workers);
                observed = epoch;
                continue;
            }
            if self.terminal.load(Ordering::Acquire) != 0 {
                return self.protocol_status();
            }
            core::hint::spin_loop();
        }
    }
}

/// Runs one validated AP slot. The boot adapter owns slot assignment, common
/// backend selection, and processor-state establishment/restoration.
pub unsafe extern "efiapi" fn inference_worker_entry(
    job: *mut InferenceWorkerJob,
    slot: u32,
    workers: u32,
    backend: u32,
) -> u32 {
    if job.is_null() {
        return 1;
    }
    (*job).run(slot as usize, workers as usize, backend)
}

struct TokenCoordinator {
    backend: fp32_sse2::InferenceBackend,
    parallel: *const InferenceWorkerJob,
}

impl TokenCoordinator {
    const fn serial(backend: fp32_sse2::InferenceBackend) -> Self {
        Self {
            backend,
            parallel: ptr::null(),
        }
    }

    unsafe fn parallel(job: *const InferenceWorkerJob) -> Self {
        let backend = if (*job).backend.load(Ordering::Acquire) == 0 {
            fp32_sse2::InferenceBackend::Sse2
        } else {
            fp32_sse2::InferenceBackend::Avx2
        };
        Self {
            backend,
            parallel: job,
        }
    }

    unsafe fn q4(
        &self,
        weights: *const u8,
        output: *mut u8,
        prepared: *mut u8,
        rows: usize,
        columns: usize,
    ) {
        self.run(MatvecJob {
            weights,
            output,
            prepared,
            rows,
            columns,
            q8: false,
        });
    }

    unsafe fn q8(
        &self,
        weights: *const u8,
        output: *mut u32,
        prepared: *mut u8,
        rows: usize,
        columns: usize,
    ) {
        self.run(MatvecJob {
            weights,
            output: output.cast(),
            prepared,
            rows,
            columns,
            q8: true,
        });
    }

    unsafe fn run(&self, descriptor: MatvecJob) {
        if self.parallel.is_null() {
            run_matvec_rows(descriptor, self.backend, 0, descriptor.rows);
        } else {
            (*self.parallel).publish(descriptor, self.backend);
        }
    }
}

unsafe fn run_matvec_rows(
    descriptor: MatvecJob,
    backend: fp32_sse2::InferenceBackend,
    first: usize,
    end: usize,
) {
    if descriptor.q8 {
        fp32_sse2::inference_q8_matvec_rows_prepared(
            backend,
            descriptor.weights,
            descriptor.output.cast(),
            descriptor.prepared,
            descriptor.rows,
            descriptor.columns,
            first,
            end,
        );
    } else {
        fp32_sse2::inference_q4_matvec_rows_prepared(
            backend,
            descriptor.weights,
            descriptor.output,
            descriptor.prepared,
            descriptor.rows,
            descriptor.columns,
            first,
            end,
        );
    }
}

unsafe fn coordinate_engine<'model, 'bytes, 'kv, 'scratch>(
    engine: *mut c_void,
    job: *mut InferenceWorkerJob,
)
where
    'bytes: 'model,
{
    let engine = &mut *engine.cast::<InferenceEngine<'model, 'bytes, 'kv, 'scratch>>();
    let job_ref = &*job;
    let logits = core::slice::from_raw_parts_mut(job_ref.logits, job_ref.logits_len);
    let coordinator = TokenCoordinator::parallel(job);
    let previous_mxcsr = fp32_sse2::inference_enter_fp();
    let result = engine.evaluate_inner(
        job_ref.token,
        logits,
        job_ref.produce_logits,
        &coordinator,
    );
    fp32_sse2::inference_exit_fp(previous_mxcsr);
    match result {
        Ok(()) => job_ref.terminal.store(1, Ordering::Release),
        Err(error) => {
            ptr::write(job_ref.result.get(), error);
            job_ref.terminal.store(2, Ordering::Release);
        }
    }
}

struct RetainedPrefix {
    len: u32,
    tokens: [u32; MAX_TOKENS],
}

impl RetainedPrefix {
    const EMPTY: Self = Self {
        len: 0,
        tokens: [0; MAX_TOKENS],
    };

    fn clear(&mut self) {
        self.tokens.fill(0);
        self.len = 0;
    }
}

fn buffer_error(storage: &[u8], needed: usize, sub: u32) -> Option<InferenceError> {
    if storage.len() < needed {
        return Some(InferenceError::new(
            InferenceStatus::CAPACITY,
            InferenceDomain::INIT,
            NO_LAYER,
            0,
            NO_TENSOR,
            needed as u64,
            storage.len() as u64,
            InferenceFieldKind::BYTES,
            sub,
        ));
    }
    let misalignment = (storage.as_ptr() as usize & 63) as u64;
    if misalignment != 0 {
        return Some(InferenceError::new(
            InferenceStatus::ALIGNMENT,
            InferenceDomain::INIT,
            NO_LAYER,
            0,
            NO_TENSOR,
            64,
            misalignment,
            InferenceFieldKind::ALIGNMENT,
            sub,
        ));
    }
    None
}

fn logits_error(length: usize, position: u32) -> Option<InferenceError> {
    if length == LOGIT_WORDS {
        None
    } else {
        Some(InferenceError::new(
            InferenceStatus::CAPACITY,
            InferenceDomain::LOGITS,
            NO_LAYER,
            position,
            NO_TENSOR,
            LOGIT_WORDS as u64,
            length as u64,
            InferenceFieldKind::LOGITS,
            0,
        ))
    }
}

fn finite(bits: u32) -> bool {
    bits & 0x7f80_0000 != 0x7f80_0000
}

/// One binary32 logit in the deterministic descending top-eight diagnostic.
///
/// The integer-only public representation keeps floating-point values out of
/// the freestanding Rust ABI.  Equal numeric values (including signed zero)
/// retain ascending token order because the input scan is ascending and the
/// insertion pass only moves a candidate past a strictly smaller value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopLogit {
    pub token: u32,
    pub logit_bits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SamplingState {
    state: u64,
    draws: u64,
}

impl SamplingState {
    const ZERO_SEED_REPLACEMENT: u64 = 0x9e37_79b9_7f4a_7c15;

    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                Self::ZERO_SEED_REPLACEMENT
            } else {
                seed
            },
            draws: 0,
        }
    }

    pub const fn state(&self) -> u64 {
        self.state
    }

    pub const fn draws(&self) -> u64 {
        self.draws
    }

    fn next_24(&mut self) -> u32 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        self.draws = self.draws.wrapping_add(1);
        ((value.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 40) as u32
    }
}

const _: [(); 16] = [(); size_of::<SamplingState>()];

pub fn top_logits_8(logits: &[u32], output: &mut [TopLogit; 8]) -> Result<(), InferenceError> {
    if let Some(error) = logits_error(logits.len(), 0) {
        return Err(error);
    }

    let mut staged = [TopLogit {
        token: 0,
        logit_bits: 0,
    }; 8];
    let mut used = 0usize;
    for (token, bits) in logits.iter().copied().enumerate() {
        if !finite(bits) {
            return Err(InferenceError::new(
                InferenceStatus::NONFINITE_OUTPUT,
                InferenceDomain::ARGMAX,
                NO_LAYER,
                0,
                NO_TENSOR,
                0,
                bits as u64,
                InferenceFieldKind::FINITE,
                token as u32,
            ));
        }

        let mut insertion = used;
        while insertion != 0
            && unsafe { fp32_sse2::inference_greater(bits, staged[insertion - 1].logit_bits) }
        {
            if insertion < staged.len() {
                staged[insertion] = staged[insertion - 1];
            }
            insertion -= 1;
        }
        if insertion < staged.len() {
            staged[insertion] = TopLogit {
                token: token as u32,
                logit_bits: bits,
            };
        }
        used = core::cmp::min(used + 1, staged.len());
    }

    *output = staged;
    Ok(())
}

pub fn greedy_token(logits: &[u32]) -> Result<u32, InferenceError> {
    if let Some(error) = logits_error(logits.len(), 0) {
        return Err(error);
    }
    for (index, bits) in logits.iter().copied().enumerate() {
        if !finite(bits) {
            return Err(InferenceError::new(
                InferenceStatus::NONFINITE_OUTPUT,
                InferenceDomain::ARGMAX,
                NO_LAYER,
                0,
                NO_TENSOR,
                0,
                bits as u64,
                InferenceFieldKind::FINITE,
                index as u32,
            ));
        }
    }
    Ok(unsafe { fp32_sse2::inference_argmax(logits.as_ptr(), logits.len()) })
}

fn validate_sample_logits(logits: &[u32]) -> Result<(), InferenceError> {
    if let Some(error) = logits_error(logits.len(), 0) {
        return Err(error);
    }
    for (index, bits) in logits.iter().copied().enumerate() {
        if !finite(bits) {
            return Err(InferenceError::new(
                InferenceStatus::NONFINITE_OUTPUT,
                InferenceDomain::ARGMAX,
                NO_LAYER,
                0,
                NO_TENSOR,
                0,
                bits as u64,
                InferenceFieldKind::FINITE,
                index as u32,
            ));
        }
    }
    Ok(())
}

fn sample_validated(logits: &[u32], state: &mut SamplingState) -> u32 {
    let random = state.next_24();
    let previous_mxcsr = unsafe { fp32_sse2::inference_enter_fp() };
    let selected =
        unsafe { fp32_sse2::inference_sample_top_k_top_p(logits.as_ptr(), logits.len(), random) };
    unsafe { fp32_sse2::inference_exit_fp(previous_mxcsr) };
    selected
}

pub fn sample_token(logits: &[u32], state: &mut SamplingState) -> Result<u32, InferenceError> {
    validate_sample_logits(logits)?;
    Ok(sample_validated(logits, state))
}

pub fn sample_token_with_repetition(
    logits: &mut [u32],
    tokens: &[u32],
    seen: &mut [u8],
    state: &mut SamplingState,
) -> Result<u32, InferenceError> {
    validate_sample_logits(logits)?;
    if seen.len() < logits.len().div_ceil(8) {
        return Err(InferenceError::new(
            InferenceStatus::CAPACITY,
            InferenceDomain::ARGMAX,
            NO_LAYER,
            0,
            NO_TENSOR,
            logits.len().div_ceil(8) as u64,
            seen.len() as u64,
            InferenceFieldKind::BYTES,
            2,
        ));
    }
    for token in tokens.iter().copied() {
        if token as usize >= logits.len() {
            return Err(InferenceError::new(
                InferenceStatus::TOKEN_ID,
                InferenceDomain::ARGMAX,
                NO_LAYER,
                0,
                NO_TENSOR,
                logits.len() as u64,
                token as u64,
                InferenceFieldKind::TOKEN,
                0,
            ));
        }
    }
    let previous_mxcsr = unsafe { fp32_sse2::inference_enter_fp() };
    unsafe {
        fp32_sse2::inference_apply_repetition_penalty(
            logits.as_mut_ptr(),
            tokens.as_ptr(),
            tokens.len(),
            seen.as_mut_ptr(),
        );
    }
    unsafe { fp32_sse2::inference_exit_fp(previous_mxcsr) };
    Ok(sample_validated(logits, state))
}

impl<'model, 'bytes, 'kv, 'scratch> InferenceEngine<'model, 'bytes, 'kv, 'scratch> {
    /// Borrows disjoint KV and scratch arenas for the engine lifetime.
    ///
    /// ```compile_fail
    /// use promptboot_core::{InferenceEngine, ModelView};
    ///
    /// fn alias_is_rejected(model: &ModelView<'_>, storage: &mut [u8]) {
    ///     let _engine = InferenceEngine::build(model, storage, storage, None);
    /// }
    /// ```
    pub fn build(
        model: &'model ModelView<'bytes>,
        kv_storage: &'kv mut [u8],
        scratch_storage: &'scratch mut [u8],
        dispatcher: Option<InferenceDispatcher<'model>>,
    ) -> Result<Self, InferenceError> {
        if let Some(error) = buffer_error(kv_storage, KV_BYTES, 0) {
            return Err(error);
        }
        if let Some(error) = buffer_error(scratch_storage, SCRATCH_BYTES, 1) {
            return Err(error);
        }
        #[cfg(test)]
        if let Some(error) = take_build_fault_for_test() {
            return Err(error);
        }
        scratch_storage[..SCRATCH_BYTES].fill(0);
        Ok(Self {
            model,
            kv: kv_storage,
            scratch: scratch_storage,
            position: 0,
            generation_reserve: 0,
            predictions_emitted: 0,
            last_selected: 0,
            state: InferenceState::RESET,
            backend: fp32_sse2::InferenceBackend::detect(),
            dispatcher,
            kv_high_water: 0,
            scratch_high_water: 0,
            retained_prefix: RetainedPrefix::EMPTY,
            #[cfg(test)]
            q8_preparations: 0,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    #[inline]
    unsafe fn prepare_q8(&mut self, input: *const u8, staging: *mut u8, columns: usize) {
        #[cfg(test)]
        {
            self.q8_preparations += 1;
        }
        fp32_sse2::inference_prepare_q8(input, staging, columns);
    }

    pub fn prefill(
        &mut self,
        prompt_tokens: &[u32],
        generation_reserve: u32,
        logits: &mut [u32],
    ) -> Result<InferenceStep, InferenceError> {
        if self.state == InferenceState::FAULTED {
            return Err(self.state_error(InferenceState::RESET, 3));
        }
        if self.state != InferenceState::RESET {
            return Err(self.state_error(InferenceState::RESET, 1));
        }
        self.prefill_inner(prompt_tokens, generation_reserve, logits, false)
    }

    /// Evaluates a fresh prompt while retaining only its fixed leading KV
    /// state. After `reset_to_prefix`, the same token prefix is verified before
    /// the remaining prompt is appended.
    pub fn prefill_with_prefix(
        &mut self,
        prompt_tokens: &[u32],
        prefix_tokens: u32,
        generation_reserve: u32,
        logits: &mut [u32],
    ) -> Result<InferenceStep, InferenceError> {
        let prefix = prefix_tokens as usize;
        if prefix == 0 || prefix >= prompt_tokens.len() || prefix > MAX_TOKENS {
            self.reset_state();
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                0,
                NO_TENSOR,
                1,
                prefix_tokens as u64,
                InferenceFieldKind::CONTEXT,
                3,
            ));
        }
        if self.retained_prefix.len != 0 {
            if self.state == InferenceState::RESET
                && self.position == self.retained_prefix.len
                && self.retained_prefix.len == prefix_tokens
                && self.retained_prefix.tokens[..prefix] == prompt_tokens[..prefix]
            {
                return self.prefill_inner(
                    &prompt_tokens[prefix..],
                    generation_reserve,
                    logits,
                    true,
                );
            }
            if self.state != InferenceState::RESET {
                let error = self.state_error(InferenceState::RESET, 4);
                self.reset_state();
                return Err(error);
            }
            self.reset_state();
        }
        let result = self.prefill(prompt_tokens, generation_reserve, logits);
        if result.is_ok() {
            self.retained_prefix.clear();
            self.retained_prefix.tokens[..prefix].copy_from_slice(&prompt_tokens[..prefix]);
            self.retained_prefix.len = prefix_tokens;
        }
        result
    }

    pub fn append_prefill(
        &mut self,
        prompt_tokens: &[u32],
        generation_reserve: u32,
        logits: &mut [u32],
    ) -> Result<InferenceStep, InferenceError> {
        match self.state {
            InferenceState::FAULTED => return Err(self.state_error(InferenceState::RESET, 3)),
            InferenceState::RESET => return Err(self.state_error(InferenceState::READY, 0)),
            InferenceState::READY | InferenceState::EOS => {}
        }
        self.prefill_inner(prompt_tokens, generation_reserve, logits, true)
    }

    fn prefill_inner(
        &mut self,
        prompt_tokens: &[u32],
        generation_reserve: u32,
        logits: &mut [u32],
        append: bool,
    ) -> Result<InferenceStep, InferenceError> {
        let start = self.position;
        if !(1..=CONTEXT_LIMIT).contains(&generation_reserve) {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                start,
                NO_TENSOR,
                if generation_reserve == 0 {
                    1
                } else {
                    CONTEXT_LIMIT as u64
                },
                generation_reserve as u64,
                InferenceFieldKind::RESERVE,
                0,
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                start,
                NO_TENSOR,
                1,
                0,
                InferenceFieldKind::CONTEXT,
                0,
            ));
        }
        if !append && prompt_tokens.len() >= CONTEXT_LIMIT as usize {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                0,
                NO_TENSOR,
                (CONTEXT_LIMIT - 1) as u64,
                prompt_tokens.len() as u64,
                InferenceFieldKind::CONTEXT,
                0,
            ));
        }
        let total = start as u64 + prompt_tokens.len() as u64 + generation_reserve as u64;
        if total > CONTEXT_LIMIT as u64 {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                start,
                NO_TENSOR,
                CONTEXT_LIMIT as u64,
                total,
                InferenceFieldKind::CONTEXT,
                1,
            ));
        }
        if let Some(error) = logits_error(logits.len(), start) {
            return Err(error);
        }
        for (index, token) in prompt_tokens.iter().copied().enumerate() {
            if token >= VOCAB_COUNT {
                return Err(InferenceError::new(
                    InferenceStatus::TOKEN_ID,
                    InferenceDomain::PROMPT,
                    NO_LAYER,
                    start + index as u32,
                    NO_TENSOR,
                    VOCAB_COUNT as u64,
                    token as u64,
                    InferenceFieldKind::TOKEN,
                    0,
                ));
            }
        }

        self.generation_reserve = generation_reserve;
        self.predictions_emitted = 0;
        for (index, token) in prompt_tokens.iter().copied().enumerate() {
            let produce_logits = index + 1 == prompt_tokens.len();
            if let Err(error) = self.evaluate(token, logits, produce_logits) {
                return Err(self.fault(error, logits));
            }
        }
        self.predictions_emitted = 1;
        self.publish_step(logits)
    }

    pub fn decode(
        &mut self,
        token: u32,
        logits: &mut [u32],
    ) -> Result<InferenceStep, InferenceError> {
        self.decode_inner(token, logits, true)
    }

    pub fn decode_selected(
        &mut self,
        token: u32,
        logits: &mut [u32],
    ) -> Result<InferenceStep, InferenceError> {
        self.decode_inner(token, logits, false)
    }

    fn decode_inner(
        &mut self,
        token: u32,
        logits: &mut [u32],
        require_previous_greedy: bool,
    ) -> Result<InferenceStep, InferenceError> {
        match self.state {
            InferenceState::FAULTED => return Err(self.state_error(InferenceState::RESET, 3)),
            InferenceState::EOS if require_previous_greedy => {
                return Err(self.state_error(InferenceState::READY, 2))
            }
            InferenceState::RESET => return Err(self.state_error(InferenceState::READY, 0)),
            InferenceState::READY | InferenceState::EOS => {}
        }
        if token >= VOCAB_COUNT {
            return Err(InferenceError::new(
                InferenceStatus::TOKEN_ID,
                InferenceDomain::PROMPT,
                NO_LAYER,
                self.position,
                NO_TENSOR,
                VOCAB_COUNT as u64,
                token as u64,
                InferenceFieldKind::TOKEN,
                1,
            ));
        }
        if require_previous_greedy && token != self.last_selected {
            return Err(InferenceError::new(
                InferenceStatus::STATE,
                InferenceDomain::ARGMAX,
                NO_LAYER,
                self.position,
                NO_TENSOR,
                self.last_selected as u64,
                token as u64,
                InferenceFieldKind::SELECTED,
                0,
            ));
        }
        if self.predictions_emitted >= self.generation_reserve {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::PROMPT,
                NO_LAYER,
                self.position,
                NO_TENSOR,
                self.generation_reserve as u64,
                self.predictions_emitted as u64,
                InferenceFieldKind::RESERVE,
                1,
            ));
        }
        if self.position >= CONTEXT_LIMIT {
            return Err(InferenceError::new(
                InferenceStatus::CONTEXT,
                InferenceDomain::KV,
                NO_LAYER,
                self.position,
                NO_TENSOR,
                CONTEXT_LIMIT as u64,
                self.position as u64,
                InferenceFieldKind::CONTEXT,
                2,
            ));
        }
        if let Some(error) = logits_error(logits.len(), self.position) {
            return Err(error);
        }
        if let Err(error) = self.evaluate(token, logits, true) {
            return Err(self.fault(error, logits));
        }
        self.predictions_emitted += 1;
        self.publish_step(logits)
    }

    fn reset_state(&mut self) {
        self.scratch[..SCRATCH_BYTES].fill(0);
        self.position = 0;
        self.generation_reserve = 0;
        self.predictions_emitted = 0;
        self.last_selected = 0;
        self.state = InferenceState::RESET;
        self.retained_prefix.clear();
    }

    pub fn reset(&mut self) -> Result<(), InferenceError> {
        self.reset_state();
        Ok(())
    }

    /// Resets generation and conversation state while keeping one previously
    /// validated fixed prefix in place. Invalid or faulted state falls back to
    /// a full logical reset.
    pub fn reset_to_prefix(&mut self) -> Result<u32, InferenceError> {
        self.scratch[..SCRATCH_BYTES].fill(0);
        let prefix = self.retained_prefix.len;
        let valid = match self.state {
            InferenceState::RESET => prefix != 0 && self.position == prefix,
            InferenceState::READY | InferenceState::EOS => {
                prefix != 0 && self.position >= prefix && self.position <= CONTEXT_LIMIT
            }
            InferenceState::FAULTED => false,
        };
        if !valid {
            self.reset_state();
            return Ok(0);
        }
        self.position = prefix;
        self.generation_reserve = 0;
        self.predictions_emitted = 0;
        self.last_selected = 0;
        self.state = InferenceState::RESET;
        Ok(prefix)
    }

    pub fn usage(&self) -> InferenceUsage {
        let model_bytes = MODEL_BYTES as u64;
        let kv_current = self.position as u64 * KV_WORDS_PER_POSITION as u64 * 4;
        InferenceUsage {
            weights: ArenaUsage {
                capacity: model_bytes,
                requested: model_bytes,
                committed: model_bytes,
                current: model_bytes,
                high_water: model_bytes,
            },
            kv: ArenaUsage {
                capacity: self.kv.len() as u64,
                requested: KV_BYTES as u64,
                committed: KV_BYTES as u64,
                current: kv_current,
                high_water: self.kv_high_water,
            },
            scratch: ArenaUsage {
                capacity: self.scratch.len() as u64,
                requested: SCRATCH_BYTES as u64,
                committed: SCRATCH_BYTES as u64,
                current: 0,
                high_water: self.scratch_high_water,
            },
            position: self.position,
            context_limit: CONTEXT_LIMIT,
            generation_reserve: self.generation_reserve,
            state: self.state as u32,
        }
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    fn state_error(&self, needed: InferenceState, sub: u32) -> InferenceError {
        InferenceError::new(
            InferenceStatus::STATE,
            InferenceDomain::PROMPT,
            NO_LAYER,
            self.position,
            NO_TENSOR,
            needed as u64,
            self.state as u64,
            InferenceFieldKind::STATE,
            sub,
        )
    }

    fn publish_step(&mut self, logits: &mut [u32]) -> Result<InferenceStep, InferenceError> {
        if let Some(error) = self.inject_and_check(
            InferenceDomain::ARGMAX,
            NO_LAYER,
            290,
            logits.as_mut_ptr(),
            logits.len(),
        ) {
            return Err(self.fault(error, logits));
        }
        let selected = unsafe { fp32_sse2::inference_argmax(logits.as_ptr(), logits.len()) };
        self.last_selected = selected;
        let eos = u32::from(selected == self.model.config().eos);
        self.state = if eos != 0 {
            InferenceState::EOS
        } else {
            InferenceState::READY
        };
        Ok(InferenceStep {
            position: self.position,
            selected_token: selected,
            selected_logit_bits: logits[selected as usize],
            eos,
        })
    }

    fn evaluate(
        &mut self,
        token: u32,
        logits: &mut [u32],
        produce_logits: bool,
    ) -> Result<(), InferenceError> {
        if let Some(dispatcher) = self.dispatcher.as_ref() {
            let context = dispatcher.context;
            let dispatch = dispatcher.dispatch;
            let mut job = core::pin::pin!(InferenceWorkerJob::new(
                (self as *mut Self).cast(),
                coordinate_engine,
                token,
                logits,
                produce_logits,
            ));
            unsafe { dispatch(context, job.as_mut().get_unchecked_mut()) };
            let job = job.as_ref().get_ref();
            return match job.terminal.load(Ordering::Acquire) {
                1 => Ok(()),
                2 => Err(unsafe { ptr::read(job.result.get()) }),
                _ => Err(InferenceError::new(
                    InferenceStatus::STATE,
                    InferenceDomain::PROMPT,
                    NO_LAYER,
                    self.position,
                    NO_TENSOR,
                    1,
                    0,
                    InferenceFieldKind::STATE,
                    6,
                )),
            };
        }
        let previous_mxcsr = unsafe { fp32_sse2::inference_enter_fp() };
        let coordinator = TokenCoordinator::serial(self.backend);
        let result = self.evaluate_inner(token, logits, produce_logits, &coordinator);
        unsafe { fp32_sse2::inference_exit_fp(previous_mxcsr) };
        result
    }

    fn evaluate_inner(
        &mut self,
        token: u32,
        logits: &mut [u32],
        produce_logits: bool,
        coordinator: &TokenCoordinator,
    ) -> Result<(), InferenceError> {
        let position = self.position;
        let base = self.scratch.as_mut_ptr();
        let hidden = unsafe { base.add(HIDDEN) };
        let residual = unsafe { base.add(RESIDUAL) };
        let norm = unsafe { base.add(NORM) };
        let q = unsafe { base.add(Q) };
        let k = unsafe { base.add(K) };
        let v = unsafe { base.add(V) };
        let attention = unsafe { base.add(ATTENTION) };
        let scores = unsafe { base.add(SCORES) };
        let gate = unsafe { base.add(GATE) };
        let up = unsafe { base.add(UP) };
        let product = unsafe { base.add(PRODUCT) };
        let q8_staging = unsafe { base.add(Q8_STAGING) };

        let embedding = self.tensor(0, InferenceDomain::EMBEDDING, NO_LAYER)?;
        self.scratch_high_water = self
            .scratch_high_water
            .max((HIDDEN + INFERENCE_SCRATCH_LAYOUT.hidden.bytes as usize) as u64);
        unsafe {
            fp32_sse2::inference_q4_row(
                embedding.data().as_ptr(),
                token as usize,
                hidden,
                HIDDEN_WORDS,
            );
        }
        self.check(
            InferenceDomain::EMBEDDING,
            NO_LAYER,
            0,
            hidden,
            HIDDEN_WORDS,
        )?;
        #[cfg(test)]
        if let Some(error) = take_arithmetic_fault_for_test(InferenceFieldKind::MUL, position) {
            return Err(error);
        }
        #[cfg(test)]
        if let Some(error) = take_arithmetic_fault_for_test(InferenceFieldKind::USIZE, position) {
            return Err(error);
        }

        for layer in 0..LAYERS {
            let layer_u32 = layer as u32;
            unsafe { ptr::copy_nonoverlapping(hidden, residual, HIDDEN_WORDS * 4) };
            let attn_norm = self.layer_tensor(layer, 10, InferenceDomain::ATTN_NORM)?;
            self.scratch_high_water = self
                .scratch_high_water
                .max((NORM + INFERENCE_SCRATCH_LAYOUT.norm.bytes as usize) as u64);
            unsafe {
                fp32_sse2::inference_rmsnorm(hidden, attn_norm.data().as_ptr(), norm, HIDDEN_WORDS);
            }
            self.check(
                InferenceDomain::ATTN_NORM,
                layer_u32,
                attn_norm.meta().id,
                norm,
                HIDDEN_WORDS,
            )?;

            let q_weight = self.layer_tensor(layer, 19, InferenceDomain::Q)?;
            self.scratch_high_water = self.scratch_high_water.max(USED_SCRATCH as u64);
            unsafe {
                self.prepare_q8(norm, q8_staging, HIDDEN_WORDS);
                coordinator.q4(
                    q_weight.data().as_ptr(),
                    q,
                    q8_staging,
                    HIDDEN_WORDS,
                    HIDDEN_WORDS,
                )
            };
            let q_bias = self.layer_tensor(layer, 18, InferenceDomain::Q)?;
            unsafe { fp32_sse2::inference_add_bias(q, q_bias.data().as_ptr(), HIDDEN_WORDS) };
            self.check(
                InferenceDomain::Q,
                layer_u32,
                q_weight.meta().id,
                q,
                HIDDEN_WORDS,
            )?;

            let k_weight = self.layer_tensor(layer, 16, InferenceDomain::K)?;
            unsafe {
                coordinator.q4(
                    k_weight.data().as_ptr(),
                    k,
                    q8_staging,
                    128,
                    HIDDEN_WORDS,
                )
            };
            let k_bias = self.layer_tensor(layer, 15, InferenceDomain::K)?;
            unsafe { fp32_sse2::inference_add_bias(k, k_bias.data().as_ptr(), 128) };
            self.check(InferenceDomain::K, layer_u32, k_weight.meta().id, k, 128)?;

            let v_weight = self.layer_tensor(layer, 21, InferenceDomain::V)?;
            unsafe {
                coordinator.q4(
                    v_weight.data().as_ptr(),
                    v,
                    q8_staging,
                    128,
                    HIDDEN_WORDS,
                )
            };
            let v_bias = self.layer_tensor(layer, 20, InferenceDomain::V)?;
            unsafe { fp32_sse2::inference_add_bias(v, v_bias.data().as_ptr(), 128) };
            self.check(InferenceDomain::V, layer_u32, v_weight.meta().id, v, 128)?;

            unsafe {
                fp32_sse2::inference_rope_in_place(q, QUERY_HEADS, position as usize);
                fp32_sse2::inference_rope_in_place(k, KV_HEADS, position as usize);
            }
            self.check(
                InferenceDomain::ROPE,
                layer_u32,
                q_weight.meta().id,
                q,
                HIDDEN_WORDS,
            )?;
            self.check(InferenceDomain::ROPE, layer_u32, k_weight.meta().id, k, 128)?;

            self.check(InferenceDomain::KV, layer_u32, k_weight.meta().id, k, 128)?;
            unsafe { self.write_kv(layer, position as usize, k, v)? };
            unsafe {
                fp32_sse2::inference_attention(
                    self.backend,
                    q,
                    self.kv.as_ptr(),
                    attention,
                    scores,
                    layer,
                    position as usize,
                    ((position as usize + 1) + 3) & !3,
                );
            }
            self.check(
                InferenceDomain::ATTENTION,
                layer_u32,
                NO_TENSOR,
                attention,
                HIDDEN_WORDS,
            )?;

            let attn_output = self.layer_tensor(layer, 17, InferenceDomain::ATTN_OUTPUT)?;
            unsafe {
                self.prepare_q8(attention, q8_staging, HIDDEN_WORDS);
                coordinator.q4(
                    attn_output.data().as_ptr(),
                    norm,
                    q8_staging,
                    HIDDEN_WORDS,
                    HIDDEN_WORDS,
                );
                fp32_sse2::inference_add_residual(residual, norm, hidden, HIDDEN_WORDS);
            }
            self.check(
                InferenceDomain::ATTN_OUTPUT,
                layer_u32,
                attn_output.meta().id,
                hidden,
                HIDDEN_WORDS,
            )?;

            unsafe { ptr::copy_nonoverlapping(hidden, residual, HIDDEN_WORDS * 4) };
            let ffn_norm = self.layer_tensor(layer, 14, InferenceDomain::FFN_NORM)?;
            unsafe {
                fp32_sse2::inference_rmsnorm(hidden, ffn_norm.data().as_ptr(), norm, HIDDEN_WORDS);
            }
            self.check(
                InferenceDomain::FFN_NORM,
                layer_u32,
                ffn_norm.meta().id,
                norm,
                HIDDEN_WORDS,
            )?;

            let gate_weight = self.layer_tensor(layer, 12, InferenceDomain::GATE)?;
            unsafe {
                self.prepare_q8(norm, q8_staging, HIDDEN_WORDS);
                coordinator.q4(
                    gate_weight.data().as_ptr(),
                    gate,
                    q8_staging,
                    FFN_WORDS,
                    HIDDEN_WORDS,
                )
            };
            self.check(
                InferenceDomain::GATE,
                layer_u32,
                gate_weight.meta().id,
                gate,
                FFN_WORDS,
            )?;
            let up_weight = self.layer_tensor(layer, 13, InferenceDomain::UP)?;
            unsafe {
                coordinator.q4(
                    up_weight.data().as_ptr(),
                    up,
                    q8_staging,
                    FFN_WORDS,
                    HIDDEN_WORDS,
                )
            };
            self.check(
                InferenceDomain::UP,
                layer_u32,
                up_weight.meta().id,
                up,
                FFN_WORDS,
            )?;
            unsafe { fp32_sse2::inference_swiglu(gate, up, product, FFN_WORDS) };
            self.check(
                InferenceDomain::SWIGLU,
                layer_u32,
                gate_weight.meta().id,
                product,
                FFN_WORDS,
            )?;

            let down_weight = self.layer_tensor(layer, 11, InferenceDomain::DOWN)?;
            unsafe {
                self.prepare_q8(product, q8_staging, FFN_WORDS);
                coordinator.q4(
                    down_weight.data().as_ptr(),
                    norm,
                    q8_staging,
                    HIDDEN_WORDS,
                    FFN_WORDS,
                );
                fp32_sse2::inference_add_residual(residual, norm, hidden, HIDDEN_WORDS);
            }
            self.check(
                InferenceDomain::DOWN,
                layer_u32,
                down_weight.meta().id,
                hidden,
                HIDDEN_WORDS,
            )?;
        }

        #[cfg(test)]
        if let Some(error) = take_arithmetic_fault_for_test(InferenceFieldKind::ADD, position) {
            return Err(error);
        }
        let next_position = self.position.checked_add(1).ok_or_else(|| {
            InferenceError::new(
                InferenceStatus::ARITHMETIC,
                InferenceDomain::PROMPT,
                NO_LAYER,
                position,
                NO_TENSOR,
                position as u64,
                1,
                InferenceFieldKind::ADD,
                0,
            )
        })?;
        if produce_logits {
            let output_norm = self.tensor(289, InferenceDomain::OUTPUT_NORM, NO_LAYER)?;
            unsafe {
                fp32_sse2::inference_rmsnorm(
                    hidden,
                    output_norm.data().as_ptr(),
                    norm,
                    HIDDEN_WORDS,
                );
            }
            self.check(
                InferenceDomain::OUTPUT_NORM,
                NO_LAYER,
                289,
                norm,
                HIDDEN_WORDS,
            )?;
            let output = self.tensor(290, InferenceDomain::LOGITS, NO_LAYER)?;
            unsafe {
                self.prepare_q8(norm, q8_staging, HIDDEN_WORDS);
                coordinator.q8(
                    output.data().as_ptr(),
                    logits.as_mut_ptr(),
                    q8_staging,
                    LOGIT_WORDS,
                    HIDDEN_WORDS,
                );
            }
            if let Some(error) = self.inject_and_check(
                InferenceDomain::LOGITS,
                NO_LAYER,
                290,
                logits.as_mut_ptr(),
                logits.len(),
            ) {
                return Err(error);
            }
        }
        self.position = next_position;
        let kv_current = self.position as u64 * KV_WORDS_PER_POSITION as u64 * 4;
        self.kv_high_water = self.kv_high_water.max(kv_current);
        self.scratch[..SCRATCH_BYTES].fill(0);
        Ok(())
    }

    fn tensor(
        &self,
        id: u32,
        domain: InferenceDomain,
        layer: u32,
    ) -> Result<TensorView<'bytes>, InferenceError> {
        self.model.tensor(id).map_err(|error| {
            InferenceError::new(
                InferenceStatus::MODEL,
                domain,
                layer,
                self.position,
                id,
                id as u64,
                error.available,
                InferenceFieldKind::TENSOR,
                0,
            )
        })
    }

    fn layer_tensor(
        &self,
        layer: usize,
        role: u16,
        domain: InferenceDomain,
    ) -> Result<TensorView<'bytes>, InferenceError> {
        self.model.tensor_for(layer as u16, role).map_err(|error| {
            InferenceError::new(
                InferenceStatus::MODEL,
                domain,
                layer as u32,
                self.position,
                NO_TENSOR,
                role as u64,
                error.available,
                InferenceFieldKind::TENSOR,
                0,
            )
        })
    }

    unsafe fn write_kv(
        &mut self,
        layer: usize,
        position: usize,
        k: *const u8,
        v: *const u8,
    ) -> Result<(), InferenceError> {
        if layer >= LAYERS || position >= CONTEXT_LIMIT as usize {
            return Err(InferenceError::new(
                InferenceStatus::ARITHMETIC,
                InferenceDomain::KV,
                layer as u32,
                position as u32,
                NO_TENSOR,
                layer as u64,
                KV_BYTES as u64 / 4,
                InferenceFieldKind::USIZE,
                0,
            ));
        }
        let final_key = kv_word(layer, 0, position, KV_HEADS - 1, HEAD_WORDS - 1);
        let final_value = kv_word(layer, 1, position, KV_HEADS - 1, HEAD_WORDS - 1);
        if final_key >= KV_BYTES / 4 || final_value >= KV_BYTES / 4 {
            return Err(InferenceError::new(
                InferenceStatus::CAPACITY,
                InferenceDomain::KV,
                layer as u32,
                position as u32,
                NO_TENSOR,
                (final_key.max(final_value) + 1) as u64,
                (KV_BYTES / 4) as u64,
                InferenceFieldKind::BYTES,
                0,
            ));
        }
        for head in 0..KV_HEADS {
            for component in 0..HEAD_WORDS {
                let source = (head * HEAD_WORDS + component) * 4;
                let k_word = kv_word(layer, 0, position, head, component);
                ptr::copy_nonoverlapping(k.add(source), self.kv.as_mut_ptr().add(k_word * 4), 4);
            }
        }
        #[cfg(test)]
        if take_kv_write_fault_for_test(0) {
            return Err(InferenceError::new(
                InferenceStatus::ARITHMETIC,
                InferenceDomain::KV,
                layer as u32,
                position as u32,
                NO_TENSOR,
                0,
                0,
                InferenceFieldKind::FAULT,
                0,
            ));
        }
        for head in 0..KV_HEADS {
            for component in 0..HEAD_WORDS {
                let source = (head * HEAD_WORDS + component) * 4;
                let v_word = kv_word(layer, 1, position, head, component);
                ptr::copy_nonoverlapping(v.add(source), self.kv.as_mut_ptr().add(v_word * 4), 4);
            }
        }
        #[cfg(test)]
        if take_kv_write_fault_for_test(1) {
            return Err(InferenceError::new(
                InferenceStatus::ARITHMETIC,
                InferenceDomain::KV,
                layer as u32,
                position as u32,
                NO_TENSOR,
                0,
                0,
                InferenceFieldKind::FAULT,
                1,
            ));
        }
        Ok(())
    }

    fn check(
        &mut self,
        domain: InferenceDomain,
        layer: u32,
        tensor_id: u32,
        values: *mut u8,
        words: usize,
    ) -> Result<(), InferenceError> {
        if let Some(error) = self.inject_and_check(domain, layer, tensor_id, values.cast(), words) {
            Err(error)
        } else {
            Ok(())
        }
    }

    fn inject_and_check(
        &mut self,
        domain: InferenceDomain,
        layer: u32,
        tensor_id: u32,
        values: *mut u32,
        words: usize,
    ) -> Option<InferenceError> {
        #[cfg(test)]
        inject_fault_for_test(domain, layer, values, words);
        for index in 0..words {
            let bits = unsafe { ptr::read_unaligned(values.add(index)) };
            if !finite(bits) {
                let status = if (domain as u32) <= InferenceDomain::DOWN as u32 {
                    InferenceStatus::NONFINITE_INPUT
                } else {
                    InferenceStatus::NONFINITE_OUTPUT
                };
                return Some(InferenceError::new(
                    status,
                    domain,
                    layer,
                    self.position,
                    tensor_id,
                    0,
                    bits as u64,
                    InferenceFieldKind::FINITE,
                    index as u32,
                ));
            }
        }
        None
    }

    fn fault(&mut self, error: InferenceError, logits: &mut [u32]) -> InferenceError {
        self.scratch[..SCRATCH_BYTES].fill(0);
        logits.fill(0);
        self.state = InferenceState::FAULTED;
        self.retained_prefix.clear();
        error
    }

    #[cfg(test)]
    pub(crate) fn force_decode_state_for_test(
        &mut self,
        position: u32,
        reserve: u32,
        emitted: u32,
        selected: u32,
    ) {
        self.position = position;
        self.generation_reserve = reserve;
        self.predictions_emitted = emitted;
        self.last_selected = selected;
        self.state = InferenceState::READY;
    }

    #[cfg(test)]
    pub(crate) fn force_state_for_test(&mut self, state: InferenceState, position: u32) {
        self.state = state;
        self.position = position;
    }

    #[cfg(test)]
    pub(crate) fn retained_prefix_for_test(&self) -> Option<u32> {
        (self.retained_prefix.len != 0).then_some(self.retained_prefix.len)
    }

    #[cfg(test)]
    pub(crate) fn force_backend_for_test(&mut self, backend: fp32_sse2::InferenceBackend) {
        self.backend = backend;
    }

    #[cfg(test)]
    pub(crate) fn kv_is_filled_for_test(&self, value: u8) -> bool {
        self.kv[..KV_BYTES].iter().all(|byte| *byte == value)
    }

    #[cfg(test)]
    pub(crate) fn kv_byte_for_test(&self, at: usize) -> u8 {
        self.kv[at]
    }

    #[cfg(test)]
    pub(crate) fn logical_kv_word_for_test(
        &self,
        layer: usize,
        kind: usize,
        position: usize,
        head: usize,
        component: usize,
    ) -> u32 {
        let word = kv_word(layer, kind, position, head, component);
        unsafe { ptr::read_unaligned(self.kv.as_ptr().cast::<u32>().add(word)) }
    }

    #[cfg(test)]
    pub(crate) fn write_kv_for_test(
        &mut self,
        layer: usize,
        position: usize,
        k: &[u32; 128],
        v: &[u32; 128],
    ) -> Result<(), InferenceError> {
        unsafe { self.write_kv(layer, position, k.as_ptr().cast(), v.as_ptr().cast()) }
    }

    #[cfg(test)]
    pub(crate) fn reset_q8_preparations_for_test(&mut self) {
        self.q8_preparations = 0;
    }

    #[cfg(test)]
    pub(crate) fn q8_preparations_for_test(&self) -> u32 {
        self.q8_preparations
    }

    #[cfg(test)]
    pub(crate) fn scratch_is_zero_for_test(&self) -> bool {
        self.scratch[..SCRATCH_BYTES].iter().all(|byte| *byte == 0)
    }

    #[cfg(test)]
    pub(crate) fn arena_digests_for_test(&self) -> ([u8; 32], [u8; 32]) {
        (
            crate::sha256::digest(&self.kv[..KV_BYTES]),
            crate::sha256::digest(&self.scratch[..SCRATCH_BYTES]),
        )
    }
}

pub(crate) const fn kv_word(
    layer: usize,
    kind: usize,
    position: usize,
    head: usize,
    component: usize,
) -> usize {
    let base = layer * KV_LAYER_WORDS;
    if kind == 0 {
        base + (((head * (CONTEXT_LIMIT as usize / KV_TILE) + position / KV_TILE) * HEAD_WORDS
            + component)
            * KV_TILE
            + position % KV_TILE)
    } else {
        base + KV_VALUE_WORDS
            + ((head * CONTEXT_LIMIT as usize + position) * HEAD_WORDS + component)
    }
}

#[cfg(test)]
use core::sync::atomic::{AtomicBool, AtomicU64};

#[cfg(test)]
static FAULT_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static FAULT_DOMAIN: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static FAULT_LAYER: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static FAULT_ELEMENT: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static FAULT_BITS: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static BUILD_FAULT_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static BUILD_FAULT_LAYER: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static BUILD_FAULT_TENSOR: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static BUILD_FAULT_NEEDED: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static BUILD_FAULT_AVAILABLE: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static BUILD_FAULT_SUB: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static ARITHMETIC_FAULT_KIND: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_DOMAIN: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_LAYER: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_TENSOR: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_NEEDED: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_AVAILABLE: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static ARITHMETIC_FAULT_SUB: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static KV_WRITE_FAULT_STAGE: AtomicU32 = AtomicU32::new(u32::MAX);

#[cfg(test)]
pub(crate) fn set_inference_fault_for_test(
    domain: InferenceDomain,
    layer: u32,
    element: u32,
    bits: u32,
) {
    FAULT_DOMAIN.store(domain as u32, Ordering::Relaxed);
    FAULT_LAYER.store(layer, Ordering::Relaxed);
    FAULT_ELEMENT.store(element, Ordering::Relaxed);
    FAULT_BITS.store(bits, Ordering::Relaxed);
    FAULT_ACTIVE.store(true, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn set_inference_build_fault_for_test(
    layer: u32,
    tensor: u32,
    needed: u32,
    available: u32,
    sub: u32,
) {
    BUILD_FAULT_LAYER.store(layer, Ordering::Relaxed);
    BUILD_FAULT_TENSOR.store(tensor, Ordering::Relaxed);
    BUILD_FAULT_NEEDED.store(needed, Ordering::Relaxed);
    BUILD_FAULT_AVAILABLE.store(available, Ordering::Relaxed);
    BUILD_FAULT_SUB.store(sub, Ordering::Relaxed);
    BUILD_FAULT_ACTIVE.store(true, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn set_inference_arithmetic_fault_for_test(
    kind: InferenceFieldKind,
    domain: InferenceDomain,
    layer: u32,
    tensor: u32,
    needed: u64,
    available: u64,
    sub: u32,
) {
    assert!(matches!(
        kind,
        InferenceFieldKind::ADD | InferenceFieldKind::MUL | InferenceFieldKind::USIZE
    ));
    ARITHMETIC_FAULT_KIND.store(kind as u32, Ordering::Relaxed);
    ARITHMETIC_FAULT_DOMAIN.store(domain as u32, Ordering::Relaxed);
    ARITHMETIC_FAULT_LAYER.store(layer, Ordering::Relaxed);
    ARITHMETIC_FAULT_TENSOR.store(tensor, Ordering::Relaxed);
    ARITHMETIC_FAULT_NEEDED.store(needed, Ordering::Relaxed);
    ARITHMETIC_FAULT_AVAILABLE.store(available, Ordering::Relaxed);
    ARITHMETIC_FAULT_SUB.store(sub, Ordering::Relaxed);
    ARITHMETIC_FAULT_ACTIVE.store(true, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn set_kv_write_fault_for_test(stage: u32) {
    KV_WRITE_FAULT_STAGE.store(stage, Ordering::Release);
}

#[cfg(test)]
fn take_kv_write_fault_for_test(stage: u32) -> bool {
    KV_WRITE_FAULT_STAGE
        .compare_exchange(stage, u32::MAX, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(test)]
fn take_build_fault_for_test() -> Option<InferenceError> {
    if !BUILD_FAULT_ACTIVE.swap(false, Ordering::AcqRel) {
        return None;
    }
    Some(InferenceError::new(
        InferenceStatus::MODEL,
        InferenceDomain::INIT,
        BUILD_FAULT_LAYER.load(Ordering::Relaxed),
        0,
        BUILD_FAULT_TENSOR.load(Ordering::Relaxed),
        BUILD_FAULT_NEEDED.load(Ordering::Relaxed) as u64,
        BUILD_FAULT_AVAILABLE.load(Ordering::Relaxed) as u64,
        InferenceFieldKind::TENSOR,
        BUILD_FAULT_SUB.load(Ordering::Relaxed),
    ))
}

#[cfg(test)]
fn take_arithmetic_fault_for_test(
    expected_kind: InferenceFieldKind,
    position: u32,
) -> Option<InferenceError> {
    if !ARITHMETIC_FAULT_ACTIVE.load(Ordering::Acquire)
        || ARITHMETIC_FAULT_KIND.load(Ordering::Relaxed) != expected_kind as u32
        || !ARITHMETIC_FAULT_ACTIVE.swap(false, Ordering::AcqRel)
    {
        return None;
    }
    let domain = match ARITHMETIC_FAULT_DOMAIN.load(Ordering::Relaxed) {
        value if value == InferenceDomain::PROMPT as u32 => InferenceDomain::PROMPT,
        value if value == InferenceDomain::KV as u32 => InferenceDomain::KV,
        value if value == InferenceDomain::LOGITS as u32 => InferenceDomain::LOGITS,
        _ => unreachable!(),
    };
    Some(InferenceError::new(
        InferenceStatus::ARITHMETIC,
        domain,
        ARITHMETIC_FAULT_LAYER.load(Ordering::Relaxed),
        position,
        ARITHMETIC_FAULT_TENSOR.load(Ordering::Relaxed),
        ARITHMETIC_FAULT_NEEDED.load(Ordering::Relaxed),
        ARITHMETIC_FAULT_AVAILABLE.load(Ordering::Relaxed),
        expected_kind,
        ARITHMETIC_FAULT_SUB.load(Ordering::Relaxed),
    ))
}

#[cfg(test)]
fn inject_fault_for_test(domain: InferenceDomain, layer: u32, values: *mut u32, words: usize) {
    if !FAULT_ACTIVE.load(Ordering::Acquire)
        || FAULT_DOMAIN.load(Ordering::Relaxed) != domain as u32
        || FAULT_LAYER.load(Ordering::Relaxed) != layer
    {
        return;
    }
    let element = FAULT_ELEMENT.load(Ordering::Relaxed) as usize;
    if element < words && FAULT_ACTIVE.swap(false, Ordering::AcqRel) {
        unsafe { ptr::write_unaligned(values.add(element), FAULT_BITS.load(Ordering::Relaxed)) };
    }
}
