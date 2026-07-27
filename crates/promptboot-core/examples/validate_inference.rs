use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use promptboot_core::{
    FrozenTokenizer, InferenceEngine, InferenceState, InferenceUsage, ModelView, CONTEXT_LIMIT,
    INDEX_BYTES, KV_BYTES, LOGIT_WORDS, SCRATCH_BYTES,
};

const PACK_SHA256: [u8; 32] = [
    0xb0, 0xf9, 0x8e, 0xd6, 0xe0, 0x55, 0x7c, 0xa3, 0x5e, 0x1b, 0xce, 0xd1, 0x00, 0x0c, 0x95, 0x0b,
    0x3c, 0x84, 0x41, 0x42, 0x51, 0xdf, 0x65, 0x29, 0x03, 0x15, 0xa7, 0x96, 0x99, 0x81, 0xd4, 0x2d,
];

struct AlignedBytes {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBytes {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut storage = vec![0xa5u8; bytes.len() + 191];
        let start = 64 + ((64 - ((storage.as_ptr() as usize + 64) & 63)) & 63);
        storage[start..start + bytes.len()].copy_from_slice(bytes);
        Self {
            storage,
            start,
            len: bytes.len(),
        }
    }

    fn zeroed(len: usize) -> Self {
        let mut storage = vec![0xa5u8; len + 191];
        let start = 64 + ((64 - ((storage.as_ptr() as usize + 64) & 63)) & 63);
        storage[start..start + len].fill(0);
        Self {
            storage,
            start,
            len,
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }

    fn split_mut(&mut self) -> (&mut [u8], &mut [u8], &mut [u8]) {
        let (left, rest) = self.storage.split_at_mut(self.start);
        let (middle, right) = rest.split_at_mut(self.len);
        (left, middle, right)
    }
}

fn assert_canaries(left: &[u8], right: &[u8]) {
    assert!(left.iter().all(|byte| *byte == 0xa5));
    assert!(right.iter().all(|byte| *byte == 0xa5));
}

fn words(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).expect("read words");
    assert_eq!(bytes.len() & 3, 0, "aligned word file {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("word")))
        .collect()
}

fn write_words(path: &Path, values: &[u32]) {
    let mut encoded = Vec::with_capacity(values.len() * 4);
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, encoded).expect("write words");
}

fn json_u32(text: &str, key: &str) -> Vec<u32> {
    let needle = format!("\"{key}\":");
    let mut values = Vec::new();
    let mut remaining = text;
    while let Some(at) = remaining.find(&needle) {
        remaining = &remaining[at + needle.len()..];
        let digits = remaining
            .bytes()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        assert!(digits != 0, "JSON integer {key}");
        values.push(remaining[..digits].parse().expect("JSON u32"));
        remaining = &remaining[digits..];
    }
    values
}

fn json_hex(text: &str, key: &str) -> Vec<u32> {
    let needle = format!("\"{key}\":\"");
    let mut values = Vec::new();
    let mut remaining = text;
    while let Some(at) = remaining.find(&needle) {
        remaining = &remaining[at + needle.len()..];
        let end = remaining.find('"').expect("JSON hex close");
        values.push(u32::from_str_radix(&remaining[..end], 16).expect("JSON hex"));
        remaining = &remaining[end + 1..];
    }
    values
}

struct ExpectedSteps {
    selected: Vec<u32>,
    selected_bits: Vec<u32>,
    top_ids: Vec<u32>,
    top_bits: Vec<u32>,
}

impl ExpectedSteps {
    fn load(path: &Path) -> Self {
        let text = fs::read_to_string(path).expect("read steps JSON");
        let steps = json_u32(&text, "step");
        let selected = json_u32(&text, "selected_id");
        let selected_bits = json_hex(&text, "selected_logit_bits");
        let top_ids = json_u32(&text, "id");
        let top_bits = json_hex(&text, "logit_bits");
        assert_eq!(steps, (0..steps.len() as u32).collect::<Vec<_>>());
        assert_eq!(selected.len(), steps.len());
        assert_eq!(selected_bits.len(), steps.len());
        assert_eq!(top_ids.len(), steps.len() * 8);
        assert_eq!(top_bits.len(), steps.len() * 8);
        Self {
            selected,
            selected_bits,
            top_ids,
            top_bits,
        }
    }
}

fn top8(logits: &[u32]) -> [(u32, u32); 8] {
    let mut best = [(u32::MAX, 0u32); 8];
    for (id, bits) in logits.iter().copied().enumerate() {
        let value = f32::from_bits(bits);
        assert!(value.is_finite());
        let mut insertion = 8;
        for (at, (prior_id, prior_bits)) in best.iter().copied().enumerate() {
            if prior_id == u32::MAX {
                insertion = at;
                break;
            }
            let prior = f32::from_bits(prior_bits);
            if value > prior || (value == prior && (id as u32) < prior_id) {
                insertion = at;
                break;
            }
        }
        if insertion != 8 {
            for at in (insertion + 1..8).rev() {
                best[at] = best[at - 1];
            }
            best[insertion] = (id as u32, bits);
        }
    }
    best
}

fn tokenizer_prompt(
    tokenizer: &FrozenTokenizer<'_, '_, '_>,
    fixture: &Path,
    user: &[u8],
) -> Vec<u32> {
    let mut rendered = vec![0xa5u8; 660];
    let mut tokens = vec![u32::MAX; 599];
    let mut scratch = AlignedBytes::zeroed(5_120);
    let (scratch_left, scratch_bytes, scratch_right) = scratch.split_mut();
    let usage = tokenizer
        .render_and_tokenize(user, &mut rendered, &mut tokens, scratch_bytes)
        .expect("public tokenizer prompt");
    let committed_prompt = fs::read(fixture.join("prompt.txt")).expect("prompt text");
    assert_eq!(usage.rendered_bytes as usize, committed_prompt.len());
    assert_eq!(&rendered[..usage.rendered_bytes as usize], committed_prompt);
    let committed_tokens = words(&fixture.join("prompt_tokens.u32le"));
    assert_eq!(usage.token_count as usize, committed_tokens.len());
    assert_eq!(&tokens[..usage.token_count as usize], committed_tokens);
    assert!(scratch_bytes.iter().all(|byte| *byte == 0));
    assert_canaries(scratch_left, scratch_right);
    committed_tokens
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeterministicUsage {
    weights: [u64; 4],
    kv: [u64; 4],
    scratch: [u64; 4],
    position: u32,
    context_limit: u32,
    generation_reserve: u32,
    state: u32,
}

impl DeterministicUsage {
    fn from_usage(usage: InferenceUsage) -> Self {
        Self {
            weights: [
                usage.weights.capacity,
                usage.weights.requested,
                usage.weights.committed,
                usage.weights.current,
            ],
            kv: [
                usage.kv.capacity,
                usage.kv.requested,
                usage.kv.committed,
                usage.kv.current,
            ],
            scratch: [
                usage.scratch.capacity,
                usage.scratch.requested,
                usage.scratch.committed,
                usage.scratch.current,
            ],
            position: usage.position,
            context_limit: usage.context_limit,
            generation_reserve: usage.generation_reserve,
            state: usage.state,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RunIdentity {
    prompt_logits: Vec<u32>,
    all_logits: Vec<u32>,
    selected: Vec<u32>,
    usage: Vec<DeterministicUsage>,
}

struct RunResult {
    identity: RunIdentity,
    prefill_seconds: f64,
    decode_seconds: f64,
    total_seconds: f64,
}

fn validate_step(
    case: &str,
    run: &str,
    at: usize,
    step: promptboot_core::InferenceStep,
    logits: &[u32],
    expected: &ExpectedSteps,
    usage: InferenceUsage,
    prompt_len: usize,
    prediction_count: usize,
    expected_kv_high_water: u64,
    evidence: &mut String,
) {
    assert_eq!(
        step.selected_token, expected.selected[at],
        "{run}/{case} selected {at}"
    );
    assert_eq!(
        step.selected_logit_bits, expected.selected_bits[at],
        "{run}/{case} selected bits {at}"
    );
    assert_eq!(
        logits[step.selected_token as usize],
        step.selected_logit_bits
    );
    let actual_top = top8(logits);
    for rank in 0..8 {
        assert_eq!(
            actual_top[rank].0,
            expected.top_ids[at * 8 + rank],
            "{run}/{case} top ID step={at} rank={rank}"
        );
        assert_eq!(
            actual_top[rank].1,
            expected.top_bits[at * 8 + rank],
            "{run}/{case} top bits step={at} rank={rank}"
        );
    }
    assert_eq!(step.position, (prompt_len + at) as u32);
    assert_eq!(step.eos, u32::from(step.selected_token == 151_645));
    assert_eq!(usage.position, step.position);
    assert_eq!(usage.context_limit, CONTEXT_LIMIT);
    assert_eq!(usage.generation_reserve, prediction_count as u32);
    assert_eq!(
        usage.state,
        if step.eos == 0 {
            InferenceState::READY as u32
        } else {
            InferenceState::EOS as u32
        }
    );
    assert_eq!(usage.weights.capacity, 426_762_944);
    assert_eq!(usage.weights.requested, 426_762_944);
    assert_eq!(usage.weights.committed, 426_762_944);
    assert_eq!(usage.weights.current, 426_762_944);
    assert_eq!(usage.weights.high_water, 426_762_944);
    assert_eq!(usage.kv.capacity, KV_BYTES as u64);
    assert_eq!(usage.kv.requested, KV_BYTES as u64);
    assert_eq!(usage.kv.committed, KV_BYTES as u64);
    assert_eq!(usage.kv.current, step.position as u64 * 24_576);
    assert_eq!(usage.kv.high_water, expected_kv_high_water);
    assert_eq!(usage.scratch.capacity, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.requested, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.committed, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.current, 0);
    assert_eq!(usage.scratch.high_water, 213_568);
    let top = actual_top
        .iter()
        .map(|(id, bits)| format!("{id}:{bits:08x}"))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(
        evidence,
        "{run}\t{case}\t{at}\t{}\t{}\t{:08x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        step.position,
        step.selected_token,
        step.selected_logit_bits,
        usage.kv.current,
        usage.kv.high_water,
        usage.scratch.high_water,
        usage.generation_reserve,
        usage.state,
        step.eos,
        top,
    )
    .expect("step evidence");
}

fn run_case(
    engine: &mut InferenceEngine<'_, '_, '_, '_>,
    fixture_root: &Path,
    case: &str,
    run: &str,
    prompt: &[u32],
    baseline_kv_high_water: u64,
    step_evidence: &mut String,
) -> RunResult {
    let fixture = fixture_root.join(case);
    let continuation = words(&fixture.join("continuation.u32le"));
    let expected = ExpectedSteps::load(&fixture.join("steps.json"));
    assert_eq!(continuation, expected.selected);
    let mut logits = vec![0u32; LOGIT_WORDS];
    let total_started = Instant::now();
    let prefill_started = Instant::now();
    let mut step = engine
        .prefill(prompt, continuation.len() as u32, &mut logits)
        .expect("real prefill");
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let prompt_logits = logits.clone();
    let mut all_logits = logits.clone();
    let mut selected = vec![step.selected_token];
    let first_usage = engine.usage();
    let mut usages = vec![DeterministicUsage::from_usage(first_usage)];
    validate_step(
        case,
        run,
        0,
        step,
        &logits,
        &expected,
        first_usage,
        prompt.len(),
        continuation.len(),
        baseline_kv_high_water.max(step.position as u64 * 24_576),
        step_evidence,
    );

    let decode_started = Instant::now();
    for at in 1..continuation.len() {
        step = engine
            .decode(step.selected_token, &mut logits)
            .expect("real incremental decode");
        selected.push(step.selected_token);
        all_logits.extend_from_slice(&logits);
        let usage = engine.usage();
        usages.push(DeterministicUsage::from_usage(usage));
        validate_step(
            case,
            run,
            at,
            step,
            &logits,
            &expected,
            usage,
            prompt.len(),
            continuation.len(),
            baseline_kv_high_water.max(step.position as u64 * 24_576),
            step_evidence,
        );
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert_eq!(selected, continuation);
    RunResult {
        identity: RunIdentity {
            prompt_logits,
            all_logits,
            selected,
            usage: usages,
        },
        prefill_seconds,
        decode_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    }
}

fn assert_reset(engine: &InferenceEngine<'_, '_, '_, '_>, expected_kv_high_water: u64) {
    let usage = engine.usage();
    assert_eq!(usage.state, InferenceState::RESET as u32);
    assert_eq!(usage.position, 0);
    assert_eq!(usage.generation_reserve, 0);
    assert_eq!(usage.context_limit, CONTEXT_LIMIT);
    assert_eq!(usage.weights.capacity, 426_762_944);
    assert_eq!(usage.weights.requested, 426_762_944);
    assert_eq!(usage.weights.committed, 426_762_944);
    assert_eq!(usage.weights.current, 426_762_944);
    assert_eq!(usage.weights.high_water, 426_762_944);
    assert_eq!(usage.kv.current, 0);
    assert_eq!(usage.kv.capacity, KV_BYTES as u64);
    assert_eq!(usage.kv.requested, KV_BYTES as u64);
    assert_eq!(usage.kv.committed, KV_BYTES as u64);
    assert_eq!(usage.kv.high_water, expected_kv_high_water);
    assert_eq!(usage.scratch.capacity, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.requested, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.committed, SCRATCH_BYTES as u64);
    assert_eq!(usage.scratch.current, 0);
    assert_eq!(usage.scratch.high_water, 213_568);
}

fn median3(mut values: [f64; 3]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[1]
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    assert_eq!(arguments.len(), 4, "model fixture-root evidence-dir");
    let packed = fs::read(&arguments[1]).expect("read packed model");
    let model_storage = AlignedBytes::from_bytes(&packed);
    let model_bytes = model_storage.bytes();
    assert_eq!(
        promptboot_core::sha256::digest(model_bytes),
        PACK_SHA256,
        "authenticate model"
    );
    let model = ModelView::open_authenticated(model_bytes).expect("open model");
    let fixture_root = PathBuf::from(&arguments[2]);
    let evidence_root = PathBuf::from(&arguments[3]);
    fs::create_dir_all(&evidence_root).expect("create evidence");

    let mut index = AlignedBytes::zeroed(INDEX_BYTES);
    let (index_left, index_bytes, index_right) = index.split_mut();
    let tokenizer = FrozenTokenizer::build(&model, index_bytes).expect("build tokenizer");
    let hello_prompt = tokenizer_prompt(&tokenizer, &fixture_root.join("hello"), b"Hello");
    let arithmetic_prompt = tokenizer_prompt(
        &tokenizer,
        &fixture_root.join("arithmetic"),
        b"What is 2+2?",
    );
    let color_prompt =
        tokenizer_prompt(&tokenizer, &fixture_root.join("color"), b"Name one color.");
    assert_canaries(index_left, index_right);
    drop(tokenizer);
    assert_canaries(index_left, index_right);

    let mut kv = AlignedBytes::zeroed(KV_BYTES);
    let mut scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
    let (kv_left, kv_bytes, kv_right) = kv.split_mut();
    let (scratch_left, scratch_bytes, scratch_right) = scratch.split_mut();
    let mut engine = InferenceEngine::build(&model, kv_bytes, scratch_bytes).expect("build engine");
    let mut step_evidence = String::from(
        "run\tcase\tstep\tposition\tselected\tselected_bits\tkv_current\tkv_high_water\tscratch_high_water\treserve\tstate\teos\ttop8\n",
    );

    let hello_a1 = run_case(
        &mut engine,
        &fixture_root,
        "hello",
        "A1",
        &hello_prompt,
        0,
        &mut step_evidence,
    );
    write_words(
        &evidence_root.join("hello-actual.f32le"),
        &hello_a1.identity.prompt_logits,
    );
    engine.reset().expect("reset A1");
    assert_reset(&engine, 45 * 24_576);
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);

    let arithmetic = run_case(
        &mut engine,
        &fixture_root,
        "arithmetic",
        "B",
        &arithmetic_prompt,
        45 * 24_576,
        &mut step_evidence,
    );
    write_words(
        &evidence_root.join("arithmetic-actual.f32le"),
        &arithmetic.identity.prompt_logits,
    );
    engine.reset().expect("reset B");
    assert_reset(&engine, 45 * 24_576);
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);

    let hello_a2 = run_case(
        &mut engine,
        &fixture_root,
        "hello",
        "A2",
        &hello_prompt,
        45 * 24_576,
        &mut step_evidence,
    );
    assert_eq!(
        hello_a2.identity, hello_a1.identity,
        "A1/reset/B/reset/A2 identity"
    );
    engine.reset().expect("reset A2");
    assert_reset(&engine, 45 * 24_576);
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);

    let color = run_case(
        &mut engine,
        &fixture_root,
        "color",
        "color",
        &color_prompt,
        45 * 24_576,
        &mut step_evidence,
    );
    write_words(
        &evidence_root.join("color-actual.f32le"),
        &color.identity.prompt_logits,
    );
    engine.reset().expect("reset color");
    assert_reset(&engine, 45 * 24_576);
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);

    let hello_a3 = run_case(
        &mut engine,
        &fixture_root,
        "hello",
        "A3",
        &hello_prompt,
        45 * 24_576,
        &mut step_evidence,
    );
    assert_eq!(
        hello_a3.identity, hello_a1.identity,
        "third fresh A identity"
    );
    engine.reset().expect("reset A3");
    assert_reset(&engine, 45 * 24_576);
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);

    fs::write(evidence_root.join("steps-evidence.tsv"), step_evidence).expect("write steps");
    let prefill = [
        hello_a1.prefill_seconds,
        hello_a2.prefill_seconds,
        hello_a3.prefill_seconds,
    ];
    let rates = [
        15.0 / hello_a1.decode_seconds,
        15.0 / hello_a2.decode_seconds,
        15.0 / hello_a3.decode_seconds,
    ];
    let total_inference_seconds = hello_a1.total_seconds
        + arithmetic.total_seconds
        + hello_a2.total_seconds
        + color.total_seconds
        + hello_a3.total_seconds;
    let prefill_threshold_seconds = 120.0;
    let decode_threshold_tokens_per_second = 0.2;
    let arithmetic_budget_seconds = 155.0;
    let color_budget_seconds = 145.0;
    let inference_budget_seconds = 885.0;
    assert!(
        prefill
            .iter()
            .all(|seconds| *seconds <= prefill_threshold_seconds),
        "hello prefill performance floor"
    );
    assert!(
        rates
            .iter()
            .all(|rate| *rate >= decode_threshold_tokens_per_second),
        "hello decode performance floor"
    );
    assert!(
        arithmetic.total_seconds <= arithmetic_budget_seconds,
        "arithmetic phase budget"
    );
    assert!(
        color.total_seconds <= color_budget_seconds,
        "color phase budget"
    );
    assert!(
        total_inference_seconds <= inference_budget_seconds,
        "full inference phase budget"
    );
    let report = format!(
        concat!(
            "{{\"schema\":1,\"result\":\"PASS\",",
            "\"sequence\":[\"A1\",\"reset\",\"B\",\"reset\",\"A2\",\"reset\",\"color\",\"reset\",\"A3\"],",
            "\"hello_identity\":\"byte-identical-all-step-logits-ids-deterministic-usage\",",
            "\"identity_note\":\"deterministic usage excludes lifetime high-water; lifetime high-water is asserted exactly at every step and reset\",",
            "\"timer_boundaries\":{{",
            "\"prefill_seconds\":\"immediately before InferenceEngine::prefill through successful return\",",
            "\"decode_seconds\":\"immediately before the first InferenceEngine::decode through the final successful decode return\",",
            "\"case_total_seconds\":\"immediately before prefill through the final successful decode return\",",
            "\"inference_total_seconds\":\"sum of the five case-total intervals; excludes model read, tokenizer, resets, evidence encoding, and process startup\"}},",
            "\"prefill_seconds\":{{\"runs\":[{:.9},{:.9},{:.9}],\"min\":{:.9},\"median\":{:.9},\"max\":{:.9}}},",
            "\"incremental_tokens_per_second\":{{\"decoded_tokens_per_run\":15,\"runs\":[{:.9},{:.9},{:.9}],\"min\":{:.9},\"median\":{:.9},\"max\":{:.9}}},",
            "\"phase_seconds\":{{\"arithmetic\":{:.9},\"color\":{:.9},\"inference_total\":{:.9}}},",
            "\"thresholds\":{{",
            "\"hello_prefill_max_seconds\":120.0,\"hello_prefill_pass\":true,",
            "\"hello_decode_min_tokens_per_second\":0.2,\"hello_decode_pass\":true,",
            "\"arithmetic_budget_seconds\":155.0,\"arithmetic_pass\":true,",
            "\"color_budget_seconds\":145.0,\"color_pass\":true,",
            "\"inference_budget_seconds\":885.0,\"inference_total_pass\":true}}}}\n"
        ),
        prefill[0], prefill[1], prefill[2],
        prefill.into_iter().fold(f64::INFINITY, f64::min), median3(prefill), prefill.into_iter().fold(0.0, f64::max),
        rates[0], rates[1], rates[2],
        rates.into_iter().fold(f64::INFINITY, f64::min), median3(rates), rates.into_iter().fold(0.0, f64::max),
        arithmetic.total_seconds, color.total_seconds, total_inference_seconds,
    );
    fs::write(evidence_root.join("run-report.json"), report).expect("write report");

    drop(engine);
    assert!(kv_bytes.iter().all(|byte| *byte == 0));
    assert!(scratch_bytes.iter().all(|byte| *byte == 0));
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);
    println!(
        "INFERENCE_HARNESS_PASS sequence=A1-B-A2-color-A3 cases=5 steps=62 tokenizer=public-api repeated_A=byte-identical"
    );
}
