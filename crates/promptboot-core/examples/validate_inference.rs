use std::env;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use promptboot_core::{
    inference_avx2_available, sample_token_with_repetition, FrozenTokenizer, InferenceEngine,
    InferenceState, InferenceUsage, ModelConfig, ModelView, SamplingState, CONTEXT_LIMIT,
    INDEX_BYTES, KV_BYTES, LOGIT_WORDS, SAMPLING_POLICY, SCRATCH_BYTES,
};

const PACK_SHA256: [u8; 32] = [
    0xb0, 0xf9, 0x8e, 0xd6, 0xe0, 0x55, 0x7c, 0xa3, 0x5e, 0x1b, 0xce, 0xd1, 0x00, 0x0c, 0x95, 0x0b,
    0x3c, 0x84, 0x41, 0x42, 0x51, 0xdf, 0x65, 0x29, 0x03, 0x15, 0xa7, 0x96, 0x99, 0x81, 0xd4, 0x2d,
];
const SHA256_BASELINE_MEDIAN_SECONDS: f64 = 1.147_311_064;
const SHA256_TARGET_MEDIAN_SECONDS: f64 = 0.573_655_532;
const SHA256_REQUIRED_REDUCTION_SECONDS: f64 = 0.004_272_708;
const PREFILL_TARGET_MEDIAN_SECONDS: f64 = 2.389_392_961;
const DECODE_TARGET_MEDIAN_SECONDS: f64 = 1.606_323_054;

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
        Self::filled(len, 0)
    }

    fn filled(len: usize, value: u8) -> Self {
        let mut storage = vec![0xa5u8; len + 191];
        let start = 64 + ((64 - ((storage.as_ptr() as usize + 64) & 63)) & 63);
        storage[start..start + len].fill(value);
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
) -> (Vec<u32>, usize) {
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
    assert!(usage.prefix_tokens > 0);
    assert!((usage.prefix_tokens as usize) < committed_tokens.len());
    assert!(scratch_bytes.iter().all(|byte| *byte == 0));
    assert_canaries(scratch_left, scratch_right);
    (committed_tokens, usage.prefix_tokens as usize)
}

#[derive(Clone)]
struct ExactPromptState {
    logits: Vec<u32>,
    kv: Vec<u32>,
}

fn copy_used_kv_words(kv: &[u8], position: usize, config: ModelConfig) -> Vec<u32> {
    let value_words = config.kv_heads as usize * CONTEXT_LIMIT as usize * config.head_dim as usize;
    let layer_words = 2 * value_words;
    let words = position
        * config.block_count as usize
        * 2
        * config.kv_heads as usize
        * config.head_dim as usize;
    let mut output = Vec::with_capacity(words);
    for layer in 0..config.block_count as usize {
        for kind in 0..2 {
            for at in 0..position {
                for head in 0..config.kv_heads as usize {
                    for component in 0..config.head_dim as usize {
                        let base = layer * layer_words;
                        let index = if kind == 0 {
                            base + (((head * (CONTEXT_LIMIT as usize / 4) + at / 4)
                                * config.head_dim as usize
                                + component)
                                * 4
                                + at % 4)
                        } else {
                            base + value_words
                                + ((head * CONTEXT_LIMIT as usize + at) * config.head_dim as usize
                                    + component)
                        };
                        let offset = index * core::mem::size_of::<u32>();
                        output.push(u32::from_ne_bytes(
                            kv[offset..offset + core::mem::size_of::<u32>()]
                                .try_into()
                                .expect("KV word"),
                        ));
                    }
                }
            }
        }
    }
    output
}

fn evaluate_exact_prompt(
    model: &ModelView<'_>,
    config: ModelConfig,
    establish: Option<(&[u32], usize)>,
    prompt: &[u32],
    prefix_tokens: Option<usize>,
) -> (ExactPromptState, f64) {
    let mut kv = AlignedBytes::filled(KV_BYTES, 0xa5);
    let mut scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
    let (kv_left, kv_bytes, kv_right) = kv.split_mut();
    let (scratch_left, scratch_bytes, scratch_right) = scratch.split_mut();
    let mut engine =
        InferenceEngine::build(model, kv_bytes, scratch_bytes, None).expect("build exact engine");
    let mut logits = vec![0u32; LOGIT_WORDS];
    if let Some((establishment_prompt, prefix)) = establish {
        engine
            .prefill_with_prefix(establishment_prompt, prefix as u32, 1, &mut logits)
            .expect("establish retained prefix");
        assert_eq!(
            engine.reset_to_prefix().expect("reset to retained prefix"),
            prefix as u32
        );
    }
    let started = Instant::now();
    match prefix_tokens {
        Some(prefix) => engine
            .prefill_with_prefix(prompt, prefix as u32, 1, &mut logits)
            .expect("retained-prefix prefill"),
        None => engine
            .prefill(prompt, 1, &mut logits)
            .expect("fresh exact prefill"),
    };
    let seconds = started.elapsed().as_secs_f64();
    let usage = engine.usage();
    drop(engine);
    let used_kv = copy_used_kv_words(kv_bytes, usage.position as usize, config);
    assert_eq!(
        used_kv.len() * core::mem::size_of::<u32>(),
        usage.kv.current as usize
    );
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);
    (
        ExactPromptState {
            logits,
            kv: used_kv,
        },
        seconds,
    )
}

fn assert_exact_prompt(actual: &ExactPromptState, expected: &ExactPromptState, label: &str) {
    assert_eq!(
        actual.logits.len(),
        expected.logits.len(),
        "{label} logits length"
    );
    for (index, (actual, expected)) in actual.logits.iter().zip(&expected.logits).enumerate() {
        assert_eq!(actual, expected, "{label} logit word {index}");
    }
    assert_eq!(actual.kv.len(), expected.kv.len(), "{label} KV length");
    for (index, (actual, expected)) in actual.kv.iter().zip(&expected.kv).enumerate() {
        assert_eq!(actual, expected, "{label} KV word {index}");
    }
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
    assert_eq!(
        prompt_logits,
        words(&fixture.join("prompt_final_logits.f32le")),
        "{run}/{case} complete prompt logits"
    );
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

#[derive(Clone, Copy)]
struct TimingStats {
    runs: [f64; 3],
    min: f64,
    median: f64,
    max: f64,
}

impl TimingStats {
    fn from_durations(values: [Duration; 3]) -> Self {
        let runs = values.map(|value| value.as_secs_f64());
        Self {
            runs,
            min: runs.into_iter().fold(f64::INFINITY, f64::min),
            median: median3(runs),
            max: runs.into_iter().fold(0.0, f64::max),
        }
    }

    fn from_seconds(runs: [f64; 3]) -> Self {
        Self {
            runs,
            min: runs.into_iter().fold(f64::INFINITY, f64::min),
            median: median3(runs),
            max: runs.into_iter().fold(0.0, f64::max),
        }
    }
}

fn push_stats(output: &mut String, stats: TimingStats) {
    write!(
        output,
        "{{\"repetitions\":3,\"runs\":[{:.9},{:.9},{:.9}],\"min\":{:.9},\"median\":{:.9},\"max\":{:.9}}}",
        stats.runs[0], stats.runs[1], stats.runs[2], stats.min, stats.median, stats.max
    )
    .expect("timing JSON");
}

fn words_sha256(values: &[u32]) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * std::mem::size_of::<u32>(),
        )
    };
    let mut result = String::with_capacity(64);
    for byte in promptboot_core::sha256::digest(bytes) {
        write!(result, "{byte:02x}").expect("hash hex");
    }
    result
}

#[derive(Default)]
struct ExactOracle {
    checkpoints: Vec<(String, String)>,
    sampled_tokens: Vec<u32>,
    sampler_state: u64,
    sampler_draws: u64,
}

impl ExactOracle {
    fn checkpoint(&mut self, name: &str, logits: &[u32]) {
        assert_eq!(logits.len(), LOGIT_WORDS, "{name} complete logits");
        self.checkpoints
            .push((name.to_owned(), words_sha256(logits)));
    }

    fn encode(&self) -> String {
        let mut output =
            String::from("schema=1\nlogit_words=151936\nseed=0123456789abcdef\nsampling_policy=");
        output.push_str(std::str::from_utf8(SAMPLING_POLICY).expect("sampling policy ASCII"));
        output.push('\n');
        for (name, hash) in &self.checkpoints {
            writeln!(output, "checkpoint={name} sha256={hash}").expect("oracle checkpoint");
        }
        output.push_str("sampled_tokens=");
        for (at, token) in self.sampled_tokens.iter().enumerate() {
            if at != 0 {
                output.push(',');
            }
            write!(output, "{token}").expect("oracle token");
        }
        writeln!(
            output,
            "\nsampler_state={:016x}\nsampler_draws={}",
            self.sampler_state, self.sampler_draws
        )
        .expect("oracle sampler");
        output
    }
}

fn env_usize(name: &str) -> usize {
    env::var(name)
        .unwrap_or_else(|_| panic!("missing {name}; use make benchmark-host"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {name}"))
}

fn json_escape(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value.is_control() => write!(output, "\\u{:04x}", value as u32).unwrap(),
            value => output.push(value),
        }
    }
    output
}

fn cpu_field(cpu: usize, field: &str) -> String {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").expect("read /proc/cpuinfo");
    let mut selected = false;
    for line in cpuinfo.lines() {
        if line.is_empty() {
            selected = false;
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim();
            let value = value.trim();
            if name == "processor" {
                selected = value.parse::<usize>().ok() == Some(cpu);
            } else if selected && name == field {
                return value.to_owned();
            }
        }
    }
    panic!("CPU {cpu} field {field}");
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    let result = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("{program}: {error}"));
    assert!(result.status.success(), "{program} failed");
    String::from_utf8(result.stdout)
        .expect("command UTF-8")
        .trim()
        .to_owned()
}

fn os_name() -> String {
    let release = fs::read_to_string("/etc/os-release").expect("read /etc/os-release");
    release
        .lines()
        .find_map(|line| {
            line.strip_prefix("PRETTY_NAME=")
                .map(|value| value.trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn affinity_list() -> String {
    fs::read_to_string("/proc/self/status")
        .expect("read /proc/self/status")
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .expect("Cpus_allowed_list")
        .to_owned()
}

fn hypervisor_kind() -> String {
    if let Ok(value) = fs::read_to_string("/sys/hypervisor/type") {
        return value.trim().to_owned();
    }
    let output = Command::new("systemd-detect-virt").output();
    match output {
        Ok(result) if result.status.success() => String::from_utf8(result.stdout)
            .expect("hypervisor UTF-8")
            .trim()
            .to_owned(),
        _ => "none".to_owned(),
    }
}

fn render_conversation(
    tokenizer: &FrozenTokenizer<'_, '_, '_>,
    history: &[u32],
    user: &[u8],
) -> Vec<u32> {
    let mut rendered = vec![0u8; 660];
    let mut staging = vec![0u32; 599];
    let mut tokens = vec![0u32; CONTEXT_LIMIT as usize];
    let mut scratch = AlignedBytes::zeroed(5_120);
    let (_, scratch_bytes, _) = scratch.split_mut();
    let mut outcome = promptboot_core::ConversationUsage::ZERO;
    let usage = tokenizer
        .render_conversation_and_tokenize(
            history,
            user,
            &mut rendered,
            &mut staging,
            &mut tokens,
            scratch_bytes,
            &mut outcome,
        )
        .expect("render conversation");
    tokens.truncate(usage.prompt_tokens as usize);
    tokens
}

fn sampled_first_turn(
    engine: &mut InferenceEngine<'_, '_, '_, '_>,
    prompt: &[u32],
    oracle: &mut ExactOracle,
) -> (Vec<u32>, TimingStats) {
    const SEED: u64 = 0x0123_4567_89ab_cdef;
    const MAX_TOKENS: usize = 128;
    let mut logits = vec![0u32; LOGIT_WORDS];
    engine
        .prefill(prompt, MAX_TOKENS as u32, &mut logits)
        .unwrap();
    oracle.checkpoint("conversation_first_hello", &logits);

    let mut sample_times = [Duration::ZERO; 3];
    for at in 0..3 {
        let mut candidate = logits.clone();
        let mut seen = vec![0u8; LOGIT_WORDS.div_ceil(8)];
        let mut sampling = SamplingState::new(SEED);
        let started = Instant::now();
        let selected =
            sample_token_with_repetition(&mut candidate, prompt, &mut seen, &mut sampling).unwrap();
        sample_times[at] = started.elapsed();
        if at == 0 {
            oracle.sampled_tokens.push(selected);
        } else {
            assert_eq!(selected, oracle.sampled_tokens[0], "sample repeat {at}");
        }
    }

    let mut history = prompt.to_vec();
    let mut sampling = SamplingState::new(SEED);
    let mut seen = vec![0u8; LOGIT_WORDS.div_ceil(8)];
    loop {
        let selected =
            sample_token_with_repetition(&mut logits, &history, &mut seen, &mut sampling).unwrap();
        if oracle.sampled_tokens.len() == 1 && history.len() == prompt.len() {
            assert_eq!(selected, oracle.sampled_tokens[0]);
        } else {
            oracle.sampled_tokens.push(selected);
        }
        history.push(if selected == 151_643 {
            151_645
        } else {
            selected
        });
        if selected == 151_643 || selected == 151_645 {
            break;
        }
        assert!(
            history.len() - prompt.len() < MAX_TOKENS,
            "sampled Hello turn did not reach EOS"
        );
        engine.decode_selected(selected, &mut logits).unwrap();
    }
    oracle.sampler_state = sampling.state();
    oracle.sampler_draws = sampling.draws();
    (history, TimingStats::from_durations(sample_times))
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    assert_eq!(
        arguments.len(),
        5,
        "model fixture-root exact-oracle evidence-dir"
    );
    let run_started = Instant::now();
    let selected_cpu = env_usize("PROMPTBOOT_BENCH_CPU");
    let allowed = affinity_list();
    assert_eq!(
        allowed,
        selected_cpu.to_string(),
        "single-CPU affinity was not established"
    );
    assert_eq!(
        std::thread::available_parallelism().unwrap().get(),
        1,
        "benchmark process sees more than one CPU"
    );
    let fixture_root = PathBuf::from(&arguments[2]);
    let oracle_path = PathBuf::from(&arguments[3]);
    let evidence_root = PathBuf::from(&arguments[4]);
    fs::create_dir_all(&evidence_root).expect("create evidence");
    match fs::remove_file(evidence_root.join("run-report.json")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("remove stale run report: {error}"),
    }

    let mut model_read_durations = [Duration::ZERO; 3];
    for duration in &mut model_read_durations {
        let started = Instant::now();
        let bytes = fs::read(&arguments[1]).expect("read packed model");
        black_box(bytes.len());
        *duration = started.elapsed();
    }
    let packed = fs::read(&arguments[1]).expect("read packed model for validation");
    let model_storage = AlignedBytes::from_bytes(&packed);
    let model_bytes = model_storage.bytes();
    let sha256_available = promptboot_core::sha256::sha_ni_available();
    let avx2_available = inference_avx2_available();

    let mut sha256_durations = [Duration::ZERO; 3];
    let mut sha256_backend = None;
    for duration in &mut sha256_durations {
        let started = Instant::now();
        let mut digest = promptboot_core::sha256::Sha256::new();
        digest.update(model_bytes);
        let backend = digest.backend_name();
        let actual = digest.finish();
        *duration = started.elapsed();
        assert_eq!(actual, PACK_SHA256, "packed model digest");
        assert_eq!(
            sha256_backend.get_or_insert(backend),
            &backend,
            "SHA-256 backend changed between repetitions"
        );
    }
    let sha256_backend = sha256_backend.expect("SHA-256 backend");
    assert_eq!(
        sha256_backend,
        if sha256_available { "sha_ni" } else { "scalar" },
        "SHA-256 backend disagrees with CPUID"
    );
    let sha256_timing = TimingStats::from_durations(sha256_durations);

    let mut authenticate_durations = [Duration::ZERO; 3];
    for duration in &mut authenticate_durations {
        let started = Instant::now();
        assert_eq!(
            promptboot_core::sha256::digest(model_bytes),
            PACK_SHA256,
            "authenticate model"
        );
        let opened = ModelView::open_authenticated(model_bytes).expect("open authenticated model");
        black_box(&opened);
        *duration = started.elapsed();
    }
    let model = ModelView::open_authenticated(model_bytes).expect("open model");
    let model_config = model.config();

    let mut tokenizer_durations = [Duration::ZERO; 3];
    for duration in &mut tokenizer_durations {
        let mut storage = AlignedBytes::zeroed(INDEX_BYTES);
        let (_, bytes, _) = storage.split_mut();
        let started = Instant::now();
        let tokenizer = FrozenTokenizer::build(&model, bytes).expect("build tokenizer benchmark");
        black_box(&tokenizer);
        *duration = started.elapsed();
    }
    let mut index = AlignedBytes::zeroed(INDEX_BYTES);
    let (index_left, index_bytes, index_right) = index.split_mut();
    let tokenizer = FrozenTokenizer::build(&model, index_bytes).expect("build tokenizer");
    let (hello_prompt, hello_prefix_tokens) =
        tokenizer_prompt(&tokenizer, &fixture_root.join("hello"), b"Hello");
    let (arithmetic_prompt, arithmetic_prefix_tokens) = tokenizer_prompt(
        &tokenizer,
        &fixture_root.join("arithmetic"),
        b"What is 2+2?",
    );
    let (color_prompt, color_prefix_tokens) =
        tokenizer_prompt(&tokenizer, &fixture_root.join("color"), b"Name one color.");
    assert_eq!(arithmetic_prefix_tokens, hello_prefix_tokens);
    assert_eq!(color_prefix_tokens, hello_prefix_tokens);
    assert_canaries(index_left, index_right);
    drop(tokenizer);
    assert_canaries(index_left, index_right);

    let mut allocation_durations = [Duration::ZERO; 3];
    for duration in &mut allocation_durations {
        let started = Instant::now();
        let kv = AlignedBytes::zeroed(KV_BYTES);
        let scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
        black_box((kv.bytes().as_ptr(), scratch.bytes().as_ptr()));
        *duration = started.elapsed();
    }
    let mut build_durations = [Duration::ZERO; 3];
    for duration in &mut build_durations {
        let mut kv = AlignedBytes::zeroed(KV_BYTES);
        let mut scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
        let (_, kv_bytes, _) = kv.split_mut();
        let (_, scratch_bytes, _) = scratch.split_mut();
        let started = Instant::now();
        let engine = InferenceEngine::build(&model, kv_bytes, scratch_bytes, None)
            .expect("build engine benchmark");
        black_box(engine.position());
        *duration = started.elapsed();
    }

    let mut index = AlignedBytes::zeroed(INDEX_BYTES);
    let (index_left, index_bytes, index_right) = index.split_mut();
    let tokenizer = FrozenTokenizer::build(&model, index_bytes).expect("build tokenizer");
    let mut kv = AlignedBytes::filled(KV_BYTES, 0xa5);
    let mut scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
    let (kv_left, kv_bytes, kv_right) = kv.split_mut();
    let (scratch_left, scratch_bytes, scratch_right) = scratch.split_mut();
    let mut engine =
        InferenceEngine::build(&model, kv_bytes, scratch_bytes, None).expect("build engine");
    let inference_backend = engine.backend_name();
    assert_eq!(
        inference_backend,
        if avx2_available { "avx2" } else { "sse2" },
        "inference backend disagrees with CPU and OS state"
    );
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
    let mut oracle = ExactOracle::default();
    oracle.checkpoint("fixture_hello", &hello_a1.identity.prompt_logits);
    let mut reset_durations = [Duration::ZERO; 3];
    let reset_started = Instant::now();
    engine.reset().expect("reset A1");
    reset_durations[0] = reset_started.elapsed();
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
    oracle.checkpoint("fixture_arithmetic", &arithmetic.identity.prompt_logits);
    let reset_started = Instant::now();
    engine.reset().expect("reset B");
    reset_durations[1] = reset_started.elapsed();
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
    let reset_started = Instant::now();
    engine.reset().expect("reset A2");
    reset_durations[2] = reset_started.elapsed();
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
    oracle.checkpoint("fixture_color", &color.identity.prompt_logits);
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

    let (first_turn_history, sampling_timing) =
        sampled_first_turn(&mut engine, &hello_prompt, &mut oracle);
    assert_eq!(
        engine.position() as usize + 1,
        first_turn_history.len(),
        "retained KV excludes selected EOS"
    );
    let second_turn = render_conversation(&tokenizer, &first_turn_history, b"What is 2+2?");
    let append_start = engine.position() as usize;
    assert_eq!(
        &second_turn[..append_start],
        &first_turn_history[..append_start],
        "retained history prefix"
    );
    let mut cached_logits = vec![0u32; LOGIT_WORDS];
    let cached_started = Instant::now();
    engine
        .append_prefill(&second_turn[append_start..], 1, &mut cached_logits)
        .expect("cached second turn");
    let cached_seconds = cached_started.elapsed().as_secs_f64();
    oracle.checkpoint("conversation_cached_second_turn", &cached_logits);

    let fresh_second_logits = {
        let mut fresh_kv = AlignedBytes::zeroed(KV_BYTES);
        let mut fresh_scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
        let (_, fresh_kv_bytes, _) = fresh_kv.split_mut();
        let (_, fresh_scratch_bytes, _) = fresh_scratch.split_mut();
        let mut fresh =
            InferenceEngine::build(&model, fresh_kv_bytes, fresh_scratch_bytes, None).unwrap();
        let mut logits = vec![0u32; LOGIT_WORDS];
        fresh.prefill(&second_turn, 1, &mut logits).unwrap();
        logits
    };
    assert_eq!(
        cached_logits, fresh_second_logits,
        "cached append versus exact flattened fresh sequence"
    );

    engine.reset().expect("reset conversation");
    let mut reset_hello_logits = vec![0u32; LOGIT_WORDS];
    engine
        .prefill(&hello_prompt, 1, &mut reset_hello_logits)
        .expect("Hello after reset");
    oracle.checkpoint("conversation_reset_hello", &reset_hello_logits);
    let fresh_hello_logits = {
        let mut fresh_kv = AlignedBytes::zeroed(KV_BYTES);
        let mut fresh_scratch = AlignedBytes::zeroed(SCRATCH_BYTES);
        let (_, fresh_kv_bytes, _) = fresh_kv.split_mut();
        let (_, fresh_scratch_bytes, _) = fresh_scratch.split_mut();
        let mut fresh =
            InferenceEngine::build(&model, fresh_kv_bytes, fresh_scratch_bytes, None).unwrap();
        let mut logits = vec![0u32; LOGIT_WORDS];
        fresh.prefill(&hello_prompt, 1, &mut logits).unwrap();
        logits
    };
    assert_eq!(
        reset_hello_logits, fresh_hello_logits,
        "reset engine versus fresh engine complete logits"
    );
    oracle.checkpoint("conversation_fresh_hello", &fresh_hello_logits);

    engine.reset().expect("reset before prefix references");
    let (fresh_hello, _) = evaluate_exact_prompt(&model, model_config, None, &hello_prompt, None);
    let (fresh_arithmetic, _) =
        evaluate_exact_prompt(&model, model_config, None, &arithmetic_prompt, None);
    let (fresh_color, _) = evaluate_exact_prompt(&model, model_config, None, &color_prompt, None);

    let (established, _) = evaluate_exact_prompt(
        &model,
        model_config,
        Some((&hello_prompt, hello_prefix_tokens)),
        &hello_prompt,
        Some(hello_prefix_tokens),
    );
    assert_exact_prompt(&established, &fresh_hello, "established prefix");

    let mut retained_hello_seconds = [0.0; 3];
    for (run, seconds) in retained_hello_seconds.iter_mut().enumerate() {
        let (retained, elapsed) = evaluate_exact_prompt(
            &model,
            model_config,
            Some((&hello_prompt, hello_prefix_tokens)),
            &hello_prompt,
            Some(hello_prefix_tokens),
        );
        assert_exact_prompt(&retained, &fresh_hello, &format!("retained Hello {run}"));
        *seconds = elapsed;
    }

    let (retained_arithmetic, _) = evaluate_exact_prompt(
        &model,
        model_config,
        Some((&hello_prompt, hello_prefix_tokens)),
        &arithmetic_prompt,
        Some(arithmetic_prefix_tokens),
    );
    assert_exact_prompt(
        &retained_arithmetic,
        &fresh_arithmetic,
        "retained arithmetic",
    );

    let (retained_color, _) = evaluate_exact_prompt(
        &model,
        model_config,
        Some((&hello_prompt, hello_prefix_tokens)),
        &color_prompt,
        Some(color_prefix_tokens),
    );
    assert_exact_prompt(&retained_color, &fresh_color, "retained color");

    engine.reset().expect("reset attention");
    let mut attention_tokens = hello_prompt.clone();
    attention_tokens.resize(256, 198);
    let mut attention_logits = vec![0u32; LOGIT_WORDS];
    let attention_started = Instant::now();
    engine
        .prefill(&attention_tokens, 257, &mut attention_logits)
        .expect("attention position 256");
    let attention_256_seconds = attention_started.elapsed().as_secs_f64();
    assert_eq!(engine.position(), 256);
    oracle.checkpoint("attention_position_256", &attention_logits);
    let attention_started = Instant::now();
    while engine.position() < 512 {
        engine
            .decode_selected(198, &mut attention_logits)
            .expect("attention growth decode");
    }
    let attention_512_increment_seconds = attention_started.elapsed().as_secs_f64();
    oracle.checkpoint("attention_position_512", &attention_logits);

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
    let decode_seconds = [
        hello_a1.decode_seconds,
        hello_a2.decode_seconds,
        hello_a3.decode_seconds,
    ];
    let prefill_timing = TimingStats::from_seconds(prefill);
    let retained_prefix_timing = TimingStats::from_seconds(retained_hello_seconds);
    let retained_prefix_speedup = prefill_timing.median / retained_prefix_timing.median;
    let retained_prefix_reduction = prefill_timing.median - retained_prefix_timing.median;
    let fresh_prefill_spread = prefill_timing.max - prefill_timing.min;
    assert!(
        retained_prefix_speedup >= 1.25,
        "retained-prefix speedup {:.6}x is below 1.25x",
        retained_prefix_speedup
    );
    assert!(
        retained_prefix_reduction > fresh_prefill_spread,
        "retained-prefix reduction {:.9}s does not exceed fresh spread {:.9}s",
        retained_prefix_reduction,
        fresh_prefill_spread
    );
    let decode_timing = TimingStats::from_seconds(decode_seconds);
    let total_inference_seconds = hello_a1.total_seconds
        + arithmetic.total_seconds
        + hello_a2.total_seconds
        + color.total_seconds
        + hello_a3.total_seconds;
    let observed_oracle = oracle.encode();
    fs::write(evidence_root.join("observed-oracle.txt"), &observed_oracle)
        .expect("write observed oracle");
    if env::var("PROMPTBOOT_ORACLE_MODE").as_deref() == Ok("record") {
        println!(
            "INFERENCE_ORACLE_RECORDED {}",
            evidence_root.join("observed-oracle.txt").display()
        );
        return;
    }
    let expected_oracle = fs::read_to_string(&oracle_path).expect("read exact oracle");
    assert_eq!(observed_oracle, expected_oracle, "exact host oracle");
    let sha256_reduction = SHA256_BASELINE_MEDIAN_SECONDS - sha256_timing.median;
    let sha256_gate_applicable = sha256_available;
    if sha256_gate_applicable {
        assert!(
            sha256_timing.median <= SHA256_TARGET_MEDIAN_SECONDS,
            "SHA-256 median {:.9}s exceeds {:.9}s target",
            sha256_timing.median,
            SHA256_TARGET_MEDIAN_SECONDS
        );
        assert!(
            sha256_reduction > SHA256_REQUIRED_REDUCTION_SECONDS,
            "SHA-256 reduction {:.9}s does not exceed {:.9}s requirement",
            sha256_reduction,
            SHA256_REQUIRED_REDUCTION_SECONDS
        );
    }
    if avx2_available {
        assert!(
            prefill_timing.median < PREFILL_TARGET_MEDIAN_SECONDS,
            "prefill median {:.9}s does not beat {:.9}s target",
            prefill_timing.median,
            PREFILL_TARGET_MEDIAN_SECONDS
        );
        assert!(
            decode_timing.median < DECODE_TARGET_MEDIAN_SECONDS,
            "decode median {:.9}s does not beat {:.9}s target",
            decode_timing.median,
            DECODE_TARGET_MEDIAN_SECONDS
        );
    }

    let elapsed = run_started.elapsed().as_secs_f64();
    assert!(
        elapsed < 180.0,
        "host benchmark exceeded 180 seconds: {elapsed}"
    );
    let flags = cpu_field(selected_cpu, "flags");
    let hypervisor = hypervisor_kind();
    let mut report = format!(
        concat!(
            "{{\"schema\":2,\"result\":\"PASS\",",
            "\"machine\":{{\"cpu\":{{\"vendor\":\"{}\",\"model\":\"{}\",\"family\":\"{}\",",
            "\"features\":{{\"sse2\":{},\"avx\":{},\"avx2\":{},\"avx512\":{},\"sha\":{}}}}},",
            "\"processors\":{{\"physical\":{},\"logical\":{},\"online\":{},\"host_affinity\":{},\"benchmark_affinity\":1,\"selected\":{}}},",
            "\"hypervisor\":{{\"detected\":{},\"kind\":\"{}\"}},",
            "\"os\":\"{}\",\"kernel\":\"{}\",",
            "\"timer\":{{\"api\":\"std::time::Instant\",\"clock\":\"CLOCK_MONOTONIC\",\"source\":\"{}\"}},",
            "\"sha256_backend\":\"{}\",\"inference_backend\":\"{}\"}},",
            "\"repetitions\":3,",
            "\"timer_boundaries\":{{",
            "\"model_file_read\":\"immediately before std::fs::read through successful complete return\",",
            "\"sha256_digest\":\"already-loaded packed model bytes; immediately before Sha256::new through digest finish, excluding file read, copy, model open, and report formatting\",",
            "\"authenticate_open\":\"immediately before SHA-256 of the complete packed bytes through successful ModelView::open_authenticated return\",",
            "\"tokenizer_index\":\"after zeroed aligned index allocation, immediately before FrozenTokenizer::build through successful return\",",
            "\"kv_scratch_allocation\":\"immediately before aligned zeroed KV allocation through completed aligned zeroed scratch allocation\",",
            "\"engine_build\":\"after arena allocation, immediately before InferenceEngine::build through successful return\",",
            "\"reset\":\"immediately before InferenceEngine::reset through successful return\",",
            "\"prompt_prefill\":\"immediately before InferenceEngine::prefill through successful final-prompt-logits return\",",
            "\"production_sampling\":\"after copying logits and allocating sampler scratch, immediately before sample_token_with_repetition through successful return\",",
            "\"incremental_decode\":\"immediately before the first InferenceEngine::decode through the final successful decode return\"}},",
            "\"timings\":{{"
        ),
        json_escape(&cpu_field(selected_cpu, "vendor_id")),
        json_escape(&cpu_field(selected_cpu, "model name")),
        json_escape(&cpu_field(selected_cpu, "cpu family")),
        std::arch::is_x86_feature_detected!("sse2"),
        std::arch::is_x86_feature_detected!("avx"),
        avx2_available,
        std::arch::is_x86_feature_detected!("avx512f"),
        sha256_available,
        env_usize("PROMPTBOOT_HOST_PHYSICAL_COUNT"),
        env_usize("PROMPTBOOT_HOST_LOGICAL_COUNT"),
        env_usize("PROMPTBOOT_HOST_ONLINE_COUNT"),
        env_usize("PROMPTBOOT_HOST_AFFINITY_COUNT"),
        selected_cpu,
        flags.split_whitespace().any(|flag| flag == "hypervisor") || hypervisor != "none",
        json_escape(&hypervisor),
        json_escape(&os_name()),
        json_escape(&command_output("uname", &["-srvm"])),
        json_escape(
            fs::read_to_string(
                "/sys/devices/system/clocksource/clocksource0/current_clocksource"
            )
            .expect("read clocksource")
            .trim()
        ),
        sha256_backend,
        inference_backend,
    );
    report.push_str("\"model_file_read\":");
    push_stats(
        &mut report,
        TimingStats::from_durations(model_read_durations),
    );
    report.push_str(",\"sha256_digest\":");
    push_stats(&mut report, sha256_timing);
    report.push_str(",\"authenticate_open\":");
    push_stats(
        &mut report,
        TimingStats::from_durations(authenticate_durations),
    );
    report.push_str(",\"tokenizer_index\":");
    push_stats(
        &mut report,
        TimingStats::from_durations(tokenizer_durations),
    );
    report.push_str(",\"kv_scratch_allocation\":");
    push_stats(
        &mut report,
        TimingStats::from_durations(allocation_durations),
    );
    report.push_str(",\"engine_build\":");
    push_stats(&mut report, TimingStats::from_durations(build_durations));
    report.push_str(",\"reset\":");
    push_stats(&mut report, TimingStats::from_durations(reset_durations));
    report.push_str(",\"prompt_prefill\":");
    push_stats(&mut report, prefill_timing);
    report.push_str(",\"retained_prefix_hello\":");
    push_stats(&mut report, retained_prefix_timing);
    report.push_str(",\"production_sampling\":");
    push_stats(&mut report, sampling_timing);
    report.push_str(",\"incremental_decode_seconds\":");
    push_stats(&mut report, decode_timing);
    report.push_str(",\"incremental_tokens_per_second\":");
    push_stats(&mut report, TimingStats::from_seconds(rates));
    write!(
        report,
        concat!(
            "}},\"long_paths\":{{\"cached_second_prefill_seconds\":{:.9},",
            "\"attention_position_256_seconds\":{:.9},",
            "\"attention_256_to_512_seconds\":{:.9},",
            "\"fixture_inference_seconds\":{:.9}}},",
            "\"exact_gate\":{{\"logit_words\":151936,\"fixture_cases\":[\"hello\",\"arithmetic\",\"color\"],",
            "\"cached_second_turn\":\"all-logit-bits-equal-fresh-flattened-sequence\",",
            "\"reset\":\"all-logit-bits-equal-fresh-engine\",",
            "\"retained_prefix\":\"all-used-f32-kv-words-and-final-logit-bits-equal-fresh\",",
            "\"attention_positions\":[256,512],\"kv_precision\":\"F32\",",
            "\"oracle\":\"{}\",\"seed\":\"0123456789abcdef\",",
            "\"sampled_token_count\":{},\"sampler_state\":\"{:016x}\",\"sampler_draws\":{}}},",
            "\"retained_prefix_gate\":{{\"required_speedup\":1.25,\"observed_speedup\":{:.9},",
            "\"fresh_spread_seconds\":{:.9},\"observed_reduction_seconds\":{:.9},\"pass\":true}},",
            "\"sha256_gate\":{{\"baseline_median_seconds\":{:.9},\"target_median_seconds\":{:.9},",
            "\"required_reduction_seconds\":{:.9},\"observed_reduction_seconds\":{:.9},",
            "\"applicable\":{},\"pass\":{}}},",
            "\"inference_gate\":{{\"prefill_target_median_seconds\":{:.9},",
            "\"decode_target_median_seconds\":{:.9},\"applicable\":{},\"pass\":{}}},",
            "\"budget\":{{\"seconds\":180.0,\"elapsed_seconds\":{:.9},\"pass\":true}}}}\n"
        ),
        cached_seconds,
        attention_256_seconds,
        attention_512_increment_seconds,
        total_inference_seconds,
        json_escape(&oracle_path.display().to_string()),
        oracle.sampled_tokens.len(),
        oracle.sampler_state,
        oracle.sampler_draws,
        retained_prefix_speedup,
        fresh_prefill_spread,
        retained_prefix_reduction,
        SHA256_BASELINE_MEDIAN_SECONDS,
        SHA256_TARGET_MEDIAN_SECONDS,
        SHA256_REQUIRED_REDUCTION_SECONDS,
        sha256_reduction,
        sha256_gate_applicable,
        if sha256_gate_applicable {
            "true"
        } else {
            "null"
        },
        PREFILL_TARGET_MEDIAN_SECONDS,
        DECODE_TARGET_MEDIAN_SECONDS,
        avx2_available,
        if avx2_available { "true" } else { "null" },
        elapsed,
    )
    .expect("finish report");

    engine.reset().expect("final reset");
    drop(tokenizer);
    assert_canaries(index_left, index_right);
    drop(engine);
    assert!(kv_bytes.iter().any(|byte| *byte != 0));
    assert_eq!(kv_bytes[KV_BYTES - 1], 0xa5);
    assert!(scratch_bytes.iter().all(|byte| *byte == 0));
    assert_canaries(kv_left, kv_right);
    assert_canaries(scratch_left, scratch_right);
    fs::write(evidence_root.join("run-report.json"), report).expect("write report");
    println!(
        "INFERENCE_HARNESS_PASS cpu={} fixtures=3 cached=exact reset=exact attention=256,512 elapsed={:.3}",
        selected_cpu, elapsed
    );
}
