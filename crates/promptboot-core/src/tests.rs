extern crate std;

use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::vec;
use std::vec::Vec;
use std::{env, fs};

use super::*;
use crate::inference::active_scratch_regions_for_test;

const EXPECTED: &[u8; 680] = include_bytes!("../../../fixtures/analytic/primitives.f32le");
const EXP_EDGE: &[u8; 128] = include_bytes!("../../../fixtures/analytic/expf-edge.u32le");

#[repr(align(64))]
struct Aligned<const N: usize>([u8; N]);

struct ScopedDispatcher {
    calls: Arc<AtomicUsize>,
    failure: Arc<AtomicU32>,
    workers: u32,
    backend: u32,
}

unsafe extern "efiapi" fn dispatch_scoped_workers(
    context: *mut core::ffi::c_void,
    job: *mut InferenceWorkerJob,
) {
    let dispatcher = &mut *context.cast::<ScopedDispatcher>();
    dispatcher.calls.fetch_add(1, Ordering::Relaxed);
    let workers = dispatcher.workers;
    let backend = dispatcher.backend;
    let failure = Arc::clone(&dispatcher.failure);
    let job_address = job as usize;
    thread::scope(|scope| {
        for slot in 1..workers {
            let failure = Arc::clone(&failure);
            scope.spawn(move || {
                let status = unsafe {
                    inference_worker_entry(job_address as *mut _, slot, workers, backend)
                };
                if status != 0 {
                    let _ =
                        failure.compare_exchange(0, status, Ordering::AcqRel, Ordering::Acquire);
                }
            });
        }
        let status = unsafe { inference_worker_entry(job, 0, workers, backend) };
        if status != 0 {
            let _ = failure.compare_exchange(0, status, Ordering::AcqRel, Ordering::Acquire);
        }
    });
}

fn test_dispatcher(
    calls: Arc<AtomicUsize>,
    failure: Arc<AtomicU32>,
    workers: u32,
    backend: u32,
) -> ScopedDispatcher {
    ScopedDispatcher {
        calls,
        failure,
        workers,
        backend,
    }
}

fn words(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn word_bytes(values: &[u32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

fn duration_median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn used_kv_words(kv: &[u8], position: usize, config: ModelConfig) -> Vec<u32> {
    let words = position
        * config.block_count as usize
        * 2
        * config.kv_heads as usize
        * config.head_dim as usize;
    let mut result = Vec::with_capacity(words);
    for layer in 0..config.block_count as usize {
        for kind in 0..2 {
            for at in 0..position {
                for head in 0..config.kv_heads as usize {
                    for component in 0..config.head_dim as usize {
                        let index = crate::inference::kv_word(layer, kind, at, head, component);
                        let offset = index * core::mem::size_of::<u32>();
                        result.push(u32::from_ne_bytes(
                            kv[offset..offset + core::mem::size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        ));
                    }
                }
            }
        }
    }
    result
}

fn hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("bad test hex"),
            };
            digit(pair[0]) * 16 + digit(pair[1])
        })
        .collect()
}

fn expected_words(offset: usize, count: usize) -> Vec<u32> {
    EXPECTED[offset..offset + count * 4]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn expected_named_model_error(name: &str) -> [u64; 7] {
    match name {
        "header_magic" => [3, 1, 0, 0, 80, 88, 17_179_869_184],
        "header_version" => [4, 1, 8, 8, 1, 2, 21_474_836_480],
        "header_endian" => [5, 1, 16, 16, 16_909_060, 0, 25_769_803_776],
        "header_identity" => [6, 1, 40, 40, 24, 25, 30_064_771_072],
        "header_reserved" => [7, 1, 192, 192, 0, 1, 34_359_738_368],
        "section_order" => [16, 2, 1, 320, 2, 1, 38_654_705_664],
        "section_alignment" => [9, 2, 1, 336, 64, 1, 47_244_640_272],
        "section_overlap" => [11, 2, 1, 336, 608_452, 608_448, 47_244_640_272],
        "section_gap" => [12, 2, 1, 608_452, 0, 1, 60_129_542_144],
        "section_trailing_extent" => [16, 2, 6, 664, 422_782_464, 422_782_463, 51_539_607_576],
        "tensor_id" => [17, 3, 0, 3_952_512, 0, 1, 64_424_509_440],
        "tensor_role" => [17, 3, 0, 3_952_518, 1, 99, 73_014_444_032],
        "tensor_dtype" => [17, 3, 0, 3_952_520, 2, 3, 77_309_411_328],
        "tensor_rank" => [17, 3, 0, 3_952_524, 2, 4, 81_604_378_624],
        "tensor_dimension" => [17, 3, 0, 3_952_528, 896, 4_294_967_295, 85_899_345_920],
        "tensor_block_length" => [17, 3, 0, 3_952_552, 76_575_744, 76_575_745, 90_194_313_218],
        "tensor_offset" => [10, 3, 0, 3_952_544, 3_980_480, 3_980_544, 47_244_640_256],
        "tensor_overlap" => [11, 3, 1, 3_952_640, 80_556_224, 80_556_160, 47_244_640_256],
        "tensor_reserved" => [7, 3, 0, 3_952_592, 0, 1, 34_359_738_368],
        "token_offset_start" => [18, 4, 0, 704, 0, 1, 94_489_280_512],
        "token_offset_monotonic" => [18, 4, 0, 708, 1, 0, 94_489_280_513],
        "token_offset_range" => [18, 4, 0, 708, 1_372_758, 1_372_759, 51_539_607_552],
        "token_type" => [18, 4, 0, 1_981_312, 58, 2, 98_784_247_808],
        "token_utf8" => [19, 4, 0, 608_512, 0, 255, 103_079_215_104],
        "merge_id" => [18, 5, 0, 2_133_248, 151_936, 151_936, 107_374_182_400],
        "merge_concat" => [18, 5, 0, 608_988, 196, 116, 124_554_051_586],
        "chat_template" => [18, 6, 0, 3_949_952, 123, 122, 128_849_018_880],
        "merge_duplicate" => [18, 5, 3, 2_133_284, 0, 3, 120_259_084_288],
        _ => panic!("missing named model error {name}"),
    }
}

fn model_error_tuple(error: ModelError) -> [u64; 7] {
    [
        error.status as u64,
        error.domain as u64,
        error.index as u64,
        error.offset,
        error.needed,
        error.available,
        error.detail,
    ]
}

fn inference_error_tuple(error: InferenceError) -> [u64; 8] {
    [
        error.status as u64,
        error.domain as u64,
        error.layer as u64,
        error.position as u64,
        error.tensor_id as u64,
        error.needed,
        error.available,
        error.detail,
    ]
}

#[test]
fn inference_scratch_regions_fit_their_runtime_requirements() {
    let regions = active_scratch_regions_for_test();
    let q8_required = 4_864usize.div_ceil(32) * 34;
    let required = [
        896 * 4,
        896 * 4,
        896 * 4,
        896 * 4,
        2 * 64 * 4,
        2 * 64 * 4,
        896 * 4,
        512 * 4,
        4_864 * 4,
        4_864 * 4,
        4_864 * 4,
        q8_required,
    ];

    for (index, ((offset, capacity), required)) in regions.into_iter().zip(required).enumerate() {
        assert_eq!(offset % 64, 0, "region {index} alignment");
        assert!(capacity >= required, "region {index} capacity");
        assert!(
            offset
                .checked_add(capacity)
                .is_some_and(|end| end <= SCRATCH_BYTES),
            "region {index} bounds"
        );
    }
    for left in 0..regions.len() {
        for right in left + 1..regions.len() {
            let (left_offset, left_capacity) = regions[left];
            let (right_offset, right_capacity) = regions[right];
            assert!(
                left_offset + left_capacity <= right_offset
                    || right_offset + right_capacity <= left_offset,
                "regions {left} and {right} overlap"
            );
        }
    }
    assert!(regions[11].1 >= q8_required, "Q8 staging capacity");
}

fn expected_index_identity_fault(fault: u8) -> [u64; 7] {
    match fault {
        1 => [26, 7, 0, 0, 42, 41, 137_438_953_472],
        2 => [26, 7, 0, 0, 50, 49, 141_733_920_768],
        3 => [26, 7, 0, 0, 43, 42, 146_028_888_064],
        4 => [26, 7, 0, 0, 51, 50, 150_323_855_360],
        5 => [26, 7, 0, 0, 57, 56, 133_143_986_176],
        _ => panic!("missing index identity fault {fault}"),
    }
}

fn invoke(
    operation: PrimitiveOp,
    input: &[u8],
    aux: &[u8],
    output_words: usize,
    dimensions: [u32; 4],
    position: u32,
) -> (Vec<u32>, PrimitiveResult) {
    let mut output = vec![0xfeed_face; output_words];
    let mut scratch = Aligned([0xa5; 4096]);
    let input_pointer = if input.is_empty() {
        ptr::null()
    } else {
        input.as_ptr()
    };
    let aux_pointer = if aux.is_empty() {
        ptr::null()
    } else {
        aux.as_ptr()
    };
    let request = PrimitiveRequest {
        abi_version: ABI_VERSION,
        operation: operation as u32,
        input: input_pointer,
        input_bytes: input.len() as u64,
        aux: aux_pointer,
        aux_bytes: aux.len() as u64,
        output: output.as_mut_ptr(),
        output_capacity_words: output.len() as u64,
        scratch: scratch.0.as_mut_ptr(),
        scratch_bytes: scratch.0.len() as u64,
        dim0: dimensions[0],
        dim1: dimensions[1],
        dim2: dimensions[2],
        dim3: dimensions[3],
        position,
        flags: 0,
    };
    let mut result = PrimitiveResult::failure_safe();
    let status = unsafe { promptboot_run_primitive(&request, &mut result) };
    assert_eq!(status, PrimitiveStatus::OK as u32);
    assert_eq!(result.status, PrimitiveStatus::OK as u32);
    assert_eq!(result.output_words as usize, output_words);
    assert_eq!(result.arena_requested, (output_words * 4) as u64);
    assert_eq!(result.arena_committed, (output_words * 4) as u64);
    assert_eq!(result.arena_current, (output_words * 4) as u64);
    assert_eq!(result.arena_high_water, (output_words * 4) as u64);
    (output, result)
}

fn assert_close(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (f32::from_bits(actual) - f32::from_bits(expected)).abs();
        assert!(
            difference <= 1.0e-5,
            "word {index}: actual={actual:08x} expected={expected:08x} difference={difference}"
        );
    }
}

#[test]
fn abi_layout_is_exact() {
    assert_eq!(size_of::<PrimitiveRequest>(), 96);
    assert_eq!(align_of::<PrimitiveRequest>(), 8);
    assert_eq!(offset_of!(PrimitiveRequest, abi_version), 0);
    assert_eq!(offset_of!(PrimitiveRequest, operation), 4);
    assert_eq!(offset_of!(PrimitiveRequest, input), 8);
    assert_eq!(offset_of!(PrimitiveRequest, input_bytes), 16);
    assert_eq!(offset_of!(PrimitiveRequest, aux), 24);
    assert_eq!(offset_of!(PrimitiveRequest, aux_bytes), 32);
    assert_eq!(offset_of!(PrimitiveRequest, output), 40);
    assert_eq!(offset_of!(PrimitiveRequest, output_capacity_words), 48);
    assert_eq!(offset_of!(PrimitiveRequest, scratch), 56);
    assert_eq!(offset_of!(PrimitiveRequest, scratch_bytes), 64);
    assert_eq!(offset_of!(PrimitiveRequest, dim0), 72);
    assert_eq!(offset_of!(PrimitiveRequest, position), 88);
    assert_eq!(offset_of!(PrimitiveRequest, flags), 92);
    assert_eq!(size_of::<PrimitiveResult>(), 80);
    assert_eq!(align_of::<PrimitiveResult>(), 8);
    assert_eq!(offset_of!(PrimitiveResult, needed_bytes), 24);
    assert_eq!(offset_of!(PrimitiveResult, available_bytes), 32);
    assert_eq!(offset_of!(PrimitiveResult, arena_capacity), 40);
    assert_eq!(offset_of!(PrimitiveResult, arena_high_water), 72);

    assert_eq!(size_of::<InferenceError>(), 48);
    assert_eq!(align_of::<InferenceError>(), 8);
    assert_eq!(offset_of!(InferenceError, status), 0);
    assert_eq!(offset_of!(InferenceError, domain), 4);
    assert_eq!(offset_of!(InferenceError, layer), 8);
    assert_eq!(offset_of!(InferenceError, position), 12);
    assert_eq!(offset_of!(InferenceError, tensor_id), 16);
    assert_eq!(offset_of!(InferenceError, reserved), 20);
    assert_eq!(offset_of!(InferenceError, needed), 24);
    assert_eq!(offset_of!(InferenceError, available), 32);
    assert_eq!(offset_of!(InferenceError, detail), 40);
    assert_eq!(size_of::<InferenceStep>(), 16);
    assert_eq!(offset_of!(InferenceStep, position), 0);
    assert_eq!(offset_of!(InferenceStep, selected_token), 4);
    assert_eq!(offset_of!(InferenceStep, selected_logit_bits), 8);
    assert_eq!(offset_of!(InferenceStep, eos), 12);
    assert_eq!(size_of::<TopLogit>(), 8);
    assert_eq!(align_of::<TopLogit>(), 4);
    assert_eq!(offset_of!(TopLogit, token), 0);
    assert_eq!(offset_of!(TopLogit, logit_bits), 4);
    assert_eq!(size_of::<InferenceUsage>(), 136);
    assert_eq!(align_of::<InferenceUsage>(), 8);
    assert_eq!(offset_of!(InferenceUsage, weights), 0);
    assert_eq!(offset_of!(InferenceUsage, kv), 40);
    assert_eq!(offset_of!(InferenceUsage, scratch), 80);
    assert_eq!(offset_of!(InferenceUsage, position), 120);
    assert_eq!(offset_of!(InferenceUsage, context_limit), 124);
    assert_eq!(offset_of!(InferenceUsage, generation_reserve), 128);
    assert_eq!(offset_of!(InferenceUsage, state), 132);
}

#[test]
fn inference_greedy_is_finite_exact_and_lowest_id_tied() {
    let mut logits = vec![(-100.0f32).to_bits(); LOGIT_WORDS];
    logits[7] = 1.5f32.to_bits();
    logits[3] = 1.5f32.to_bits();
    assert_eq!(greedy_token(&logits), Ok(3));

    let short = greedy_token(&logits[..LOGIT_WORDS - 1]).unwrap_err();
    assert_eq!(
        inference_error_tuple(short),
        [
            4,
            18,
            4_294_967_295,
            0,
            4_294_967_295,
            151_936,
            151_935,
            51_539_607_552
        ]
    );
    let long = vec![0u32; LOGIT_WORDS + 1];
    assert_eq!(
        inference_error_tuple(greedy_token(&long).unwrap_err()),
        [
            4,
            18,
            4_294_967_295,
            0,
            4_294_967_295,
            151_936,
            151_937,
            51_539_607_552
        ]
    );

    logits[11] = f32::NAN.to_bits();
    let nonfinite = greedy_token(&logits).unwrap_err();
    assert_eq!(
        inference_error_tuple(nonfinite),
        [
            7,
            19,
            4_294_967_295,
            0,
            4_294_967_295,
            0,
            2_143_289_344,
            30_064_771_083
        ]
    );
    inference_top_logits_8_is_exact_stable_and_transactional();
    inference_top_logits_8_orders_negative_values_and_signed_zero_by_token();
    inference_probability_transfer_replaces_all_sentinel_bytes();
}

#[test]
fn inference_sampling_is_seeded_bounded_and_transactional() {
    let mut logits = vec![(-100.0f32).to_bits(); LOGIT_WORDS];
    for token in 0..41 {
        logits[token] = (4.0f32 - token as f32 * 0.05).to_bits();
    }
    let mut first = SamplingState::new(0x0123_4567_89ab_cdef);
    let mut second = first;
    for _ in 0..8 {
        let selected = sample_token(&logits, &mut first).unwrap();
        assert_eq!(selected, sample_token(&logits, &mut second).unwrap());
        assert!(selected < SAMPLING_TOP_K);
    }
    assert_eq!(first, second);
    assert_eq!(first.draws(), 8);
    assert_eq!(first.state(), 0x37fe_8664_ce10_9d2f);
    assert_eq!(SamplingState::new(0).state(), 0x9e37_79b9_7f4a_7c15);
    assert_eq!(
        (
            SAMPLING_TEMPERATURE_MILLI,
            SAMPLING_TOP_K,
            SAMPLING_TOP_P_MILLI,
            SAMPLING_REPETITION_PENALTY_MILLI,
            SAMPLING_POLICY,
        ),
        (
            700,
            20,
            800,
            1_100,
            b"temperature_0p7_top_k_20_top_p_0p8_repetition_penalty_1p1" as &[u8],
        )
    );

    let before = first;
    let error = sample_token(&logits[..LOGIT_WORDS - 1], &mut first).unwrap_err();
    assert_eq!(error.status, InferenceStatus::CAPACITY as u32);
    assert_eq!(first, before);
    logits[17] = f32::NAN.to_bits();
    let error = sample_token(&logits, &mut first).unwrap_err();
    assert_eq!(error.status, InferenceStatus::NONFINITE_OUTPUT as u32);
    assert_eq!(first, before);
}

#[test]
fn inference_repetition_penalty_applies_once_per_seen_token_and_clears_scratch() {
    let mut logits = vec![0.0f32.to_bits(); LOGIT_WORDS];
    logits[7] = 11.0f32.to_bits();
    logits[9] = (-2.0f32).to_bits();
    logits[12] = (-0.0f32).to_bits();
    let mut seen = vec![0u8; LOGIT_WORDS.div_ceil(8)];
    let mut sampling = SamplingState::new(1);
    sample_token_with_repetition(&mut logits, &[7, 9, 7, 12, 9], &mut seen, &mut sampling).unwrap();
    assert_eq!(logits[7], (11.0f32 / 1.1).to_bits());
    assert_eq!(logits[9], (-2.0f32 * 1.1).to_bits());
    assert_eq!(logits[12], (-0.0f32).to_bits());
    assert!(seen.iter().all(|byte| *byte == 0));
    assert_eq!(sampling.draws(), 1);

    let before = logits.clone();
    let before_sampling = sampling;
    let error =
        sample_token_with_repetition(&mut logits, &[LOGIT_WORDS as u32], &mut seen, &mut sampling)
            .unwrap_err();
    assert_eq!(error.status, InferenceStatus::TOKEN_ID as u32);
    assert_eq!(logits, before);
    assert!(seen.iter().all(|byte| *byte == 0));
    assert_eq!(sampling, before_sampling);
}

#[test]
fn inference_sampling_can_select_eos_and_collapses_dominant_top_p() {
    let mut logits = vec![(-100.0f32).to_bits(); LOGIT_WORDS];
    let eos = 151_645usize;
    logits[eos] = 10.0f32.to_bits();
    logits[9] = 0.0f32.to_bits();
    for seed in [0, 1, u64::MAX, 0xfeed_face_dead_beef] {
        let mut sampling = SamplingState::new(seed);
        assert_eq!(sample_token(&logits, &mut sampling), Ok(eos as u32));
        assert_eq!(sampling.draws(), 1);
    }
}

fn inference_top_logits_8_is_exact_stable_and_transactional() {
    let mut logits = vec![(-100.0f32).to_bits(); LOGIT_WORDS];
    for (token, value) in [
        (7, 9.0f32),
        (2, 9.0),
        (11, 8.0),
        (4, 7.0),
        (15, 6.0),
        (5, 5.0),
        (20, 4.0),
        (6, 3.0),
        (30, 2.0),
    ] {
        logits[token] = value.to_bits();
    }
    let sentinel = TopLogit {
        token: 0xfeed_face,
        logit_bits: 0xdead_beef,
    };
    let mut actual = [sentinel; 8];
    top_logits_8(&logits, &mut actual).unwrap();
    assert_eq!(
        actual,
        [
            TopLogit {
                token: 2,
                logit_bits: 9.0f32.to_bits()
            },
            TopLogit {
                token: 7,
                logit_bits: 9.0f32.to_bits()
            },
            TopLogit {
                token: 11,
                logit_bits: 8.0f32.to_bits()
            },
            TopLogit {
                token: 4,
                logit_bits: 7.0f32.to_bits()
            },
            TopLogit {
                token: 15,
                logit_bits: 6.0f32.to_bits()
            },
            TopLogit {
                token: 5,
                logit_bits: 5.0f32.to_bits()
            },
            TopLogit {
                token: 20,
                logit_bits: 4.0f32.to_bits()
            },
            TopLogit {
                token: 6,
                logit_bits: 3.0f32.to_bits()
            },
        ]
    );

    let unchanged = [sentinel; 8];
    actual = unchanged;
    assert!(top_logits_8(&logits[..LOGIT_WORDS - 1], &mut actual).is_err());
    assert_eq!(actual, unchanged);
    logits[42] = f32::INFINITY.to_bits();
    let error = top_logits_8(&logits, &mut actual).unwrap_err();
    assert_eq!(error.status, InferenceStatus::NONFINITE_OUTPUT as u32);
    assert_eq!(error.domain, InferenceDomain::ARGMAX as u32);
    assert_eq!(
        error.detail,
        ((InferenceFieldKind::FINITE as u64) << 32) | 42
    );
    assert_eq!(actual, unchanged);
}

fn inference_top_logits_8_orders_negative_values_and_signed_zero_by_token() {
    let mut logits = vec![(-100.0f32).to_bits(); LOGIT_WORDS];
    logits[1] = (-0.0f32).to_bits();
    logits[2] = 0.0f32.to_bits();
    logits[3] = (-1.0f32).to_bits();
    logits[4] = (-2.0f32).to_bits();
    logits[5] = (-3.0f32).to_bits();
    logits[6] = (-4.0f32).to_bits();
    logits[7] = (-5.0f32).to_bits();
    logits[8] = (-6.0f32).to_bits();
    let mut actual = [TopLogit {
        token: 0,
        logit_bits: 0,
    }; 8];
    top_logits_8(&logits, &mut actual).unwrap();
    assert_eq!(actual.map(|item| item.token), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(actual[0].logit_bits, (-0.0f32).to_bits());
    assert_eq!(actual[1].logit_bits, 0.0f32.to_bits());
}

fn inference_probability_transfer_replaces_all_sentinel_bytes() {
    let source = [0x0123_4567u32, 0x89ab_cdef, 0x1020_3040, 0xfedc_ba98];
    let mut destination = [0xa5u8; 16];
    crate::fp32_sse2::copy_probability_words_for_test(&source, &mut destination);
    assert_eq!(
        destination,
        *word_bytes(&source).first_chunk::<16>().unwrap()
    );
    assert!(!destination.iter().any(|byte| *byte == 0xa5));
}

#[test]
fn tiled_kv_layout_is_exact_nonoverlapping_and_bounded() {
    const POSITIONS: [usize; 10] = [0, 1, 3, 4, 5, 255, 256, 511, 512, 32_767];
    const CONTEXT: usize = CONTEXT_LIMIT as usize;
    const HEADS: usize = 2;
    const DIMENSION: usize = 64;
    const TILES: usize = CONTEXT / 4;
    const LAYER_WORDS: usize = 2 * HEADS * CONTEXT * DIMENSION;

    assert_eq!(crate::inference::KV_LAYER_WORDS, LAYER_WORDS);
    assert_eq!(crate::inference::LAYERS * LAYER_WORDS * 4, KV_BYTES);
    assert_eq!(KV_BYTES & 63, 0);
    assert_eq!(LAYER_WORDS * 4 & 63, 0);

    let expected = |layer: usize, kind: usize, position: usize, head: usize, component: usize| {
        let base = layer * LAYER_WORDS;
        if kind == 0 {
            base + (((head * TILES + position / 4) * DIMENSION + component) * 4 + position % 4)
        } else {
            base + HEADS * CONTEXT * DIMENSION
                + ((head * CONTEXT + position) * DIMENSION + component)
        }
    };

    let mut addresses = Vec::new();
    for layer in [0, crate::inference::LAYERS - 1] {
        for kind in 0..2 {
            for position in POSITIONS {
                for head in 0..HEADS {
                    let mut component_addresses = Vec::with_capacity(DIMENSION);
                    for component in 0..DIMENSION {
                        let address =
                            crate::inference::kv_word(layer, kind, position, head, component);
                        assert_eq!(address, expected(layer, kind, position, head, component));
                        assert!(address < KV_BYTES / 4);
                        addresses.push(address);
                        component_addresses.push(address);
                    }
                    let first = *component_addresses.iter().min().unwrap();
                    let last = *component_addresses.iter().max().unwrap();
                    let mut canary = vec![0xa5a5_a5a5u32; last - first + 3];
                    for address in component_addresses {
                        canary[address - first + 1] = address as u32;
                    }
                    assert_eq!(canary[0], 0xa5a5_a5a5);
                    assert_eq!(*canary.last().unwrap(), 0xa5a5_a5a5);
                }
            }
        }
    }
    addresses.sort_unstable();
    let sampled = addresses.len();
    addresses.dedup();
    assert_eq!(addresses.len(), sampled);
    assert_eq!(
        crate::inference::kv_word(
            crate::inference::LAYERS - 1,
            1,
            CONTEXT - 1,
            HEADS - 1,
            DIMENSION - 1,
        ),
        KV_BYTES / 4 - 1
    );
}

#[test]
#[ignore = "host AVX2 performance gate; run pinned on one CPU"]
fn tiled_kv_attention_backends_are_exact_and_faster() {
    const HEADS: usize = 2;
    const CONTEXT: usize = CONTEXT_LIMIT as usize;
    const DIMENSION: usize = 64;
    const HEAD_WORDS: usize = CONTEXT * DIMENSION;
    const LAYER_WORDS: usize = HEADS * HEAD_WORDS * 2;

    assert!(inference_avx2_available());
    let value_bits = |index: usize| {
        let mixed = (index as u32)
            .wrapping_mul(0x9e37_79b9)
            .rotate_left((index & 31) as u32);
        let signed = (mixed & 0x1fff) as i32 - 0x1000;
        ((signed as f32) / 4096.0).to_bits()
    };
    let mut query = vec![0u32; 14 * DIMENSION];
    for (index, value) in query.iter_mut().enumerate() {
        *value = value_bits(index + 1);
    }
    query[..6].copy_from_slice(&[
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x8000_0001,
    ]);
    let mut tiled_kv = vec![0u32; LAYER_WORDS];
    for kind in 0..2 {
        for head in 0..HEADS {
            for position in 0..CONTEXT {
                for component in 0..DIMENSION {
                    let logical =
                        (((kind * HEADS + head) * CONTEXT + position) * DIMENSION) + component;
                    let bits = value_bits(logical + query.len());
                    let tiled = crate::inference::kv_word(0, kind, position, head, component);
                    tiled_kv[tiled] = bits;
                }
            }
        }
    }
    for (component, bits) in query[..6].iter().copied().enumerate() {
        tiled_kv[crate::inference::kv_word(0, 0, 0, 0, component)] = bits;
        tiled_kv[crate::inference::kv_word(0, 1, 0, 1, component)] = bits;
    }

    let previous_mxcsr = unsafe { crate::fp32_sse2::inference_enter_fp() };
    for positions in [
        1usize, 2, 3, 4, 5, 6, 30, 255, 256, 257, 511, 512, 513, 2_048, 8_192, 32_768,
    ] {
        let span = (positions + 3) & !3;
        let mut scalar_output = vec![0u32; 14 * DIMENSION];
        let mut candidate_output = vec![0u32; 14 * DIMENSION];
        let mut scalar_scores = vec![0u32; span];
        let mut candidate_scores = vec![0u32; span];
        let mut scalar_raw = vec![0u32; 14 * span];
        let mut candidate_raw = vec![0u32; 14 * span];
        let mut scalar_probabilities = vec![0u32; 14 * span];
        let mut candidate_probabilities = vec![0u32; 14 * span];
        let checks_masked_tail = matches!(positions, 1 | 2 | 3 | 5 | 30);
        if checks_masked_tail {
            unsafe { crate::fp32_sse2::inference_clear_fp_exceptions_for_test() };
        }
        unsafe {
            crate::fp32_sse2::inference_attention_captured_for_test(
                crate::fp32_sse2::InferenceBackend::Sse2,
                query.as_ptr().cast(),
                tiled_kv.as_ptr().cast(),
                scalar_output.as_mut_ptr().cast(),
                scalar_scores.as_mut_ptr().cast(),
                0,
                positions - 1,
                span,
                scalar_raw.as_mut_ptr(),
                scalar_probabilities.as_mut_ptr(),
            );
        }
        if checks_masked_tail {
            let exceptions = unsafe { crate::fp32_sse2::inference_fp_exceptions_for_test() };
            assert_eq!(
                exceptions & 1,
                0,
                "N={positions} scalar attention raised invalid"
            );
        }
        let mut disabled_keys = Vec::new();
        if checks_masked_tail {
            for head in 0..HEADS {
                for position in positions..span {
                    for component in 0..DIMENSION {
                        let word = crate::inference::kv_word(0, 0, position, head, component);
                        disabled_keys.push((word, tiled_kv[word]));
                        tiled_kv[word] = 0x7f80_0001;
                    }
                }
            }
            unsafe { crate::fp32_sse2::inference_clear_fp_exceptions_for_test() };
        }
        unsafe {
            crate::fp32_sse2::inference_attention_captured_for_test(
                crate::fp32_sse2::InferenceBackend::Avx2,
                query.as_ptr().cast(),
                tiled_kv.as_ptr().cast(),
                candidate_output.as_mut_ptr().cast(),
                candidate_scores.as_mut_ptr().cast(),
                0,
                positions - 1,
                span,
                candidate_raw.as_mut_ptr(),
                candidate_probabilities.as_mut_ptr(),
            );
        }
        if !disabled_keys.is_empty() {
            let exceptions = unsafe { crate::fp32_sse2::inference_fp_exceptions_for_test() };
            for (word, bits) in disabled_keys {
                tiled_kv[word] = bits;
            }
            assert_eq!(
                exceptions & 1,
                0,
                "N={positions} accessed a disabled signaling-NaN K lane"
            );
        }
        assert_eq!(candidate_raw, scalar_raw, "N={positions} raw scores");
        assert_eq!(
            candidate_probabilities, scalar_probabilities,
            "N={positions} probabilities"
        );
        assert_eq!(candidate_output, scalar_output, "N={positions} output");

        let repetitions = match positions {
            1 | 30 | 256 | 512 => 100,
            2_048 | 8_192 => 30,
            _ => continue,
        };
        let mut scalar_times = Vec::with_capacity(repetitions);
        let mut candidate_times = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let started = Instant::now();
            unsafe {
                crate::fp32_sse2::inference_attention(
                    crate::fp32_sse2::InferenceBackend::Sse2,
                    query.as_ptr().cast(),
                    tiled_kv.as_ptr().cast(),
                    scalar_output.as_mut_ptr().cast(),
                    scalar_scores.as_mut_ptr().cast(),
                    0,
                    positions - 1,
                    span,
                );
            }
            scalar_times.push(started.elapsed());
            black_box(&scalar_output);
        }
        for _ in 0..repetitions {
            let started = Instant::now();
            unsafe {
                crate::fp32_sse2::inference_attention(
                    crate::fp32_sse2::InferenceBackend::Avx2,
                    query.as_ptr().cast(),
                    tiled_kv.as_ptr().cast(),
                    candidate_output.as_mut_ptr().cast(),
                    candidate_scores.as_mut_ptr().cast(),
                    0,
                    positions - 1,
                    span,
                );
            }
            candidate_times.push(started.elapsed());
            black_box(&candidate_output);
        }
        let scalar_median = duration_median(scalar_times.clone());
        let candidate_median = duration_median(candidate_times);
        let scalar_min = *scalar_times.iter().min().unwrap();
        let scalar_max = *scalar_times.iter().max().unwrap();
        let scalar_range = scalar_max - scalar_min;
        let reduction = scalar_median.saturating_sub(candidate_median);
        let speedup = scalar_median.as_secs_f64() / candidate_median.as_secs_f64();
        std::println!(
            "TILED_KV_ATTENTION N={positions} reps={repetitions} scalar_median={:.9} candidate_median={:.9} scalar_range={:.9} reduction={:.9} speedup={speedup:.6}",
            scalar_median.as_secs_f64(),
            candidate_median.as_secs_f64(),
            scalar_range.as_secs_f64(),
            reduction.as_secs_f64(),
        );
        if positions == 30 {
            assert!(
                candidate_median <= scalar_median + scalar_range,
                "N=30 regressed beyond scalar range"
            );
        }
        if matches!(positions, 512 | 2_048 | 8_192) {
            assert!(speedup >= 2.0, "N={positions} speedup below 2x");
            assert!(
                reduction > scalar_range,
                "N={positions} reduction did not exceed scalar range"
            );
        }
    }
    unsafe { crate::fp32_sse2::inference_exit_fp(previous_mxcsr) };
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn inference_prechecks_state_usage_and_faults_are_exact() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();
    let mut kv_storage = vec![0xa5u8; KV_BYTES + 191];
    let kv_start = 64 + ((64 - ((kv_storage.as_ptr() as usize + 64) & 63)) & 63);
    let mut scratch_storage = vec![0x5au8; SCRATCH_BYTES + 191];
    let scratch_start = 64 + ((64 - ((scratch_storage.as_ptr() as usize + 64) & 63)) & 63);
    let kv_base = kv_storage.as_ptr();
    let kv_total = kv_storage.len();
    let scratch_base = scratch_storage.as_ptr();
    let scratch_total = scratch_storage.len();
    let canaries_intact = || unsafe {
        core::slice::from_raw_parts(kv_base, kv_start)
            .iter()
            .all(|byte| *byte == 0xa5)
            && core::slice::from_raw_parts(
                kv_base.add(kv_start + KV_BYTES),
                kv_total - kv_start - KV_BYTES,
            )
            .iter()
            .all(|byte| *byte == 0xa5)
            && core::slice::from_raw_parts(scratch_base, scratch_start)
                .iter()
                .all(|byte| *byte == 0x5a)
            && core::slice::from_raw_parts(
                scratch_base.add(scratch_start + SCRATCH_BYTES),
                scratch_total - scratch_start - SCRATCH_BYTES,
            )
            .iter()
            .all(|byte| *byte == 0x5a)
    };
    let nl = NO_LAYER as u64;
    let nt = NO_TENSOR as u64;
    let k = |field: InferenceFieldKind, sub: u32| ((field as u64) << 32) | sub as u64;

    let error = InferenceEngine::build(
        &model,
        &mut kv_storage[kv_start..kv_start + KV_BYTES - 1],
        &mut scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES],
        None,
    )
    .err()
    .expect("short KV must fail");
    assert_eq!(
        inference_error_tuple(error),
        [
            4,
            1,
            nl,
            0,
            nt,
            KV_BYTES as u64,
            (KV_BYTES - 1) as u64,
            k(InferenceFieldKind::BYTES, 0)
        ]
    );
    let error = InferenceEngine::build(
        &model,
        &mut kv_storage[kv_start..kv_start + KV_BYTES],
        &mut scratch_storage[scratch_start + 1..scratch_start + 1 + SCRATCH_BYTES],
        None,
    )
    .err()
    .expect("misaligned scratch must fail");
    assert_eq!(
        inference_error_tuple(error),
        [5, 1, 4_294_967_295, 0, 4_294_967_295, 64, 1, 21_474_836_481]
    );
    let error = InferenceEngine::build(
        &model,
        &mut kv_storage[kv_start..kv_start + KV_BYTES],
        &mut scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES - 1],
        None,
    )
    .err()
    .expect("short scratch must fail");
    assert_eq!(
        inference_error_tuple(error),
        [
            4,
            1,
            nl,
            0,
            nt,
            SCRATCH_BYTES as u64,
            (SCRATCH_BYTES - 1) as u64,
            k(InferenceFieldKind::BYTES, 1)
        ]
    );
    let error = InferenceEngine::build(
        &model,
        &mut kv_storage[kv_start + 1..kv_start + 1 + KV_BYTES],
        &mut scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES],
        None,
    )
    .err()
    .expect("misaligned KV must fail");
    assert_eq!(
        inference_error_tuple(error),
        [
            4 + 1,
            1,
            nl,
            0,
            nt,
            64,
            1,
            k(InferenceFieldKind::ALIGNMENT, 0)
        ]
    );

    for (name, layer, tensor, needed, available, sub, expected) in [
        (
            "role",
            7,
            88,
            19,
            18,
            0,
            [8, 1, 7, 0, 88, 19, 18, 25_769_803_776],
        ),
        (
            "dtype",
            7,
            88,
            2,
            3,
            1,
            [8, 1, 7, 0, 88, 2, 3, 25_769_803_777],
        ),
    ] {
        crate::inference::set_inference_build_fault_for_test(layer, tensor, needed, available, sub);
        let error = InferenceEngine::build(
            &model,
            &mut kv_storage[kv_start..kv_start + KV_BYTES],
            &mut scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES],
            None,
        )
        .err()
        .unwrap_or_else(|| panic!("injected {name} mismatch must fail"));
        assert_eq!(inference_error_tuple(error), expected, "{name}");
    }
    assert!(kv_storage[kv_start..kv_start + KV_BYTES]
        .iter()
        .all(|byte| *byte == 0xa5));
    assert!(
        scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES]
            .iter()
            .all(|byte| *byte == 0x5a)
    );

    let mut logits = vec![0xfeed_faceu32; LOGIT_WORDS];
    let mut engine = InferenceEngine::build(
        &model,
        &mut kv_storage[kv_start..kv_start + KV_BYTES],
        &mut scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let source = [0x3f80_0000u32; 128];
    let before_invalid_write = engine.arena_digests_for_test();
    assert!(engine
        .write_kv_for_test(crate::inference::LAYERS, 0, &source, &source)
        .is_err());
    assert!(engine
        .write_kv_for_test(0, CONTEXT_LIMIT as usize, &source, &source)
        .is_err());
    assert_eq!(engine.arena_digests_for_test(), before_invalid_write);
    assert!(canaries_intact());
    macro_rules! precheck_error {
        ($buffer:ident, $call:expr) => {{
            let before_logits = crate::sha256::digest(word_bytes(&$buffer));
            let before_arenas = engine.arena_digests_for_test();
            let before_usage = engine.usage();
            let error = $call.unwrap_err();
            assert_eq!(crate::sha256::digest(word_bytes(&$buffer)), before_logits);
            assert_eq!(engine.arena_digests_for_test(), before_arenas);
            assert_eq!(engine.usage(), before_usage);
            assert!(canaries_intact());
            error
        }};
    }
    let usage = engine.usage();
    assert_eq!(usage.weights.current, 426_762_944);
    assert_eq!(usage.kv.capacity, KV_BYTES as u64);
    assert_eq!(usage.kv.current, 0);
    assert_eq!(usage.scratch.current, 0);
    assert_eq!(usage.scratch.high_water, 0);
    assert_eq!(usage.state, InferenceState::RESET as u32);

    let cases: [(&str, Vec<u32>, u32, usize, [u64; 8]); 7] = [
        (
            "reserve_zero",
            vec![0],
            0,
            LOGIT_WORDS,
            [2, 2, nl, 0, nt, 1, 0, k(InferenceFieldKind::RESERVE, 0)],
        ),
        (
            "reserve_above_context",
            vec![0],
            CONTEXT_LIMIT + 1,
            LOGIT_WORDS,
            [
                2,
                2,
                nl,
                0,
                nt,
                CONTEXT_LIMIT as u64,
                CONTEXT_LIMIT as u64 + 1,
                k(InferenceFieldKind::RESERVE, 0),
            ],
        ),
        (
            "empty",
            vec![],
            1,
            LOGIT_WORDS,
            [2, 2, nl, 0, nt, 1, 0, k(InferenceFieldKind::CONTEXT, 0)],
        ),
        (
            "length_at_context",
            vec![0; CONTEXT_LIMIT as usize],
            1,
            LOGIT_WORDS,
            [
                2,
                2,
                nl,
                0,
                nt,
                CONTEXT_LIMIT as u64 - 1,
                CONTEXT_LIMIT as u64,
                k(InferenceFieldKind::CONTEXT, 0),
            ],
        ),
        (
            "total_above_context",
            vec![0; CONTEXT_LIMIT as usize - 1],
            2,
            LOGIT_WORDS,
            [
                2,
                2,
                nl,
                0,
                nt,
                CONTEXT_LIMIT as u64,
                CONTEXT_LIMIT as u64 + 1,
                k(InferenceFieldKind::CONTEXT, 1),
            ],
        ),
        (
            "short_logits",
            vec![0],
            1,
            LOGIT_WORDS - 1,
            [
                4,
                18,
                nl,
                0,
                nt,
                LOGIT_WORDS as u64,
                (LOGIT_WORDS - 1) as u64,
                k(InferenceFieldKind::LOGITS, 0),
            ],
        ),
        (
            "invalid_token",
            vec![0, LOGIT_WORDS as u32],
            1,
            LOGIT_WORDS,
            [
                3,
                2,
                nl,
                1,
                nt,
                LOGIT_WORDS as u64,
                LOGIT_WORDS as u64,
                k(InferenceFieldKind::TOKEN, 0),
            ],
        ),
    ];
    for (name, prompt, reserve, logit_words, expected) in cases {
        let error = precheck_error!(
            logits,
            engine.prefill(&prompt, reserve, &mut logits[..logit_words])
        );
        assert_eq!(inference_error_tuple(error), expected, "{name}");
        assert!(logits.iter().all(|word| *word == 0xfeed_face));
    }
    let mut long_logits = vec![0xfeed_faceu32; LOGIT_WORDS + 1];
    assert_eq!(
        inference_error_tuple(precheck_error!(
            long_logits,
            engine.prefill(&[0], 1, &mut long_logits)
        )),
        [
            4,
            18,
            4_294_967_295,
            0,
            4_294_967_295,
            151_936,
            151_937,
            51_539_607_552
        ]
    );
    assert!(long_logits.iter().all(|word| *word == 0xfeed_face));
    assert!(engine.kv_is_filled_for_test(0xa5));
    assert!(engine.scratch_is_zero_for_test());
    let before = precheck_error!(logits, engine.decode(0, &mut logits));
    assert_eq!(
        inference_error_tuple(before),
        [1, 2, nl, 0, nt, 1, 0, k(InferenceFieldKind::STATE, 0)]
    );

    let nonfinite_cases = [
        (
            "embedding",
            InferenceDomain::EMBEDDING,
            NO_LAYER,
            [6, 3, 4_294_967_295, 0, 0, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "attn_norm",
            InferenceDomain::ATTN_NORM,
            0,
            [6, 4, 0, 0, 1, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "q",
            InferenceDomain::Q,
            0,
            [6, 5, 0, 0, 10, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "k",
            InferenceDomain::K,
            0,
            [6, 6, 0, 0, 7, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "v",
            InferenceDomain::V,
            0,
            [6, 7, 0, 0, 12, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "rope",
            InferenceDomain::ROPE,
            0,
            [6, 8, 0, 0, 10, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "kv",
            InferenceDomain::KV,
            0,
            [6, 9, 0, 0, 7, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "attention",
            InferenceDomain::ATTENTION,
            0,
            [6, 10, 0, 0, 4_294_967_295, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "attn_output",
            InferenceDomain::ATTN_OUTPUT,
            0,
            [6, 11, 0, 0, 8, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "ffn_norm",
            InferenceDomain::FFN_NORM,
            0,
            [6, 12, 0, 0, 5, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "gate",
            InferenceDomain::GATE,
            0,
            [6, 13, 0, 0, 3, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "up",
            InferenceDomain::UP,
            0,
            [6, 14, 0, 0, 4, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "swiglu",
            InferenceDomain::SWIGLU,
            0,
            [6, 15, 0, 0, 3, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "down",
            InferenceDomain::DOWN,
            0,
            [6, 16, 0, 0, 2, 0, 2_143_289_344, 30_064_771_072],
        ),
        (
            "output_norm",
            InferenceDomain::OUTPUT_NORM,
            NO_LAYER,
            [
                7,
                17,
                4_294_967_295,
                0,
                289,
                0,
                2_143_289_344,
                30_064_771_072,
            ],
        ),
        (
            "logits",
            InferenceDomain::LOGITS,
            NO_LAYER,
            [
                7,
                18,
                4_294_967_295,
                0,
                290,
                0,
                2_143_289_344,
                30_064_771_072,
            ],
        ),
        (
            "argmax",
            InferenceDomain::ARGMAX,
            NO_LAYER,
            [
                7,
                19,
                4_294_967_295,
                1,
                290,
                0,
                2_143_289_344,
                30_064_771_072,
            ],
        ),
    ];
    assert_eq!(nonfinite_cases.len(), 17);
    for (name, domain, layer, expected) in nonfinite_cases {
        logits.fill(0xfeed_face);
        crate::inference::set_inference_fault_for_test(domain, layer, 0, f32::NAN.to_bits());
        let fault = engine
            .prefill(&[0], 1, &mut logits)
            .err()
            .unwrap_or_else(|| panic!("{name} nonfinite injection must fail"));
        assert_eq!(inference_error_tuple(fault), expected, "{name}");
        let fault_usage = engine.usage();
        assert_eq!(fault_usage.state, 3, "{name} state");
        let expected_high_water = match name {
            "embedding" => 3_584,
            "attn_norm" => 10_752,
            _ => 213_568,
        };
        assert_eq!(
            fault_usage.scratch.high_water, expected_high_water,
            "{name} touched scratch"
        );
        assert!(logits.iter().all(|word| *word == 0), "{name} logits");
        assert!(engine.scratch_is_zero_for_test(), "{name} scratch");
        if domain == InferenceDomain::KV {
            assert!(engine.kv_is_filled_for_test(0xa5), "invalid KV published");
        }
        if name == "embedding" {
            let state_error = precheck_error!(logits, engine.prefill(&[0], 1, &mut logits));
            assert_eq!(
                inference_error_tuple(state_error),
                [1, 2, 4_294_967_295, 0, 4_294_967_295, 0, 3, 4_294_967_299]
            );
            let state_error = precheck_error!(logits, engine.decode(0, &mut logits));
            assert_eq!(
                inference_error_tuple(state_error),
                [1, 2, 4_294_967_295, 0, 4_294_967_295, 0, 3, 4_294_967_299]
            );
        }
        engine.reset().unwrap();
        let reset_usage = engine.usage();
        assert_eq!(reset_usage.state, 0, "{name} reset state");
        assert_eq!(reset_usage.position, 0, "{name} reset position");
        assert_eq!(
            reset_usage.scratch.high_water,
            fault_usage.scratch.high_water
        );
        assert_eq!(reset_usage.kv.high_water, fault_usage.kv.high_water);
        assert!(engine.scratch_is_zero_for_test(), "{name} reset scratch");
        assert!(canaries_intact(), "{name} canaries");
    }

    let arithmetic_cases = [
        (
            "add",
            InferenceFieldKind::ADD,
            InferenceDomain::PROMPT,
            NO_LAYER,
            NO_TENSOR,
            u32::MAX as u64,
            1,
            0,
            [
                9,
                2,
                4_294_967_295,
                0,
                4_294_967_295,
                4_294_967_295,
                1,
                34_359_738_368,
            ],
        ),
        (
            "mul",
            InferenceFieldKind::MUL,
            InferenceDomain::KV,
            3,
            7,
            512,
            256,
            1,
            [9, 9, 3, 0, 7, 512, 256, 38_654_705_665],
        ),
        (
            "usize",
            InferenceFieldKind::USIZE,
            InferenceDomain::LOGITS,
            NO_LAYER,
            290,
            123,
            usize::MAX as u64,
            2,
            [
                9,
                18,
                4_294_967_295,
                0,
                290,
                123,
                18_446_744_073_709_551_615,
                42_949_672_962,
            ],
        ),
    ];
    assert_eq!(arithmetic_cases.len(), 3);
    for (name, kind, domain, layer, tensor, needed, available, sub, expected) in arithmetic_cases {
        logits.fill(0xfeed_face);
        crate::inference::set_inference_arithmetic_fault_for_test(
            kind, domain, layer, tensor, needed, available, sub,
        );
        let fault = engine
            .prefill(&[0], 1, &mut logits)
            .err()
            .unwrap_or_else(|| panic!("{name} arithmetic injection must fail"));
        assert_eq!(inference_error_tuple(fault), expected, "{name}");
        let fault_usage = engine.usage();
        assert_eq!(fault_usage.state, 3, "{name} state");
        assert_eq!(
            fault_usage.scratch.high_water, 213_568,
            "{name} touched scratch"
        );
        assert!(logits.iter().all(|word| *word == 0), "{name} logits");
        assert!(engine.scratch_is_zero_for_test(), "{name} scratch");
        engine.reset().unwrap();
        let reset_usage = engine.usage();
        assert_eq!(reset_usage.state, 0, "{name} reset state");
        assert_eq!(
            reset_usage.scratch.high_water,
            fault_usage.scratch.high_water
        );
        assert_eq!(reset_usage.kv.high_water, fault_usage.kv.high_water);
        assert!(engine.scratch_is_zero_for_test(), "{name} reset scratch");
        assert!(canaries_intact(), "{name} canaries");
    }
    assert_eq!(engine.usage().state, InferenceState::RESET as u32);

    let kv_words = |engine: &InferenceEngine<'_, '_, '_, '_>, kind: usize, position: usize| {
        let mut words = Vec::with_capacity(128);
        for head in 0..2 {
            for component in 0..64 {
                words.push(engine.logical_kv_word_for_test(0, kind, position, head, component));
            }
        }
        words
    };
    logits.fill(0xfeed_face);
    engine
        .prefill_with_prefix(&[0, 1], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);
    let key_before = kv_words(&engine, 0, 1);
    let value_before = kv_words(&engine, 1, 1);
    crate::inference::set_kv_write_fault_for_test(0);
    let key_fault = engine
        .prefill_with_prefix(&[0, 2], 1, 32, &mut logits)
        .unwrap_err();
    assert_eq!(key_fault.domain, InferenceDomain::KV as u32);
    assert_eq!(key_fault.position, 1);
    assert_ne!(kv_words(&engine, 0, 1), key_before);
    assert_eq!(kv_words(&engine, 1, 1), value_before);
    assert_eq!(engine.usage().state, InferenceState::FAULTED as u32);
    assert_eq!(engine.retained_prefix_for_test(), None);
    assert!(logits.iter().all(|word| *word == 0));
    assert!(engine.scratch_is_zero_for_test());
    engine.reset().unwrap();
    assert_eq!(engine.position(), 0);

    let key_before = kv_words(&engine, 0, 0);
    let value_before = kv_words(&engine, 1, 0);
    logits.fill(0xfeed_face);
    crate::inference::set_kv_write_fault_for_test(1);
    let value_fault = engine.prefill(&[2], 32, &mut logits).unwrap_err();
    assert_eq!(value_fault.domain, InferenceDomain::KV as u32);
    assert_eq!(value_fault.position, 0);
    assert_ne!(kv_words(&engine, 0, 0), key_before);
    assert_ne!(kv_words(&engine, 1, 0), value_before);
    assert_eq!(engine.usage().state, InferenceState::FAULTED as u32);
    assert_eq!(engine.retained_prefix_for_test(), None);
    assert!(logits.iter().all(|word| *word == 0));
    assert!(engine.scratch_is_zero_for_test());
    engine.reset().unwrap();
    assert_eq!(engine.position(), 0);
    assert!(canaries_intact());

    logits.fill(0xfeed_face);
    let successful = engine.prefill(&[0], 32, &mut logits).unwrap();
    assert_eq!(successful.position, 1);
    let ready_usage = engine.usage();
    assert_eq!(ready_usage.position, 1);
    assert_eq!(ready_usage.generation_reserve, 32);
    assert_eq!(ready_usage.kv.current, 24_576);
    assert_eq!(ready_usage.scratch.current, 0);
    assert_eq!(ready_usage.scratch.high_water, 213_568);
    assert!(engine.scratch_is_zero_for_test());
    if ready_usage.state == InferenceState::EOS as u32 {
        engine.force_decode_state_for_test(1, 1, 1, successful.selected_token);
    }
    assert_eq!(
        inference_error_tuple(precheck_error!(
            logits,
            engine.prefill(&[0], 1, &mut logits)
        )),
        [1, 2, 4_294_967_295, 1, 4_294_967_295, 0, 1, 4_294_967_297]
    );
    engine.reset().unwrap();
    let ready_reset = engine.usage();
    assert_eq!(ready_reset.state, 0);
    assert_eq!(ready_reset.position, 0);
    assert_eq!(ready_reset.kv.current, 0);
    assert_eq!(ready_reset.kv.high_water, ready_usage.kv.high_water);
    assert_eq!(
        ready_reset.scratch.high_water,
        ready_usage.scratch.high_water
    );
    assert!(engine.scratch_is_zero_for_test());

    engine.force_decode_state_for_test(5, 2, 1, 42);
    let long_decode = precheck_error!(long_logits, engine.decode(42, &mut long_logits));
    assert_eq!(
        inference_error_tuple(long_decode),
        [
            4,
            18,
            4_294_967_295,
            5,
            4_294_967_295,
            151_936,
            151_937,
            51_539_607_552
        ]
    );
    let invalid = precheck_error!(logits, engine.decode(LOGIT_WORDS as u32, &mut logits));
    assert_eq!(
        inference_error_tuple(invalid),
        [
            3,
            2,
            nl,
            5,
            nt,
            LOGIT_WORDS as u64,
            LOGIT_WORDS as u64,
            k(InferenceFieldKind::TOKEN, 1),
        ]
    );
    let wrong = precheck_error!(logits, engine.decode(41, &mut logits));
    assert_eq!(
        inference_error_tuple(wrong),
        [1, 19, nl, 5, nt, 42, 41, k(InferenceFieldKind::SELECTED, 0)]
    );
    engine.force_decode_state_for_test(5, 3, 1, 42);
    let selected = engine.decode_selected(41, &mut logits).unwrap();
    assert_eq!(selected.position, 6);
    engine.force_state_for_test(InferenceState::EOS, 6);
    let selected = engine.decode_selected(41, &mut logits).unwrap();
    assert_eq!(selected.position, 7);
    engine.force_decode_state_for_test(100, 1, 1, 0);
    let exhausted = precheck_error!(logits, engine.decode(0, &mut logits));
    assert_eq!(
        inference_error_tuple(exhausted),
        [2, 2, nl, 100, nt, 1, 1, k(InferenceFieldKind::RESERVE, 1)]
    );
    engine.force_decode_state_for_test(CONTEXT_LIMIT, 2, 1, 0);
    let context = precheck_error!(logits, engine.decode(0, &mut logits));
    assert_eq!(
        inference_error_tuple(context),
        [
            2,
            9,
            nl,
            CONTEXT_LIMIT as u64,
            nt,
            CONTEXT_LIMIT as u64,
            CONTEXT_LIMIT as u64,
            k(InferenceFieldKind::CONTEXT, 2)
        ]
    );
    engine.force_decode_state_for_test(100, 2, 1, 0);
    let short = precheck_error!(logits, engine.decode(0, &mut logits[..LOGIT_WORDS - 1]));
    assert_eq!(
        inference_error_tuple(short),
        [
            4,
            18,
            nl,
            100,
            nt,
            LOGIT_WORDS as u64,
            (LOGIT_WORDS - 1) as u64,
            k(InferenceFieldKind::LOGITS, 0),
        ]
    );
    engine.force_state_for_test(InferenceState::EOS, 5);
    assert_eq!(
        inference_error_tuple(precheck_error!(logits, engine.decode(0, &mut logits))),
        [1, 2, 4_294_967_295, 5, 4_294_967_295, 1, 2, 4_294_967_298]
    );
    let eos_high_water = engine.usage();
    engine.reset().unwrap();
    assert_eq!(engine.usage().kv.high_water, eos_high_water.kv.high_water);
    assert_eq!(
        engine.usage().scratch.high_water,
        eos_high_water.scratch.high_water
    );
    assert!(engine.scratch_is_zero_for_test());
    let reset_high_water = engine.usage();
    engine.reset().unwrap();
    assert_eq!(engine.usage().kv.high_water, reset_high_water.kv.high_water);
    assert_eq!(
        engine.usage().scratch.high_water,
        reset_high_water.scratch.high_water
    );

    let full = engine.prefill(&[42, 43], 32, &mut logits).unwrap();
    let full_logits = crate::sha256::digest(word_bytes(&logits));
    let full_arenas = engine.arena_digests_for_test();
    engine.reset().unwrap();
    engine.prefill(&[42], 32, &mut logits).unwrap();
    let appended = engine.append_prefill(&[43], 32, &mut logits).unwrap();
    assert_eq!(appended, full);
    assert_eq!(crate::sha256::digest(word_bytes(&logits)), full_logits);
    assert_eq!(engine.arena_digests_for_test(), full_arenas);

    engine.reset().unwrap();
    let retained = engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(retained, full);
    assert_eq!(engine.retained_prefix_for_test(), Some(1));
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);
    assert_eq!(engine.position(), 1);
    let reused = engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(reused, full);
    assert_eq!(crate::sha256::digest(word_bytes(&logits)), full_logits);
    assert_eq!(engine.arena_digests_for_test(), full_arenas);
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);

    let invalid_boundary = engine
        .prefill_with_prefix(&[42, 43], 0, 32, &mut logits)
        .unwrap_err();
    assert_eq!(invalid_boundary.status, InferenceStatus::CONTEXT as u32);
    assert_eq!(engine.position(), 0);
    assert_eq!(engine.retained_prefix_for_test(), None);

    crate::inference::set_inference_fault_for_test(
        InferenceDomain::EMBEDDING,
        NO_LAYER,
        0,
        f32::NAN.to_bits(),
    );
    assert!(engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .is_err());
    assert_eq!(engine.usage().state, InferenceState::FAULTED as u32);
    assert_eq!(engine.retained_prefix_for_test(), None);
    assert_eq!(engine.reset_to_prefix().unwrap(), 0);
    assert_eq!(engine.position(), 0);

    engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .unwrap();
    let non_reset_mismatch = engine
        .prefill_with_prefix(&[44, 43], 1, 32, &mut logits)
        .unwrap_err();
    assert_eq!(non_reset_mismatch.status, InferenceStatus::STATE as u32);
    assert_eq!(engine.usage().state, InferenceState::RESET as u32);
    assert_eq!(engine.position(), 0);
    assert_eq!(engine.retained_prefix_for_test(), None);

    engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);
    crate::inference::set_inference_fault_for_test(
        InferenceDomain::EMBEDDING,
        NO_LAYER,
        0,
        f32::NAN.to_bits(),
    );
    assert!(engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .is_err());
    assert_eq!(engine.usage().state, InferenceState::FAULTED as u32);
    assert_eq!(engine.retained_prefix_for_test(), None);
    assert_eq!(engine.reset_to_prefix().unwrap(), 0);
    assert_eq!(engine.position(), 0);

    let divergent = engine.prefill(&[44, 43], 32, &mut logits).unwrap();
    let divergent_logits = crate::sha256::digest(word_bytes(&logits));
    engine.reset().unwrap();
    engine
        .prefill_with_prefix(&[42, 43], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(engine.reset_to_prefix().unwrap(), 1);
    let divergent_after_prefix = engine
        .prefill_with_prefix(&[44, 43], 1, 32, &mut logits)
        .unwrap();
    assert_eq!(divergent_after_prefix, divergent);
    assert_eq!(crate::sha256::digest(word_bytes(&logits)), divergent_logits);
    assert_eq!(engine.retained_prefix_for_test(), Some(1));
    engine.force_state_for_test(InferenceState::RESET, 0);
    assert_eq!(engine.reset_to_prefix().unwrap(), 0);
    assert_eq!(engine.retained_prefix_for_test(), None);

    engine.reset().unwrap();
    assert_eq!(engine.kv_byte_for_test(KV_BYTES - 1), 0xa5);
    drop(engine);
    assert_eq!(kv_storage[kv_start + KV_BYTES - 1], 0xa5);
    assert!(
        scratch_storage[scratch_start..scratch_start + SCRATCH_BYTES]
            .iter()
            .all(|byte| *byte == 0)
    );
    assert!(kv_storage[..kv_start].iter().all(|byte| *byte == 0xa5));
    assert!(kv_storage[kv_start + KV_BYTES..]
        .iter()
        .all(|byte| *byte == 0xa5));
    assert!(scratch_storage[..scratch_start]
        .iter()
        .all(|byte| *byte == 0x5a));
    assert!(scratch_storage[scratch_start + SCRATCH_BYTES..]
        .iter()
        .all(|byte| *byte == 0x5a));
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn inference_dirty_and_stale_kv_match_fresh_paths() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();

    let mut dirty_kv = vec![0xa5u8; KV_BYTES + 63];
    let dirty_start = (64 - (dirty_kv.as_ptr() as usize & 63)) & 63;
    let mut fresh_kv = vec![0u8; KV_BYTES + 63];
    let fresh_start = (64 - (fresh_kv.as_ptr() as usize & 63)) & 63;
    let mut dirty_scratch = vec![0x5au8; SCRATCH_BYTES + 63];
    let dirty_scratch_start = (64 - (dirty_scratch.as_ptr() as usize & 63)) & 63;
    let mut fresh_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let fresh_scratch_start = (64 - (fresh_scratch.as_ptr() as usize & 63)) & 63;
    let mut dirty = InferenceEngine::build(
        &model,
        &mut dirty_kv[dirty_start..dirty_start + KV_BYTES],
        &mut dirty_scratch[dirty_scratch_start..dirty_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let mut fresh = InferenceEngine::build(
        &model,
        &mut fresh_kv[fresh_start..fresh_start + KV_BYTES],
        &mut fresh_scratch[fresh_scratch_start..fresh_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    dirty.force_backend_for_test(crate::fp32_sse2::InferenceBackend::Sse2);
    if crate::fp32_sse2::InferenceBackend::detect() == crate::fp32_sse2::InferenceBackend::Avx2 {
        fresh.force_backend_for_test(crate::fp32_sse2::InferenceBackend::Avx2);
        assert_eq!(
            (dirty.backend_name(), fresh.backend_name()),
            ("sse2", "avx2")
        );
    }
    assert!(dirty.kv_is_filled_for_test(0xa5));
    assert!(fresh.kv_is_filled_for_test(0));
    assert!(dirty.scratch_is_zero_for_test());
    assert!(fresh.scratch_is_zero_for_test());

    let mut dirty_logits = vec![0u32; LOGIT_WORDS];
    let mut fresh_logits = vec![0u32; LOGIT_WORDS];
    dirty.reset_q8_preparations_for_test();
    dirty.prefill(&[42], 8, &mut dirty_logits).unwrap();
    let full_logit_preparations = dirty.q8_preparations_for_test();
    dirty.reset().unwrap();
    dirty.reset_q8_preparations_for_test();
    dirty.prefill(&[42, 43], 8, &mut dirty_logits).unwrap();
    let non_logit_preparations = dirty.q8_preparations_for_test() - full_logit_preparations;
    assert_eq!((full_logit_preparations, non_logit_preparations), (97, 96));
    dirty.reset().unwrap();

    dirty.prefill(&[42], 8, &mut dirty_logits).unwrap();
    fresh.prefill(&[42], 8, &mut fresh_logits).unwrap();
    assert_eq!(dirty_logits, fresh_logits, "dirty build first prefill");
    dirty.decode_selected(43, &mut dirty_logits).unwrap();
    fresh.decode_selected(43, &mut fresh_logits).unwrap();
    assert_eq!(dirty_logits, fresh_logits, "dirty build decode");

    dirty.reset().unwrap();
    dirty.reset().unwrap();
    fresh.reset().unwrap();
    dirty.prefill(&[42, 43, 44], 8, &mut dirty_logits).unwrap();
    fresh.prefill(&[42, 43, 44], 8, &mut fresh_logits).unwrap();
    assert_eq!(
        dirty_logits, fresh_logits,
        "multi-token prefill after reset"
    );

    dirty.reset().unwrap();
    fresh.reset().unwrap();
    dirty.prefill(&[42], 8, &mut dirty_logits).unwrap();
    dirty
        .append_prefill(&[43, 44], 8, &mut dirty_logits)
        .unwrap();
    fresh.prefill(&[42, 43, 44], 8, &mut fresh_logits).unwrap();
    assert_eq!(dirty_logits, fresh_logits, "cached append versus fresh");

    dirty.reset().unwrap();
    fresh.reset().unwrap();
    dirty.prefill(&[1, 2, 3], 8, &mut dirty_logits).unwrap();
    fresh.prefill(&[4, 5], 8, &mut fresh_logits).unwrap();
    dirty.reset().unwrap();
    dirty.reset().unwrap();
    fresh.reset().unwrap();
    dirty.prefill(&[42, 43], 8, &mut dirty_logits).unwrap();
    fresh.prefill(&[42, 43], 8, &mut fresh_logits).unwrap();
    assert_eq!(
        dirty_logits, fresh_logits,
        "divergent stale sessions after reset"
    );
    assert_eq!(dirty.kv_byte_for_test(KV_BYTES - 1), 0xa5);
    assert!(dirty.scratch_is_zero_for_test());
    assert!(fresh.scratch_is_zero_for_test());
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn inference_dispatch_preflight_failures_make_zero_worker_calls() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicU32::new(0));
    let mut capacity_context =
        test_dispatcher(Arc::clone(&calls), Arc::clone(&failure), 3, 0);
    let mut tiny_kv = [0u8; 1];
    let mut tiny_scratch = [0u8; 1];
    let capacity = InferenceEngine::build(
        &model,
        &mut tiny_kv,
        &mut tiny_scratch,
        Some(unsafe {
            InferenceDispatcher::new(&mut capacity_context, dispatch_scoped_workers)
        }),
    )
    .err()
    .expect("capacity preflight");
    assert_eq!(capacity.status, InferenceStatus::CAPACITY as u32);
    assert_eq!(calls.load(Ordering::Acquire), 0);

    let mut kv = vec![0xa5u8; KV_BYTES + 64];
    let kv_start = (64 - (kv.as_ptr() as usize & 63)) & 63;
    let mut scratch = vec![0x5au8; SCRATCH_BYTES + 64];
    let scratch_start = (64 - (scratch.as_ptr() as usize & 63)) & 63;
    let mut alignment_context =
        test_dispatcher(Arc::clone(&calls), Arc::clone(&failure), 3, 0);
    let alignment = InferenceEngine::build(
        &model,
        &mut kv[kv_start + 1..kv_start + 1 + KV_BYTES],
        &mut scratch[scratch_start..scratch_start + SCRATCH_BYTES],
        Some(unsafe {
            InferenceDispatcher::new(&mut alignment_context, dispatch_scoped_workers)
        }),
    )
    .err()
    .expect("alignment preflight");
    assert_eq!(alignment.status, InferenceStatus::ALIGNMENT as u32);
    assert_eq!(calls.load(Ordering::Acquire), 0);

    let mut context = test_dispatcher(Arc::clone(&calls), Arc::clone(&failure), 3, 0);
    let dispatcher = unsafe { InferenceDispatcher::new(&mut context, dispatch_scoped_workers) };
    let mut engine = InferenceEngine::build(
        &model,
        &mut kv[kv_start..kv_start + KV_BYTES],
        &mut scratch[scratch_start..scratch_start + SCRATCH_BYTES],
        Some(dispatcher),
    )
    .unwrap();
    let mut logits = vec![0u32; LOGIT_WORDS];
    let mut short_logits = [0u32; 1];

    assert_eq!(
        engine.decode_selected(42, &mut logits).unwrap_err().status,
        InferenceStatus::STATE as u32
    );
    assert_eq!(
        engine.prefill(&[], 1, &mut logits).unwrap_err().status,
        InferenceStatus::CONTEXT as u32
    );
    assert_eq!(
        engine.prefill(&[42], 0, &mut logits).unwrap_err().status,
        InferenceStatus::CONTEXT as u32
    );
    assert_eq!(
        engine
            .prefill(&[model.config().vocab_count], 1, &mut logits)
            .unwrap_err()
            .status,
        InferenceStatus::TOKEN_ID as u32
    );
    assert_eq!(
        engine
            .prefill(&[42], 1, &mut short_logits)
            .unwrap_err()
            .status,
        InferenceStatus::CAPACITY as u32
    );
    assert_eq!(calls.load(Ordering::Acquire), 0);

    engine.prefill(&[42], 1, &mut logits).unwrap();
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.prefill(&[42], 1, &mut logits).unwrap_err().status,
        InferenceStatus::STATE as u32
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(failure.load(Ordering::Acquire), 0);
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn inference_dispatch_cooperatively_recovers_first_middle_and_last_faults() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();
    let faults = [
        (InferenceDomain::EMBEDDING, NO_LAYER),
        (InferenceDomain::UP, 12),
        (InferenceDomain::LOGITS, NO_LAYER),
    ];

    let mut kv = vec![0xa5u8; KV_BYTES + 63];
    let kv_start = (64 - (kv.as_ptr() as usize & 63)) & 63;
    let mut scratch = vec![0x5au8; SCRATCH_BYTES + 63];
    let scratch_start = (64 - (scratch.as_ptr() as usize & 63)) & 63;
    let mut serial_errors = Vec::new();
    {
        let mut serial = InferenceEngine::build(
            &model,
            &mut kv[kv_start..kv_start + KV_BYTES],
            &mut scratch[scratch_start..scratch_start + SCRATCH_BYTES],
            None,
        )
        .unwrap();
        let mut logits = vec![0xfeed_face; LOGIT_WORDS];
        for (domain, layer) in faults {
            crate::inference::set_inference_fault_for_test(
                domain,
                layer,
                0,
                f32::NAN.to_bits(),
            );
            let error = serial.prefill(&[42], 1, &mut logits).unwrap_err();
            serial_errors.push(inference_error_tuple(error));
            assert_eq!(serial.usage().state, InferenceState::FAULTED as u32);
            assert!(serial.scratch_is_zero_for_test());
            assert!(logits.iter().all(|word| *word == 0));
            assert_eq!(serial.retained_prefix_for_test(), None);
            serial.reset().unwrap();
            assert_eq!(serial.position(), 0);
            assert_eq!(serial.usage().state, InferenceState::RESET as u32);
            logits.fill(0xfeed_face);
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicU32::new(0));
    let mut context = test_dispatcher(Arc::clone(&calls), Arc::clone(&failure), 3, 0);
    let dispatcher = unsafe { InferenceDispatcher::new(&mut context, dispatch_scoped_workers) };
    let mut parallel = InferenceEngine::build(
        &model,
        &mut kv[kv_start..kv_start + KV_BYTES],
        &mut scratch[scratch_start..scratch_start + SCRATCH_BYTES],
        Some(dispatcher),
    )
    .unwrap();
    let mut logits = vec![0xfeed_face; LOGIT_WORDS];
    for ((domain, layer), expected) in faults.into_iter().zip(serial_errors) {
        crate::inference::set_inference_fault_for_test(domain, layer, 0, f32::NAN.to_bits());
        let error = parallel.prefill(&[42], 1, &mut logits).unwrap_err();
        assert_eq!(inference_error_tuple(error), expected);
        assert_eq!(parallel.usage().state, InferenceState::FAULTED as u32);
        assert!(parallel.scratch_is_zero_for_test());
        assert!(logits.iter().all(|word| *word == 0));
        assert_eq!(parallel.retained_prefix_for_test(), None);
        parallel.reset().unwrap();
        assert_eq!(parallel.position(), 0);
        assert_eq!(parallel.usage().state, InferenceState::RESET as u32);
        logits.fill(0xfeed_face);
    }
    assert_eq!(calls.load(Ordering::Acquire), 3);
    assert_eq!(failure.load(Ordering::Acquire), 0);
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn inference_multiworker_schedule_matches_serial_sessions_and_boundaries() {
    fn assert_exact(
        serial: &InferenceEngine<'_, '_, '_, '_>,
        parallel: &InferenceEngine<'_, '_, '_, '_>,
        serial_logits: &[u32],
        parallel_logits: &[u32],
    ) {
        assert_eq!(serial_logits, parallel_logits);
        assert_eq!(serial.position(), parallel.position());
        assert_eq!(serial.usage(), parallel.usage());
        let position = serial.position() as usize;
        for layer in 0..crate::inference::LAYERS {
            for kind in 0..2 {
                for at in 0..position {
                    for head in 0..2 {
                        for component in 0..64 {
                            assert_eq!(
                                serial.logical_kv_word_for_test(
                                    layer, kind, at, head, component,
                                ),
                                parallel.logical_kv_word_for_test(
                                    layer, kind, at, head, component,
                                ),
                                "position={position} layer={layer} kind={kind} at={at} head={head} component={component}"
                            );
                        }
                    }
                }
            }
        }
    }

    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();

    let mut serial_kv = vec![0xa5u8; KV_BYTES + 63];
    let serial_kv_start = (64 - (serial_kv.as_ptr() as usize & 63)) & 63;
    let mut parallel_kv = vec![0x5au8; KV_BYTES + 63];
    let parallel_kv_start = (64 - (parallel_kv.as_ptr() as usize & 63)) & 63;
    let mut serial_scratch = vec![0xa5u8; SCRATCH_BYTES + 63];
    let serial_scratch_start = (64 - (serial_scratch.as_ptr() as usize & 63)) & 63;
    let mut parallel_scratch = vec![0x5au8; SCRATCH_BYTES + 63];
    let parallel_scratch_start = (64 - (parallel_scratch.as_ptr() as usize & 63)) & 63;
    let calls = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicU32::new(0));
    let mut context = test_dispatcher(Arc::clone(&calls), Arc::clone(&failure), 4, 0);
    let dispatcher = unsafe { InferenceDispatcher::new(&mut context, dispatch_scoped_workers) };
    let mut serial = InferenceEngine::build(
        &model,
        &mut serial_kv[serial_kv_start..serial_kv_start + KV_BYTES],
        &mut serial_scratch[serial_scratch_start..serial_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let mut parallel = InferenceEngine::build(
        &model,
        &mut parallel_kv[parallel_kv_start..parallel_kv_start + KV_BYTES],
        &mut parallel_scratch[parallel_scratch_start..parallel_scratch_start + SCRATCH_BYTES],
        Some(dispatcher),
    )
    .unwrap();
    serial.force_backend_for_test(crate::fp32_sse2::InferenceBackend::Sse2);
    let mut serial_logits = vec![0u32; LOGIT_WORDS];
    let mut parallel_logits = vec![0u32; LOGIT_WORDS];

    assert_eq!(
        serial.prefill(&[198], 1024, &mut serial_logits).unwrap(),
        parallel
            .prefill(&[198], 1024, &mut parallel_logits)
            .unwrap()
    );
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    for target in [3u32, 4, 5, 255, 256, 511, 512] {
        while serial.position() < target {
            assert_eq!(
                serial
                    .decode_selected(198, &mut serial_logits)
                    .unwrap(),
                parallel
                    .decode_selected(198, &mut parallel_logits)
                    .unwrap()
            );
        }
        assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    }

    serial.reset().unwrap();
    parallel.reset().unwrap();
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    assert_eq!(
        serial
            .prefill_with_prefix(&[42, 43], 1, 32, &mut serial_logits)
            .unwrap(),
        parallel
            .prefill_with_prefix(&[42, 43], 1, 32, &mut parallel_logits)
            .unwrap()
    );
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    assert_eq!(serial.reset_to_prefix().unwrap(), 1);
    assert_eq!(parallel.reset_to_prefix().unwrap(), 1);
    assert_eq!(
        serial
            .prefill_with_prefix(&[42, 44], 1, 32, &mut serial_logits)
            .unwrap(),
        parallel
            .prefill_with_prefix(&[42, 44], 1, 32, &mut parallel_logits)
            .unwrap()
    );
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    assert_eq!(
        serial
            .append_prefill(&[45, 46], 32, &mut serial_logits)
            .unwrap(),
        parallel
            .append_prefill(&[45, 46], 32, &mut parallel_logits)
            .unwrap()
    );
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);

    serial.reset().unwrap();
    parallel.reset().unwrap();
    assert_eq!(
        serial.prefill(&[99, 100], 32, &mut serial_logits).unwrap(),
        parallel
            .prefill(&[99, 100], 32, &mut parallel_logits)
            .unwrap()
    );
    assert_exact(&serial, &parallel, &serial_logits, &parallel_logits);
    assert_eq!(failure.load(Ordering::Acquire), 0);
    assert_eq!(calls.load(Ordering::Acquire), 519);

    drop(serial);
    drop(parallel);
    assert!(serial_kv[..serial_kv_start].iter().all(|byte| *byte == 0xa5));
    assert!(serial_kv[serial_kv_start + KV_BYTES..]
        .iter()
        .all(|byte| *byte == 0xa5));
    assert!(parallel_kv[..parallel_kv_start]
        .iter()
        .all(|byte| *byte == 0x5a));
    assert!(parallel_kv[parallel_kv_start + KV_BYTES..]
        .iter()
        .all(|byte| *byte == 0x5a));
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn real_model_attention_backends_match_all_boundary_state() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();
    assert!(inference_avx2_available());

    let mut sse2_kv = vec![0xa5u8; KV_BYTES + 63];
    let sse2_kv_start = (64 - (sse2_kv.as_ptr() as usize & 63)) & 63;
    let mut avx2_kv = vec![0x5au8; KV_BYTES + 63];
    let avx2_kv_start = (64 - (avx2_kv.as_ptr() as usize & 63)) & 63;
    let mut sse2_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let sse2_scratch_start = (64 - (sse2_scratch.as_ptr() as usize & 63)) & 63;
    let mut avx2_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let avx2_scratch_start = (64 - (avx2_scratch.as_ptr() as usize & 63)) & 63;
    let mut sse2 = InferenceEngine::build(
        &model,
        &mut sse2_kv[sse2_kv_start..sse2_kv_start + KV_BYTES],
        &mut sse2_scratch[sse2_scratch_start..sse2_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let mut avx2 = InferenceEngine::build(
        &model,
        &mut avx2_kv[avx2_kv_start..avx2_kv_start + KV_BYTES],
        &mut avx2_scratch[avx2_scratch_start..avx2_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    sse2.force_backend_for_test(crate::fp32_sse2::InferenceBackend::Sse2);
    avx2.force_backend_for_test(crate::fp32_sse2::InferenceBackend::Avx2);
    let mut sse2_logits = vec![0u32; LOGIT_WORDS];
    let mut avx2_logits = vec![0u32; LOGIT_WORDS];
    sse2.prefill(&[198], 512, &mut sse2_logits).unwrap();
    avx2.prefill(&[198], 512, &mut avx2_logits).unwrap();

    for target in [1usize, 3, 4, 5, 255, 256, 511, 512] {
        while sse2.position() < target as u32 {
            sse2.decode_selected(198, &mut sse2_logits).unwrap();
            avx2.decode_selected(198, &mut avx2_logits).unwrap();
        }
        assert_eq!(sse2.position(), target as u32);
        assert_eq!(avx2.position(), target as u32);
        assert_eq!(sse2_logits, avx2_logits, "N={target} logits");
        for layer in 0..crate::inference::LAYERS {
            for kind in 0..2 {
                for position in 0..target {
                    for head in 0..2 {
                        for component in 0..64 {
                            assert_eq!(
                                sse2.logical_kv_word_for_test(
                                    layer, kind, position, head, component,
                                ),
                                avx2.logical_kv_word_for_test(
                                    layer, kind, position, head, component,
                                ),
                                "N={target} layer={layer} kind={kind} position={position} head={head} component={component}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn real_model_near_limit_reset_reuses_exact_prefix_and_faults_closed() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 63];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();
    let config = model.config();

    let mut index_storage = vec![0u8; INDEX_BYTES + 63];
    let index_start = (64 - (index_storage.as_ptr() as usize & 63)) & 63;
    let tokenizer = FrozenTokenizer::build(
        &model,
        &mut index_storage[index_start..index_start + INDEX_BYTES],
    )
    .unwrap();
    let mut rendered = vec![0u8; 660];
    let mut staging = vec![0u32; 599];
    let mut working = vec![0u32; CONTEXT_LIMIT as usize];
    let mut tokenizer_scratch = vec![0u8; 5_120 + 63];
    let tokenizer_scratch_start = (64 - (tokenizer_scratch.as_ptr() as usize & 63)) & 63;
    let tokenizer_scratch =
        &mut tokenizer_scratch[tokenizer_scratch_start..tokenizer_scratch_start + 5_120];

    let hello = tokenizer
        .render_and_tokenize(b"Hello", &mut rendered, &mut staging, tokenizer_scratch)
        .unwrap();
    let hello_tokens = staging[..hello.token_count as usize].to_vec();

    let mut near_history = vec![198u32; PROMPT_LIMIT];
    *near_history.last_mut().unwrap() = crate::tokenizer::IM_END;
    tokenizer
        .validate_conversation_history(&near_history)
        .unwrap();
    working.fill(0xfeed_face);
    let mut overflow_outcome = ConversationUsage::ZERO;
    let overflow = tokenizer
        .render_conversation_and_tokenize(
            &near_history,
            b"Name one color.",
            &mut rendered,
            &mut staging,
            &mut working,
            tokenizer_scratch,
            &mut overflow_outcome,
        )
        .unwrap_err();
    assert_eq!(overflow.status, ModelStatus::OUTPUT_CAPACITY as u32);
    assert_eq!(
        decide_history(true, overflow_outcome.prompt_tokens as usize, None),
        HistoryDecision::RetryFresh
    );
    assert!(working.iter().all(|token| *token == 0xfeed_face));

    let mut fresh_outcome = ConversationUsage::ZERO;
    let fresh = tokenizer
        .render_conversation_and_tokenize(
            &[],
            b"Name one color.",
            &mut rendered,
            &mut staging,
            &mut working,
            tokenizer_scratch,
            &mut fresh_outcome,
        )
        .unwrap();
    assert_eq!(
        decide_history(
            true,
            overflow_outcome.prompt_tokens as usize,
            Some(fresh.prompt_tokens as usize),
        ),
        HistoryDecision::Reset {
            prompt_tokens: fresh.prompt_tokens as usize,
        }
    );
    assert_eq!(fresh.prefix_tokens, hello.prefix_tokens);
    let fresh_tokens = working[..fresh.prompt_tokens as usize].to_vec();

    let mut retained_kv = vec![0xa5u8; KV_BYTES + 63];
    let retained_kv_start = (64 - (retained_kv.as_ptr() as usize & 63)) & 63;
    let mut retained_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let retained_scratch_start = (64 - (retained_scratch.as_ptr() as usize & 63)) & 63;
    let mut retained = InferenceEngine::build(
        &model,
        &mut retained_kv[retained_kv_start..retained_kv_start + KV_BYTES],
        &mut retained_scratch[retained_scratch_start..retained_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let mut retained_logits = vec![0u32; LOGIT_WORDS];
    retained
        .prefill_with_prefix(&hello_tokens, hello.prefix_tokens, 1, &mut retained_logits)
        .unwrap();

    let mut committed = near_history;
    let mut cached_len = PROMPT_LIMIT - 1;
    let mut committed_len = PROMPT_LIMIT;
    let mut history_turns = 7;
    let retained_position = reset_conversation_for_fresh_prompt(
        &mut retained,
        &mut committed,
        &mut working,
        fresh.prompt_tokens as usize,
        &mut cached_len,
        &mut committed_len,
        &mut history_turns,
    )
    .unwrap();
    assert_eq!(retained_position, fresh.prefix_tokens);
    assert_eq!(retained.position(), fresh.prefix_tokens);
    assert_eq!((cached_len, committed_len, history_turns), (0, 0, 0));
    assert!(committed.iter().all(|token| *token == 0));
    assert_eq!(&working[..fresh_tokens.len()], fresh_tokens);
    assert!(working[fresh_tokens.len()..]
        .iter()
        .all(|token| *token == 0));
    retained
        .prefill_with_prefix(&fresh_tokens, fresh.prefix_tokens, 1, &mut retained_logits)
        .unwrap();
    let retained_position = retained.position() as usize;
    drop(retained);
    let retained_kv_words = used_kv_words(
        &retained_kv[retained_kv_start..retained_kv_start + KV_BYTES],
        retained_position,
        config,
    );
    drop(retained_kv);
    drop(retained_scratch);

    let mut fresh_kv = vec![0u8; KV_BYTES + 63];
    let fresh_kv_start = (64 - (fresh_kv.as_ptr() as usize & 63)) & 63;
    let mut fresh_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let fresh_scratch_start = (64 - (fresh_scratch.as_ptr() as usize & 63)) & 63;
    let mut fresh_engine = InferenceEngine::build(
        &model,
        &mut fresh_kv[fresh_kv_start..fresh_kv_start + KV_BYTES],
        &mut fresh_scratch[fresh_scratch_start..fresh_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    let mut fresh_logits = vec![0u32; LOGIT_WORDS];
    fresh_engine
        .prefill(&fresh_tokens, 1, &mut fresh_logits)
        .unwrap();
    let fresh_position = fresh_engine.position() as usize;
    drop(fresh_engine);
    let fresh_kv_words = used_kv_words(
        &fresh_kv[fresh_kv_start..fresh_kv_start + KV_BYTES],
        fresh_position,
        config,
    );
    assert_eq!(retained_position, fresh_position);
    assert_eq!(retained_logits, fresh_logits);
    assert_eq!(retained_kv_words, fresh_kv_words);
    drop(fresh_kv);
    drop(fresh_scratch);

    working[..fresh_tokens.len()].copy_from_slice(&fresh_tokens);
    working[fresh_tokens.len()..].fill(0xfeed_face);
    committed.fill(0xfeed_face);
    *committed.last_mut().unwrap() = crate::tokenizer::IM_END;
    cached_len = PROMPT_LIMIT - 1;
    committed_len = PROMPT_LIMIT;
    history_turns = 9;
    let mut fault_kv = vec![0xa5u8; KV_BYTES + 63];
    let fault_kv_start = (64 - (fault_kv.as_ptr() as usize & 63)) & 63;
    let mut fault_scratch = vec![0u8; SCRATCH_BYTES + 63];
    let fault_scratch_start = (64 - (fault_scratch.as_ptr() as usize & 63)) & 63;
    let mut fault_engine = InferenceEngine::build(
        &model,
        &mut fault_kv[fault_kv_start..fault_kv_start + KV_BYTES],
        &mut fault_scratch[fault_scratch_start..fault_scratch_start + SCRATCH_BYTES],
        None,
    )
    .unwrap();
    fault_engine
        .prefill_with_prefix(&hello_tokens, hello.prefix_tokens, 1, &mut retained_logits)
        .unwrap();
    assert_eq!(
        reset_conversation_for_fresh_prompt(
            &mut fault_engine,
            &mut committed,
            &mut working,
            fresh_tokens.len(),
            &mut cached_len,
            &mut committed_len,
            &mut history_turns,
        )
        .unwrap(),
        fresh.prefix_tokens
    );
    crate::inference::set_inference_fault_for_test(
        InferenceDomain::EMBEDDING,
        NO_LAYER,
        0,
        f32::NAN.to_bits(),
    );
    assert!(fault_engine
        .prefill_with_prefix(&fresh_tokens, fresh.prefix_tokens, 1, &mut retained_logits,)
        .is_err());
    assert_eq!(fault_engine.usage().state, InferenceState::FAULTED as u32);
    committed.fill(0xfeed_face);
    working[fresh_tokens.len()..].fill(0xfeed_face);
    cached_len = fresh_tokens.len() - 1;
    committed_len = fresh_tokens.len();
    history_turns = 1;
    assert_eq!(
        reset_conversation_for_fresh_prompt(
            &mut fault_engine,
            &mut committed,
            &mut working,
            fresh_tokens.len(),
            &mut cached_len,
            &mut committed_len,
            &mut history_turns,
        )
        .unwrap(),
        0
    );
    assert_eq!(fault_engine.position(), 0);
    assert_eq!(fault_engine.usage().state, InferenceState::RESET as u32);
    assert_eq!((cached_len, committed_len, history_turns), (0, 0, 0));
    assert!(committed.iter().all(|token| *token == 0));
    assert!(working[fresh_tokens.len()..]
        .iter()
        .all(|token| *token == 0));
    fault_engine
        .prefill_with_prefix(&fresh_tokens, fresh.prefix_tokens, 1, &mut retained_logits)
        .unwrap();
    let recovered_position = fault_engine.position() as usize;
    drop(fault_engine);
    let recovered_kv_words = used_kv_words(
        &fault_kv[fault_kv_start..fault_kv_start + KV_BYTES],
        recovered_position,
        config,
    );
    assert_eq!(recovered_position, fresh_position);
    assert_eq!(retained_logits, fresh_logits);
    assert_eq!(recovered_kv_words, fresh_kv_words);
}

#[test]
fn model_runtime_abi_and_sha256_vectors_are_exact() {
    assert_eq!(size_of::<ModelError>(), 48);
    assert_eq!(align_of::<ModelError>(), 8);
    assert_eq!(offset_of!(ModelError, status), 0);
    assert_eq!(offset_of!(ModelError, domain), 4);
    assert_eq!(offset_of!(ModelError, index), 8);
    assert_eq!(offset_of!(ModelError, reserved), 12);
    assert_eq!(offset_of!(ModelError, offset), 16);
    assert_eq!(offset_of!(ModelError, needed), 24);
    assert_eq!(offset_of!(ModelError, available), 32);
    assert_eq!(offset_of!(ModelError, detail), 40);
    assert_eq!(size_of::<ModelConfig>(), 80);
    assert_eq!(offset_of!(ModelConfig, model_bytes), 64);
    assert_eq!(offset_of!(ModelConfig, tensor_data_bytes), 72);
    assert_eq!(size_of::<TensorMeta>(), 64);
    assert_eq!(offset_of!(TensorMeta, dims), 16);
    assert_eq!(offset_of!(TensorMeta, data_offset), 32);
    assert_eq!(offset_of!(TensorMeta, reserved), 56);
    assert_eq!(size_of::<TokenizerUsage>(), 40);
    assert_eq!(size_of::<PromptUsage>(), 20);
    assert_eq!(size_of::<ConversationUsage>(), 24);
    assert_eq!(size_of::<PieceUsage>(), 8);

    let vector = |input: &[u8], expected: &str| {
        let expected: [u8; 32] = hex(expected).try_into().unwrap();
        assert_eq!(crate::sha256::digest(input), expected);
    };
    vector(
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    vector(
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    vector(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
    let mut chunked = crate::sha256::Sha256::new();
    for byte in b"abc" {
        chunked.update(core::slice::from_ref(byte));
    }
    let chunked_expected: [u8; 32] =
        hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            .try_into()
            .unwrap();
    assert_eq!(chunked.finish(), chunked_expected);
    let million_a = vec![b'a'; 1_000_000];
    vector(
        &million_a,
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
    );
}

#[test]
fn sha256_boundaries_partitions_and_backends_are_exact() {
    const CASES: &[(usize, &str)] = &[
        (
            55,
            "2900465fcb533e05a158fd2b3be0e5e3b03740d83060aa3580e0d98a96bf2384",
        ),
        (
            56,
            "31454ff48ef36af2f08fd511bdc37d9d5855ac23e992e5ff5445cb6b7674a674",
        ),
        (
            63,
            "5f6401b96532c36de4e65beec0409b69b1d181864c8009b7a04f43e5d56350d1",
        ),
        (
            64,
            "94eb5de4943613fd048dc93393ab06877405faa39c11f53e9386083339833e7e",
        ),
        (
            65,
            "fc518669b6eb4b4dd91827ecacef86689c725bd5bab888fd3b26dbb196eec954",
        ),
        (
            119,
            "b0dc41b1a384e2f1203f0351b38fbeaafceef577ce1191d5bfc25da39f721eae",
        ),
        (
            120,
            "5df24dd802ac26132ce608dcb5f09841eef039ee0f152acf98d26d17fe4e88e6",
        ),
        (
            127,
            "0fe729ff19257bd6fec853acc2ea355f6b34b58e6c0f684c3e188fcdfcd9baae",
        ),
        (
            128,
            "0aedd4856f8eba0963627336ad5144a9a7dbe12498e6066f0165fc97d8ddee4c",
        ),
    ];

    let partitioned = |state: &mut crate::sha256::Sha256, input: &[u8]| {
        const WIDTHS: [usize; 8] = [0, 1, 7, 3, 16, 5, 31, 2];
        state.update(&[]);
        let mut offset = 0;
        let mut partition = 0;
        while offset < input.len() {
            let width = WIDTHS[partition % WIDTHS.len()];
            partition += 1;
            if width == 0 {
                state.update(&[]);
                continue;
            }
            let end = core::cmp::min(offset + width, input.len());
            state.update(&input[offset..end]);
            offset = end;
        }
        state.update(&[]);
    };

    for &(length, expected_hex) in CASES {
        let input: Vec<u8> = (0..length)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let expected: [u8; 32] = hex(expected_hex).try_into().unwrap();
        assert_eq!(crate::sha256::digest(&input), expected, "length {length}");

        let mut scalar = crate::sha256::Sha256::new();
        scalar.force_scalar_for_test();
        partitioned(&mut scalar, &input);
        assert_eq!(scalar.finish(), expected, "scalar length {length}");

        let mut sha_ni = crate::sha256::Sha256::new();
        if sha_ni.force_sha_ni_for_test() {
            partitioned(&mut sha_ni, &input);
            assert_eq!(sha_ni.finish(), expected, "SHA-NI length {length}");
        }
    }
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn sha256_packed_model_backends_are_exact() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let model = fs::read(path).expect("read packed model");
    let expected: [u8; 32] =
        hex("b0f98ed6e0557ca35e1bced1000c950b3c84414251df65290315a7969981d42d")
            .try_into()
            .unwrap();

    let mut scalar = crate::sha256::Sha256::new();
    scalar.force_scalar_for_test();
    scalar.update(&model);
    assert_eq!(scalar.finish(), expected, "packed model scalar");

    let mut sha_ni = crate::sha256::Sha256::new();
    if sha_ni.force_sha_ni_for_test() {
        sha_ni.update(&model);
        assert_eq!(sha_ni.finish(), expected, "packed model SHA-NI");
    }
}

#[test]
#[ignore = "requires PROMPTBOOT_TEST_MODEL; run make validate-host"]
fn real_model_index_identity_failures_roll_back_zero_prefix() {
    let path =
        env::var("PROMPTBOOT_TEST_MODEL").expect("PROMPTBOOT_TEST_MODEL; run make validate-host");
    let file = fs::read(path).unwrap();
    let mut model_storage = vec![0u8; file.len() + 64];
    let model_start = (64 - (model_storage.as_ptr() as usize & 63)) & 63;
    model_storage[model_start..model_start + file.len()].copy_from_slice(&file);
    let model =
        ModelView::open_authenticated(&model_storage[model_start..model_start + file.len()])
            .unwrap();

    let mut short_index = vec![0u8; INDEX_BYTES - 1 + 63];
    let short_start = (64 - (short_index.as_ptr() as usize & 63)) & 63;
    let short_before = short_index.clone();
    let error = FrozenTokenizer::build(
        &model,
        &mut short_index[short_start..short_start + INDEX_BYTES - 1],
    )
    .err()
    .expect("short index must reject");
    assert_eq!(error.status, ModelStatus::INDEX_CAPACITY as u32);
    assert_eq!(short_index, short_before);

    let mut misaligned_index = vec![0u8; INDEX_BYTES + 64];
    let aligned_start = (64 - (misaligned_index.as_ptr() as usize & 63)) & 63;
    let misaligned_before = misaligned_index.clone();
    let error = FrozenTokenizer::build(
        &model,
        &mut misaligned_index[aligned_start + 1..aligned_start + 1 + INDEX_BYTES],
    )
    .err()
    .expect("misaligned index must reject");
    assert_eq!(error.status, ModelStatus::INDEX_ALIGNMENT as u32);
    assert_eq!(misaligned_index, misaligned_before);

    let mut dirty_index = vec![0u8; INDEX_BYTES + 63];
    let dirty_start = (64 - (dirty_index.as_ptr() as usize & 63)) & 63;
    dirty_index[dirty_start + 123] = 7;
    let dirty_before = dirty_index.clone();
    let error = FrozenTokenizer::build(
        &model,
        &mut dirty_index[dirty_start..dirty_start + INDEX_BYTES],
    )
    .err()
    .expect("dirty index must reject");
    assert_eq!(error.status, ModelStatus::STATE as u32);
    assert_eq!(dirty_index, dirty_before);

    for fault in 1..=5 {
        let mut storage = vec![0xa5u8; INDEX_BYTES + 191];
        let start = 64 + ((64 - ((storage.as_ptr() as usize + 64) & 63)) & 63);
        storage[start..start + INDEX_BYTES].fill(0);
        crate::tokenizer::set_index_identity_fault_for_test(fault);
        let error = FrozenTokenizer::build(&model, &mut storage[start..start + INDEX_BYTES])
            .err()
            .expect("fault must reject");
        let actual = model_error_tuple(error);
        assert_eq!(
            actual,
            expected_index_identity_fault(fault),
            "fault {fault}"
        );
        assert!(storage[start..start + INDEX_BYTES]
            .iter()
            .all(|byte| *byte == 0));
        assert!(storage[..start].iter().all(|byte| *byte == 0xa5));
        assert!(storage[start + INDEX_BYTES..]
            .iter()
            .all(|byte| *byte == 0xa5));
    }
    crate::tokenizer::set_index_identity_fault_for_test(0);

    let mut index = vec![0u8; INDEX_BYTES + 63];
    let index_start = (64 - (index.as_ptr() as usize & 63)) & 63;
    let tokenizer =
        FrozenTokenizer::build(&model, &mut index[index_start..index_start + INDEX_BYTES]).unwrap();

    let mut piece = [0xa5u8; 128];
    let piece_usage = tokenizer.decode_piece(0, &mut piece).unwrap();
    assert!(piece_usage.bytes > 0);
    assert!(piece[piece_usage.bytes as usize..]
        .iter()
        .all(|byte| *byte == 0xa5));

    for token in [151_643, 151_645] {
        piece.fill(0xa5);
        let usage = tokenizer.decode_piece(token, &mut piece).unwrap();
        assert_eq!(usage.kind, PieceKind::EOS as u32);
        assert_eq!(usage.bytes, 0);
        assert!(piece.iter().all(|byte| *byte == 0xa5));
    }

    let mut short_piece = [0xa5u8; 127];
    let error = tokenizer
        .decode_piece(0, &mut short_piece[..piece_usage.bytes as usize - 1])
        .unwrap_err();
    assert_eq!(error.status, ModelStatus::OUTPUT_CAPACITY as u32);
    assert!(short_piece.iter().all(|byte| *byte == 0xa5));

    piece.fill(0xa5);
    let error = tokenizer.decode_piece(151_936, &mut piece).unwrap_err();
    assert_eq!(error.status, ModelStatus::TOKEN_ID as u32);
    assert!(piece.iter().all(|byte| *byte == 0xa5));

    let mut rendered = [0xa5u8; 660];
    let mut tokens = [0xfeed_faceu32; 599];
    let mut scratch = Aligned([0x3cu8; 5_120]);

    let error = tokenizer
        .render_and_tokenize(b"", &mut rendered[..147], &mut tokens, &mut scratch.0)
        .unwrap_err();
    assert_eq!(error.status, ModelStatus::OUTPUT_CAPACITY as u32);
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert!(scratch.0.iter().all(|byte| *byte == 0x3c));

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_and_tokenize(b"", &mut rendered, &mut tokens[..598], &mut scratch.0)
        .unwrap_err();
    assert_eq!(error.status, ModelStatus::OUTPUT_CAPACITY as u32);
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert!(scratch.0.iter().all(|byte| *byte == 0x3c));

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_and_tokenize(b"", &mut rendered, &mut tokens, &mut scratch.0[..5_119])
        .unwrap_err();
    assert_eq!(error.status, ModelStatus::SCRATCH_CAPACITY as u32);
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert!(scratch.0.iter().all(|byte| *byte == 0x3c));

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    let mut misaligned_scratch = vec![0x3cu8; 5_184];
    let scratch_start = (64 - (misaligned_scratch.as_ptr() as usize & 63)) & 63;
    let scratch_before = misaligned_scratch.clone();
    let error = tokenizer
        .render_and_tokenize(
            b"",
            &mut rendered,
            &mut tokens,
            &mut misaligned_scratch[scratch_start + 1..scratch_start + 1 + 5_120],
        )
        .unwrap_err();
    assert_eq!(error.status, ModelStatus::ALIGNMENT as u32);
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert_eq!(misaligned_scratch, scratch_before);

    scratch.0.fill(0x3c);
    crate::tokenizer::set_prompt_after_write_fault_for_test(true);
    let error = tokenizer
        .render_and_tokenize(b"Hello", &mut rendered, &mut tokens, &mut scratch.0)
        .unwrap_err();
    crate::tokenizer::set_prompt_after_write_fault_for_test(false);
    assert_eq!(error.status, ModelStatus::STATE as u32);
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert!(scratch.0.iter().all(|byte| *byte == 0));

    rendered.fill(0);
    tokens.fill(0);
    let mut conversation = vec![0xfeed_faceu32; CONTEXT_LIMIT as usize];
    let mut conversation_outcome = ConversationUsage::ZERO;
    let fresh = tokenizer
        .render_conversation_and_tokenize(
            &[],
            b"Hello",
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap();
    assert_eq!(fresh.prompt_tokens, 30);
    assert_eq!(fresh.fresh_prompt_tokens, 30);
    assert_eq!(fresh.user_tokens, 1);
    assert_eq!(fresh.prefix_tokens, 24);
    let mut history = [0u32; 32];
    history[..30].copy_from_slice(&conversation[..30]);
    history[30] = 9707;
    history[31] = 151_645;
    conversation.fill(0xfeed_face);
    let second = tokenizer
        .render_conversation_and_tokenize(
            &history,
            b"Color?",
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap();
    assert_eq!(second.history_tokens, 32);
    assert_eq!(second.prompt_tokens, 43);
    assert_eq!(&conversation[..32], &history);
    assert_eq!(conversation[32], 198);
    let before = conversation.clone();
    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &[1, 2, 3],
            b"x",
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [27, 10, 2, 0, 151_645, 3, 167_503_724_555]
    );
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert_eq!(conversation, before);
    assert!(scratch.0.iter().all(|byte| *byte == 0x3c));

    let oversized_history = vec![151_645u32; CONTEXT_LIMIT as usize + 1];
    let oversized_user = [b'x'; 513];
    let mut short_staging = [0xfeed_faceu32; 598];
    let mut short_output = vec![0xfeed_faceu32; CONTEXT_LIMIT as usize - 1];

    let check_unchanged = |rendered: &[u8], staging: &[u32], output: &[u32], scratch: &[u8]| {
        assert!(rendered.iter().all(|byte| *byte == 0xa5));
        assert!(staging.iter().all(|word| *word == 0xfeed_face));
        assert!(output.iter().all(|word| *word == 0xfeed_face));
        assert!(scratch.iter().all(|byte| *byte == 0x3c));
    };

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    conversation.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &oversized_history,
            b"x",
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [
            1,
            10,
            0,
            0,
            CONTEXT_LIMIT as u64,
            CONTEXT_LIMIT as u64 + 1,
            163_208_757_258,
        ]
    );
    check_unchanged(&rendered, &tokens, &conversation, &scratch.0);

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    conversation.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &[],
            &oversized_user,
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [1, 8, 0, 0, 512, 513, 163_208_757_248]
    );
    check_unchanged(&rendered, &tokens, &conversation, &scratch.0);

    rendered.fill(0xa5);
    short_staging.fill(0xfeed_face);
    conversation.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &[],
            b"x",
            &mut rendered,
            &mut short_staging,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [22, 10, 0, 0, 599, 598, 163_208_757_260]
    );
    check_unchanged(&rendered, &short_staging, &conversation, &scratch.0);

    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    short_output.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &[],
            b"x",
            &mut rendered,
            &mut tokens,
            &mut short_output,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [
            22,
            10,
            0,
            0,
            CONTEXT_LIMIT as u64,
            CONTEXT_LIMIT as u64 - 1,
            163_208_757_261,
        ]
    );
    check_unchanged(&rendered, &tokens, &short_output, &scratch.0);

    let valid_full_history = vec![151_645u32; CONTEXT_LIMIT as usize];
    rendered.fill(0xa5);
    tokens.fill(0xfeed_face);
    conversation.fill(0xfeed_face);
    scratch.0.fill(0x3c);
    let error = tokenizer
        .render_conversation_and_tokenize(
            &valid_full_history,
            b"x",
            &mut rendered,
            &mut tokens,
            &mut conversation,
            &mut scratch.0,
            &mut conversation_outcome,
        )
        .unwrap_err();
    assert_eq!(
        model_error_tuple(error),
        [
            22,
            10,
            0,
            0,
            CONTEXT_LIMIT as u64,
            CONTEXT_LIMIT as u64 + 10,
            163_208_757_264,
        ]
    );
    assert!(rendered.iter().all(|byte| *byte == 0xa5));
    assert!(tokens.iter().all(|word| *word == 0xfeed_face));
    assert!(conversation.iter().all(|word| *word == 0xfeed_face));
    assert!(scratch.0.iter().all(|byte| *byte == 0));
    assert_eq!(conversation_outcome.user_tokens, 1);
    assert_eq!(conversation_outcome.fresh_prompt_tokens, 30);
    assert_eq!(conversation_outcome.history_tokens, CONTEXT_LIMIT);
    assert_eq!(
        conversation_outcome.prompt_tokens,
        CONTEXT_LIMIT.saturating_add(10)
    );
    assert_eq!(conversation_outcome.prefix_tokens, 24);

    drop(tokenizer);
    drop(model);
    let bytes = &mut model_storage[model_start..model_start + file.len()];
    let mut mutations: Vec<(&str, usize, Vec<u8>)> = vec![
        ("header_magic", 0, vec![b'X']),
        ("header_version", 8, 2u32.to_le_bytes().to_vec()),
        ("header_endian", 16, 0u32.to_le_bytes().to_vec()),
        ("header_identity", 40, 25u32.to_le_bytes().to_vec()),
        ("header_reserved", 192, vec![1]),
        ("section_order", 320, 1u32.to_le_bytes().to_vec()),
        ("section_alignment", 336, 608_513u64.to_le_bytes().to_vec()),
        ("section_overlap", 336, 608_448u64.to_le_bytes().to_vec()),
        ("section_gap", 608_452, vec![1]),
        (
            "section_trailing_extent",
            664,
            422_782_463u64.to_le_bytes().to_vec(),
        ),
        ("tensor_id", 3_952_512, 1u32.to_le_bytes().to_vec()),
        ("tensor_role", 3_952_518, 99u16.to_le_bytes().to_vec()),
        ("tensor_dtype", 3_952_520, 3u32.to_le_bytes().to_vec()),
        ("tensor_rank", 3_952_524, 4u32.to_le_bytes().to_vec()),
        (
            "tensor_dimension",
            3_952_528,
            u32::MAX.to_le_bytes().to_vec(),
        ),
        (
            "tensor_block_length",
            3_952_552,
            76_575_745u64.to_le_bytes().to_vec(),
        ),
        (
            "tensor_offset",
            3_952_544,
            3_980_544u64.to_le_bytes().to_vec(),
        ),
        (
            "tensor_overlap",
            3_952_640,
            80_556_160u64.to_le_bytes().to_vec(),
        ),
        ("tensor_reserved", 3_952_592, vec![1]),
        ("token_offset_start", 704, 1u32.to_le_bytes().to_vec()),
        ("token_offset_monotonic", 708, 0u32.to_le_bytes().to_vec()),
        (
            "token_offset_range",
            708,
            1_372_759u32.to_le_bytes().to_vec(),
        ),
        ("token_type", 1_981_312, vec![2]),
        ("token_utf8", 608_512, vec![0xff]),
        ("merge_id", 2_133_248, 151_936u32.to_le_bytes().to_vec()),
        ("merge_concat", 2_133_256, 270u32.to_le_bytes().to_vec()),
        ("chat_template", 3_949_952, vec![b'z']),
    ];
    mutations.push(("merge_duplicate", 2_133_288, 220u32.to_le_bytes().to_vec()));
    for (name, offset, replacement) in mutations {
        let original = bytes[offset..offset + replacement.len()].to_vec();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        let error = ModelView::open_authenticated(bytes)
            .err()
            .unwrap_or_else(|| panic!("{name} unexpectedly accepted"));
        let actual = [
            error.status as u64,
            error.domain as u64,
            error.index as u64,
            error.offset,
            error.needed,
            error.available,
            error.detail,
        ];
        assert_eq!(actual, expected_named_model_error(name), "{name}");
        bytes[offset..offset + replacement.len()].copy_from_slice(&original);
    }

    let short_error = ModelView::open_authenticated(&bytes[..bytes.len() - 1])
        .err()
        .expect("short model must fail");
    let short_tuple = [
        short_error.status as u64,
        short_error.domain as u64,
        short_error.index as u64,
        short_error.offset,
        short_error.needed,
        short_error.available,
        short_error.detail,
    ];
    assert_eq!(
        short_tuple,
        [1, 1, 0, 0, 426_762_944, 426_762_943, 4_294_967_296]
    );
    let mut actual_hash = [0u8; 32];
    actual_hash[0] = 53;
    let mut expected_hash = actual_hash;
    expected_hash[0] = 52;
    let full_hash_error = ModelError::full_hash_mismatch(&expected_hash, &actual_hash)
        .expect("different hashes must fail");
    let full_hash_tuple = model_error_tuple(full_hash_error);
    assert_eq!(full_hash_tuple, [2, 1, 0, 0, 52, 53, 12_884_901_888]);
    let long = &model_storage[model_start..model_start + file.len() + 1];
    let long_error = ModelView::open_authenticated(long)
        .err()
        .expect("long model must fail");
    let long_tuple = [
        long_error.status as u64,
        long_error.domain as u64,
        long_error.index as u64,
        long_error.offset,
        long_error.needed,
        long_error.available,
        long_error.detail,
    ];
    assert_eq!(
        long_tuple,
        [1, 1, 0, 0, 426_762_944, 426_762_945, 4_294_967_296]
    );
    let misaligned = &model_storage[model_start + 1..model_start + 1 + file.len()];
    let misaligned_error = ModelView::open_authenticated(misaligned)
        .err()
        .expect("misaligned model must fail");
    let misaligned_tuple = model_error_tuple(misaligned_error);
    assert_eq!(misaligned_tuple, [9, 1, 0, 0, 64, 1, 8_589_934_592]);
}

#[test]
fn analytic_fixture_all_nine_operations() {
    let (actual, _) = invoke(
        PrimitiveOp::BIAS_RESIDUAL,
        &words(&[
            1.25, -2.0, 0.5, 16.0, 0.25, 0.75, -1.5, -0.125, -0.5, 1.0, 2.0, -15.0,
        ]),
        &[],
        4,
        [4, 0, 0, 0],
        0,
    );
    assert_eq!(actual, expected_words(0, 4));

    let vector = words(&[
        -1.125, -0.25, 0.625, -0.875, 0.0, 0.875, -0.625, 0.25, 1.125, -0.375, 0.5, -1.0, -0.125,
        0.75, -0.75, 0.125, 1.0, -0.5, 0.375, -1.125, -0.25, 0.625, -0.875, 0.0, 0.875, -0.625,
        0.25, 1.125, -0.375, 0.5, -1.0, -0.125,
    ]);
    let (actual, _) = invoke(
        PrimitiveOp::Q4,
        &hex("00383388dd2277cc1166bb0055aaff4499ee00b4ffeeddccbbaa99887766554433221100"),
        &vector,
        67,
        [2, 32, 0, 0],
        0,
    );
    assert_eq!(actual, expected_words(16, 67));
    let (actual, _) = invoke(
        PrimitiveOp::Q8,
        &hex("0030e8f3fe09141febf6010c17e3eef9040f1ae6f1fc07121de9f4ff0a15e1ecf70200ac1f160d04fbf2e91f160d04fbf2e91f160d04fbf2e91f160d04fbf2e91f160d04"),
        &vector, 67, [2, 32, 0, 0], 0,
    );
    assert_eq!(actual, expected_words(284, 67));

    let (actual, _) = invoke(
        PrimitiveOp::RMSNORM,
        &words(&[1.0, -2.0, 3.0, -4.0]),
        &words(&[0.5, 1.5, -0.75, 2.0]),
        4,
        [4, 0, 0, 0],
        0,
    );
    assert_close(&actual, &expected_words(552, 4));
    let (actual, _) = invoke(
        PrimitiveOp::ROPE,
        &words(&[1.0, 2.0, -3.0, 4.0]),
        &[],
        4,
        [4, 1, 0, 0],
        7,
    );
    assert_close(&actual, &expected_words(568, 4));
    let (actual, _) = invoke(
        PrimitiveOp::SOFTMAX,
        &words(&[1.0, -2.0, 3.0, 3.0]),
        &[],
        4,
        [4, 0, 0, 0],
        0,
    );
    assert_close(&actual, &expected_words(584, 4));

    let attention = words(&[
        1.0, 0.5, -0.25, 2.0, 0.5, -1.0, 1.5, 0.25, 2.0, -0.5, 0.25, 3.0,
    ]);
    let (actual, _) = invoke(
        PrimitiveOp::GQA_ATTENTION,
        &attention,
        &[],
        12,
        [2, 1, 2, 2],
        0,
    );
    assert_close(&actual, &expected_words(600, 12));
    let (actual, _) = invoke(
        PrimitiveOp::SILU_SWIGLU,
        &words(&[-3.0, -0.5, 0.0, 2.0]),
        &words(&[0.25, -2.0, 4.0, 1.5]),
        8,
        [4, 0, 0, 0],
        0,
    );
    assert_close(&actual, &expected_words(648, 8));
    let (actual, _) = invoke(
        PrimitiveOp::ARGMAX,
        &words(&[1.0, 4.0, 4.0, -2.0, 4.0]),
        &[],
        1,
        [5, 0, 0, 0],
        0,
    );
    assert_eq!(actual, [1]);
}

#[test]
fn expf_edge_oracle_is_within_one_ulp() {
    for (index, pair) in EXP_EDGE.chunks_exact(8).enumerate() {
        let input = u32::from_le_bytes(pair[..4].try_into().unwrap());
        let expected = u32::from_le_bytes(pair[4..].try_into().unwrap());
        let actual = super::fp32_sse2::expf_bits_for_test(input);
        if expected & 0x7fff_ffff == 0 || expected & 0x7f80_0000 == 0x7f80_0000 {
            assert_eq!(actual, expected, "edge {index}");
        } else {
            assert!(
                actual.abs_diff(expected) <= 1,
                "edge {index}: actual={actual:08x} expected={expected:08x}"
            );
        }
    }
}

#[test]
fn arena_lifecycle_alignment_zeroing_and_views_are_transactional() {
    for alignment in [1, 2, 4, 8, 16, 32, 64] {
        let mut storage = Aligned([0x5a; 192]);
        let mut arena = Arena::new(&mut storage.0).unwrap();
        let region = arena.allocate(1, alignment).unwrap();
        assert_eq!(region.offset % u64::from(alignment), 0);
    }
    let mut storage = Aligned([0x5a; 192]);
    let mut arena = Arena::new(&mut storage.0).unwrap();
    let first = arena.allocate(1, 1).unwrap();
    let second = arena.allocate(1, 64).unwrap();
    assert_eq!(
        first,
        Region {
            offset: 0,
            length: 1
        }
    );
    assert_eq!(
        second,
        Region {
            offset: 64,
            length: 1
        }
    );
    assert_eq!(
        arena.usage(),
        ArenaUsage {
            capacity: 192,
            requested: 2,
            committed: 65,
            current: 2,
            high_water: 2
        }
    );
    assert_eq!(arena.reset().unwrap_err().status, PrimitiveStatus::STATE);
    arena.seal().unwrap();
    assert_eq!(arena.seal().unwrap_err().status, PrimitiveStatus::STATE);
    assert_eq!(
        arena.allocate(1, 1).unwrap_err().status,
        PrimitiveStatus::ARENA_SEALED
    );
    arena
        .region_mut(Region {
            offset: 0,
            length: 65,
        })
        .unwrap()
        .fill(0xa5);
    assert_eq!(
        arena
            .region(Region {
                offset: 64,
                length: 1
            })
            .unwrap(),
        &[0xa5]
    );
    assert_eq!(
        arena
            .region(Region {
                offset: u64::MAX,
                length: 2
            })
            .unwrap_err()
            .status,
        PrimitiveStatus::ARITHMETIC_OVERFLOW
    );
    assert_eq!(
        arena
            .region(Region {
                offset: 65,
                length: 0
            })
            .unwrap_err()
            .status,
        PrimitiveStatus::LENGTH
    );
    arena.reset().unwrap();
    assert_eq!(
        arena.usage(),
        ArenaUsage {
            capacity: 192,
            requested: 0,
            committed: 0,
            current: 0,
            high_water: 2
        }
    );
    assert!(storage.0[..65].iter().all(|byte| *byte == 0));
}

#[test]
fn arena_rejects_empty_misaligned_short_and_invalid_requests_without_mutation() {
    let mut empty = [];
    assert_eq!(
        Arena::new(&mut empty).err().unwrap().status,
        PrimitiveStatus::ARENA_CAPACITY
    );
    let mut raw = Aligned([0x33; 192]);
    assert_eq!(
        Arena::new(&mut raw.0[1..]).err().unwrap().status,
        PrimitiveStatus::ALIGNMENT
    );
    let before = raw.0;
    let mut arena = Arena::new(&mut raw.0).unwrap();
    for (bytes, alignment, status) in [
        (0, 1, PrimitiveStatus::LENGTH),
        (1, 0, PrimitiveStatus::ALIGNMENT),
        (1, 3, PrimitiveStatus::ALIGNMENT),
        (1, 128, PrimitiveStatus::ALIGNMENT),
    ] {
        assert_eq!(arena.allocate(bytes, alignment).unwrap_err().status, status);
        assert_eq!(arena.usage().requested, 0);
    }
    assert_eq!(
        arena.allocate(193, 1).unwrap_err().status,
        PrimitiveStatus::ARENA_CAPACITY
    );
    assert_eq!(arena.usage().committed, 0);
    drop(arena);
    assert_eq!(raw.0, before);
}

fn base_argmax_request(
    input: &[u8],
    output: &mut [u32],
    scratch: &mut Aligned<64>,
) -> PrimitiveRequest {
    PrimitiveRequest {
        abi_version: 1,
        operation: PrimitiveOp::ARGMAX as u32,
        input: input.as_ptr(),
        input_bytes: input.len() as u64,
        aux: ptr::null(),
        aux_bytes: 0,
        output: output.as_mut_ptr(),
        output_capacity_words: output.len() as u64,
        scratch: scratch.0.as_mut_ptr(),
        scratch_bytes: scratch.0.len() as u64,
        dim0: (input.len() / 4) as u32,
        dim1: 0,
        dim2: 0,
        dim3: 0,
        position: 0,
        flags: 0,
    }
}

#[test]
fn runner_preserves_output_on_capacity_nonfinite_and_schema_failures() {
    let input = words(&[1.0, 2.0]);
    let mut output = [0xdead_beef];
    let mut scratch = Aligned([0x7b; 64]);
    let mut request = base_argmax_request(&input, &mut output, &mut scratch);
    let mut result = PrimitiveResult::failure_safe();
    request.scratch_bytes = 3;
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::ARENA_CAPACITY as u32
    );
    assert_eq!(result.needed_bytes, 4);
    assert_eq!(result.available_bytes, 3);
    assert_eq!(output, [0xdead_beef]);
    assert!(scratch.0.iter().all(|byte| *byte == 0x7b));

    let nonfinite = words(&[1.0, f32::NAN]);
    request = base_argmax_request(&nonfinite, &mut output, &mut scratch);
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::NONFINITE_INPUT as u32
    );
    assert_eq!(result.error_index, 1);
    assert_eq!(output, [0xdead_beef]);

    request = base_argmax_request(&input, &mut output, &mut scratch);
    request.dim1 = 1;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::DIMENSION as u32
    );
    assert_eq!(output, [0xdead_beef]);
}

#[test]
fn runner_enforces_no_write_alias_and_null_zero_rules() {
    let input = words(&[1.0]);
    let mut output = [0x1122_3344];
    let mut scratch = Aligned([0x77; 64]);
    let mut request = base_argmax_request(&input, &mut output, &mut scratch);
    let mut result = PrimitiveResult {
        status: 0xaaaa_aaaa,
        output_words: 0xbbbb_bbbb,
        error_operation: 0,
        error_arena: 0,
        error_index: 0,
        reserved: 0,
        needed_bytes: 0,
        available_bytes: 0,
        arena_capacity: 0,
        arena_requested: 0,
        arena_committed: 0,
        arena_current: 0,
        arena_high_water: 0,
    };
    let null_before = result;
    assert_eq!(
        unsafe { promptboot_run_primitive(ptr::null(), &mut result) },
        PrimitiveStatus::NULL as u32
    );
    assert_eq!(result, null_before);

    request.output = (&mut result as *mut PrimitiveResult).cast();
    request.output_capacity_words = 1;
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );
    assert_eq!(result.status, 0xaaaa_aaaa);
    assert_eq!(result.output_words, 0xbbbb_bbbb);

    request = base_argmax_request(&input, &mut output, &mut scratch);
    request.output = scratch.0.as_mut_ptr().cast();
    request.output_capacity_words = 1;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );
    assert_eq!(result.status, PrimitiveStatus::OVERLAP as u32);
    assert!(scratch.0.iter().all(|byte| *byte == 0x77));

    request = base_argmax_request(&input, &mut output, &mut scratch);
    request.output = ptr::null_mut();
    request.output_capacity_words = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::LENGTH as u32
    );
    assert_eq!((result.needed_bytes, result.available_bytes), (1, 0));

    request = base_argmax_request(&input, &mut output, &mut scratch);
    request.scratch = ptr::null_mut();
    request.scratch_bytes = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::ARENA_CAPACITY as u32
    );
    assert_eq!((result.needed_bytes, result.available_bytes), (4, 0));
}

#[test]
fn runner_preserves_scratch_red_zones_and_publishes_only_exact_output() {
    let input = words(&[1.0, 4.0, 4.0]);
    let mut output = [0xcccc_cccc; 3];
    let mut guarded = Aligned([0x6d; 192]);
    let request = PrimitiveRequest {
        abi_version: 1,
        operation: PrimitiveOp::ARGMAX as u32,
        input: input.as_ptr(),
        input_bytes: input.len() as u64,
        aux: ptr::null(),
        aux_bytes: 0,
        output: output.as_mut_ptr(),
        output_capacity_words: output.len() as u64,
        scratch: unsafe { guarded.0.as_mut_ptr().add(64) },
        scratch_bytes: 4,
        dim0: 3,
        dim1: 0,
        dim2: 0,
        dim3: 0,
        position: 0,
        flags: 0,
    };
    let mut result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::OK as u32
    );
    assert_eq!(output, [1, 0xcccc_cccc, 0xcccc_cccc]);
    assert!(guarded.0[..64].iter().all(|byte| *byte == 0x6d));
    assert!(guarded.0[68..].iter().all(|byte| *byte == 0x6d));
}

#[test]
fn unisolated_fixed_and_declared_range_failures_never_write_result() {
    let input = words(&[1.0]);
    let mut output = [0u32; 1];
    let mut scratch = Aligned([0u8; 64]);
    let request = base_argmax_request(&input, &mut output, &mut scratch);

    let mut result_bytes = Aligned([0x5cu8; 128]);
    let before = result_bytes.0;
    let misaligned_result = unsafe { result_bytes.0.as_mut_ptr().add(1).cast::<PrimitiveResult>() };
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, misaligned_result) },
        PrimitiveStatus::ALIGNMENT as u32
    );
    assert_eq!(result_bytes.0, before);

    let wrapped_result = (usize::MAX - 7) as *mut PrimitiveResult;
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, wrapped_result) },
        PrimitiveStatus::ARITHMETIC_OVERFLOW as u32
    );

    let mut result = PrimitiveResult {
        status: 0x1234_5678,
        output_words: 0x8765_4321,
        error_operation: 1,
        error_arena: 2,
        error_index: 3,
        reserved: 4,
        needed_bytes: 5,
        available_bytes: 6,
        arena_capacity: 7,
        arena_requested: 8,
        arena_committed: 9,
        arena_current: 10,
        arena_high_water: 11,
    };
    let sentinel = result;
    let wrapped_request = (usize::MAX - 7) as *const PrimitiveRequest;
    assert_eq!(
        unsafe { promptboot_run_primitive(wrapped_request, &mut result) },
        PrimitiveStatus::ARITHMETIC_OVERFLOW as u32
    );
    assert_eq!(result, sentinel);

    let mut request_bytes = Aligned([0u8; 128]);
    unsafe { ptr::write_unaligned(request_bytes.0.as_mut_ptr().add(1).cast(), request) };
    assert_eq!(
        unsafe {
            promptboot_run_primitive(
                request_bytes.0.as_ptr().add(1).cast::<PrimitiveRequest>(),
                &mut result,
            )
        },
        PrimitiveStatus::ALIGNMENT as u32
    );
    assert_eq!(result, sentinel);

    let mut overlap_bytes = Aligned([0x3au8; 128]);
    let overlap_before = overlap_bytes.0;
    let common = overlap_bytes.0.as_mut_ptr();
    assert_eq!(
        unsafe { promptboot_run_primitive(common.cast(), common.cast()) },
        PrimitiveStatus::OVERLAP as u32
    );
    assert_eq!(overlap_bytes.0, overlap_before);

    for field in [0, 1] {
        let mut local = request;
        if field == 0 {
            local.input = (&result as *const PrimitiveResult).cast();
            local.input_bytes = 4;
        } else {
            local.scratch = (&mut result as *mut PrimitiveResult).cast();
            local.scratch_bytes = 4;
        }
        result = sentinel;
        assert_eq!(
            unsafe { promptboot_run_primitive(&local, &mut result) },
            PrimitiveStatus::OVERLAP as u32
        );
        assert_eq!(result, sentinel);
    }

    let mut wrapped = request;
    wrapped.input = (usize::MAX - 3) as *const u8;
    wrapped.input_bytes = 8;
    result = sentinel;
    assert_eq!(
        unsafe { promptboot_run_primitive(&wrapped, &mut result) },
        PrimitiveStatus::ARITHMETIC_OVERFLOW as u32
    );
    assert_eq!(result, sentinel);
}

#[test]
fn runner_enforces_complete_mutable_alias_matrix_after_result_isolation() {
    let mut input = words(&[1.0]);
    let mut output = [0xaaaa_aaaa];
    let mut scratch = Aligned([0x44; 64]);
    let base = base_argmax_request(&input, &mut output, &mut scratch);

    let mut cases = Vec::new();
    let mut output_input = base;
    output_input.output = input.as_mut_ptr().cast();
    cases.push(output_input);

    let mut scratch_input = base;
    scratch_input.input = scratch.0.as_ptr();
    scratch_input.input_bytes = 4;
    cases.push(scratch_input);

    for request in &cases {
        let mut result = PrimitiveResult::failure_safe();
        assert_eq!(
            unsafe { promptboot_run_primitive(request, &mut result) },
            PrimitiveStatus::OVERLAP as u32
        );
        assert_eq!(result.status, PrimitiveStatus::OVERLAP as u32);
        assert_eq!(result.output_words, 0);
    }

    let mut output_request = base;
    output_request.output = (&output_request as *const PrimitiveRequest)
        .cast_mut()
        .cast();
    let mut result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&output_request, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );
    let mut scratch_request = base;
    scratch_request.scratch = (&scratch_request as *const PrimitiveRequest)
        .cast_mut()
        .cast();
    scratch_request.scratch_bytes = 64;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&scratch_request, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );

    let rms_input = words(&[1.0]);
    let mut rms_output = [0u32; 1];
    let mut rms_scratch = Aligned([0u8; 64]);
    let mut rms = PrimitiveRequest {
        abi_version: 1,
        operation: PrimitiveOp::RMSNORM as u32,
        input: rms_input.as_ptr(),
        input_bytes: 4,
        aux: rms_input.as_ptr(),
        aux_bytes: 4,
        output: rms_output.as_mut_ptr(),
        output_capacity_words: 1,
        scratch: rms_scratch.0.as_mut_ptr(),
        scratch_bytes: 64,
        dim0: 1,
        dim1: 0,
        dim2: 0,
        dim3: 0,
        position: 0,
        flags: 0,
    };
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&rms, &mut result) },
        PrimitiveStatus::OK as u32
    );

    rms.output = rms.aux.cast_mut().cast();
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&rms, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );
    rms.output = rms_output.as_mut_ptr();
    rms.scratch = rms.aux.cast_mut();
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&rms, &mut result) },
        PrimitiveStatus::OVERLAP as u32
    );
}

#[test]
fn runner_null_zero_alignment_overflow_and_operation_errors_are_exact() {
    let input = words(&[1.0]);
    let mut output = [0xface_cafe];
    let mut scratch = Aligned([0x55; 64]);
    let base = base_argmax_request(&input, &mut output, &mut scratch);

    let mut null_nonzero = base;
    null_nonzero.input = ptr::null();
    let mut result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&null_nonzero, &mut result) },
        PrimitiveStatus::NULL as u32
    );
    let mut nonnull_zero = base;
    nonnull_zero.input_bytes = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&nonnull_zero, &mut result) },
        PrimitiveStatus::LENGTH as u32
    );
    let mut null_output = base;
    null_output.output = ptr::null_mut();
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&null_output, &mut result) },
        PrimitiveStatus::NULL as u32
    );
    let mut zero_output = base;
    zero_output.output_capacity_words = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&zero_output, &mut result) },
        PrimitiveStatus::LENGTH as u32
    );
    let mut null_scratch = base;
    null_scratch.scratch = ptr::null_mut();
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&null_scratch, &mut result) },
        PrimitiveStatus::NULL as u32
    );
    let mut zero_scratch = base;
    zero_scratch.scratch_bytes = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&zero_scratch, &mut result) },
        PrimitiveStatus::LENGTH as u32
    );

    let mut misaligned_input_storage = Aligned([0u8; 8]);
    misaligned_input_storage.0[1..5].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
    let mut misaligned_input = base;
    misaligned_input.input = unsafe { misaligned_input_storage.0.as_ptr().add(1) };
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&misaligned_input, &mut result) },
        PrimitiveStatus::ALIGNMENT as u32
    );
    let mut misaligned_output_storage = Aligned([0u8; 16]);
    let mut misaligned_output = base;
    misaligned_output.output = unsafe { misaligned_output_storage.0.as_mut_ptr().add(1).cast() };
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&misaligned_output, &mut result) },
        PrimitiveStatus::ALIGNMENT as u32
    );
    let mut misaligned_scratch = base;
    misaligned_scratch.scratch = unsafe { scratch.0.as_mut_ptr().add(1) };
    misaligned_scratch.scratch_bytes = 63;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&misaligned_scratch, &mut result) },
        PrimitiveStatus::ALIGNMENT as u32
    );

    let mut overflow = base;
    overflow.operation = PrimitiveOp::Q4 as u32;
    overflow.dim0 = u32::MAX;
    overflow.dim1 = 32;
    overflow.input = ptr::null();
    overflow.input_bytes = 0;
    overflow.aux = ptr::null();
    overflow.aux_bytes = 0;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&overflow, &mut result) },
        PrimitiveStatus::ARITHMETIC_OVERFLOW as u32
    );

    let mut bad_abi = base;
    bad_abi.abi_version = 2;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&bad_abi, &mut result) },
        PrimitiveStatus::STATE as u32
    );
    let mut unsupported = base;
    unsupported.operation = 99;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&unsupported, &mut result) },
        PrimitiveStatus::UNSUPPORTED_OPERATION as u32
    );
    let mut bad_index = base;
    bad_index.operation = PrimitiveOp::ROPE as u32;
    bad_index.dim0 = 4;
    bad_index.dim1 = 1;
    bad_index.position = CONTEXT_LIMIT;
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&bad_index, &mut result) },
        PrimitiveStatus::INDEX as u32
    );
    assert_eq!(result.error_index, CONTEXT_LIMIT);
}

#[test]
fn runner_rejects_block_encoding_and_nonfinite_output_transactionally() {
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    let vector = words(&[1.0; 32]);
    let mut output = [0xaaaa_aaaa; 34];
    let mut scratch = Aligned([0x66u8; 192]);
    let request = PrimitiveRequest {
        abi_version: 1,
        operation: PrimitiveOp::Q4 as u32,
        input: block.as_ptr(),
        input_bytes: 18,
        aux: vector.as_ptr(),
        aux_bytes: 128,
        output: output.as_mut_ptr(),
        output_capacity_words: 34,
        scratch: scratch.0.as_mut_ptr(),
        scratch_bytes: 192,
        dim0: 1,
        dim1: 32,
        dim2: 0,
        dim3: 0,
        position: 0,
        flags: 0,
    };
    let mut result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&request, &mut result) },
        PrimitiveStatus::BLOCK_ENCODING as u32
    );
    assert_eq!(output, [0xaaaa_aaaa; 34]);

    let overflow_input = words(&[f32::MAX, f32::MAX, 0.0]);
    let mut scalar_output = [0x1234_5678];
    let mut scalar_scratch = Aligned([0x77; 64]);
    let overflow_request = PrimitiveRequest {
        abi_version: 1,
        operation: PrimitiveOp::BIAS_RESIDUAL as u32,
        input: overflow_input.as_ptr(),
        input_bytes: 12,
        aux: ptr::null(),
        aux_bytes: 0,
        output: scalar_output.as_mut_ptr(),
        output_capacity_words: 1,
        scratch: scalar_scratch.0.as_mut_ptr(),
        scratch_bytes: 64,
        dim0: 1,
        dim1: 0,
        dim2: 0,
        dim3: 0,
        position: 0,
        flags: 0,
    };
    result = PrimitiveResult::failure_safe();
    assert_eq!(
        unsafe { promptboot_run_primitive(&overflow_request, &mut result) },
        PrimitiveStatus::NONFINITE_OUTPUT as u32
    );
    assert_eq!(scalar_output, [0x1234_5678]);
}

#[test]
fn inference_q4_block_dot_matches_scalar_nibble_order_and_extrema() {
    let scalar_dot = |weights: &[u8; 16], activation: &[i8; 32]| {
        (0..16).fold(0i32, |total, index| {
            let packed = weights[index];
            total
                + (i32::from(packed & 0x0f) - 8) * i32::from(activation[index])
                + (i32::from(packed >> 4) - 8) * i32::from(activation[16 + index])
        })
    };

    let ordered_weights = core::array::from_fn(|index| (((15 - index) << 4) | index) as u8);
    let ordered_activation = core::array::from_fn(|index| {
        if index < 16 {
            (index + 1) as i8
        } else {
            -((index - 15) as i8)
        }
    });
    let ordered_expected = scalar_dot(&ordered_weights, &ordered_activation);
    let swapped_expected = (0..16).fold(0i32, |total, index| {
        let packed = ordered_weights[index];
        total
            + (i32::from(packed >> 4) - 8) * i32::from(ordered_activation[index])
            + (i32::from(packed & 0x0f) - 8) * i32::from(ordered_activation[16 + index])
    });
    assert_ne!(ordered_expected, swapped_expected, "ordering fixture");
    for backend in [
        crate::fp32_sse2::InferenceBackend::Sse2,
        crate::fp32_sse2::InferenceBackend::detect(),
    ] {
        assert_eq!(
            crate::fp32_sse2::inference_q4_block_dot_for_test(
                backend,
                &ordered_weights,
                &ordered_activation
            ),
            ordered_expected
        );
    }

    let extrema_weights = core::array::from_fn(|index| if index & 1 == 0 { 0x0f } else { 0xf0 });
    let extrema_activation = core::array::from_fn(|index| {
        if (index < 16) == (index & 1 == 0) {
            i8::MIN
        } else {
            i8::MAX
        }
    });
    for backend in [
        crate::fp32_sse2::InferenceBackend::Sse2,
        crate::fp32_sse2::InferenceBackend::detect(),
    ] {
        assert_eq!(
            crate::fp32_sse2::inference_q4_block_dot_for_test(
                backend,
                &extrema_weights,
                &extrema_activation
            ),
            scalar_dot(&extrema_weights, &extrema_activation)
        );
    }
}

#[test]
fn inference_q8_block_dot_matches_scalar_signed_extrema() {
    let weights = core::array::from_fn(|index| match index & 3 {
        0 | 1 => i8::MIN,
        _ => i8::MAX,
    });
    let activation = core::array::from_fn(|index| match index & 3 {
        0 | 2 => i8::MIN,
        _ => i8::MAX,
    });
    let expected = weights
        .iter()
        .zip(&activation)
        .fold(0i32, |total, (&weight, &activation)| {
            total + i32::from(weight) * i32::from(activation)
        });
    assert_eq!(expected, 8, "signed-extrema fixture");
    for backend in [
        crate::fp32_sse2::InferenceBackend::Sse2,
        crate::fp32_sse2::InferenceBackend::detect(),
    ] {
        assert_eq!(
            crate::fp32_sse2::inference_q8_block_dot_for_test(backend, &weights, &activation),
            expected
        );
    }
}

#[test]
fn inference_block_dots_match_scalar_oracles_across_deterministic_inputs() {
    let q4_scalar = |weights: &[u8; 16], activation: &[i8; 32]| {
        (0..16).fold(0i32, |total, index| {
            let packed = weights[index];
            total
                + (i32::from(packed & 0x0f) - 8) * i32::from(activation[index])
                + (i32::from(packed >> 4) - 8) * i32::from(activation[16 + index])
        })
    };
    let q8_scalar = |weights: &[i8; 32], activation: &[i8; 32]| {
        weights
            .iter()
            .zip(activation)
            .fold(0i32, |total, (&weight, &value)| {
                total + i32::from(weight) * i32::from(value)
            })
    };
    let mut state = 0x243f_6a88u32;
    for case in 0..256 {
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let q4_weights = core::array::from_fn(|_| next());
        let q8_weights = core::array::from_fn(|_| next() as i8);
        let activation = core::array::from_fn(|_| next() as i8);
        for backend in [
            crate::fp32_sse2::InferenceBackend::Sse2,
            crate::fp32_sse2::InferenceBackend::detect(),
        ] {
            assert_eq!(
                crate::fp32_sse2::inference_q4_block_dot_for_test(
                    backend,
                    &q4_weights,
                    &activation
                ),
                q4_scalar(&q4_weights, &activation),
                "Q4 case {case} backend {}",
                backend.name()
            );
            assert_eq!(
                crate::fp32_sse2::inference_q8_block_dot_for_test(
                    backend,
                    &q8_weights,
                    &activation
                ),
                q8_scalar(&q8_weights, &activation),
                "Q8 case {case} backend {}",
                backend.name()
            );
        }
    }
}

#[test]
fn inference_kernel_bit_fixtures_match_pinned_llama_sse2() {
    let word_digest = |words: &[u32]| crate::sha256::digest(word_bytes(words));
    let expected_digest = |value: &str| -> [u8; 32] { hex(value).try_into().unwrap() };

    let mut quant_input = vec![0u32; 32 * 6];
    quant_input[32] = 1.0f32.to_bits();
    quant_input[33] = (-1.0f32).to_bits();
    quant_input[34] = 1.0f32.to_bits();
    quant_input[35] = (-1.0f32).to_bits();
    quant_input[64] = 127.0f32.to_bits();
    quant_input[65] = 0.5f32.to_bits();
    quant_input[66] = (-0.5f32).to_bits();
    quant_input[67] = 1.5f32.to_bits();
    quant_input[68] = (-1.5f32).to_bits();
    quant_input[96] = f32::from_bits(0x3400_0000).to_bits();
    quant_input[97] = f32::from_bits(0x3380_0000).to_bits();
    quant_input[98] = f32::from_bits(0x3300_0000).to_bits();
    quant_input[128] = 126.5f32.to_bits();
    quant_input[129] = (-126.5f32).to_bits();
    quant_input[130] = 63.25f32.to_bits();
    quant_input[160] = (-0.0f32).to_bits();
    quant_input[161] = 0.0f32.to_bits();
    let mut quant_stage = vec![0xa5u8; 5_184];
    crate::fp32_sse2::inference_quantize_q8_for_test(&quant_input, &mut quant_stage);
    assert_eq!(
        &quant_stage[..204],
        include_bytes!("../../../fixtures/inference/kernels/q8.bin")
    );
    assert!(quant_stage[204..].iter().all(|byte| *byte == 0));

    let mut mat_input = vec![0u32; 64];
    for (index, word) in mat_input.iter_mut().enumerate() {
        *word = (((index as i32 % 31) - 15) as f32).to_bits();
    }
    mat_input[31] = 127.0f32.to_bits();
    mat_input[63] = (-127.0f32).to_bits();
    let mut q4 = vec![0u8; 36];
    for block in 0..2 {
        q4[block * 18..block * 18 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..16 {
            q4[block * 18 + 2 + index] = ((15 - index) << 4 | index) as u8;
        }
    }
    let mut q8 = vec![0u8; 68];
    for block in 0..2 {
        q8[block * 34..block * 34 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for index in 0..32 {
            q8[block * 34 + 2 + index] = ((index as i32 % 17) - 8) as i8 as u8;
        }
    }
    let mut stage = vec![0xa5u8; 5_184 + 64];
    let mut q4_output = [0u32; 1];
    let mut q8_output = [0u32; 1];
    unsafe {
        crate::fp32_sse2::inference_prepare_q8(mat_input.as_ptr().cast(), stage.as_mut_ptr(), 64);
    }
    let prepared = stage[..5_184].to_vec();
    assert!(prepared[68..].iter().all(|byte| *byte == 0));
    unsafe {
        crate::fp32_sse2::inference_q4_matvec_rows_prepared(
            crate::fp32_sse2::InferenceBackend::Sse2,
            q4.as_ptr(),
            q4_output.as_mut_ptr().cast(),
            stage.as_mut_ptr(),
            1,
            64,
            0,
            1,
        );
    }
    assert_eq!(&stage[..5_184], prepared);
    unsafe {
        crate::fp32_sse2::inference_prepare_q8(mat_input.as_ptr().cast(), stage.as_mut_ptr(), 64);
        crate::fp32_sse2::inference_q8_matvec_rows_prepared(
            crate::fp32_sse2::InferenceBackend::Sse2,
            q8.as_ptr(),
            q8_output.as_mut_ptr(),
            stage.as_mut_ptr(),
            1,
            64,
            0,
            1,
        );
    }
    assert_eq!(q4_output, [0x43e0_8000]);
    assert_eq!(q8_output, [0x4438_8000]);
    assert_eq!(
        [word_bytes(&q4_output), word_bytes(&q8_output)].concat(),
        include_bytes!("../../../fixtures/inference/kernels/dots.bin")
    );
    if crate::fp32_sse2::InferenceBackend::detect() == crate::fp32_sse2::InferenceBackend::Avx2 {
        let mut avx_q4 = [0u32; 1];
        let mut avx_q8 = [0u32; 1];
        unsafe {
            crate::fp32_sse2::inference_prepare_q8(
                mat_input.as_ptr().cast(),
                stage.as_mut_ptr(),
                64,
            );
            crate::fp32_sse2::inference_q4_matvec_rows_prepared(
                crate::fp32_sse2::InferenceBackend::Avx2,
                q4.as_ptr(),
                avx_q4.as_mut_ptr().cast(),
                stage.as_mut_ptr(),
                1,
                64,
                0,
                1,
            );
            crate::fp32_sse2::inference_q8_matvec_rows_prepared(
                crate::fp32_sse2::InferenceBackend::Avx2,
                q8.as_ptr(),
                avx_q8.as_mut_ptr(),
                stage.as_mut_ptr(),
                1,
                64,
                0,
                1,
            );
        }
        assert_eq!(avx_q4, q4_output);
        assert_eq!(avx_q8, q8_output);
    }
    assert_eq!(&stage[..5_184], prepared);
    assert!(stage[5_184..].iter().all(|byte| *byte == 0xa5));

    let rms_input: Vec<u32> = (0..896)
        .map(|index| (((index % 17) as i32 - 8) as f32 / 8.0).to_bits())
        .collect();
    let rms_weight: Vec<u32> = (0..896)
        .map(|index| (((index % 7 + 1) as f32) / 4.0).to_bits())
        .collect();
    let mut rms_output = vec![0u32; 896];
    unsafe {
        crate::fp32_sse2::inference_rmsnorm(
            rms_input.as_ptr().cast(),
            rms_weight.as_ptr().cast(),
            rms_output.as_mut_ptr().cast(),
            896,
        )
    };
    assert_eq!(
        word_digest(&rms_output),
        expected_digest("74ca732ed2538165b72cbfefc766de0bb0d14117ea3ab8a588e097e576560a5c")
    );
    assert_eq!(
        word_bytes(&rms_output),
        include_bytes!("../../../fixtures/inference/kernels/rms.bin")
    );
    assert_eq!(
        (rms_output[0], rms_output[447], rms_output[895]),
        (0xbed1_5d27, 0xbf89_6521, 0x3f89_6521)
    );

    let gate: Vec<u32> = (0..4_864)
        .map(|index| (((index % 33) as i32 - 16) as f32 / 16.0).to_bits())
        .collect();
    let up: Vec<u32> = (0..4_864)
        .map(|index| (((index % 19) as i32 - 9) as f32 / 8.0).to_bits())
        .collect();
    let mut product = vec![0u32; 4_864];
    unsafe {
        crate::fp32_sse2::inference_swiglu(
            gate.as_ptr().cast(),
            up.as_ptr().cast(),
            product.as_mut_ptr().cast(),
            4_864,
        )
    };
    assert_eq!(
        word_digest(&product),
        expected_digest("1763ddc60a335d5253dd4a3dcb69bfb12b85e7104c760be3870eaf75c0fc3aa0")
    );
    assert_eq!(
        word_bytes(&product),
        include_bytes!("../../../fixtures/inference/kernels/swiglu.bin")
    );
    assert_eq!(
        (product[0], product[2_431], product[4_863]),
        (0x3e9a_e906, 0x3e80_0417, 0xbdfc_2fb4)
    );

    let rope_table = include_bytes!("../../../fixtures/inference/rope-table.f32le");
    assert_eq!(
        crate::sha256::digest(rope_table),
        expected_digest("cd75fcc63f7514055daf75917521bfe4612ce6417419de5ce77ca766473c01c5")
    );
    assert_eq!(
        u32::from_le_bytes(rope_table[0..4].try_into().unwrap()),
        0x3f80_0000
    );
    assert_eq!(
        u32::from_le_bytes(rope_table[65_532..65_536].try_into().unwrap()),
        0x39cd_e0cf
    );
    assert_eq!(
        u32::from_le_bytes(rope_table[131_068..131_072].try_into().unwrap()),
        0x3a4e_4824
    );
    assert_eq!(
        u32::from_le_bytes(rope_table[rope_table.len() - 4..].try_into().unwrap(),),
        0x3d4e_9771
    );
    let mut rope_values: Vec<u32> = (0..64)
        .map(|index| (((index as i32) - 32) as f32 / 16.0).to_bits())
        .collect();
    unsafe { crate::fp32_sse2::inference_rope_in_place(rope_values.as_mut_ptr().cast(), 1, 511) };
    assert_eq!(
        word_digest(&rope_values),
        expected_digest("2428f74055d47f641740f2a914403b95118155c29a9d5d5d32b38fb12b2ad672")
    );
    assert_eq!(
        word_bytes(&rope_values),
        include_bytes!("../../../fixtures/inference/kernels/rope.bin")
    );
    assert_eq!(
        (rope_values[0], rope_values[31], rope_values[63]),
        (0x3f71_7fe5, 0xbd83_1f55, 0x3ff7_fe5e)
    );
    let mut final_rope_values: Vec<u32> = (0..64)
        .map(|index| (((index as i32) - 32) as f32 / 16.0).to_bits())
        .collect();
    unsafe {
        crate::fp32_sse2::inference_rope_in_place(
            final_rope_values.as_mut_ptr().cast(),
            1,
            CONTEXT_LIMIT as usize - 1,
        )
    };
    assert_eq!(
        word_digest(&final_rope_values),
        expected_digest("8121af7bd1111f604c69d651761737255b00bf43931b14c8fe64640bfd76aa61")
    );
    assert_eq!(
        (
            final_rope_values[0],
            final_rope_values[31],
            final_rope_values[63],
        ),
        (0xbffb_759c, 0xbe23_fc80, 0x3ff7_47e5)
    );

    let mut query: Vec<u32> = (0..896)
        .map(|index| (((index % 23) as i32 - 11) as f32 / 16.0).to_bits())
        .collect();
    let mut kv = vec![0u32; crate::inference::KV_LAYER_WORDS];
    let kv_index = |kind: usize, position: usize, head: usize, component: usize| {
        crate::inference::kv_word(0, kind, position, head, component)
    };
    for position in 0..=256 {
        for head in 0..2 {
            for component in 0..64 {
                kv[kv_index(0, position, head, component)] =
                    ((((position + component + head) % 29) as i32 - 14) as f32 / 32.0).to_bits();
                kv[kv_index(1, position, head, component)] =
                    ((((position * 3 + component + head) % 31) as i32 - 15) as f32 / 32.0)
                        .to_bits();
            }
        }
    }
    let mut attention = vec![0u32; 896];
    let mut scores = vec![0u32; 512];
    unsafe {
        crate::fp32_sse2::inference_attention(
            crate::fp32_sse2::InferenceBackend::Sse2,
            query.as_ptr().cast(),
            kv.as_ptr().cast(),
            attention.as_mut_ptr().cast(),
            scores.as_mut_ptr().cast(),
            0,
            2,
            256,
        )
    };
    assert_eq!(
        word_digest(&attention),
        expected_digest("623d8a942d109f5f0606ce1f00b4757758d567ab40842e8e4ce4671295e59834")
    );
    assert_eq!(
        word_bytes(&attention),
        include_bytes!("../../../fixtures/inference/kernels/attention-256.bin")
    );
    assert_eq!(
        word_digest(&scores[..256]),
        expected_digest("0698daf221562bfa6d8d3a954abb35489155e53b73c1feab1904708300e74e58")
    );
    assert_eq!(
        word_bytes(&scores[..256]),
        include_bytes!("../../../fixtures/inference/kernels/scores-256.bin")
    );
    assert_eq!(
        (attention[0], attention[447], attention[895]),
        (0xbebe_c356, 0xbeb0_8850, 0xbe9f_8fbd)
    );
    query.rotate_left(7);
    unsafe {
        crate::fp32_sse2::inference_attention(
            crate::fp32_sse2::InferenceBackend::Sse2,
            query.as_ptr().cast(),
            kv.as_ptr().cast(),
            attention.as_mut_ptr().cast(),
            scores.as_mut_ptr().cast(),
            0,
            256,
            512,
        )
    };
    assert_eq!(
        word_digest(&attention),
        expected_digest("1a386ebbbf897b7709d8cade1549041415ac80d03f5e89b30c39424987fab043")
    );
    assert_eq!(
        word_bytes(&attention),
        include_bytes!("../../../fixtures/inference/kernels/attention-512.bin")
    );
    assert_eq!(
        word_digest(&scores),
        expected_digest("a027538d38b56e069ddcfafe848c17b361a92e01463c0c98f743ba011e6cadd0")
    );
    assert_eq!(
        word_bytes(&scores),
        include_bytes!("../../../fixtures/inference/kernels/scores-512.bin")
    );
    assert_eq!(
        (attention[0], attention[447], attention[895]),
        (0xba51_5bd2, 0xbb34_df23, 0xbb36_98c2)
    );
}
