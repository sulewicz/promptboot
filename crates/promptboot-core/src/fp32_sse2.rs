//! Deterministic scalar binary32 primitives.
//!
//! Every hardware floating-point operation in `promptboot-core` is confined to
//! this module. Rust's UEFI target keeps its global soft-float ABI, so even the
//! private helpers carry binary32 as `u32` and execute scalar SSE2 through
//! explicit inline assembly. No float is visible at a function or crate ABI.

use core::arch::x86_64::{
    __cpuid, __cpuid_count, _mm_add_pd, _mm_castsi128_ps, _mm_cvtpd_ps, _mm_cvtps_pd, _mm_loadu_ps,
    _mm_movehl_ps, _mm_mul_ps, _mm_set1_epi32, _mm_setzero_pd, _mm_storeu_ps,
};
use core::arch::{asm, global_asm};
use core::ptr;

use crate::PrimitiveStatus;

const AVX_STATE_MASK: u64 = 0x6;
const AVX_LEAF1_MASK: u32 = (1 << 26) | (1 << 27) | (1 << 28);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InferenceBackend {
    Sse2,
    Avx2,
}

impl InferenceBackend {
    pub(crate) fn detect() -> Self {
        if inference_avx2_available() {
            Self::Avx2
        } else {
            Self::Sse2
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Sse2 => "sse2",
            Self::Avx2 => "avx2",
        }
    }
}

pub(crate) fn inference_avx2_available() -> bool {
    let max_basic_leaf = __cpuid(0).eax;
    let leaf1_ecx = if max_basic_leaf >= 1 {
        __cpuid(1).ecx
    } else {
        0
    };
    resolve_avx2(
        max_basic_leaf,
        leaf1_ecx,
        || unsafe { xgetbv0() },
        || __cpuid_count(7, 0).ebx,
    )
}

fn resolve_avx2<X, L>(max_basic_leaf: u32, leaf1_ecx: u32, read_xcr0: X, read_leaf7_ebx: L) -> bool
where
    X: FnOnce() -> u64,
    L: FnOnce() -> u32,
{
    if max_basic_leaf < 7 || leaf1_ecx & AVX_LEAF1_MASK != AVX_LEAF1_MASK {
        return false;
    }
    select_avx2(max_basic_leaf, leaf1_ecx, read_xcr0(), read_leaf7_ebx())
}

fn select_avx2(max_basic_leaf: u32, leaf1_ecx: u32, xcr0: u64, leaf7_ebx: u32) -> bool {
    max_basic_leaf >= 7
        && leaf1_ecx & AVX_LEAF1_MASK == AVX_LEAF1_MASK
        && xcr0 & AVX_STATE_MASK == AVX_STATE_MASK
        && leaf7_ebx & (1 << 5) != 0
}

unsafe fn xgetbv0() -> u64 {
    let low: u32;
    let high: u32;
    asm!(
        "xgetbv",
        in("ecx") 0u32,
        lateout("eax") low,
        lateout("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    u64::from(low) | (u64::from(high) << 32)
}

#[cfg(test)]
mod backend_tests {
    use core::cell::Cell;

    use super::{resolve_avx2, select_avx2, AVX_LEAF1_MASK};

    #[test]
    fn avx2_selector_requires_every_cpu_and_os_prerequisite() {
        let avx2 = 1 << 5;
        assert!(!select_avx2(6, AVX_LEAF1_MASK, 0x6, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK & !(1 << 26), 0x6, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK & !(1 << 27), 0x6, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK & !(1 << 28), 0x6, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK, 0x4, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK, 0x2, avx2));
        assert!(!select_avx2(7, AVX_LEAF1_MASK, 0x6, 0));
        assert!(select_avx2(7, AVX_LEAF1_MASK, 0x6, avx2));
    }

    #[test]
    fn avx2_detection_does_not_read_xcr0_without_osxsave() {
        let xgetbv_called = Cell::new(false);
        let selected = resolve_avx2(
            7,
            AVX_LEAF1_MASK & !(1 << 27),
            || {
                xgetbv_called.set(true);
                0x6
            },
            || 1 << 5,
        );
        assert!(!selected);
        assert!(!xgetbv_called.get());
    }
}

static ROPE_TABLE: &[u8; 8_388_608] = include_bytes!("../../../fixtures/analytic/rope-table.f32le");
static INFERENCE_ROPE_TABLE: &[u8; 8_388_608] =
    include_bytes!("../../../fixtures/inference/rope-table.f32le");

pub(crate) struct ExecOutcome {
    pub status: PrimitiveStatus,
    pub index: u32,
}

impl ExecOutcome {
    const fn ok() -> Self {
        Self {
            status: PrimitiveStatus::OK,
            index: 0,
        }
    }

    const fn error(status: PrimitiveStatus, index: u32) -> Self {
        Self { status, index }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn add(left: u32, right: u32) -> u32 {
    let mut saved = [0u8; 32];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0",
        "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movd xmm0, {left:e}",
        "movd xmm1, {right:e}",
        "addss xmm0, xmm1",
        "movd {output:e}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]",
        "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left,
        right = in(reg) right,
        output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sub(left: u32, right: u32) -> u32 {
    let mut saved = [0u8; 32];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movd xmm0, {left:e}", "movd xmm1, {right:e}", "subss xmm0, xmm1", "movd {output:e}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn mul(left: u32, right: u32) -> u32 {
    let mut saved = [0u8; 32];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movd xmm0, {left:e}", "movd xmm1, {right:e}", "mulss xmm0, xmm1", "movd {output:e}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn div(left: u32, right: u32) -> u32 {
    let mut saved = [0u8; 32];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movd xmm0, {left:e}", "movd xmm1, {right:e}", "divss xmm0, xmm1", "movd {output:e}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sqrt(value: u32) -> u32 {
    let mut saved = [0u8; 16];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movd xmm0, {value:e}", "sqrtss xmm0, xmm0", "movd {output:e}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) value, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn greater(left: u32, right: u32) -> bool {
    let mut saved = [0u8; 32];
    let output: u8;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movd xmm0, {left:e}", "movd xmm1, {right:e}", "ucomiss xmm0, xmm1", "seta {output}",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg_byte) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack),
    );
    output != 0
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn from_i32(value: i32) -> u32 {
    let mut saved = [0u8; 16];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "cvtsi2ss xmm0, {value:e}", "movd {output:e}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) value, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn from_u32(value: u32) -> u32 {
    let wide = u64::from(value);
    let mut saved = [0u8; 16];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "cvtsi2ss xmm0, {value}", "movd {output:e}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) wide, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f32_to_f64(value: u32) -> u64 {
    let mut saved = [0u8; 16];
    let output: u64;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movd xmm0, {value:e}",
        "cvtss2sd xmm0, xmm0", "movq {output}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) value, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f64_to_f32(value: u64) -> u32 {
    let mut saved = [0u8; 16];
    let output: u32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movq xmm0, {value}",
        "cvtsd2ss xmm0, xmm0", "movd {output:e}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) value, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f64_add(left: u64, right: u64) -> u64 {
    let mut saved = [0u8; 32];
    let output: u64;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movq xmm0, {left}", "movq xmm1, {right}", "addsd xmm0, xmm1", "movq {output}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f64_div(left: u64, right: u64) -> u64 {
    let mut saved = [0u8; 32];
    let output: u64;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movdqu xmmword ptr [{saved} + 16], xmm1",
        "movq xmm0, {left}", "movq xmm1, {right}", "divsd xmm0, xmm1", "movq {output}, xmm0",
        "movdqu xmm1, xmmword ptr [{saved} + 16]", "movdqu xmm0, xmmword ptr [{saved}]",
        left = in(reg) left, right = in(reg) right, output = out(reg) output,
        saved = in(reg) saved.as_mut_ptr(), options(nostack, preserves_flags),
    );
    output
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f64_from_u32(value: u32) -> u64 {
    f32_to_f64(from_u32(value))
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn trunc_i32(value: u32) -> i32 {
    let mut saved = [0u8; 16];
    let output: i32;
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0", "movd xmm0, {value:e}", "cvttss2si {output:e}, xmm0", "movdqu xmm0, xmmword ptr [{saved}]",
        value = in(reg) value, output = out(reg) output, saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[inline]
fn finite(bits: u32) -> bool {
    bits & 0x7f80_0000 != 0x7f80_0000
}

#[inline]
unsafe fn read_bits(base: *const u8, word: usize) -> u32 {
    u32::from_le(ptr::read_unaligned(base.add(word * 4).cast::<u32>()))
}

#[inline]
unsafe fn write_bits(base: *mut u8, word: usize, bits: u32) {
    ptr::write_unaligned(base.add(word * 4).cast::<u32>(), bits.to_le());
}

fn half_to_f32_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    match (exponent, fraction) {
        (0, 0) => sign,
        (0, _) => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((113 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        (31, _) => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}

#[target_feature(enable = "sse2")]
unsafe fn scalbnf(mut value: u32, mut exponent: i32) -> u32 {
    if exponent > 127 {
        value = mul(value, 0x7f00_0000);
        exponent -= 127;
        if exponent > 127 {
            value = mul(value, 0x7f00_0000);
            exponent -= 127;
            if exponent > 127 {
                exponent = 127;
            }
        }
    } else if exponent < -126 {
        value = mul(value, mul(0x0080_0000, 0x4b80_0000));
        exponent += 126 - 24;
        if exponent < -126 {
            value = mul(value, mul(0x0080_0000, 0x4b80_0000));
            exponent += 126 - 24;
            if exponent < -126 {
                exponent = -126;
            }
        }
    }
    mul(value, ((0x7f + exponent) as u32) << 23)
}

/* origin: FreeBSD /usr/src/lib/msun/src/e_expf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
// Adapted from rust-lang/libm 0.2.11 expf/scalbnf. The source branches and
// constants are retained; only force_eval exception-flag effects are omitted.
#[target_feature(enable = "sse2")]
unsafe fn expf(mut value: u32) -> u32 {
    let magnitude = value & 0x7fff_ffff;
    let sign = value >> 31;
    if magnitude >= 0x42ae_ac50 {
        if magnitude > 0x7f80_0000 {
            return value;
        }
        if magnitude >= 0x42b1_7218 && sign == 0 {
            return mul(value, 0x7f00_0000);
        }
        if sign != 0 && magnitude >= 0x42cf_f1b5 {
            return 0;
        }
    }

    let reduction: i32;
    let high: u32;
    let low: u32;
    if magnitude > 0x3eb1_7218 {
        if magnitude > 0x3f85_1592 {
            let half = if sign == 0 { 0x3f00_0000 } else { 0xbf00_0000 };
            reduction = trunc_i32(add(mul(0x3fb8_aa3b, value), half));
        } else {
            reduction = 1 - sign as i32 - sign as i32;
        }
        let reduction_bits = from_i32(reduction);
        high = sub(value, mul(reduction_bits, 0x3f31_7200));
        low = mul(reduction_bits, 0x35bf_be8e);
        value = sub(high, low);
    } else if magnitude > 0x3900_0000 {
        reduction = 0;
        high = value;
        low = 0;
    } else {
        return add(0x3f80_0000, value);
    }
    let square = mul(value, value);
    let coefficient = sub(
        value,
        mul(square, add(0x3e2a_aa8f, mul(square, 0xbb35_5215))),
    );
    let fraction = div(mul(value, coefficient), sub(0x4000_0000, coefficient));
    let result = add(0x3f80_0000, add(sub(fraction, low), high));
    if reduction == 0 {
        result
    } else {
        scalbnf(result, reduction)
    }
}

#[cfg(test)]
pub(crate) fn expf_bits_for_test(bits: u32) -> u32 {
    unsafe { expf(bits) }
}

unsafe fn check_finite(base: *const u8, words: usize) -> Result<(), u32> {
    for index in 0..words {
        if !finite(read_bits(base, index)) {
            return Err(index as u32);
        }
    }
    Ok(())
}

#[target_feature(enable = "sse2")]
unsafe fn bias_residual(input: *const u8, output: *mut u8, count: usize) {
    for index in 0..count {
        let sum = add(read_bits(input, index), read_bits(input, count + index));
        write_bits(output, index, add(sum, read_bits(input, count * 2 + index)));
    }
}

#[target_feature(enable = "sse2")]
unsafe fn quantized(
    input: *const u8,
    aux: *const u8,
    output: *mut u8,
    rows: usize,
    columns: usize,
    q4: bool,
) -> Result<(), u32> {
    let block_bytes = if q4 { 18 } else { 34 };
    let blocks_per_row = columns >> 5;
    for row in 0..rows {
        for block_index in 0..blocks_per_row {
            let block_number = row * blocks_per_row + block_index;
            let block = input.add(block_number * block_bytes);
            let scale = half_to_f32_bits(u16::from_le(ptr::read_unaligned(block.cast::<u16>())));
            if !finite(scale) {
                return Err(block_number as u32);
            }
            for element in 0..32 {
                let quant = if q4 {
                    let packed = ptr::read(block.add(2 + (element & 15)));
                    if element < 16 {
                        i32::from(packed & 0x0f) - 8
                    } else {
                        i32::from(packed >> 4) - 8
                    }
                } else {
                    i32::from(ptr::read(block.add(2 + element).cast::<i8>()))
                };
                write_bits(
                    output,
                    row * columns + block_index * 32 + element,
                    mul(scale, from_i32(quant)),
                );
            }
        }
    }
    let dot_base = rows * columns;
    for row in 0..rows {
        let mut total = 0;
        for column in 0..columns {
            total = add(
                total,
                mul(
                    read_bits(output, row * columns + column),
                    read_bits(aux, column),
                ),
            );
        }
        if row == 0 {
            write_bits(output, dot_base, total);
        }
        write_bits(output, dot_base + 1 + row, total);
    }
    Ok(())
}

#[target_feature(enable = "sse2")]
unsafe fn rmsnorm(input: *const u8, aux: *const u8, output: *mut u8, count: usize) {
    let mut squares = 0;
    for index in 0..count {
        let value = read_bits(input, index);
        squares = add(squares, mul(value, value));
    }
    let inverse = div(
        0x3f80_0000,
        sqrt(add(div(squares, from_u32(count as u32)), 0x3586_37bd)),
    );
    for index in 0..count {
        write_bits(
            output,
            index,
            mul(mul(read_bits(input, index), inverse), read_bits(aux, index)),
        );
    }
}

fn rope_bits(position: usize, pair: usize, component: usize) -> u32 {
    let byte = ((position * 32 + pair) * 2 + component) * 4;
    unsafe {
        u32::from_le(ptr::read_unaligned(
            ROPE_TABLE.as_ptr().add(byte).cast::<u32>(),
        ))
    }
}

fn inference_rope_bits(position: usize, pair: usize, component: usize) -> u32 {
    let byte = ((position * 32 + pair) * 2 + component) * 4;
    unsafe {
        u32::from_le(ptr::read_unaligned(
            INFERENCE_ROPE_TABLE.as_ptr().add(byte).cast::<u32>(),
        ))
    }
}

#[target_feature(enable = "sse2")]
unsafe fn rope(input: *const u8, output: *mut u8, head_dim: usize, heads: usize, position: usize) {
    let half = head_dim >> 1;
    let table_stride = if head_dim == 4 { 16 } else { 1 };
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..half {
            let cosine = rope_bits(position, pair * table_stride, 0);
            let sine = rope_bits(position, pair * table_stride, 1);
            let left = read_bits(input, base + pair);
            let right = read_bits(input, base + pair + half);
            write_bits(
                output,
                base + pair,
                sub(mul(left, cosine), mul(right, sine)),
            );
            write_bits(
                output,
                base + pair + half,
                add(mul(left, sine), mul(right, cosine)),
            );
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn softmax(input: *const u8, output: *mut u8, count: usize) {
    let mut maximum = read_bits(input, 0);
    for index in 1..count {
        let value = read_bits(input, index);
        if greater(value, maximum) {
            maximum = value;
        }
    }
    let mut total = 0;
    for index in 0..count {
        let value = expf(sub(read_bits(input, index), maximum));
        write_bits(output, index, value);
        total = add(total, value);
    }
    for index in 0..count {
        write_bits(output, index, div(read_bits(output, index), total));
    }
}

fn unsigned_divide(numerator: usize, denominator: usize) -> usize {
    let mut quotient = 0usize;
    let mut remainder = 0usize;
    let mut bit = usize::BITS;
    while bit != 0 {
        bit -= 1;
        remainder = (remainder << 1) | ((numerator >> bit) & 1);
        if remainder >= denominator {
            remainder -= denominator;
            quotient |= 1usize << bit;
        }
    }
    quotient
}

#[target_feature(enable = "sse2")]
unsafe fn gqa_attention(
    input: *const u8,
    output: *mut u8,
    query_heads: usize,
    kv_heads: usize,
    positions: usize,
    head_dim: usize,
) {
    let queries_words = query_heads * head_dim;
    let kv_words = positions * kv_heads * head_dim;
    let keys_base = queries_words;
    let values_base = keys_base + kv_words;
    let group = unsigned_divide(query_heads, kv_heads);
    let scale = div(0x3f80_0000, sqrt(from_u32(head_dim as u32)));
    let per_head = head_dim + positions * 2;
    for query_head in 0..query_heads {
        let kv_head = unsigned_divide(query_head, group);
        let result_base = query_head * per_head;
        let probability_base = result_base + head_dim;
        let score_base = probability_base + positions;
        let mut maximum = 0xff80_0000;
        for position in 0..positions {
            let mut dot = 0;
            for element in 0..head_dim {
                let key = keys_base + (position * kv_heads + kv_head) * head_dim + element;
                dot = add(
                    dot,
                    mul(
                        read_bits(input, query_head * head_dim + element),
                        read_bits(input, key),
                    ),
                );
            }
            let score = mul(dot, scale);
            write_bits(output, score_base + position, score);
            if greater(score, maximum) {
                maximum = score;
            }
        }
        let mut sum = 0;
        for position in 0..positions {
            let probability = expf(sub(read_bits(output, score_base + position), maximum));
            write_bits(output, probability_base + position, probability);
            sum = add(sum, probability);
        }
        for position in 0..positions {
            write_bits(
                output,
                probability_base + position,
                div(read_bits(output, probability_base + position), sum),
            );
        }
        for element in 0..head_dim {
            let mut total = 0;
            for position in 0..positions {
                let value = values_base + (position * kv_heads + kv_head) * head_dim + element;
                total = add(
                    total,
                    mul(
                        read_bits(output, probability_base + position),
                        read_bits(input, value),
                    ),
                );
            }
            write_bits(output, result_base + element, total);
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn silu_swiglu(input: *const u8, aux: *const u8, output: *mut u8, count: usize) {
    for index in 0..count {
        let x = read_bits(input, index);
        let exponent = expf(x ^ 0x8000_0000);
        let sigmoid = div(0x3f80_0000, add(0x3f80_0000, exponent));
        let silu = mul(x, sigmoid);
        write_bits(output, index, mul(silu, read_bits(aux, index)));
        write_bits(output, count + index, silu);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn argmax(input: *const u8, output: *mut u8, count: usize) {
    let mut selected = 0usize;
    let mut best = read_bits(input, 0);
    for index in 1..count {
        let value = read_bits(input, index);
        if greater(value, best) {
            selected = index;
            best = value;
        }
    }
    write_bits(output, 0, selected as u32);
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn execute(
    operation: u32,
    input: *const u8,
    aux: *const u8,
    output: *mut u8,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    position: u32,
) -> ExecOutcome {
    let input_words = match operation {
        1 => dim0 as usize * 3,
        2 | 3 => 0,
        4 | 6 | 8 | 9 => dim0 as usize,
        5 => dim0 as usize * dim1 as usize,
        7 => dim0 as usize * dim3 as usize + dim2 as usize * dim1 as usize * dim3 as usize * 2,
        _ => 0,
    };
    if input_words != 0 {
        if let Err(index) = check_finite(input, input_words) {
            return ExecOutcome::error(PrimitiveStatus::NONFINITE_INPUT, index);
        }
    }
    let aux_words = match operation {
        2 | 3 => dim1 as usize,
        4 | 8 => dim0 as usize,
        _ => 0,
    };
    if aux_words != 0 {
        if let Err(index) = check_finite(aux, aux_words) {
            return ExecOutcome::error(PrimitiveStatus::NONFINITE_INPUT, index);
        }
    }
    let output_words = match operation {
        1 => {
            bias_residual(input, output, dim0 as usize);
            dim0 as usize
        }
        2 | 3 => {
            if let Err(index) = quantized(
                input,
                aux,
                output,
                dim0 as usize,
                dim1 as usize,
                operation == 2,
            ) {
                return ExecOutcome::error(PrimitiveStatus::BLOCK_ENCODING, index);
            }
            dim0 as usize * dim1 as usize + 1 + dim0 as usize
        }
        4 => {
            rmsnorm(input, aux, output, dim0 as usize);
            dim0 as usize
        }
        5 => {
            rope(
                input,
                output,
                dim0 as usize,
                dim1 as usize,
                position as usize,
            );
            dim0 as usize * dim1 as usize
        }
        6 => {
            softmax(input, output, dim0 as usize);
            dim0 as usize
        }
        7 => {
            gqa_attention(
                input,
                output,
                dim0 as usize,
                dim1 as usize,
                dim2 as usize,
                dim3 as usize,
            );
            dim0 as usize * (dim3 as usize + dim2 as usize * 2)
        }
        8 => {
            silu_swiglu(input, aux, output, dim0 as usize);
            dim0 as usize * 2
        }
        9 => {
            argmax(input, output, dim0 as usize);
            return ExecOutcome::ok();
        }
        _ => return ExecOutcome::error(PrimitiveStatus::UNSUPPORTED_OPERATION, 0),
    };
    if let Err(index) = check_finite(output, output_words) {
        return ExecOutcome::error(PrimitiveStatus::NONFINITE_OUTPUT, index);
    }
    ExecOutcome::ok()
}

// Inference-only kernels. Their public-to-crate boundary remains pointers,
// integer lengths and f32 bit words; all hardware FP stays in this module.
// Quantized matvecs reproduce the pinned generic Q8-activation block order;
// attention softmax and SiLU reproduce its explicit four-lane SSE2 paths.

pub(crate) unsafe fn inference_enter_fp() -> u32 {
    let mut previous = 0u32;
    let required = 0x1f80u32;
    asm!(
        "stmxcsr dword ptr [{previous}]",
        "ldmxcsr dword ptr [{required}]",
        previous = in(reg) &mut previous,
        required = in(reg) &required,
        options(nostack, preserves_flags),
    );
    previous
}

pub(crate) unsafe fn inference_exit_fp(previous: u32) {
    asm!(
        "ldmxcsr dword ptr [{previous}]",
        previous = in(reg) &previous,
        options(nostack, preserves_flags),
    );
}

#[cfg(test)]
pub(crate) unsafe fn inference_clear_fp_exceptions_for_test() {
    let mut current = 0u32;
    asm!(
        "stmxcsr dword ptr [{current}]",
        current = in(reg) &mut current,
        options(nostack, preserves_flags),
    );
    current &= !0x3f;
    asm!(
        "ldmxcsr dword ptr [{current}]",
        current = in(reg) &current,
        options(nostack, preserves_flags),
    );
}

#[cfg(test)]
pub(crate) unsafe fn inference_fp_exceptions_for_test() -> u32 {
    let mut current = 0u32;
    asm!(
        "stmxcsr dword ptr [{current}]",
        current = in(reg) &mut current,
        options(nostack, preserves_flags),
    );
    current & 0x3f
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_q4_row(
    weights: *const u8,
    row: usize,
    output: *mut u8,
    columns: usize,
) {
    let blocks_per_row = columns >> 5;
    let row_base = weights.add(row * blocks_per_row * 18);
    for block_index in 0..blocks_per_row {
        let block = row_base.add(block_index * 18);
        let scale = half_to_f32_bits(u16::from_le(ptr::read_unaligned(block.cast::<u16>())));
        for element in 0..32 {
            let packed = ptr::read(block.add(2 + (element & 15)));
            let quant = if element < 16 {
                i32::from(packed & 0x0f) - 8
            } else {
                i32::from(packed >> 4) - 8
            };
            write_bits(
                output,
                block_index * 32 + element,
                mul(scale, from_i32(quant)),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn f32_bits_to_half(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    let base = mul(mul(magnitude, 0x7780_0000), 0x0880_0000);
    let doubled = bits.wrapping_add(bits);
    let mut bias = doubled & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    let rounded = add((bias >> 1).wrapping_add(0x0780_0000), base);
    let exponent = (rounded >> 13) & 0x7c00;
    let mantissa = rounded & 0x0fff;
    if doubled > 0xff00_0000 {
        sign | 0x7e00
    } else {
        sign | (exponent + mantissa) as u16
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn round_half_away(value: u32) -> i32 {
    let half = if value >> 31 == 0 {
        0x3f00_0000
    } else {
        0xbf00_0000
    };
    trunc_i32(add(value, half))
}

const INFERENCE_Q8_STAGING_BYTES: usize = 5_184;

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_prepare_q8(input: *const u8, staging: *mut u8, columns: usize) {
    ptr::write_bytes(staging, 0, INFERENCE_Q8_STAGING_BYTES);
    for block_index in 0..(columns >> 5) {
        let input_base = block_index * 32;
        let block = staging.add(block_index * 34);
        let mut maximum = 0u32;
        for element in 0..32 {
            let magnitude = read_bits(input, input_base + element) & 0x7fff_ffff;
            if greater(magnitude, maximum) {
                maximum = magnitude;
            }
        }
        let scale = div(maximum, 0x42fe_0000);
        ptr::write_unaligned(block.cast::<u16>(), f32_bits_to_half(scale).to_le());
        let inverse = if scale == 0 {
            0
        } else {
            div(0x3f80_0000, scale)
        };
        for element in 0..32 {
            let quant = round_half_away(mul(read_bits(input, input_base + element), inverse));
            ptr::write(block.add(2 + element).cast::<i8>(), quant as i8);
        }
    }
}

#[cfg(test)]
pub(crate) fn inference_quantize_q8_for_test(input: &[u32], staging: &mut [u8]) {
    assert!(input.len() % 32 == 0 && input.len() <= 4_864);
    assert!(staging.len() >= INFERENCE_Q8_STAGING_BYTES);
    unsafe {
        inference_prepare_q8(
            input.as_ptr().cast::<u8>(),
            staging.as_mut_ptr(),
            input.len(),
        )
    }
}

global_asm!(
    ".text",
    ".p2align 4",
    ".globl promptboot_inference_q4_block_dot",
    "promptboot_inference_q4_block_dot:",
    "pxor xmm2, xmm2",
    "pxor xmm5, xmm5",
    "movdqu xmm0, xmmword ptr [rcx]",
    "pand xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "psubb xmm0, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "movdqu xmm1, xmmword ptr [rdx]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpcklbw xmm0, xmm3",
    "punpcklbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx]",
    "pand xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "psubb xmm0, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "movdqu xmm1, xmmword ptr [rdx]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpckhbw xmm0, xmm3",
    "punpckhbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx]",
    "psrlw xmm0, 4",
    "pand xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "psubb xmm0, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpcklbw xmm0, xmm3",
    "punpcklbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx]",
    "psrlw xmm0, 4",
    "pand xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "psubb xmm0, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpckhbw xmm0, xmm3",
    "punpckhbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqa xmm0, xmm5",
    "psrldq xmm0, 8",
    "paddd xmm5, xmm0",
    "movdqa xmm0, xmm5",
    "psrldq xmm0, 4",
    "paddd xmm5, xmm0",
    "movd eax, xmm5",
    "ret",
    ".p2align 4",
    ".globl promptboot_inference_q8_block_dot",
    "promptboot_inference_q8_block_dot:",
    "pxor xmm2, xmm2",
    "movdqu xmm0, xmmword ptr [rcx]",
    "movdqu xmm1, xmmword ptr [rdx]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpcklbw xmm0, xmm3",
    "punpcklbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "movdqa xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx]",
    "movdqu xmm1, xmmword ptr [rdx]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpckhbw xmm0, xmm3",
    "punpckhbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx + 16]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpcklbw xmm0, xmm3",
    "punpcklbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqu xmm0, xmmword ptr [rcx + 16]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqa xmm3, xmm2",
    "pcmpgtb xmm3, xmm0",
    "movdqa xmm4, xmm2",
    "pcmpgtb xmm4, xmm1",
    "punpckhbw xmm0, xmm3",
    "punpckhbw xmm1, xmm4",
    "pmaddwd xmm0, xmm1",
    "paddd xmm5, xmm0",
    "movdqa xmm0, xmm5",
    "psrldq xmm0, 8",
    "paddd xmm5, xmm0",
    "movdqa xmm0, xmm5",
    "psrldq xmm0, 4",
    "paddd xmm5, xmm0",
    "movd eax, xmm5",
    "ret",
    ".p2align 4",
    ".globl promptboot_inference_q4_block_dot_avx2",
    "promptboot_inference_q4_block_dot_avx2:",
    "vmovdqu xmm0, xmmword ptr [rcx]",
    "vpand xmm1, xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "vpsrlw xmm0, xmm0, 4",
    "vpand xmm0, xmm0, xmmword ptr [rip + .Lpromptboot_q4_nibble_mask]",
    "vpsubb xmm1, xmm1, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "vpsubb xmm0, xmm0, xmmword ptr [rip + .Lpromptboot_q4_zero_point]",
    "vpmovsxbw ymm1, xmm1",
    "vpmovsxbw ymm0, xmm0",
    "vmovdqu xmm2, xmmword ptr [rdx]",
    "vmovdqu xmm3, xmmword ptr [rdx + 16]",
    "vpmovsxbw ymm2, xmm2",
    "vpmovsxbw ymm3, xmm3",
    "vpmaddwd ymm1, ymm1, ymm2",
    "vpmaddwd ymm0, ymm0, ymm3",
    "vpaddd ymm0, ymm0, ymm1",
    "vextracti128 xmm1, ymm0, 1",
    "vpaddd xmm0, xmm0, xmm1",
    "vpsrldq xmm1, xmm0, 8",
    "vpaddd xmm0, xmm0, xmm1",
    "vpsrldq xmm1, xmm0, 4",
    "vpaddd xmm0, xmm0, xmm1",
    "vmovd eax, xmm0",
    "vzeroupper",
    "ret",
    ".p2align 4",
    ".globl promptboot_inference_q8_block_dot_avx2",
    "promptboot_inference_q8_block_dot_avx2:",
    "vmovdqu ymm0, ymmword ptr [rcx]",
    "vmovdqu ymm1, ymmword ptr [rdx]",
    "vextracti128 xmm2, ymm0, 1",
    "vextracti128 xmm3, ymm1, 1",
    "vpmovsxbw ymm0, xmm0",
    "vpmovsxbw ymm1, xmm1",
    "vpmovsxbw ymm2, xmm2",
    "vpmovsxbw ymm3, xmm3",
    "vpmaddwd ymm0, ymm0, ymm1",
    "vpmaddwd ymm2, ymm2, ymm3",
    "vpaddd ymm0, ymm0, ymm2",
    "vextracti128 xmm1, ymm0, 1",
    "vpaddd xmm0, xmm0, xmm1",
    "vpsrldq xmm1, xmm0, 8",
    "vpaddd xmm0, xmm0, xmm1",
    "vpsrldq xmm1, xmm0, 4",
    "vpaddd xmm0, xmm0, xmm1",
    "vmovd eax, xmm0",
    "vzeroupper",
    "ret",
    ".p2align 4",
    ".Lpromptboot_q4_nibble_mask:",
    ".byte 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15",
    ".Lpromptboot_q4_zero_point:",
    ".byte 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8",
);

extern "efiapi" {
    #[link_name = "promptboot_inference_q4_block_dot"]
    fn inference_q4_block_dot_sse2(weights: *const u8, activation: *const u8) -> i32;
    #[link_name = "promptboot_inference_q8_block_dot"]
    fn inference_q8_block_dot_sse2(weights: *const u8, activation: *const u8) -> i32;
    #[link_name = "promptboot_inference_q4_block_dot_avx2"]
    fn inference_q4_block_dot_avx2(weights: *const u8, activation: *const u8) -> i32;
    #[link_name = "promptboot_inference_q8_block_dot_avx2"]
    fn inference_q8_block_dot_avx2(weights: *const u8, activation: *const u8) -> i32;
}

#[cfg(test)]
pub(crate) fn inference_q4_block_dot_for_test(
    backend: InferenceBackend,
    weights: &[u8; 16],
    activation: &[i8; 32],
) -> i32 {
    unsafe { q4_block_dot(backend)(weights.as_ptr(), activation.as_ptr().cast()) }
}

#[cfg(test)]
pub(crate) fn inference_q8_block_dot_for_test(
    backend: InferenceBackend,
    weights: &[i8; 32],
    activation: &[i8; 32],
) -> i32 {
    unsafe { q8_block_dot(backend)(weights.as_ptr().cast(), activation.as_ptr().cast()) }
}

type BlockDot = unsafe extern "efiapi" fn(*const u8, *const u8) -> i32;

fn q4_block_dot(backend: InferenceBackend) -> BlockDot {
    match backend {
        InferenceBackend::Sse2 => inference_q4_block_dot_sse2,
        InferenceBackend::Avx2 => inference_q4_block_dot_avx2,
    }
}

fn q8_block_dot(backend: InferenceBackend) -> BlockDot {
    match backend {
        InferenceBackend::Sse2 => inference_q8_block_dot_sse2,
        InferenceBackend::Avx2 => inference_q8_block_dot_avx2,
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_q4_matvec_rows_prepared(
    backend: InferenceBackend,
    weights: *const u8,
    output: *mut u8,
    staging: *mut u8,
    rows: usize,
    columns: usize,
    first: usize,
    end: usize,
) {
    debug_assert!(first <= end && end <= rows);
    let blocks_per_row = columns >> 5;
    let block_dot = q4_block_dot(backend);
    for row in first..end {
        let mut total = 0u32;
        let row_base = weights.add(row * blocks_per_row * 18);
        for block_index in 0..blocks_per_row {
            let block = row_base.add(block_index * 18);
            let activation = staging.add(block_index * 34);
            let integer = block_dot(block.add(2), activation.add(2));
            let weight_scale = half_to_f32_bits(u16::from_le(ptr::read_unaligned(block.cast())));
            let activation_scale =
                half_to_f32_bits(u16::from_le(ptr::read_unaligned(activation.cast())));
            total = add(
                total,
                mul(mul(from_i32(integer), weight_scale), activation_scale),
            );
        }
        write_bits(output, row, total);
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_q8_matvec_rows_prepared(
    backend: InferenceBackend,
    weights: *const u8,
    output: *mut u32,
    staging: *mut u8,
    rows: usize,
    columns: usize,
    first: usize,
    end: usize,
) {
    debug_assert!(first <= end && end <= rows);
    let blocks_per_row = columns >> 5;
    let block_dot = q8_block_dot(backend);
    for row in first..end {
        let mut total = 0u32;
        let row_base = weights.add(row * blocks_per_row * 34);
        for block_index in 0..blocks_per_row {
            let block = row_base.add(block_index * 34);
            let activation = staging.add(block_index * 34);
            let integer = block_dot(block.add(2), activation.add(2));
            let weight_scale = half_to_f32_bits(u16::from_le(ptr::read_unaligned(block.cast())));
            let activation_scale =
                half_to_f32_bits(u16::from_le(ptr::read_unaligned(activation.cast())));
            total = add(
                total,
                mul(from_i32(integer), mul(weight_scale, activation_scale)),
            );
        }
        ptr::write(output.add(row), total);
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_add_bias(values: *mut u8, bias: *const u8, words: usize) {
    for index in 0..words {
        write_bits(
            values,
            index,
            add(read_bits(values, index), read_bits(bias, index)),
        );
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_add_residual(
    residual: *const u8,
    update: *const u8,
    output: *mut u8,
    words: usize,
) {
    for index in 0..words {
        write_bits(
            output,
            index,
            add(read_bits(residual, index), read_bits(update, index)),
        );
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_rmsnorm(
    input: *const u8,
    weight: *const u8,
    output: *mut u8,
    words: usize,
) {
    let mut sum = 0u64;
    for index in 0..words {
        let value = read_bits(input, index);
        sum = f64_add(sum, f32_to_f64(mul(value, value)));
    }
    let mean = f64_to_f32(f64_div(sum, f64_from_u32(words as u32)));
    let inverse = div(0x3f80_0000, sqrt(add(mean, 0x3586_37bd)));
    for index in 0..words {
        write_bits(
            output,
            index,
            mul(
                mul(read_bits(input, index), inverse),
                read_bits(weight, index),
            ),
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn inference_expf4_in_place(values: *mut u32) {
    for lane in 0..4 {
        let value = ptr::read_unaligned(values.add(lane));
        if value == 0xff80_0000 {
            ptr::write_unaligned(values.add(lane), 0);
            continue;
        }
        let z = add(mul(value, 0x3fb8_aa3b), 0x4b40_0000);
        let n = sub(z, 0x4b40_0000);
        let inner = sub(value, mul(n, 0x3f31_7200));
        let b = sub(inner, mul(n, 0x35bf_be8e));
        let e = z << 23;
        let k = e.wrapping_add(0x3f80_0000);
        let absolute_n = n & 0x7fff_ffff;
        let u = mul(b, b);
        let first = add(mul(0x3c07_2010, b), 0x3d2b_9f17);
        let second = add(mul(0x3e2a_af33, b), 0x3eff_fedb);
        let polynomial = add(mul(first, u), second);
        let j = add(mul(polynomial, u), mul(0x3f7f_fff6, b));
        let result = if !greater(absolute_n, 0x42fc_0000) {
            add(mul(j, k), k)
        } else {
            let g = if !greater(n, 0) { 0x8200_0000u32 } else { 0 };
            let s1 = g.wrapping_add(0x7f00_0000);
            let s2 = e.wrapping_sub(g);
            let exceptional = mul(add(mul(s2, j), s2), s1);
            if greater(absolute_n, 0x4340_0000) {
                mul(s1, s1)
            } else {
                exceptional
            }
        };
        ptr::write_unaligned(values.add(lane), result);
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn copy_probability_words(source: *const u32, destination: *mut u8) {
    // rustc 1.97.1's x86_64-unknown-uefi lowering of four unaligned u32
    // stores (and of the equivalent copy_nonoverlapping) writes alternating
    // bytes.  Keep the fixed 16-byte transfer explicit and preserve the XMM
    // register so the surrounding integer-only Rust ABI remains unchanged.
    let mut saved = [0u8; 16];
    asm!(
        "movdqu xmmword ptr [{saved}], xmm0",
        "movdqu xmm0, xmmword ptr [{source}]",
        "movdqu xmmword ptr [{destination}], xmm0",
        "movdqu xmm0, xmmword ptr [{saved}]",
        saved = in(reg) saved.as_mut_ptr(),
        source = in(reg) source,
        destination = in(reg) destination,
        options(nostack, preserves_flags),
    );
}

#[cfg(test)]
pub(crate) fn copy_probability_words_for_test(source: &[u32; 4], destination: &mut [u8; 16]) {
    unsafe { copy_probability_words(source.as_ptr(), destination.as_mut_ptr()) }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum4(values: *const u32) -> u32 {
    let first = add(
        ptr::read_unaligned(values),
        ptr::read_unaligned(values.add(1)),
    );
    let second = add(
        ptr::read_unaligned(values.add(2)),
        ptr::read_unaligned(values.add(3)),
    );
    add(first, second)
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_rope_in_place(values: *mut u8, heads: usize, position: usize) {
    for head in 0..heads {
        let base = head * 64;
        for pair in 0..32 {
            let left = read_bits(values, base + pair);
            let right = read_bits(values, base + pair + 32);
            let cosine = inference_rope_bits(position, pair, 0);
            let sine = inference_rope_bits(position, pair, 1);
            let rotated_left = sub(mul(left, cosine), mul(right, sine));
            let rotated_right = add(mul(left, sine), mul(right, cosine));
            write_bits(values, base + pair, rotated_left);
            write_bits(values, base + pair + 32, rotated_right);
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn attention_scores_sse2(
    query: *const u8,
    kv: *const u8,
    scores: *mut u8,
    layer: usize,
    query_head: usize,
    kv_head: usize,
    positions: usize,
) {
    for position in 0..positions {
        let mut dot = 0u64;
        for component in 0..64 {
            let key = crate::inference::kv_word(layer, 0, position, kv_head, component);
            dot = f64_add(
                dot,
                f32_to_f64(mul(
                    read_bits(query, query_head * 64 + component),
                    read_bits(kv, key),
                )),
            );
        }
        write_bits(scores, position, mul(f64_to_f32(dot), 0x3e00_0000));
    }
}

#[target_feature(enable = "avx2")]
unsafe fn attention_maskload4(source: *const f32, remaining: usize) -> [u32; 4] {
    let mask = [
        if remaining > 0 { u32::MAX } else { 0 },
        if remaining > 1 { u32::MAX } else { 0 },
        if remaining > 2 { u32::MAX } else { 0 },
        if remaining > 3 { u32::MAX } else { 0 },
    ];
    let mut output = [0u32; 4];
    let mut saved = [0u8; 32];
    asm!(
        "vmovdqu xmmword ptr [{saved}], xmm0",
        "vmovdqu xmmword ptr [{saved} + 16], xmm1",
        "vmovdqu xmm0, xmmword ptr [{mask}]",
        "vmaskmovps xmm1, xmm0, xmmword ptr [{source}]",
        "vmovdqu xmmword ptr [{output}], xmm1",
        "vmovdqu xmm1, xmmword ptr [{saved} + 16]",
        "vmovdqu xmm0, xmmword ptr [{saved}]",
        mask = in(reg) mask.as_ptr(),
        source = in(reg) source,
        output = in(reg) output.as_mut_ptr(),
        saved = in(reg) saved.as_mut_ptr(),
        options(nostack, preserves_flags),
    );
    output
}

#[target_feature(enable = "avx2")]
unsafe fn attention_scores_avx2(
    query: *const u8,
    kv: *const u8,
    scores: *mut u8,
    layer: usize,
    query_head: usize,
    kv_head: usize,
    positions: usize,
    softmax_span: usize,
) {
    let scale = _mm_castsi128_ps(_mm_set1_epi32(0x3e00_0000u32 as i32));
    for position in (0..softmax_span).step_by(4) {
        let remaining = positions.saturating_sub(position).min(4);
        let mut low = _mm_setzero_pd();
        let mut high = _mm_setzero_pd();
        for component in 0..64 {
            let query_bits = read_bits(query, query_head * 64 + component);
            let query_lanes = _mm_castsi128_ps(_mm_set1_epi32(query_bits as i32));
            let key = crate::inference::kv_word(layer, 0, position, kv_head, component) * 4;
            let keys = if remaining == 4 {
                _mm_loadu_ps(kv.add(key).cast::<f32>())
            } else {
                let loaded = attention_maskload4(kv.add(key).cast::<f32>(), remaining);
                _mm_loadu_ps(loaded.as_ptr().cast::<f32>())
            };
            let products = _mm_mul_ps(query_lanes, keys);
            low = _mm_add_pd(low, _mm_cvtps_pd(products));
            high = _mm_add_pd(high, _mm_cvtps_pd(_mm_movehl_ps(products, products)));
        }
        let mut converted = [0u32; 8];
        _mm_storeu_ps(converted.as_mut_ptr().cast::<f32>(), _mm_cvtpd_ps(low));
        _mm_storeu_ps(
            converted.as_mut_ptr().add(4).cast::<f32>(),
            _mm_cvtpd_ps(high),
        );
        let dot_bits = [converted[0], converted[1], converted[4], converted[5]];
        let scaled = _mm_mul_ps(_mm_loadu_ps(dot_bits.as_ptr().cast::<f32>()), scale);
        _mm_storeu_ps(scores.add(position * 4).cast::<f32>(), scaled);
    }
    asm!("vzeroupper", options(nomem, nostack, preserves_flags));
}

#[target_feature(enable = "sse2")]
unsafe fn attention_values_sse2(
    kv: *const u8,
    probabilities: *const u8,
    output: *mut u8,
    layer: usize,
    query_head: usize,
    kv_head: usize,
    positions: usize,
) {
    for component in 0..64 {
        let mut total = 0u64;
        for position in 0..positions {
            let value = crate::inference::kv_word(layer, 1, position, kv_head, component);
            total = f64_add(
                total,
                f32_to_f64(mul(
                    read_bits(probabilities, position),
                    read_bits(kv, value),
                )),
            );
        }
        write_bits(output, query_head * 64 + component, f64_to_f32(total));
    }
}

#[target_feature(enable = "avx2")]
unsafe fn attention_values_avx2(
    kv: *const u8,
    probabilities: *const u8,
    output: *mut u8,
    layer: usize,
    query_head: usize,
    kv_head: usize,
    positions: usize,
) {
    for component in (0..64).step_by(4) {
        let mut low = _mm_setzero_pd();
        let mut high = _mm_setzero_pd();
        for position in 0..positions {
            let probability =
                _mm_castsi128_ps(_mm_set1_epi32(read_bits(probabilities, position) as i32));
            let value = crate::inference::kv_word(layer, 1, position, kv_head, component) * 4;
            let products = _mm_mul_ps(probability, _mm_loadu_ps(kv.add(value).cast::<f32>()));
            low = _mm_add_pd(low, _mm_cvtps_pd(products));
            high = _mm_add_pd(high, _mm_cvtps_pd(_mm_movehl_ps(products, products)));
        }
        let mut converted = [0u32; 8];
        _mm_storeu_ps(converted.as_mut_ptr().cast::<f32>(), _mm_cvtpd_ps(low));
        _mm_storeu_ps(
            converted.as_mut_ptr().add(4).cast::<f32>(),
            _mm_cvtpd_ps(high),
        );
        let total_bits = [converted[0], converted[1], converted[4], converted[5]];
        ptr::copy_nonoverlapping(
            total_bits.as_ptr(),
            output.cast::<u32>().add(query_head * 64 + component),
            4,
        );
    }
    asm!("vzeroupper", options(nomem, nostack, preserves_flags));
}

#[target_feature(enable = "sse2")]
unsafe fn inference_attention_inner<const AVX2: bool, const CAPTURE: bool>(
    query: *const u8,
    kv: *const u8,
    output: *mut u8,
    scores: *mut u8,
    layer: usize,
    current_position: usize,
    softmax_span: usize,
    raw_scores: *mut u32,
    normalized_probabilities: *mut u32,
) {
    let positions = current_position + 1;
    debug_assert!(
        softmax_span >= positions && softmax_span <= crate::inference::CONTEXT_LIMIT as usize
    );
    debug_assert_eq!(softmax_span & 3, 0);
    for query_head in 0..14 {
        let kv_head = unsigned_divide(query_head, 7);
        if AVX2 {
            attention_scores_avx2(
                query,
                kv,
                scores,
                layer,
                query_head,
                kv_head,
                positions,
                softmax_span,
            );
        } else {
            attention_scores_sse2(query, kv, scores, layer, query_head, kv_head, positions);
        }
        let mut maximum = 0xff80_0000;
        for position in 0..positions {
            let score = read_bits(scores, position);
            if greater(score, maximum) {
                maximum = score;
            }
        }
        for masked in positions..softmax_span {
            write_bits(scores, masked, 0xff80_0000);
        }
        if CAPTURE {
            ptr::copy_nonoverlapping(
                scores.cast::<u32>(),
                raw_scores.add(query_head * softmax_span),
                softmax_span,
            );
        }
        let mut sum = 0u64;
        let mut position = 0usize;
        while position < softmax_span {
            let mut probabilities = [0u32; 4];
            for lane in 0..4 {
                probabilities[lane] = sub(read_bits(scores, position + lane), maximum);
            }
            inference_expf4_in_place(probabilities.as_mut_ptr());
            copy_probability_words(probabilities.as_ptr(), scores.add(position * 4));
            sum = f64_add(sum, f32_to_f64(horizontal_sum4(probabilities.as_ptr())));
            position += 4;
        }
        let normalizer = f64_to_f32(f64_div(0x3ff0_0000_0000_0000, sum));
        for position in 0..positions {
            write_bits(
                scores,
                position,
                mul(read_bits(scores, position), normalizer),
            );
        }
        if CAPTURE {
            ptr::copy_nonoverlapping(
                scores.cast::<u32>(),
                normalized_probabilities.add(query_head * softmax_span),
                softmax_span,
            );
        }
        if AVX2 {
            attention_values_avx2(kv, scores, output, layer, query_head, kv_head, positions);
        } else {
            attention_values_sse2(kv, scores, output, layer, query_head, kv_head, positions);
        }
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_attention(
    backend: InferenceBackend,
    query: *const u8,
    kv: *const u8,
    output: *mut u8,
    scores: *mut u8,
    layer: usize,
    current_position: usize,
    softmax_span: usize,
) {
    match backend {
        InferenceBackend::Sse2 => inference_attention_inner::<false, false>(
            query,
            kv,
            output,
            scores,
            layer,
            current_position,
            softmax_span,
            ptr::null_mut(),
            ptr::null_mut(),
        ),
        InferenceBackend::Avx2 => inference_attention_inner::<true, false>(
            query,
            kv,
            output,
            scores,
            layer,
            current_position,
            softmax_span,
            ptr::null_mut(),
            ptr::null_mut(),
        ),
    }
}

#[cfg(test)]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_attention_captured_for_test(
    backend: InferenceBackend,
    query: *const u8,
    kv: *const u8,
    output: *mut u8,
    scores: *mut u8,
    layer: usize,
    current_position: usize,
    softmax_span: usize,
    raw_scores: *mut u32,
    probabilities: *mut u32,
) {
    match backend {
        InferenceBackend::Sse2 => inference_attention_inner::<false, true>(
            query,
            kv,
            output,
            scores,
            layer,
            current_position,
            softmax_span,
            raw_scores,
            probabilities,
        ),
        InferenceBackend::Avx2 => inference_attention_inner::<true, true>(
            query,
            kv,
            output,
            scores,
            layer,
            current_position,
            softmax_span,
            raw_scores,
            probabilities,
        ),
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_swiglu(
    gate: *const u8,
    up: *const u8,
    output: *mut u8,
    words: usize,
) {
    for index in (0..words).step_by(4) {
        let mut exponent_words = [0u32; 4];
        for lane in 0..4 {
            exponent_words[lane] = sub(0, read_bits(gate, index + lane));
        }
        inference_expf4_in_place(exponent_words.as_mut_ptr());
        for lane in 0..4 {
            let x = read_bits(gate, index + lane);
            let silu = div(x, add(0x3f80_0000, exponent_words[lane]));
            write_bits(output, index + lane, mul(silu, read_bits(up, index + lane)));
        }
    }
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_argmax(logits: *const u32, words: usize) -> u32 {
    let mut selected = 0usize;
    let mut best = ptr::read(logits);
    for index in 1..words {
        let value = ptr::read(logits.add(index));
        if greater(value, best) {
            best = value;
            selected = index;
        }
    }
    selected as u32
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_sample_top_k_top_p(
    logits: *const u32,
    words: usize,
    random_24: u32,
) -> u32 {
    const TOP_K: usize = 20;
    const TEMPERATURE: u32 = 0x3f33_3333;
    const TOP_P: u32 = 0x3f4c_cccd;
    const TWO_POW_24: u32 = 0x4b80_0000;

    let mut ids = [u32::MAX; TOP_K];
    let mut values = [0xff80_0000; TOP_K];
    for token in 0..words {
        let value = ptr::read(logits.add(token));
        let mut insertion = TOP_K;
        for at in 0..TOP_K {
            if ids[at] == u32::MAX || greater(value, values[at]) {
                insertion = at;
                break;
            }
        }
        if insertion < TOP_K {
            for at in (insertion + 1..TOP_K).rev() {
                ids[at] = ids[at - 1];
                values[at] = values[at - 1];
            }
            ids[insertion] = token as u32;
            values[insertion] = value;
        }
    }

    let maximum = values[0];
    let mut weights = [0u32; TOP_K];
    let mut total = 0u64;
    for at in 0..TOP_K {
        weights[at] = expf(div(sub(values[at], maximum), TEMPERATURE));
        total = f64_add(total, f32_to_f64(weights[at]));
    }
    let cutoff = mul(f64_to_f32(total), TOP_P);
    let mut kept = TOP_K;
    let mut kept_total = 0u64;
    for at in 0..TOP_K {
        kept_total = f64_add(kept_total, f32_to_f64(weights[at]));
        if !greater(cutoff, f64_to_f32(kept_total)) {
            kept = at + 1;
            break;
        }
    }

    let unit = div(from_u32(random_24 & 0x00ff_ffff), TWO_POW_24);
    let target = mul(f64_to_f32(kept_total), unit);
    let mut cumulative = 0u64;
    for at in 0..kept {
        cumulative = f64_add(cumulative, f32_to_f64(weights[at]));
        if greater(f64_to_f32(cumulative), target) {
            return ids[at];
        }
    }
    ids[kept - 1]
}

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inference_apply_repetition_penalty(
    logits: *mut u32,
    tokens: *const u32,
    token_count: usize,
    seen: *mut u8,
) {
    const PENALTY: u32 = 0x3f8c_cccd;
    for at in 0..token_count {
        let token = ptr::read(tokens.add(at)) as usize;
        let byte = seen.add(token / 8);
        let bit = 1u8 << (token & 7);
        if ptr::read(byte) & bit == 0 {
            ptr::write(byte, ptr::read(byte) | bit);
            let score = ptr::read(logits.add(token));
            let adjusted = if score & 0x8000_0000 != 0 && score & 0x7fff_ffff != 0 {
                mul(score, PENALTY)
            } else {
                div(score, PENALTY)
            };
            ptr::write(logits.add(token), adjusted);
        }
    }
    for at in 0..token_count {
        let token = ptr::read(tokens.add(at)) as usize;
        let byte = seen.add(token / 8);
        ptr::write(byte, ptr::read(byte) & !(1u8 << (token & 7)));
    }
}

/// Strict binary32 comparison for integer/status-only inference diagnostics.
/// Callers establish that both operands are finite before entering this ABI.
pub(crate) unsafe fn inference_greater(left: u32, right: u32) -> bool {
    greater(left, right)
}
