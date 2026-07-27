//! Small dependency-free SHA-256 used to authenticate the frozen model image.

use core::arch::global_asm;
use core::arch::x86_64::{__cpuid, __cpuid_count};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Backend {
    Unresolved,
    Scalar,
    ShaNi,
}

pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    used: usize,
    bytes: u64,
    backend: Backend,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            used: 0,
            bytes: 0,
            backend: Backend::Unresolved,
        }
    }

    pub fn update(&mut self, mut input: &[u8]) {
        self.bytes = self.bytes.wrapping_add(input.len() as u64);
        if self.used != 0 {
            let take = core::cmp::min(64 - self.used, input.len());
            self.block[self.used..self.used + take].copy_from_slice(&input[..take]);
            self.used += take;
            input = &input[take..];
            if self.used == 64 {
                let block = self.block;
                self.compress(&block);
                self.used = 0;
            } else {
                return;
            }
        }
        while input.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[..64]);
            self.compress(&block);
            input = &input[64..];
        }
        self.block[..input.len()].copy_from_slice(input);
        self.used = input.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bit_length = self.bytes.wrapping_mul(8);
        self.block[self.used] = 0x80;
        self.used += 1;
        if self.used > 56 {
            self.block[self.used..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
        } else {
            self.block[self.used..56].fill(0);
        }
        self.block[56..64].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.block;
        self.compress(&block);
        let mut output = [0u8; 32];
        for (chunk, word) in output.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        output
    }

    fn compress(&mut self, block: &[u8; 64]) {
        if self.backend == Backend::Unresolved {
            self.backend = if sha_ni_available() {
                Backend::ShaNi
            } else {
                Backend::Scalar
            };
        }
        let schedule = message_schedule(block);
        match self.backend {
            Backend::Scalar => compress_scalar(&mut self.state, &schedule),
            Backend::ShaNi => unsafe { compress_sha_ni(&mut self.state, &schedule) },
            Backend::Unresolved => unreachable!(),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Unresolved => "unresolved",
            Backend::Scalar => "scalar",
            Backend::ShaNi => "sha_ni",
        }
    }

    #[cfg(test)]
    pub(crate) fn force_scalar_for_test(&mut self) {
        assert!(self.backend == Backend::Unresolved);
        self.backend = Backend::Scalar;
    }

    #[cfg(test)]
    pub(crate) fn force_sha_ni_for_test(&mut self) -> bool {
        assert!(self.backend == Backend::Unresolved);
        if sha_ni_available() {
            self.backend = Backend::ShaNi;
            true
        } else {
            false
        }
    }
}

pub fn sha_ni_available() -> bool {
    let max_basic_leaf = __cpuid(0).eax;
    let leaf7_ebx = if max_basic_leaf >= 7 {
        __cpuid_count(7, 0).ebx
    } else {
        0
    };
    select_sha_ni(max_basic_leaf, leaf7_ebx)
}

fn select_sha_ni(max_basic_leaf: u32, leaf7_ebx: u32) -> bool {
    max_basic_leaf >= 7 && leaf7_ebx & (1 << 29) != 0
}

fn message_schedule(block: &[u8; 64]) -> [u32; 64] {
    let mut schedule = [0u32; 64];
    for (index, chunk) in block.chunks_exact(4).enumerate() {
        schedule[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    schedule
}

fn compress_scalar(state: &mut [u32; 8], schedule: &[u32; 64]) {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choice = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(choice)
            .wrapping_add(K[index])
            .wrapping_add(schedule[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(value);
    }
}

#[target_feature(enable = "sha,sse2")]
unsafe fn compress_sha_ni(state: &mut [u32; 8], schedule: &[u32; 64]) {
    let mut round_words = [0u32; 64];
    for index in 0..64 {
        round_words[index] = schedule[index].wrapping_add(K[index]);
    }
    promptboot_sha256_rounds(state.as_mut_ptr(), round_words.as_ptr());
}

unsafe extern "sysv64" {
    fn promptboot_sha256_rounds(state: *mut u32, round_words: *const u32);
}

global_asm!(
    r#"
    .text
    .p2align 4
    .globl promptboot_sha256_rounds
promptboot_sha256_rounds:
    movdqu xmm1, xmmword ptr [rdi]
    movdqu xmm2, xmmword ptr [rdi + 16]
    pshufd xmm3, xmm1, 0xb1
    pshufd xmm2, xmm2, 0x1b
    movdqa xmm1, xmm2
    psrldq xmm1, 8
    movdqa xmm6, xmm3
    pslldq xmm6, 8
    por xmm1, xmm6
    movdqa xmm6, xmm3
    psrldq xmm6, 8
    punpcklqdq xmm2, xmm6
    movdqa xmm4, xmm1
    movdqa xmm5, xmm2
    xor eax, eax
1:
    movdqu xmm0, xmmword ptr [rsi + rax]
    sha256rnds2 xmm2, xmm1
    pshufd xmm0, xmm0, 0x0e
    sha256rnds2 xmm1, xmm2
    add eax, 16
    cmp eax, 256
    jne 1b
    paddd xmm1, xmm4
    paddd xmm2, xmm5
    pshufd xmm3, xmm1, 0x1b
    pshufd xmm2, xmm2, 0xb1
    movdqa xmm6, xmm3
    psrldq xmm6, 8
    punpcklqdq xmm6, xmm2
    psrldq xmm2, 8
    punpcklqdq xmm3, xmm2
    movdqu xmmword ptr [rdi], xmm3
    movdqu xmmword ptr [rdi + 16], xmm6
    ret
"#
);

#[cfg(test)]
mod backend_tests {
    use super::select_sha_ni;

    #[test]
    fn cpuid_selection_requires_leaf7_sha_bit() {
        assert!(!select_sha_ni(6, 1 << 29));
        assert!(!select_sha_ni(7, 0));
        assert!(select_sha_ni(7, 1 << 29));
    }
}

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut state = Sha256::new();
    state.update(bytes);
    state.finish()
}
