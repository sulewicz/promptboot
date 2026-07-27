//! Allocation-free REPL history and strict incremental UTF-8 contracts.

pub const CONTEXT_TOKENS: usize = promptboot_core::CONTEXT_LIMIT as usize;

pub fn is_event_toggle_command(line: &[u8]) -> bool {
    line == b"/events"
}

pub fn is_help_command(line: &[u8]) -> bool {
    line == b"/help"
}

pub fn is_new_command(line: &[u8]) -> bool {
    line == b"/new"
}

pub const GENERATION_RESERVE: usize = 1_024;
pub const PROMPT_LIMIT: usize = CONTEXT_TOKENS - 1;
pub const REPETITION_BITMAP_BYTES: usize = promptboot_core::LOGIT_WORDS.div_ceil(8);
pub const SESSION_BYTES: usize = CONTEXT_TOKENS * 2 * core::mem::size_of::<u32>()
    + GENERATION_RESERVE * 48
    + REPETITION_BITMAP_BYTES;
pub const IM_END: u32 = 151_645;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedTokenFailure {
    pub generated: u32,
    pub partial: bool,
}

pub const fn selected_token_failure(
    completed_tokens: u32,
    visible_tokens: u32,
) -> SelectedTokenFailure {
    SelectedTokenFailure {
        generated: completed_tokens.saturating_add(1),
        partial: visible_tokens != 0,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryDecision {
    Use { prompt_tokens: usize },
    RetryFresh,
    Reset { prompt_tokens: usize },
    Reject { fresh_prompt_tokens: usize },
}

pub const fn decide_history(
    has_history: bool,
    prospective_tokens: usize,
    fresh_prompt_tokens: Option<usize>,
) -> HistoryDecision {
    if prospective_tokens <= PROMPT_LIMIT {
        HistoryDecision::Use {
            prompt_tokens: prospective_tokens,
        }
    } else if has_history {
        match fresh_prompt_tokens {
            None => HistoryDecision::RetryFresh,
            Some(fresh_prompt_tokens) if fresh_prompt_tokens <= PROMPT_LIMIT => {
                HistoryDecision::Reset {
                    prompt_tokens: fresh_prompt_tokens,
                }
            }
            Some(fresh_prompt_tokens) => HistoryDecision::Reject {
                fresh_prompt_tokens,
            },
        }
    } else {
        HistoryDecision::Reject {
            fresh_prompt_tokens: match fresh_prompt_tokens {
                Some(value) => value,
                None => prospective_tokens,
            },
        }
    }
}

pub fn commit_eos(
    committed: &mut [u32; CONTEXT_TOKENS],
    working: &[u32; CONTEXT_TOKENS],
    working_len: usize,
) -> Result<usize, ()> {
    if working_len == 0 || working_len > CONTEXT_TOKENS || working[working_len - 1] != IM_END {
        return Err(());
    }
    committed.fill(0);
    committed[..working_len].copy_from_slice(&working[..working_len]);
    Ok(working_len)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Utf8Error {
    Invalid,
    Incomplete,
    Nul,
    Capacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Utf8Decoder {
    carry: [u8; 3],
    carry_len: u8,
}

impl Utf8Decoder {
    pub const fn new() -> Self {
        Self {
            carry: [0; 3],
            carry_len: 0,
        }
    }

    pub const fn carry_len(&self) -> usize {
        self.carry_len as usize
    }

    pub fn reset(&mut self) {
        self.carry = [0; 3];
        self.carry_len = 0;
    }

    pub fn finish(&self) -> Result<(), Utf8Error> {
        if self.carry_len == 0 {
            Ok(())
        } else {
            Err(Utf8Error::Incomplete)
        }
    }

    pub fn push(&mut self, piece: &[u8], output: &mut [u16]) -> Result<usize, Utf8Error> {
        if piece.len() > 128 {
            return Err(Utf8Error::Capacity);
        }
        let mut bytes = [0u8; 131];
        let old_carry = self.carry_len as usize;
        bytes[..old_carry].copy_from_slice(&self.carry[..old_carry]);
        bytes[old_carry..old_carry + piece.len()].copy_from_slice(piece);
        let length = old_carry + piece.len();
        let mut staged = [0u16; 132];
        let mut units = 0usize;
        let mut at = 0usize;
        let mut next_carry = [0u8; 3];
        let mut next_carry_len = 0usize;

        while at < length {
            let first = bytes[at];
            let needed = match first {
                0 => return Err(Utf8Error::Nul),
                1..=0x7f => 1,
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return Err(Utf8Error::Invalid),
            };
            if needed >= 2 && length - at >= 2 {
                let second = bytes[at + 1];
                if second & 0xc0 != 0x80
                    || (first == 0xe0 && second < 0xa0)
                    || (first == 0xed && second >= 0xa0)
                    || (first == 0xf0 && second < 0x90)
                    || (first == 0xf4 && second >= 0x90)
                {
                    return Err(Utf8Error::Invalid);
                }
            }
            if length - at < needed {
                for index in at + 1..length {
                    if bytes[index] & 0xc0 != 0x80 {
                        return Err(Utf8Error::Invalid);
                    }
                }
                next_carry_len = length - at;
                next_carry[..next_carry_len].copy_from_slice(&bytes[at..length]);
                break;
            }
            for index in 1..needed {
                if bytes[at + index] & 0xc0 != 0x80 {
                    return Err(Utf8Error::Invalid);
                }
            }
            let point = match needed {
                1 => first as u32,
                2 => ((first & 0x1f) as u32) << 6 | (bytes[at + 1] & 0x3f) as u32,
                3 => {
                    let second = bytes[at + 1];
                    if (first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0) {
                        return Err(Utf8Error::Invalid);
                    }
                    ((first & 0x0f) as u32) << 12
                        | ((second & 0x3f) as u32) << 6
                        | (bytes[at + 2] & 0x3f) as u32
                }
                4 => {
                    let second = bytes[at + 1];
                    if (first == 0xf0 && second < 0x90) || (first == 0xf4 && second >= 0x90) {
                        return Err(Utf8Error::Invalid);
                    }
                    ((first & 7) as u32) << 18
                        | ((second & 0x3f) as u32) << 12
                        | ((bytes[at + 2] & 0x3f) as u32) << 6
                        | (bytes[at + 3] & 0x3f) as u32
                }
                _ => unreachable!(),
            };
            if point == 0 || (0xd800..=0xdfff).contains(&point) || point > 0x10ffff {
                return Err(if point == 0 {
                    Utf8Error::Nul
                } else {
                    Utf8Error::Invalid
                });
            }
            if point <= 0xffff {
                if units == staged.len() {
                    return Err(Utf8Error::Capacity);
                }
                staged[units] = point as u16;
                units += 1;
            } else {
                if units + 2 > staged.len() {
                    return Err(Utf8Error::Capacity);
                }
                let value = point - 0x10000;
                staged[units] = 0xd800 | (value >> 10) as u16;
                staged[units + 1] = 0xdc00 | (value & 0x3ff) as u16;
                units += 2;
            }
            at += needed;
        }
        if output.len() < units {
            return Err(Utf8Error::Capacity);
        }
        output[..units].copy_from_slice(&staged[..units]);
        self.carry = next_carry;
        self.carry_len = next_carry_len as u8;
        Ok(units)
    }
}

impl Default for Utf8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn render_console_text(
    input: &[u16],
    previous_was_cr: &mut bool,
    output: &mut [u16],
) -> Result<usize, Utf8Error> {
    let mut needed = input.len();
    let mut preceding_cr = *previous_was_cr;
    for unit in input.iter().copied() {
        if unit == b'\n' as u16 && !preceding_cr {
            needed = needed.checked_add(1).ok_or(Utf8Error::Capacity)?;
        }
        preceding_cr = unit == b'\r' as u16;
    }
    if output.len() < needed {
        return Err(Utf8Error::Capacity);
    }

    let mut written = 0;
    preceding_cr = *previous_was_cr;
    for unit in input.iter().copied() {
        if unit == b'\n' as u16 && !preceding_cr {
            output[written] = b'\r' as u16;
            written += 1;
        }
        output[written] = unit;
        written += 1;
        preceding_cr = unit == b'\r' as u16;
    }
    *previous_was_cr = preceding_cr;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_limits_match_the_native_model_contract() {
        assert_eq!(CONTEXT_TOKENS, 32_768);
        assert_eq!(GENERATION_RESERVE, 1_024);
        assert_eq!(PROMPT_LIMIT, 32_767);
        assert_eq!(REPETITION_BITMAP_BYTES, 18_992);
        assert_eq!(SESSION_BYTES, 330_288);
    }

    #[test]
    fn commands_match_only_complete_lines() {
        assert!(is_event_toggle_command(b"/events"));
        assert!(is_help_command(b"/help"));
        assert!(is_new_command(b"/new"));
        assert!(!is_event_toggle_command(b"/events "));
        assert!(!is_help_command(b"/helps"));
        assert!(!is_new_command(b"/new "));
    }

    #[test]
    fn history_decisions_cover_use_retry_fresh_reset_and_reject() {
        assert_eq!(
            decide_history(true, PROMPT_LIMIT, None),
            HistoryDecision::Use {
                prompt_tokens: PROMPT_LIMIT
            }
        );
        assert_eq!(
            decide_history(true, PROMPT_LIMIT + 1, None),
            HistoryDecision::RetryFresh
        );
        assert_eq!(
            decide_history(true, PROMPT_LIMIT + 1, Some(PROMPT_LIMIT)),
            HistoryDecision::Reset {
                prompt_tokens: PROMPT_LIMIT
            }
        );
        assert_eq!(
            decide_history(false, PROMPT_LIMIT + 1, None),
            HistoryDecision::Reject {
                fresh_prompt_tokens: PROMPT_LIMIT + 1
            }
        );
        assert_eq!(
            decide_history(
                true,
                PROMPT_LIMIT + 1,
                Some(PROMPT_LIMIT + 1),
            ),
            HistoryDecision::Reject {
                fresh_prompt_tokens: PROMPT_LIMIT + 1
            }
        );
    }

    #[test]
    fn eos_commit_is_transactional() {
        let mut committed = [7u32; CONTEXT_TOKENS];
        let before = committed;
        let mut working = [0u32; CONTEXT_TOKENS];
        working[..3].copy_from_slice(&[1, 2, 3]);
        assert_eq!(commit_eos(&mut committed, &working, 3), Err(()));
        assert_eq!(committed, before);
        working[2] = IM_END;
        assert_eq!(commit_eos(&mut committed, &working, 3), Ok(3));
        assert_eq!(&committed[..3], &[1, 2, IM_END]);
        assert!(committed[3..].iter().all(|token| *token == 0));
    }

    #[test]
    fn selected_token_failure_tracks_visibility_and_saturates() {
        assert_eq!(
            selected_token_failure(0, 0),
            SelectedTokenFailure {
                generated: 1,
                partial: false,
            }
        );
        assert_eq!(
            selected_token_failure(0, 1),
            SelectedTokenFailure {
                generated: 1,
                partial: true,
            }
        );
        assert_eq!(
            selected_token_failure(u32::MAX, 0),
            SelectedTokenFailure {
                generated: u32::MAX,
                partial: false,
            }
        );
    }

    #[test]
    fn console_text_renders_lf_as_crlf_across_piece_boundaries() {
        let mut output = [0u16; 16];
        let mut previous_was_cr = false;
        let written = render_console_text(
            &[b'a' as u16, b'\n' as u16, b'\n' as u16],
            &mut previous_was_cr,
            &mut output,
        )
        .unwrap();
        assert_eq!(
            &output[..written],
            &[
                b'a' as u16,
                b'\r' as u16,
                b'\n' as u16,
                b'\r' as u16,
                b'\n' as u16,
            ]
        );
        assert!(!previous_was_cr);

        let written =
            render_console_text(&[b'\r' as u16], &mut previous_was_cr, &mut output).unwrap();
        assert_eq!(&output[..written], &[b'\r' as u16]);
        assert!(previous_was_cr);
        let written = render_console_text(
            &[b'\n' as u16, b'b' as u16],
            &mut previous_was_cr,
            &mut output,
        )
        .unwrap();
        assert_eq!(&output[..written], &[b'\n' as u16, b'b' as u16]);
        assert!(!previous_was_cr);
    }

    #[test]
    fn console_text_capacity_failure_is_transactional() {
        let mut output = [0xa5a5u16; 1];
        let before = output;
        let mut previous_was_cr = false;
        assert_eq!(
            render_console_text(&[b'\n' as u16], &mut previous_was_cr, &mut output),
            Err(Utf8Error::Capacity)
        );
        assert_eq!(output, before);
        assert!(!previous_was_cr);
    }

    #[test]
    fn utf8_all_split_points_and_surrogates_are_exact() {
        let cases: &[(&[u8], &[u16])] = &[
            (b"A", &[0x41]),
            (&[0xc2, 0xa2], &[0x00a2]),
            (&[0xe2, 0x82, 0xac], &[0x20ac]),
            (&[0xf0, 0x9f, 0x98, 0x80], &[0xd83d, 0xde00]),
        ];
        for (input, expected) in cases {
            for split in 0..=input.len() {
                let mut decoder = Utf8Decoder::new();
                let mut output = [0xaaaau16; 8];
                let first = decoder.push(&input[..split], &mut output).unwrap();
                let second = decoder.push(&input[split..], &mut output[first..]).unwrap();
                decoder.finish().unwrap();
                assert_eq!(&output[..first + second], *expected, "split={split}");
            }
        }
    }

    #[test]
    fn utf8_invalids_incomplete_nul_and_capacity_preserve_state_and_output() {
        let invalid: &[&[u8]] = &[
            &[0x80],
            &[0xc0, 0x80],
            &[0xc1, 0xbf],
            &[0xf5, 0x80, 0x80, 0x80],
            &[0xff],
            &[0xe0, 0x80, 0x80],
            &[0xed, 0xa0, 0x80],
            &[0xf0, 0x80, 0x80, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
        ];
        for input in invalid {
            let mut decoder = Utf8Decoder::new();
            let before = decoder;
            let mut output = [0xaaaau16; 8];
            assert_eq!(decoder.push(input, &mut output), Err(Utf8Error::Invalid));
            assert_eq!(decoder, before);
            assert!(output.iter().all(|unit| *unit == 0xaaaa));
        }
        let mut decoder = Utf8Decoder::new();
        let mut output = [0xaaaau16; 8];
        assert_eq!(decoder.push(&[0xe2, 0x82], &mut output), Ok(0));
        assert_eq!(decoder.carry_len(), 2);
        assert_eq!(decoder.finish(), Err(Utf8Error::Incomplete));
        let before = decoder;
        assert_eq!(decoder.push(&[0], &mut output), Err(Utf8Error::Invalid));
        assert_eq!(decoder, before);
        let mut fresh = Utf8Decoder::new();
        assert_eq!(fresh.push(&[0], &mut output), Err(Utf8Error::Nul));
        assert_eq!(fresh.push(b"A", &mut []), Err(Utf8Error::Capacity));
        assert_eq!(fresh, Utf8Decoder::new());
    }
}
