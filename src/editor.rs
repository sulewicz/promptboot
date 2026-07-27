pub const LINE_CAPACITY: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flow {
    Continue,
    PromptIndexOverflow,
}

pub trait Output {
    fn live_ascii(&mut self, byte: u8);
    fn erase_last(&mut self);
    fn accepted(&mut self, prompt_index: u64, line: &[u8]);
    fn rejected(&mut self, prompt_index: u64);
    fn prompt(&mut self, prompt_index: u64);
}

#[repr(C)]
pub struct Editor {
    buffer: [u8; LINE_CAPACITY],
    length: usize,
    prompt_index: u64,
    discard_until_enter: bool,
    pending_terminator: u8,
}

impl Editor {
    pub const fn new() -> Self {
        Self {
            buffer: [0; LINE_CAPACITY],
            length: 0,
            prompt_index: 1,
            discard_until_enter: false,
            pending_terminator: 0,
        }
    }

    pub const fn prompt_index(&self) -> u64 {
        self.prompt_index
    }
    pub const fn length(&self) -> usize {
        self.length
    }
    pub const fn discarding(&self) -> bool {
        self.discard_until_enter
    }
    pub const fn pending_terminator(&self) -> u8 {
        self.pending_terminator
    }

    pub fn process_key<O: Output>(&mut self, unicode_char: u16, output: &mut O) -> Flow {
        let current = if unicode_char <= u8::MAX as u16 {
            unicode_char as u8
        } else {
            0
        };
        if (self.pending_terminator == b'\r' && current == b'\n')
            || (self.pending_terminator == b'\n' && current == b'\r')
        {
            self.pending_terminator = 0;
            return Flow::Continue;
        }
        self.pending_terminator = 0;

        if self.discard_until_enter {
            if current == b'\r' || current == b'\n' {
                self.pending_terminator = current;
                self.discard_until_enter = false;
                self.prompt_index = match self.prompt_index.checked_add(1) {
                    Some(value) => value,
                    None => return Flow::PromptIndexOverflow,
                };
                output.prompt(self.prompt_index);
            }
            return Flow::Continue;
        }

        if current == b'\r' || current == b'\n' {
            let next = match self.prompt_index.checked_add(1) {
                Some(value) => value,
                None => {
                    self.clear();
                    return Flow::PromptIndexOverflow;
                }
            };
            output.accepted(self.prompt_index, &self.buffer[..self.length]);
            self.clear();
            self.prompt_index = next;
            output.prompt(self.prompt_index);
            self.pending_terminator = current;
            return Flow::Continue;
        }

        if current == 0x08 {
            if self.length != 0 {
                self.length -= 1;
                self.buffer[self.length] = 0;
                output.erase_last();
            }
            return Flow::Continue;
        }

        if (0x20..=0x7e).contains(&current) && unicode_char <= u8::MAX as u16 {
            if self.length == LINE_CAPACITY {
                if self.prompt_index.checked_add(1).is_none() {
                    self.clear();
                    return Flow::PromptIndexOverflow;
                }
                self.clear();
                self.discard_until_enter = true;
                output.rejected(self.prompt_index);
            } else {
                self.buffer[self.length] = current;
                self.length += 1;
                output.live_ascii(current);
            }
        }
        Flow::Continue
    }

    fn clear(&mut self) {
        self.buffer.fill(0);
        self.length = 0;
    }
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    #[derive(Debug, Eq, PartialEq)]
    enum Event {
        Live(u8),
        Erase,
        Accepted(u64, Vec<u8>),
        Rejected(u64),
        Prompt(u64),
    }

    #[derive(Default)]
    struct Log(Vec<Event>);
    impl Output for Log {
        fn live_ascii(&mut self, byte: u8) {
            self.0.push(Event::Live(byte));
        }
        fn erase_last(&mut self) {
            self.0.push(Event::Erase);
        }
        fn accepted(&mut self, index: u64, line: &[u8]) {
            self.0.push(Event::Accepted(index, line.to_vec()));
        }
        fn rejected(&mut self, index: u64) {
            self.0.push(Event::Rejected(index));
        }
        fn prompt(&mut self, index: u64) {
            self.0.push(Event::Prompt(index));
        }
    }

    fn key(editor: &mut Editor, log: &mut Log, value: u16) -> Flow {
        editor.process_key(value, log)
    }

    #[test]
    fn representative_printable_control_and_non_ascii_keys() {
        for value in [b' ' as u16, b'A' as u16, b'~' as u16] {
            let mut editor = Editor::new();
            let mut log = Log::default();
            assert_eq!(key(&mut editor, &mut log, value), Flow::Continue);
            assert_eq!(editor.length, 1);
            assert_eq!(log.0, vec![Event::Live(value as u8)]);
        }
        for value in [0, 0x01, 0x1f, 0x7f, 0xff, 0x100, u16::MAX] {
            let mut editor = Editor::new();
            let mut log = Log::default();
            assert_eq!(key(&mut editor, &mut log, value), Flow::Continue);
            assert_eq!(editor.length, 0);
            assert!(editor.buffer.iter().all(|byte| *byte == 0));
            assert!(log.0.is_empty());
        }
    }

    #[test]
    fn editing_empty_exact_limit_and_acceptance_zero_storage() {
        let mut editor = Editor::new();
        let mut log = Log::default();
        key(&mut editor, &mut log, 0x08);
        assert!(log.0.is_empty());
        for _ in 0..LINE_CAPACITY {
            key(&mut editor, &mut log, b'x' as u16);
        }
        assert_eq!(editor.length, LINE_CAPACITY);
        key(&mut editor, &mut log, 0x08);
        key(&mut editor, &mut log, b'y' as u16);
        assert_eq!(editor.length, LINE_CAPACITY);
        key(&mut editor, &mut log, b'\r' as u16);
        assert_eq!(editor.prompt_index, 2);
        assert_eq!(editor.length, 0);
        assert!(editor.buffer.iter().all(|byte| *byte == 0));
        assert!(matches!(log.0[LINE_CAPACITY], Event::Erase));
        assert!(
            matches!(&log.0[LINE_CAPACITY + 2], Event::Accepted(1, line) if line.len() == 512 && line[511] == b'y')
        );
    }

    #[test]
    fn terminator_pairs_and_repeated_identical_terminators_are_exact() {
        for (first, second) in [(b'\r', b'\n'), (b'\n', b'\r')] {
            let mut editor = Editor::new();
            let mut log = Log::default();
            key(&mut editor, &mut log, first as u16);
            assert_eq!(editor.pending_terminator(), first);
            key(&mut editor, &mut log, second as u16);
            assert_eq!(editor.prompt_index(), 2);
            assert_eq!(editor.pending_terminator(), 0);
            assert_eq!(
                log.0,
                vec![Event::Accepted(1, vec![]), Event::Prompt(2)],
                "pair={first:#x},{second:#x}"
            );
        }

        for terminator in [b'\r', b'\n'] {
            let mut editor = Editor::new();
            let mut log = Log::default();
            key(&mut editor, &mut log, terminator as u16);
            key(&mut editor, &mut log, terminator as u16);
            assert_eq!(editor.prompt_index(), 3);
            assert_eq!(editor.pending_terminator(), terminator);
            assert_eq!(
                log.0,
                vec![
                    Event::Accepted(1, vec![]),
                    Event::Prompt(2),
                    Event::Accepted(2, vec![]),
                    Event::Prompt(3),
                ],
                "terminator={terminator:#x}"
            );
        }
    }

    #[test]
    fn terminator_pair_after_discard_recovery_creates_only_one_prompt() {
        for (first, opposite) in [(b'\r', b'\n'), (b'\n', b'\r')] {
            let mut editor = Editor::new();
            let mut log = Log::default();
            for _ in 0..=LINE_CAPACITY {
                key(&mut editor, &mut log, b'x' as u16);
            }
            assert!(editor.discarding());
            assert_eq!(log.0.last(), Some(&Event::Rejected(1)));

            key(&mut editor, &mut log, first as u16);
            assert!(!editor.discarding());
            assert_eq!(editor.prompt_index(), 2);
            assert_eq!(editor.pending_terminator(), first);
            assert_eq!(log.0.last(), Some(&Event::Prompt(2)));
            let event_count = log.0.len();

            key(&mut editor, &mut log, opposite as u16);
            assert_eq!(log.0.len(), event_count);
            assert_eq!(editor.prompt_index(), 2);
            assert_eq!(editor.pending_terminator(), 0);
            assert_eq!(editor.length(), 0);
        }
    }

    #[test]
    fn overflow_zeroes_discards_backspace_and_recovers_cleanly() {
        let mut editor = Editor::new();
        let mut log = Log::default();
        for _ in 0..=LINE_CAPACITY {
            key(&mut editor, &mut log, b'x' as u16);
        }
        assert!(editor.discarding());
        assert_eq!(editor.length, 0);
        assert!(editor.buffer.iter().all(|byte| *byte == 0));
        assert_eq!(
            log.0
                .iter()
                .filter(|event| matches!(event, Event::Live(_)))
                .count(),
            512
        );
        assert_eq!(log.0.last(), Some(&Event::Rejected(1)));
        key(&mut editor, &mut log, 0x08);
        key(&mut editor, &mut log, b'z' as u16);
        assert_eq!(log.0.last(), Some(&Event::Rejected(1)));
        key(&mut editor, &mut log, b'\n' as u16);
        assert!(!editor.discarding());
        assert_eq!(editor.prompt_index, 2);
        assert_eq!(log.0.last(), Some(&Event::Prompt(2)));
        key(&mut editor, &mut log, b'o' as u16);
        key(&mut editor, &mut log, b'k' as u16);
        key(&mut editor, &mut log, b'\r' as u16);
        assert!(matches!(&log.0[log.0.len() - 2], Event::Accepted(2, line) if line == b"ok"));
        assert!(editor.buffer.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn checked_prompt_counter_overflow_clears_storage() {
        let mut overflow = Editor::new();
        overflow.prompt_index = u64::MAX;
        overflow.buffer[0] = b'x';
        overflow.length = 1;
        let mut overflow_log = Log::default();
        assert_eq!(
            key(&mut overflow, &mut overflow_log, b'\r' as u16),
            Flow::PromptIndexOverflow
        );
        assert!(overflow.buffer.iter().all(|byte| *byte == 0));
        assert!(overflow_log.0.is_empty());

        let mut long_overflow = Editor::new();
        long_overflow.prompt_index = u64::MAX;
        long_overflow.buffer.fill(b'x');
        long_overflow.length = LINE_CAPACITY;
        assert_eq!(
            key(&mut long_overflow, &mut overflow_log, b'x' as u16),
            Flow::PromptIndexOverflow
        );
        assert!(long_overflow.buffer.iter().all(|byte| *byte == 0));
        assert!(!long_overflow.discarding());
    }
}
