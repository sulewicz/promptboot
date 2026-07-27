pub const MAX_COLUMNS: usize = 256;
#[cfg(not(test))]
pub const HISTORY_LINES: usize = 2_048;
#[cfg(test)]
pub const HISTORY_LINES: usize = 64;

pub struct ConsoleHistory {
    cells: [u16; MAX_COLUMNS * HISTORY_LINES],
    lengths: [u16; HISTORY_LINES],
    hard_breaks: [bool; HISTORY_LINES],
    width: usize,
    rows: usize,
    head: usize,
    count: usize,
    column: usize,
    view_offset: usize,
}

impl ConsoleHistory {
    pub const fn new() -> Self {
        Self {
            cells: [0; MAX_COLUMNS * HISTORY_LINES],
            lengths: [0; HISTORY_LINES],
            hard_breaks: [false; HISTORY_LINES],
            width: 80,
            rows: 25,
            head: 0,
            count: 1,
            column: 0,
            view_offset: 0,
        }
    }

    pub fn configure(&mut self, columns: usize, rows: usize) {
        self.width = columns.clamp(1, MAX_COLUMNS);
        self.rows = rows.max(1);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.cells.fill(0);
        self.lengths.fill(0);
        self.hard_breaks.fill(false);
        self.head = 0;
        self.count = 1;
        self.column = 0;
        self.view_offset = 0;
    }

    pub fn return_to_bottom(&mut self) -> bool {
        let changed = self.view_offset != 0;
        self.view_offset = 0;
        changed
    }

    pub fn write(&mut self, units: &[u16]) {
        for unit in units.iter().copied() {
            match unit {
                0x0008 => {
                    self.column = self.column.saturating_sub(1);
                }
                0x000d => {
                    self.column = 0;
                }
                0x000a => {
                    self.hard_breaks[self.head] = true;
                    self.advance_line();
                }
                value => {
                    let at = self.head * MAX_COLUMNS + self.column;
                    self.cells[at] = value;
                    self.column += 1;
                    self.lengths[self.head] = self.lengths[self.head].max(self.column as u16);
                    if self.column == self.width {
                        self.advance_line();
                    }
                }
            }
        }
    }

    pub fn page_up(&mut self) -> bool {
        let maximum = self.count.saturating_sub(self.rows);
        let next = core::cmp::min(maximum, self.view_offset.saturating_add(self.rows));
        let changed = next != self.view_offset;
        self.view_offset = next;
        changed
    }

    pub fn page_down(&mut self) -> bool {
        let next = self.view_offset.saturating_sub(self.rows);
        let changed = next != self.view_offset;
        self.view_offset = next;
        changed
    }

    pub fn viewport_len(&self) -> usize {
        let end = self.count.saturating_sub(self.view_offset);
        let start = end.saturating_sub(self.rows);
        end - start
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub fn viewport_cursor(&self) -> (usize, usize) {
        if self.view_offset != 0 {
            let last = self.viewport_len().saturating_sub(1);
            return (
                self.viewport_line(last)
                    .0
                    .len()
                    .min(self.width.saturating_sub(1)),
                last,
            );
        }
        (self.column, self.viewport_len().saturating_sub(1))
    }

    pub fn viewport_line(&self, row: usize) -> (&[u16], bool) {
        let end = self.count.saturating_sub(self.view_offset);
        let start = end.saturating_sub(self.rows);
        let logical = start + row;
        let oldest = (self.head + HISTORY_LINES + 1 - self.count) % HISTORY_LINES;
        let physical = (oldest + logical) % HISTORY_LINES;
        let start = physical * MAX_COLUMNS;
        (
            &self.cells[start..start + self.lengths[physical] as usize],
            self.hard_breaks[physical],
        )
    }

    fn advance_line(&mut self) {
        self.head = (self.head + 1) % HISTORY_LINES;
        if self.count < HISTORY_LINES {
            self.count += 1;
        }
        let start = self.head * MAX_COLUMNS;
        self.cells[start..start + MAX_COLUMNS].fill(0);
        self.lengths[self.head] = 0;
        self.hard_breaks[self.head] = false;
        self.column = 0;
    }
}

impl Default for ConsoleHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;

    #[test]
    fn pages_clamp_and_live_output_returns_to_bottom() {
        let mut history = Box::new(ConsoleHistory::new());
        history.configure(8, 2);
        let text: std::vec::Vec<u16> = "one\r\ntwo\r\nthree\r\nfour".encode_utf16().collect();
        history.write(&text);

        assert_eq!(
            history.viewport_line(0).0,
            [
                b't' as u16,
                b'h' as u16,
                b'r' as u16,
                b'e' as u16,
                b'e' as u16
            ]
        );
        assert_eq!(
            history.viewport_line(1).0,
            [b'f' as u16, b'o' as u16, b'u' as u16, b'r' as u16]
        );
        assert!(history.page_up());
        assert_eq!(
            history.viewport_line(0).0,
            [b'o' as u16, b'n' as u16, b'e' as u16]
        );
        assert_eq!(
            history.viewport_line(1).0,
            [b't' as u16, b'w' as u16, b'o' as u16]
        );
        assert!(!history.page_up());
        assert!(history.page_down());
        assert!(!history.page_down());

        assert!(history.page_up());
        assert!(history.return_to_bottom());
        assert!(!history.return_to_bottom());
    }

    #[test]
    fn reset_removes_prior_lines_and_wrapping_is_not_a_hard_break() {
        let mut history = Box::new(ConsoleHistory::new());
        history.configure(4, 2);
        let text: std::vec::Vec<u16> = "abcdEF\r\nG".encode_utf16().collect();
        history.write(&text);
        assert_eq!(
            history.viewport_line(0),
            (&[b'E' as u16, b'F' as u16][..], true)
        );
        assert_eq!(history.viewport_line(1).0, [b'G' as u16]);

        history.reset();
        history.write(&[b'n' as u16, b'e' as u16, b'w' as u16]);
        assert_eq!(history.viewport_len(), 1);
        assert_eq!(
            history.viewport_line(0),
            (&[b'n' as u16, b'e' as u16, b'w' as u16][..], false)
        );
    }

    #[test]
    fn ring_discards_only_the_oldest_complete_lines() {
        let mut history = Box::new(ConsoleHistory::new());
        history.configure(8, 2);
        for index in 0..70 {
            history.write(&[b'A' as u16 + index % 26, 0x000d, 0x000a]);
        }
        while history.page_up() {}
        assert_eq!(history.viewport_line(0).0, [b'H' as u16]);
        assert_eq!(history.viewport_line(1).0, [b'I' as u16]);
    }

    #[test]
    fn cursor_stays_inside_configured_content_rows() {
        let mut history = Box::new(ConsoleHistory::new());
        history.configure(4, 2);
        history.write(&[b'a' as u16, b'b' as u16, b'c' as u16, b'd' as u16]);
        assert_eq!(history.viewport_cursor(), (0, 1));
        history.write(&[b'e' as u16, 0x000d, 0x000a, b'f' as u16]);
        assert_eq!(history.viewport_cursor(), (1, 1));
        assert_eq!(history.width(), 4);
        assert_eq!(history.rows(), 2);
    }

    #[test]
    fn paged_full_safe_width_line_preserves_every_cell_and_cursor() {
        let mut history = Box::new(ConsoleHistory::new());
        history.configure(4, 2);
        history.write(
            &"abcdefghijklmnopq"
                .encode_utf16()
                .collect::<std::vec::Vec<_>>(),
        );

        assert!(history.page_up());
        assert_eq!(history.viewport_line(1).0, "ijkl".encode_utf16().collect::<std::vec::Vec<_>>());
        assert_eq!(history.viewport_cursor(), (3, 1));
        assert!(history.page_down());
        assert_eq!(history.viewport_cursor(), (1, 1));
    }
}
