use crate::console_history::MAX_COLUMNS;

pub const STATUS_PERIOD_100NS: u64 = 5_000_000;
pub const TIMER_CANCEL: u32 = 0;
pub const TIMER_PERIODIC: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitSource {
    Key,
    Timer,
}

pub const fn wait_source(status: usize, index: usize) -> Result<WaitSource, usize> {
    if status != 0 {
        Err(status)
    } else {
        match index {
            0 => Ok(WaitSource::Key),
            1 => Ok(WaitSource::Timer),
            _ => Err(usize::MAX),
        }
    }
}

pub const fn first_error(first: Option<usize>, status: usize) -> Option<usize> {
    if first.is_some() {
        first
    } else if status == 0 {
        None
    } else {
        Some(status)
    }
}

pub fn render_status_row(
    old_attribute: usize,
    old_column: usize,
    old_row: usize,
    status_row: usize,
    units: &[u16],
    mut set_cursor: impl FnMut(usize, usize) -> usize,
    mut set_attribute: impl FnMut(usize) -> usize,
    mut output: impl FnMut(&[u16]) -> usize,
) -> Result<(), usize> {
    let mut first = None;
    let cursor_status = set_cursor(0, status_row);
    first = first_error(first, cursor_status);
    if first.is_none() {
        let attribute_status = set_attribute(inverted_attribute(old_attribute as i32));
        first = first_error(first, attribute_status);
    }
    if first.is_none() {
        first = first_error(first, output(units));
    }
    first = first_error(first, set_attribute(old_attribute));
    first = first_error(first, set_cursor(old_column, old_row));
    match first {
        Some(status) => Err(status),
        None => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerFailure {
    Create(usize),
    Set(usize),
    Cancel(usize),
    Close(usize),
}

pub fn start_periodic_timer<E: Copy>(
    mut create: impl FnMut() -> Result<E, usize>,
    mut set_timer: impl FnMut(E, u32, u64) -> usize,
    mut close: impl FnMut(E) -> usize,
) -> Result<E, TimerFailure> {
    let event = create().map_err(TimerFailure::Create)?;
    let status = set_timer(event, TIMER_PERIODIC, STATUS_PERIOD_100NS);
    if status != 0 {
        let _ = cleanup_periodic_timer(event, &mut set_timer, &mut close);
        return Err(TimerFailure::Set(status));
    }
    Ok(event)
}

pub fn cleanup_periodic_timer<E: Copy>(
    event: E,
    mut set_timer: impl FnMut(E, u32, u64) -> usize,
    mut close: impl FnMut(E) -> usize,
) -> Result<(), TimerFailure> {
    let cancel = set_timer(event, TIMER_CANCEL, 0);
    let close_status = close(event);
    if cancel != 0 {
        Err(TimerFailure::Cancel(cancel))
    } else if close_status != 0 {
        Err(TimerFailure::Close(close_status))
    } else {
        Ok(())
    }
}

pub fn primary_or_cleanup<T, E>(
    primary: Result<T, E>,
    cleanup: Result<(), E>,
) -> Result<T, E> {
    match primary {
        Err(error) => Err(error),
        Ok(value) => cleanup.map(|()| value),
    }
}

pub fn wait_key_first<E: Copy>(
    key: E,
    timer: E,
    mut wait: impl FnMut(&[E; 2], &mut usize) -> usize,
) -> Result<WaitSource, usize> {
    let events = [key, timer];
    let mut index = usize::MAX;
    let status = wait(&events, &mut index);
    wait_source(status, index)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatusMemoryFailure;

pub fn measure_status_memory(
    mut measure: impl FnMut() -> Option<u64>,
) -> Result<u64, StatusMemoryFailure> {
    measure().ok_or(StatusMemoryFailure)
}

#[derive(Debug, Eq, PartialEq)]
pub enum RuntimeMemoryFailure<E> {
    Memory,
    Emit(E),
}

pub fn with_runtime_memory<T, E>(
    measure: impl FnMut() -> Option<u64>,
    mut emit: impl FnMut(u64) -> Result<T, E>,
) -> Result<(u64, T), RuntimeMemoryFailure<E>> {
    let memory = measure_status_memory(measure).map_err(|_| RuntimeMemoryFailure::Memory)?;
    let emitted = emit(memory).map_err(RuntimeMemoryFailure::Emit)?;
    Ok((memory, emitted))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnCompletion {
    Success,
    RecoverableFailure,
    FatalFailure,
}

pub const fn waiting_render_failure_is_terminal(turn: TurnCompletion) -> bool {
    !matches!(turn, TurnCompletion::FatalFailure)
}

pub fn is_serial_evidence_line(units: &[u16]) -> bool {
    for prefix in [
        b"PROMPTBOOT_EVENT".as_slice(),
        b"events: on".as_slice(),
        b"events: off".as_slice(),
    ] {
        if units.len() >= prefix.len() {
            let mut index = 0usize;
            while index < prefix.len() && units[index] == prefix[index] as u16 {
                index += 1;
            }
            if index == prefix.len() {
                return true;
            }
        }
    }
    false
}

pub fn is_event_boundary_line(units: &[u16]) -> bool {
    for prefix in [b"events: on".as_slice(), b"events: off".as_slice()] {
        if units.len() >= prefix.len() {
            let mut index = 0usize;
            while index < prefix.len() && units[index] == prefix[index] as u16 {
                index += 1;
            }
            if index == prefix.len() {
                return true;
            }
        }
    }
    false
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeState {
    Loading,
    Waiting,
    Inferring,
}

impl RuntimeState {
    pub const fn label(self) -> &'static [u8] {
        match self {
            Self::Loading => b"loading",
            Self::Waiting => b"waiting",
            Self::Inferring => b"inferring",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuMode {
    Serial,
    Mp,
}

impl CpuMode {
    pub const fn label(self) -> &'static [u8] {
        match self {
            Self::Serial => b"serial",
            Self::Mp => b"mp",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuTopology {
    pub mode: CpuMode,
    pub active: u32,
    pub workers: u32,
    pub enabled: Option<u32>,
    pub total: Option<u32>,
}

impl CpuTopology {
    pub const UNKNOWN_SERIAL: Self = Self {
        mode: CpuMode::Serial,
        active: 1,
        workers: 0,
        enabled: None,
        total: None,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeSnapshot {
    pub state: RuntimeState,
    pub cpu: CpuTopology,
    pub memory_baseline: u64,
    pub memory_free: u64,
    pub committed_tokens: u32,
    pub engine_position: u32,
    pub context_limit: u32,
    pub generation_reserve: u32,
}

impl RuntimeSnapshot {
    pub const fn consumed_since_boot(self) -> u64 {
        self.memory_baseline.saturating_sub(self.memory_free)
    }

    pub const fn bar_context(self) -> u32 {
        match self.state {
            RuntimeState::Inferring => self.engine_position,
            RuntimeState::Loading | RuntimeState::Waiting => self.committed_tokens,
        }
    }
}

pub const fn bounded_status_geometry(columns: usize, rows: usize) -> Option<(usize, usize)> {
    if columns < 2 || rows < 2 {
        None
    } else {
        let bounded_columns = if columns > MAX_COLUMNS {
            MAX_COLUMNS
        } else {
            columns
        };
        Some((bounded_columns, rows))
    }
}

pub const fn inverted_attribute(attribute: i32) -> usize {
    let foreground = attribute as usize & 0x0f;
    let background = (attribute as usize >> 4) & 0x07;
    background | ((foreground & 0x07) << 4)
}

pub fn native_scrolls_for_output(
    units: &[u16],
    mut column: usize,
    mut row: usize,
    width: usize,
    content_rows: usize,
) -> Option<usize> {
    if width == 0 || content_rows == 0 || column >= width || row >= content_rows {
        return None;
    }
    let mut scrolls = 0usize;
    for unit in units {
        match *unit {
            0x0008 => column = column.saturating_sub(1),
            0x000d => column = 0,
            0x000a => {
                row += 1;
                if row >= content_rows {
                    row = content_rows - 1;
                    scrolls += 1;
                }
            }
            _ => {
                column += 1;
                if column == width {
                    column = 0;
                    row += 1;
                    if row >= content_rows {
                        row = content_rows - 1;
                        scrolls += 1;
                    }
                }
            }
        }
    }
    Some(scrolls)
}

pub fn write_content_with_wrap(
    units: &[u16],
    mut column: usize,
    width: usize,
    mut output: impl FnMut(&[u16]) -> usize,
) -> Result<usize, usize> {
    if width == 0 || column >= width {
        return Err(usize::MAX);
    }
    let mut start = 0usize;
    for (index, unit) in units.iter().copied().enumerate() {
        match unit {
            0x0008 => column = column.saturating_sub(1),
            0x000d => column = 0,
            0x000a => {}
            _ => {
                column += 1;
                if column == width {
                    let status = output(&units[start..=index]);
                    if status != 0 {
                        return Err(status);
                    }
                    let status = output(&[0x000d, 0x000a]);
                    if status != 0 {
                        return Err(status);
                    }
                    column = 0;
                    start = index + 1;
                }
            }
        }
    }
    if start < units.len() {
        let status = output(&units[start..]);
        if status != 0 {
            return Err(status);
        }
    }
    Ok(column)
}

pub const fn inference_status_due(completed_tokens: usize) -> bool {
    completed_tokens != 0 && completed_tokens.is_multiple_of(4)
}

pub fn loading_line(frame: u8, output: &mut [u8]) -> usize {
    let mut line = Line::new(output);
    line.push(b" promptboot [");
    line.push_byte(activity(frame));
    line.push(b"] loading");
    line.len()
}

pub fn runtime_line(snapshot: RuntimeSnapshot, frame: u8, output: &mut [u8]) -> usize {
    let mut line = Line::new(output);
    line.push(b" promptboot [");
    line.push_byte(activity(frame));
    line.push(b"] ");
    line.push(snapshot.state.label());
    line.push(b" | CPU ");
    line.push_decimal(snapshot.cpu.active as u64);
    line.push_byte(b'/');
    line.push_optional(snapshot.cpu.enabled);
    line.push(b" | MEM ");
    line.push_decimal(snapshot.memory_free / (1024 * 1024));
    line.push(b" MiB free | CTX ");
    line.push_decimal(snapshot.bar_context() as u64);
    line.push_byte(b'/');
    line.push_decimal(snapshot.context_limit as u64);
    line.len()
}

const fn activity(frame: u8) -> u8 {
    match frame & 3 {
        0 => b'|',
        1 => b'/',
        2 => b'-',
        _ => b'\\',
    }
}

struct Line<'a> {
    output: &'a mut [u8],
    length: usize,
}

impl<'a> Line<'a> {
    fn new(output: &'a mut [u8]) -> Self {
        Self { output, length: 0 }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn push(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.push_byte(*byte);
        }
    }

    fn push_byte(&mut self, byte: u8) {
        if self.length < self.output.len() {
            self.output[self.length] = byte;
            self.length += 1;
        }
    }

    fn push_decimal(&mut self, mut value: u64) {
        let mut digits = [0u8; 20];
        let mut count = 0usize;
        loop {
            digits[count] = b'0' + (value % 10) as u8;
            count += 1;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        while count != 0 {
            count -= 1;
            self.push_byte(digits[count]);
        }
    }

    fn push_optional(&mut self, value: Option<u32>) {
        match value {
            Some(value) => self.push_decimal(value as u64),
            None => self.push(b"unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    fn snapshot(state: RuntimeState) -> RuntimeSnapshot {
        RuntimeSnapshot {
            state,
            cpu: CpuTopology {
                mode: CpuMode::Mp,
                active: 4,
                workers: 3,
                enabled: Some(4),
                total: Some(4),
            },
            memory_baseline: 2_000,
            memory_free: 1_000,
            committed_tokens: 17,
            engine_position: 23,
            context_limit: promptboot_core::CONTEXT_LIMIT,
            generation_reserve: 1_024,
        }
    }

    #[test]
    fn geometry_bounds_status_and_history_widths() {
        assert_eq!(bounded_status_geometry(1, 2), None);
        assert_eq!(bounded_status_geometry(2, 2), Some((2, 2)));
        assert_eq!(bounded_status_geometry(80, 25), Some((80, 25)));
        assert_eq!(
            bounded_status_geometry(MAX_COLUMNS, 25),
            Some((MAX_COLUMNS, 25))
        );
        assert_eq!(
            bounded_status_geometry(MAX_COLUMNS + 1, 25),
            Some((MAX_COLUMNS, 25))
        );
        assert_eq!(
            bounded_status_geometry(usize::MAX, 2),
            Some((MAX_COLUMNS, 2))
        );
        assert_eq!(bounded_status_geometry(80, 1), None);

        let (columns, rows) = bounded_status_geometry(usize::MAX, 2).unwrap();
        assert_eq!(columns - 1, MAX_COLUMNS - 1);
        assert_eq!(rows - 1, 1);
        let mut status = [0u8; MAX_COLUMNS - 1];
        assert_eq!(loading_line(0, &mut status[..columns - 1]), 23);
    }

    #[test]
    fn inversion_obeys_uefi_attribute_asymmetry() {
        assert_eq!(inverted_attribute(0x1e), 0x61);
        assert_eq!(inverted_attribute(0x70), 0x07);
        assert_eq!(inverted_attribute(0xf9), 0x17);
    }

    #[test]
    fn compact_line_clips_to_the_safe_cell_buffer() {
        let mut full = [0u8; 128];
        let length = runtime_line(snapshot(RuntimeState::Waiting), 1, &mut full);
        assert_eq!(
            &full[..length],
            b" promptboot [/] waiting | CPU 4/4 | MEM 0 MiB free | CTX 17/32768"
        );
        let mut clipped = [0u8; 12];
        assert_eq!(runtime_line(snapshot(RuntimeState::Inferring), 2, &mut clipped), 12);
        assert_eq!(&clipped, b" promptboot ");
    }

    #[test]
    fn context_source_changes_only_with_runtime_state() {
        assert_eq!(snapshot(RuntimeState::Waiting).bar_context(), 17);
        assert_eq!(snapshot(RuntimeState::Inferring).bar_context(), 23);
        assert_eq!(snapshot(RuntimeState::Waiting).consumed_since_boot(), 1_000);
    }

    #[test]
    fn native_scroll_count_covers_line_breaks_and_wrapping() {
        assert_eq!(
            native_scrolls_for_output(
                &[b'a' as u16, 0x000d, 0x000a, b'b' as u16],
                0,
                0,
                8,
                2,
            ),
            Some(0)
        );
        assert_eq!(
            native_scrolls_for_output(&[0x000d, 0x000a], 3, 1, 8, 2),
            Some(1)
        );
        assert_eq!(
            native_scrolls_for_output(&[b'x' as u16], 6, 1, 8, 2),
            Some(0)
        );
        assert_eq!(
            native_scrolls_for_output(&[b'x' as u16], 7, 1, 8, 2),
            Some(1)
        );
        assert_eq!(
            native_scrolls_for_output(
                &[0x000d, 0x000a, 0x000d, 0x000a, 0x000d, 0x000a],
                0,
                1,
                8,
                2,
            ),
            Some(3)
        );
        assert_eq!(native_scrolls_for_output(&[], 8, 0, 8, 2), None);
    }

    #[test]
    fn safe_width_output_preserves_every_character_and_inserts_wrap() {
        let mut calls = std::vec::Vec::new();
        assert_eq!(
            write_content_with_wrap(
                &[b'a' as u16, b'b' as u16, b'c' as u16, b'd' as u16],
                0,
                3,
                |units| {
                    calls.extend_from_slice(units);
                    0
                },
            ),
            Ok(1)
        );
        assert_eq!(
            calls,
            [
                b'a' as u16,
                b'b' as u16,
                b'c' as u16,
                0x000d,
                0x000a,
                b'd' as u16,
            ]
        );

        let mut calls = std::vec::Vec::new();
        assert_eq!(
            write_content_with_wrap(&[b'x' as u16], 2, 3, |units| {
                calls.extend_from_slice(units);
                0
            }),
            Ok(0)
        );
        assert_eq!(calls, [b'x' as u16, 0x000d, 0x000a]);
    }

    #[test]
    fn safe_width_output_stops_at_the_first_firmware_error() {
        let mut calls = 0usize;
        assert_eq!(
            write_content_with_wrap(&[b'a' as u16, b'b' as u16], 0, 2, |_| {
                calls += 1;
                if calls == 1 { 7 } else { 0 }
            }),
            Err(7)
        );
        assert_eq!(calls, 1);
        assert_eq!(
            write_content_with_wrap(&[], 2, 2, |_| 0),
            Err(usize::MAX)
        );
    }

    #[test]
    fn inference_status_refresh_is_bounded_to_four_token_cadence() {
        assert!(!inference_status_due(0));
        assert!(!inference_status_due(1));
        assert!(!inference_status_due(3));
        assert!(inference_status_due(4));
        assert!(!inference_status_due(5));
        assert!(inference_status_due(8));
    }

    #[test]
    fn key_first_wait_and_cleanup_error_precedence_are_exact() {
        assert_eq!(wait_source(0, 0), Ok(WaitSource::Key));
        assert_eq!(wait_source(0, 1), Ok(WaitSource::Timer));
        assert_eq!(wait_source(7, 0), Err(7));
        assert_eq!(wait_source(0, 2), Err(usize::MAX));
        assert_eq!(first_error(None, 0), None);
        assert_eq!(first_error(None, 7), Some(7));
        assert_eq!(first_error(Some(7), 9), Some(7));
        assert!(is_serial_evidence_line(
            &"PROMPTBOOT_EVENT v=1"
                .encode_utf16()
                .collect::<std::vec::Vec<_>>()
        ));
        assert!(is_serial_evidence_line(
            &"events: on".encode_utf16().collect::<std::vec::Vec<_>>()
        ));
        assert!(is_event_boundary_line(
            &"events: off\r\n"
                .encode_utf16()
                .collect::<std::vec::Vec<_>>()
        ));
        assert!(!is_event_boundary_line(
            &"PROMPTBOOT_EVENT v=1"
                .encode_utf16()
                .collect::<std::vec::Vec<_>>()
        ));
        assert!(!is_serial_evidence_line(
            &"promptboot> ".encode_utf16().collect::<std::vec::Vec<_>>()
        ));
    }

    #[test]
    fn renderer_orders_output_and_restores_after_every_failure() {
        let calls = Rc::new(RefCell::new(std::vec::Vec::new()));
        let cursor_calls = Rc::clone(&calls);
        let attribute_calls = Rc::clone(&calls);
        let output_calls = Rc::clone(&calls);
        assert_eq!(
            render_status_row(
                0x1e,
                7,
                3,
                0,
                &[b'o' as u16, b'k' as u16],
                move |column, row| {
                    cursor_calls.borrow_mut().push((b'C', column, row));
                    0
                },
                move |attribute| {
                    attribute_calls.borrow_mut().push((b'A', attribute, 0));
                    0
                },
                move |units| {
                    output_calls.borrow_mut().push((b'O', units.len(), 0));
                    0
                },
            ),
            Ok(())
        );
        assert_eq!(
            *calls.borrow(),
            [
                (b'C', 0, 0),
                (b'A', 0x61, 0),
                (b'O', 2, 0),
                (b'A', 0x1e, 0),
                (b'C', 7, 3),
            ]
        );

        let calls = Rc::new(RefCell::new(std::vec::Vec::new()));
        let cursor_count = Rc::new(Cell::new(0usize));
        let cursor_calls = Rc::clone(&calls);
        let cursor_count_for_call = Rc::clone(&cursor_count);
        let attribute_calls = Rc::clone(&calls);
        let output_calls = Rc::clone(&calls);
        assert_eq!(
            render_status_row(
                0x70,
                5,
                2,
                0,
                &[b'x' as u16],
                move |column, row| {
                    cursor_calls.borrow_mut().push((b'C', column, row));
                    let count = cursor_count_for_call.get();
                    cursor_count_for_call.set(count + 1);
                    if count == 0 { 7 } else { 11 }
                },
                move |attribute| {
                    attribute_calls.borrow_mut().push((b'A', attribute, 0));
                    9
                },
                move |units| {
                    output_calls.borrow_mut().push((b'O', units.len(), 0));
                    13
                },
            ),
            Err(7)
        );
        assert_eq!(
            *calls.borrow(),
            [(b'C', 0, 0), (b'A', 0x70, 0), (b'C', 5, 2)]
        );

        let cursor_count = Rc::new(Cell::new(0usize));
        let attribute_count = Rc::new(Cell::new(0usize));
        let cursor_count_for_call = Rc::clone(&cursor_count);
        let attribute_count_for_call = Rc::clone(&attribute_count);
        assert_eq!(
            render_status_row(
                0x1e,
                1,
                1,
                0,
                &[b'x' as u16],
                move |_, _| {
                    cursor_count_for_call.set(cursor_count_for_call.get() + 1);
                    0
                },
                move |_| {
                    let count = attribute_count_for_call.get();
                    attribute_count_for_call.set(count + 1);
                    if count == 0 { 0 } else { 17 }
                },
                |_| 13,
            ),
            Err(13)
        );
        assert_eq!(cursor_count.get(), 2);
        assert_eq!(attribute_count.get(), 2);
    }

    #[test]
    fn timer_policy_orders_set_cancel_close_and_preserves_primary() {
        let calls = Rc::new(RefCell::new(std::vec::Vec::new()));
        let create_calls = Rc::clone(&calls);
        let set_calls = Rc::clone(&calls);
        let close_calls = Rc::clone(&calls);
        let event = start_periodic_timer(
            move || {
                create_calls.borrow_mut().push((b'C', 0, 0));
                Ok(41usize)
            },
            move |event, mode, period| {
                set_calls.borrow_mut().push((b'S', mode as u64, period));
                assert_eq!(event, 41);
                0
            },
            move |event| {
                close_calls.borrow_mut().push((b'X', event as u64, 0));
                0
            },
        )
        .unwrap();
        assert_eq!(*calls.borrow(), [(b'C', 0, 0), (b'S', 1, 5_000_000)]);

        let cleanup_calls = Rc::new(RefCell::new(std::vec::Vec::new()));
        let set_calls = Rc::clone(&cleanup_calls);
        let close_calls = Rc::clone(&cleanup_calls);
        assert_eq!(
            cleanup_periodic_timer(
                event,
                move |event, mode, period| {
                    set_calls
                        .borrow_mut()
                        .push((b'S', event as u64, ((mode as u64) << 32) | period));
                    7
                },
                move |event| {
                    close_calls.borrow_mut().push((b'X', event as u64, 0));
                    9
                },
            ),
            Err(TimerFailure::Cancel(7))
        );
        assert_eq!(
            *cleanup_calls.borrow(),
            [(b'S', 41, 0), (b'X', 41, 0)]
        );

        let failed_calls = Rc::new(RefCell::new(std::vec::Vec::new()));
        let create_calls = Rc::clone(&failed_calls);
        let set_calls = Rc::clone(&failed_calls);
        let close_calls = Rc::clone(&failed_calls);
        assert_eq!(
            start_periodic_timer(
                move || {
                    create_calls.borrow_mut().push((b'C', 0));
                    Ok(73usize)
                },
                move |_, mode, _| {
                    set_calls.borrow_mut().push((b'S', mode));
                    if mode == TIMER_PERIODIC { 5 } else { 7 }
                },
                move |_| {
                    close_calls.borrow_mut().push((b'X', 0));
                    9
                },
            ),
            Err(TimerFailure::Set(5))
        );
        assert_eq!(
            *failed_calls.borrow(),
            [(b'C', 0), (b'S', TIMER_PERIODIC), (b'S', TIMER_CANCEL), (b'X', 0)]
        );
        assert_eq!(
            primary_or_cleanup::<(), _>(Err(3), Err(5)),
            Err(3)
        );
        assert_eq!(primary_or_cleanup(Ok(7), Err(5)), Err(5));
    }

    #[test]
    fn wait_geometry_memory_and_completion_policies_fail_closed() {
        let both_signaled = [true, true];
        assert_eq!(
            wait_key_first(11, 22, |events, index| {
                assert_eq!(events, &[11, 22]);
                *index = both_signaled.iter().position(|signaled| *signaled).unwrap();
                0
            }),
            Ok(WaitSource::Key)
        );

        let emitted = Cell::new(0usize);
        assert_eq!(
            with_runtime_memory::<(), ()>(|| None, |_| {
                emitted.set(emitted.get() + 1);
                Ok(())
            }),
            Err(RuntimeMemoryFailure::Memory)
        );
        assert_eq!(emitted.get(), 0);
        assert_eq!(measure_status_memory(|| None), Err(StatusMemoryFailure));
        assert_eq!(
            with_runtime_memory(|| Some(1_024), |memory| Ok::<_, ()>(memory / 2)),
            Ok((1_024, 512))
        );

        assert!(waiting_render_failure_is_terminal(TurnCompletion::Success));
        assert!(waiting_render_failure_is_terminal(
            TurnCompletion::RecoverableFailure
        ));
        assert!(!waiting_render_failure_is_terminal(
            TurnCompletion::FatalFailure
        ));
    }
}
