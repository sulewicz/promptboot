pub const EFI_SUCCESS: usize = 0;
pub const EFI_ERROR_BIT: usize = 1usize << (usize::BITS - 1);
pub const EFI_NOT_READY: usize = EFI_ERROR_BIT | 6;
const EFI_SHIFT_STATE_VALID: u32 = 0x8000_0000;
const EFI_RIGHT_CONTROL_PRESSED: u32 = 0x0000_0004;
const EFI_LEFT_CONTROL_PRESSED: u32 = 0x0000_0008;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsoleFailure {
    Clear,
    InputMissing,
    ResetMissing,
    ReadMissing,
    EventMissing,
    WaitMissing,
    Reset,
    Wait,
    WaitIndex,
    Read,
}

impl ConsoleFailure {
    pub const fn code(self) -> &'static [u8] {
        match self {
            Self::Clear => b"CONSOLE_CLEAR",
            Self::InputMissing => b"CONSOLE_INPUT_MISSING",
            Self::ResetMissing => b"CONSOLE_RESET_MISSING",
            Self::ReadMissing => b"CONSOLE_READ_MISSING",
            Self::EventMissing => b"CONSOLE_EVENT_MISSING",
            Self::WaitMissing => b"CONSOLE_WAIT_MISSING",
            Self::Reset => b"CONSOLE_RESET",
            Self::Wait => b"CONSOLE_WAIT",
            Self::WaitIndex => b"CONSOLE_WAIT_INDEX",
            Self::Read => b"CONSOLE_READ",
        }
    }
}

pub fn clear_result(status: usize) -> Result<(), ConsoleFailure> {
    if status == EFI_SUCCESS {
        Ok(())
    } else {
        Err(ConsoleFailure::Clear)
    }
}

pub fn validate_bindings(
    con_in: usize,
    reset: usize,
    read: usize,
    wait_for_key: usize,
    wait_for_event: usize,
) -> Result<(), ConsoleFailure> {
    if con_in == 0 {
        return Err(ConsoleFailure::InputMissing);
    }
    if reset == 0 {
        return Err(ConsoleFailure::ResetMissing);
    }
    if read == 0 {
        return Err(ConsoleFailure::ReadMissing);
    }
    if wait_for_key == 0 {
        return Err(ConsoleFailure::EventMissing);
    }
    if wait_for_event == 0 {
        return Err(ConsoleFailure::WaitMissing);
    }
    Ok(())
}

pub fn reset_result(status: usize) -> Result<(), ConsoleFailure> {
    if status == EFI_SUCCESS {
        Ok(())
    } else {
        Err(ConsoleFailure::Reset)
    }
}

pub fn wait_result(status: usize, index: usize) -> Result<(), ConsoleFailure> {
    if status != EFI_SUCCESS {
        Err(ConsoleFailure::Wait)
    } else if index != 0 {
        Err(ConsoleFailure::WaitIndex)
    } else {
        Ok(())
    }
}

pub fn read_result(status: usize) -> Result<bool, ConsoleFailure> {
    if status == EFI_SUCCESS {
        Ok(true)
    } else if status == EFI_NOT_READY {
        Ok(false)
    } else {
        Err(ConsoleFailure::Read)
    }
}

pub fn is_generation_interrupt(unicode_char: u16, key_shift_state: u32) -> bool {
    if unicode_char == 0x0003 {
        return true;
    }
    let control = key_shift_state & (EFI_RIGHT_CONTROL_PRESSED | EFI_LEFT_CONTROL_PRESSED) != 0;
    key_shift_state & EFI_SHIFT_STATE_VALID != 0
        && control
        && matches!(unicode_char, 0x0043 | 0x0063)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullable_bindings_have_exact_first_failure_order_and_codes() {
        let full = [1usize; 5];
        for missing in 0..5 {
            let mut values = full;
            values[missing] = 0;
            let expected = [
                ConsoleFailure::InputMissing,
                ConsoleFailure::ResetMissing,
                ConsoleFailure::ReadMissing,
                ConsoleFailure::EventMissing,
                ConsoleFailure::WaitMissing,
            ][missing];
            let actual = validate_bindings(values[0], values[1], values[2], values[3], values[4]);
            assert_eq!(actual, Err(expected));
        }
        assert_eq!(validate_bindings(1, 1, 1, 1, 1), Ok(()));
        assert_eq!(
            ConsoleFailure::InputMissing.code(),
            b"CONSOLE_INPUT_MISSING"
        );
        assert_eq!(
            ConsoleFailure::ResetMissing.code(),
            b"CONSOLE_RESET_MISSING"
        );
        assert_eq!(ConsoleFailure::ReadMissing.code(), b"CONSOLE_READ_MISSING");
        assert_eq!(
            ConsoleFailure::EventMissing.code(),
            b"CONSOLE_EVENT_MISSING"
        );
        assert_eq!(ConsoleFailure::WaitMissing.code(), b"CONSOLE_WAIT_MISSING");
    }

    #[test]
    fn status_and_wait_index_mappings_are_exact() {
        assert_eq!(clear_result(EFI_SUCCESS), Ok(()));
        assert_eq!(clear_result(1), Err(ConsoleFailure::Clear));
        assert_eq!(ConsoleFailure::Clear.code(), b"CONSOLE_CLEAR");
        assert_eq!(reset_result(EFI_SUCCESS), Ok(()));
        assert_eq!(reset_result(1), Err(ConsoleFailure::Reset));
        assert_eq!(wait_result(EFI_SUCCESS, 0), Ok(()));
        assert_eq!(wait_result(1, 0), Err(ConsoleFailure::Wait));
        assert_eq!(wait_result(EFI_SUCCESS, 1), Err(ConsoleFailure::WaitIndex));
        assert_eq!(read_result(EFI_SUCCESS), Ok(true));
        assert_eq!(read_result(EFI_NOT_READY), Ok(false));
        assert_eq!(read_result(1), Err(ConsoleFailure::Read));
    }

    #[test]
    fn generation_interrupt_accepts_legacy_etx_and_valid_extended_control_c() {
        assert!(is_generation_interrupt(0x0003, 0));
        assert!(is_generation_interrupt(
            b'c' as u16,
            EFI_SHIFT_STATE_VALID | EFI_LEFT_CONTROL_PRESSED
        ));
        assert!(is_generation_interrupt(
            b'C' as u16,
            EFI_SHIFT_STATE_VALID | EFI_RIGHT_CONTROL_PRESSED
        ));
        assert!(!is_generation_interrupt(
            b'c' as u16,
            EFI_LEFT_CONTROL_PRESSED
        ));
        assert!(!is_generation_interrupt(b'c' as u16, EFI_SHIFT_STATE_VALID));
        assert!(!is_generation_interrupt(
            b'x' as u16,
            EFI_SHIFT_STATE_VALID | EFI_LEFT_CONTROL_PRESSED
        ));
    }
}
