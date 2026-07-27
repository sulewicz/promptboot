//! Pure, allocation-free contracts shared by the UEFI model target and host tests.

pub const RECORD_CAPACITY: usize = 511;
pub const MODEL_RECORD_FATAL: &[u8] = b"PROMPTBOOT_EVENT v=1 event=FATAL code=MODEL_RECORD phase=record efi_status=none model_error=none inference_error=none region=none cleanup_attempted=00000000 cleanup_ok=00000000\r\n";

pub const FILE_BIT: u32 = 1 << 0;
pub const ROOT_BIT: u32 = 1 << 1;
pub const SIMPLE_FS_BIT: u32 = 1 << 2;
pub const LOADED_IMAGE_BIT: u32 = 1 << 3;

pub const fn region_cleanup_bit(index: usize) -> u32 {
    1u32 << (9 - index)
}

pub fn acquired_mask(
    file: bool,
    root: bool,
    simple_fs: bool,
    loaded_image: bool,
    regions: [bool; 6],
) -> u32 {
    let mut mask = 0;
    if file {
        mask |= FILE_BIT;
    }
    if root {
        mask |= ROOT_BIT;
    }
    if simple_fs {
        mask |= SIMPLE_FS_BIT;
    }
    if loaded_image {
        mask |= LOADED_IMAGE_BIT;
    }
    let mut index = 0;
    while index < regions.len() {
        if regions[index] {
            mask |= region_cleanup_bit(index);
        }
        index += 1;
    }
    mask
}

pub const fn primary_survives_cleanup<T: Copy>(
    primary: Option<T>,
    cleanup: Option<T>,
) -> Option<T> {
    match primary {
        Some(value) => Some(value),
        None => cleanup,
    }
}

pub struct ModelRecord {
    data: [u8; RECORD_CAPACITY],
    length: usize,
    overflow: bool,
}

impl ModelRecord {
    pub const fn new() -> Self {
        Self {
            data: [0; RECORD_CAPACITY],
            length: 0,
            overflow: false,
        }
    }

    pub fn push(&mut self, value: &[u8]) {
        if self.overflow || value.len() > self.data.len() - self.length {
            self.overflow = true;
            return;
        }
        self.data[self.length..self.length + value.len()].copy_from_slice(value);
        self.length += value.len();
    }

    pub fn push_decimal(&mut self, mut value: u64) {
        let mut digits = [0u8; 20];
        let mut used = 0;
        if value == 0 {
            self.push(b"0");
            return;
        }
        while value != 0 {
            digits[used] = b'0' + (value % 10) as u8;
            used += 1;
            value /= 10;
        }
        while used != 0 {
            used -= 1;
            self.push(&digits[used..used + 1]);
        }
    }

    pub fn overflowed(&self) -> bool {
        self.overflow
    }

    pub fn as_bytes(&self) -> &[u8] {
        if self.overflow {
            MODEL_RECORD_FATAL
        } else {
            &self.data[..self.length]
        }
    }
}

impl Default for ModelRecord {
    fn default() -> Self {
        Self::new()
    }
}

pub fn record_emission(record: &ModelRecord) -> Result<&[u8], ErrorTuple> {
    if record.overflowed() {
        Err(error_tuple(
            "MODEL_RECORD",
            "record",
            None,
            None,
            None,
            None,
            0,
            0,
        ))
    } else {
        Ok(record.as_bytes())
    }
}

pub const EFI_SUCCESS: u64 = 0;
pub const EFI_DEVICE_ERROR: u64 = 0x8000_0000_0000_0007;
pub const EFI_OUT_OF_RESOURCES: u64 = 0x8000_0000_0000_0009;
pub const EFI_NOT_FOUND: u64 = 0x8000_0000_0000_000e;
pub const ALL_CLEANUP: u32 = 0x0000_03ff;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    LoadOnly,
    FirstToken,
    Identity,
    Repl,
}

impl Mode {
    pub const fn parse(value: &str) -> Option<Self> {
        match value.as_bytes() {
            b"model_load_only" => Some(Self::LoadOnly),
            b"model_first_token" => Some(Self::FirstToken),
            b"model_identity_self_test" => Some(Self::Identity),
            b"model_repl" => Some(Self::Repl),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerClass {
    TimedInvariant,
    UntimedNonInvariant,
}

pub const fn timer_class(mode: Mode, rdtscp: bool, invariant: bool) -> Result<TimerClass, ()> {
    if !rdtscp {
        return Err(());
    }
    if invariant {
        return Ok(TimerClass::TimedInvariant);
    }
    if matches!(mode, Mode::LoadOnly | Mode::Repl) {
        Ok(TimerClass::UntimedNonInvariant)
    } else {
        Err(())
    }
}

pub fn derive_frequency(deltas: [u64; 3], aux: [u32; 3]) -> Result<u64, ()> {
    if deltas.iter().any(|value| *value == 0) || aux[0] != aux[1] || aux[0] != aux[2] {
        return Err(());
    }
    let mut sorted = deltas;
    sorted.sort_unstable();
    let hz = sorted[1].checked_mul(10).ok_or(())?;
    if !(1_000_000_000..=10_000_000_000).contains(&hz) {
        return Err(());
    }
    Ok(hz)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadDecision {
    Advance(usize),
    Complete,
    Truncated,
    Oversized,
    Read,
}

pub const fn read_decision(
    requested: usize,
    returned: usize,
    status: u64,
    eof: bool,
) -> ReadDecision {
    if status != EFI_SUCCESS || returned > requested {
        return ReadDecision::Read;
    }
    if eof {
        return if returned == 0 {
            ReadDecision::Complete
        } else {
            ReadDecision::Oversized
        };
    }
    if returned == 0 {
        ReadDecision::Truncated
    } else {
        ReadDecision::Advance(returned)
    }
}

pub fn checked_read_offset(offset: usize, returned: usize, total: usize) -> Result<usize, ()> {
    let next = offset.checked_add(returned).ok_or(())?;
    if next > total {
        Err(())
    } else {
        Ok(next)
    }
}

pub const fn read_failure(decision: ReadDecision, status: u64) -> Option<ErrorTuple> {
    let code = match decision {
        ReadDecision::Truncated => "MODEL_TRUNCATED",
        ReadDecision::Oversized => "MODEL_OVERSIZED",
        ReadDecision::Read => "MODEL_READ",
        _ => return None,
    };
    Some(error_tuple(
        code,
        "read",
        if status == EFI_SUCCESS {
            None
        } else {
            Some(status)
        },
        None,
        None,
        None,
        ALL_CLEANUP,
        ALL_CLEANUP,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadProgress {
    pub offset: usize,
    pub total: usize,
}

impl ReadProgress {
    pub const fn new(total: usize) -> Self {
        Self { offset: 0, total }
    }

    /// Apply one callback result transactionally. The offset changes only for
    /// an accepted non-EOF partial read.
    pub fn apply(
        &mut self,
        requested: usize,
        returned: usize,
        status: u64,
        eof: bool,
    ) -> ReadDecision {
        let decision = read_decision(requested, returned, status, eof);
        if let ReadDecision::Advance(count) = decision {
            match checked_read_offset(self.offset, count, self.total) {
                Ok(next) => self.offset = next,
                Err(()) => return ReadDecision::Read,
            }
        }
        decision
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CleanupPlan {
    pub attempted: u32,
    pub ok: u32,
}

pub const fn cleanup_plan(acquired: u32, callable: u32, failed: u32) -> CleanupPlan {
    CleanupPlan {
        attempted: acquired,
        ok: acquired & callable & !failed,
    }
}

/// Select the next acquired resource in the target's mandatory cleanup order:
/// file, root, protocols, then prompt back through weights.
pub const fn next_cleanup_bit(pending: u32) -> Option<u32> {
    let mut bit = 0;
    while bit < 10 {
        if pending & (1 << bit) != 0 {
            return Some(bit);
        }
        bit += 1;
    }
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardFailure {
    WeightsTail,
    IndexTail,
    LogitsTail,
    PromptSlack,
    PromptTail,
}

pub const fn first_guard_failure(valid: [bool; 5]) -> Option<GuardFailure> {
    if !valid[0] {
        Some(GuardFailure::WeightsTail)
    } else if !valid[1] {
        Some(GuardFailure::IndexTail)
    } else if !valid[2] {
        Some(GuardFailure::LogitsTail)
    } else if !valid[3] {
        Some(GuardFailure::PromptSlack)
    } else if !valid[4] {
        Some(GuardFailure::PromptTail)
    } else {
        None
    }
}

pub const fn guard_failure(failure: GuardFailure) -> ErrorTuple {
    let region = match failure {
        GuardFailure::WeightsTail => "weights",
        GuardFailure::IndexTail => "index",
        GuardFailure::LogitsTail => "logits",
        GuardFailure::PromptSlack | GuardFailure::PromptTail => "prompt",
    };
    error_tuple(
        "MODEL_GUARD",
        "guard",
        None,
        None,
        None,
        Some(region),
        ALL_CLEANUP,
        ALL_CLEANUP,
    )
}

pub const fn timer_failure(status: Option<u64>) -> ErrorTuple {
    error_tuple("MODEL_TIMER", "timer", status, None, None, None, 0, 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorTuple {
    pub code: &'static str,
    pub phase: &'static str,
    pub efi: Option<u64>,
    pub model: Option<[u64; 7]>,
    pub inference: Option<[u64; 8]>,
    pub region: Option<&'static str>,
    pub attempted: u32,
    pub ok: u32,
}

pub const fn error_tuple(
    code: &'static str,
    phase: &'static str,
    efi: Option<u64>,
    model: Option<[u64; 7]>,
    inference: Option<[u64; 8]>,
    region: Option<&'static str>,
    attempted: u32,
    ok: u32,
) -> ErrorTuple {
    ErrorTuple {
        code,
        phase,
        efi,
        model,
        inference,
        region,
        attempted,
        ok,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageFailure {
    LoadedStatus,
    LoadedAbi,
    SimpleStatus,
    SimpleAbi,
    OpenVolumeStatus,
    RootNull,
    RootAbi { close_callable: bool },
    FileOpen { missing: bool },
    FileNull,
    FileAbi { close_callable: bool },
}

pub const fn storage_failure(kind: StorageFailure, status: Option<u64>) -> ErrorTuple {
    let (code, phase, acquired, callable) = match kind {
        StorageFailure::LoadedStatus => ("MODEL_LOADED_IMAGE", "loaded_image", 0, 0),
        StorageFailure::LoadedAbi => ("MODEL_LOADED_IMAGE", "loaded_image", 0x008, 0x008),
        StorageFailure::SimpleStatus => ("MODEL_SIMPLE_FS", "simple_fs", 0x008, 0x008),
        StorageFailure::SimpleAbi => ("MODEL_SIMPLE_FS", "simple_fs", 0x00c, 0x00c),
        StorageFailure::OpenVolumeStatus => ("MODEL_OPEN_VOLUME", "open_volume", 0x00c, 0x00c),
        StorageFailure::RootNull => ("MODEL_OPEN_VOLUME", "open_volume", 0x00c, 0x00c),
        StorageFailure::RootAbi { close_callable } => (
            "MODEL_FILE_ABI",
            "root_abi",
            0x00e,
            if close_callable { 0x00e } else { 0x00c },
        ),
        StorageFailure::FileOpen { missing: true } => ("MODEL_MISSING", "open", 0x00e, 0x00e),
        StorageFailure::FileOpen { missing: false } => ("MODEL_OPEN", "open", 0x00e, 0x00e),
        StorageFailure::FileNull => ("MODEL_OPEN", "open", 0x00e, 0x00e),
        StorageFailure::FileAbi { close_callable } => (
            "MODEL_FILE_ABI",
            "file_abi",
            0x00f,
            if close_callable { 0x00f } else { 0x00e },
        ),
    };
    error_tuple(code, phase, status, None, None, None, acquired, callable)
}

pub const fn loaded_transition(
    status: u64,
    interface: bool,
    revision: u32,
    device: bool,
) -> Option<ErrorTuple> {
    if status != EFI_SUCCESS {
        Some(storage_failure(StorageFailure::LoadedStatus, Some(status)))
    } else if !interface || revision < 0x1000 || !device {
        Some(storage_failure(StorageFailure::LoadedAbi, None))
    } else {
        None
    }
}

pub const fn simple_transition(
    status: u64,
    interface: bool,
    revision: u64,
    open_volume: bool,
) -> Option<ErrorTuple> {
    if status != EFI_SUCCESS {
        Some(storage_failure(StorageFailure::SimpleStatus, Some(status)))
    } else if !interface || revision < 0x1_0000 || !open_volume {
        Some(storage_failure(StorageFailure::SimpleAbi, None))
    } else {
        None
    }
}

pub const fn volume_transition(status: u64, root: bool) -> Option<ErrorTuple> {
    if status != EFI_SUCCESS {
        Some(storage_failure(
            StorageFailure::OpenVolumeStatus,
            Some(status),
        ))
    } else if !root {
        Some(storage_failure(StorageFailure::RootNull, None))
    } else {
        None
    }
}

pub const fn file_abi_transition(
    root: bool,
    revision: u64,
    open: bool,
    close: bool,
    read: bool,
) -> Option<ErrorTuple> {
    if revision >= 0x1_0000 && open && close && read {
        None
    } else if root {
        Some(storage_failure(
            StorageFailure::RootAbi {
                close_callable: close,
            },
            None,
        ))
    } else {
        Some(storage_failure(
            StorageFailure::FileAbi {
                close_callable: close,
            },
            None,
        ))
    }
}

pub const fn file_open_transition(status: u64, interface: bool) -> Option<ErrorTuple> {
    if status != EFI_SUCCESS {
        Some(storage_failure(
            StorageFailure::FileOpen {
                missing: status == EFI_NOT_FOUND,
            },
            Some(status),
        ))
    } else if !interface {
        Some(storage_failure(StorageFailure::FileNull, None))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractFailureKind {
    ModelChecksum,
    ModelIdentity,
    ModelIndex,
    Engine,
    Prefill,
    Top8,
}

pub const fn classified_failure(kind: ContractFailureKind, fields: [u64; 8]) -> ErrorTuple {
    match kind {
        ContractFailureKind::ModelChecksum => error_tuple(
            "MODEL_CHECKSUM",
            "verify",
            None,
            Some([
                fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
            ]),
            None,
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
        ContractFailureKind::ModelIdentity => error_tuple(
            "MODEL_IDENTITY",
            "verify",
            None,
            Some([
                fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
            ]),
            None,
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
        ContractFailureKind::ModelIndex => error_tuple(
            "MODEL_INDEX",
            "index",
            None,
            Some([
                fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
            ]),
            None,
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
        ContractFailureKind::Engine => error_tuple(
            "MODEL_ENGINE",
            "init",
            None,
            None,
            Some(fields),
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
        ContractFailureKind::Prefill => error_tuple(
            "MODEL_PREFILL",
            "prefill",
            None,
            None,
            Some(fields),
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
        ContractFailureKind::Top8 => error_tuple(
            "MODEL_TOP8",
            "top8",
            None,
            None,
            Some(fields),
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        ),
    }
}

pub const fn allocation_failure(index: usize) -> ErrorTuple {
    allocation_status_failure(index, EFI_OUT_OF_RESOURCES)
}

pub const fn allocation_status_failure(index: usize, status: u64) -> ErrorTuple {
    let prior = [0x00f, 0x20f, 0x30f, 0x38f, 0x3cf, 0x3ef];
    let region = ["weights", "index", "kv", "scratch", "logits", "prompt"];
    error_tuple(
        "MODEL_OOM",
        "allocate",
        Some(status),
        None,
        None,
        Some(region[index]),
        prior[index],
        prior[index],
    )
}

pub const fn cleanup_failure(bit: u32, primary: Option<ErrorTuple>) -> ErrorTuple {
    let ok = ALL_CLEANUP & !(1 << bit);
    match primary {
        Some(mut value) => {
            value.attempted = ALL_CLEANUP;
            value.ok = ok;
            value
        }
        None => {
            let region = match bit {
                4 => Some("prompt"),
                5 => Some("logits"),
                6 => Some("scratch"),
                7 => Some("kv"),
                8 => Some("index"),
                9 => Some("weights"),
                _ => None,
            };
            error_tuple(
                "MODEL_CLEANUP",
                "cleanup",
                Some(EFI_DEVICE_ERROR),
                None,
                None,
                region,
                ALL_CLEANUP,
                ok,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_mask_is_exact_for_every_acquisition_prefix() {
        let expected = [
            0x000, 0x008, 0x00c, 0x00e, 0x00f, 0x20f, 0x30f, 0x38f, 0x3cf, 0x3ef, 0x3ff,
        ];
        for (prefix, wanted) in expected.iter().copied().enumerate() {
            let file = prefix >= 4;
            let root = prefix >= 3;
            let simple = prefix >= 2;
            let loaded = prefix >= 1;
            let mut regions = [false; 6];
            let count = prefix.saturating_sub(4).min(6);
            regions[..count].fill(true);
            assert_eq!(acquired_mask(file, root, simple, loaded, regions), wanted);
        }
    }

    #[test]
    fn primary_failure_always_survives_cleanup_failure() {
        assert_eq!(primary_survives_cleanup(Some(7u8), Some(9)), Some(7));
        assert_eq!(primary_survives_cleanup(None, Some(9u8)), Some(9));
        assert_eq!(primary_survives_cleanup::<u8>(None, None), None);
    }

    #[test]
    fn record_accepts_exact_capacity_and_latches_overflow() {
        let mut below = ModelRecord::new();
        below.push(&[b'x'; 510]);
        assert!(!below.overflowed());

        let mut exact = ModelRecord::new();
        exact.push(&[b'x'; RECORD_CAPACITY]);
        assert!(!exact.overflowed());
        assert_eq!(exact.as_bytes().len(), RECORD_CAPACITY);

        exact.push(b"y");
        assert!(exact.overflowed());
        assert_eq!(exact.as_bytes(), MODEL_RECORD_FATAL);
        exact.push(b"ignored");
        assert_eq!(exact.as_bytes(), MODEL_RECORD_FATAL);
    }

    #[test]
    fn file_read_and_eof_decisions_are_transactional() {
        assert_eq!(
            read_decision(1_048_576, 1, 0, false),
            ReadDecision::Advance(1)
        );
        assert_eq!(
            read_decision(1_048_576, 1_048_575, 0, false),
            ReadDecision::Advance(1_048_575)
        );
        assert_eq!(read_decision(1, 0, 0, false), ReadDecision::Truncated);
        assert_eq!(read_decision(1, 2, 0, false), ReadDecision::Read);
        assert_eq!(
            read_decision(1, 1, EFI_DEVICE_ERROR, false),
            ReadDecision::Read
        );
        assert_eq!(read_decision(1, 0, 0, true), ReadDecision::Complete);
        assert_eq!(read_decision(1, 1, 0, true), ReadDecision::Oversized);
        assert_eq!(
            checked_read_offset(426_762_943, 1, 426_762_944),
            Ok(426_762_944)
        );
        assert_eq!(checked_read_offset(usize::MAX, 1, usize::MAX), Err(()));
        assert_eq!(checked_read_offset(426_762_944, 1, 426_762_944), Err(()));
    }

    #[test]
    fn partial_reads_advance_by_only_the_accepted_count() {
        let bytes = b"abcdefghij";
        let mut progress = ReadProgress::new(10);
        let mut sha256 = promptboot_core::sha256::Sha256::new();
        let old_offset = progress.offset;
        assert_eq!(
            progress.apply(8, 3, EFI_SUCCESS, false),
            ReadDecision::Advance(3)
        );
        sha256.update(&bytes[old_offset..old_offset + 3]);
        assert_eq!(progress.offset, 3);
        let old_offset = progress.offset;
        assert_eq!(
            progress.apply(7, 7, EFI_SUCCESS, false),
            ReadDecision::Advance(7)
        );
        sha256.update(&bytes[old_offset..old_offset + 7]);
        assert_eq!(progress.offset, 10);
        assert_eq!(
            progress.apply(1, 0, EFI_SUCCESS, true),
            ReadDecision::Complete
        );
        assert_eq!(progress.offset, 10);
        assert_eq!(sha256.finish(), promptboot_core::sha256::digest(bytes));
    }

    #[test]
    fn read_error_with_nonzero_count_does_not_advance() {
        let bytes = b"abcdefghij";
        let mut progress = ReadProgress::new(10);
        let mut sha256 = promptboot_core::sha256::Sha256::new();
        let old_offset = progress.offset;
        assert_eq!(
            progress.apply(8, 3, EFI_SUCCESS, false),
            ReadDecision::Advance(3)
        );
        sha256.update(&bytes[old_offset..old_offset + 3]);
        let rejected_offset = progress.offset;
        assert_eq!(
            progress.apply(7, 4, EFI_DEVICE_ERROR, false),
            ReadDecision::Read
        );
        assert_eq!(progress.offset, 3);
        assert_eq!(
            sha256.finish(),
            promptboot_core::sha256::digest(&bytes[..rejected_offset])
        );
    }

    #[test]
    fn correction16_timer_branches_and_calibration_are_exact() {
        assert_eq!(
            timer_class(Mode::LoadOnly, true, false),
            Ok(TimerClass::UntimedNonInvariant)
        );
        assert_eq!(timer_class(Mode::FirstToken, true, false), Err(()));
        assert_eq!(timer_class(Mode::Identity, true, false), Err(()));
        assert_eq!(
            timer_class(Mode::Repl, true, false),
            Ok(TimerClass::UntimedNonInvariant)
        );
        assert_eq!(
            timer_class(Mode::FirstToken, true, true),
            Ok(TimerClass::TimedInvariant)
        );
        assert_eq!(timer_class(Mode::LoadOnly, false, false), Err(()));
        assert_eq!(
            derive_frequency([399_000_000, 400_000_000, 401_000_000], [3; 3]),
            Ok(4_000_000_000)
        );
        assert!(derive_frequency([0, 400_000_000, 401_000_000], [3; 3]).is_err());
        assert!(derive_frequency([400_000_000; 3], [3, 4, 3]).is_err());
        assert!(derive_frequency([99_999_999; 3], [3; 3]).is_err());
        assert!(derive_frequency([1_000_000_001; 3], [3; 3]).is_err());
    }

    #[test]
    fn allocation_cleanup_and_primary_precedence_are_exact() {
        let masks = [0x00f, 0x20f, 0x30f, 0x38f, 0x3cf, 0x3ef];
        let regions = ["weights", "index", "kv", "scratch", "logits", "prompt"];
        for index in 0..6 {
            let failure = allocation_failure(index);
            assert_eq!(failure.region, Some(regions[index]));
            assert_eq!(
                (failure.attempted, failure.ok),
                (masks[index], masks[index])
            );
        }

        let read = error_tuple(
            "MODEL_READ",
            "read",
            Some(EFI_DEVICE_ERROR),
            None,
            None,
            None,
            ALL_CLEANUP,
            ALL_CLEANUP,
        );
        for bit in 0..10 {
            let cleanup = cleanup_failure(bit, None);
            assert_eq!(cleanup.ok, ALL_CLEANUP & !(1 << bit));
            let preserved = cleanup_failure(bit, Some(read));
            assert_eq!(preserved.code, "MODEL_READ");
            assert_eq!(preserved.efi, Some(EFI_DEVICE_ERROR));
            assert_eq!(preserved.ok, ALL_CLEANUP & !(1 << bit));
        }
        assert_eq!(
            cleanup_plan(0x0e, 0x0c, 0),
            CleanupPlan {
                attempted: 0x0e,
                ok: 0x0c
            }
        );
        assert_eq!(
            cleanup_plan(0x0f, 0x0e, 0),
            CleanupPlan {
                attempted: 0x0f,
                ok: 0x0e
            }
        );
    }

    #[test]
    fn guard_first_failure_and_positive_byte_counts_are_exact() {
        assert_eq!(
            first_guard_failure([false; 5]),
            Some(GuardFailure::WeightsTail)
        );
        assert_eq!(
            first_guard_failure([true, false, false, false, false]),
            Some(GuardFailure::IndexTail)
        );
        assert_eq!(
            first_guard_failure([true, true, false, false, false]),
            Some(GuardFailure::LogitsTail)
        );
        assert_eq!(
            first_guard_failure([true, true, true, false, false]),
            Some(GuardFailure::PromptSlack)
        );
        assert_eq!(
            first_guard_failure([true, true, true, true, false]),
            Some(GuardFailure::PromptTail)
        );
        assert_eq!(first_guard_failure([true; 5]), None);
        assert_eq!(
            [3392usize, 3072, 2560, 1928, 2048]
                .iter()
                .sum::<usize>(),
            13000
        );
    }
}
