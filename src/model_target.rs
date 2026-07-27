//! Compile-bound UEFI model loading and first-token target path.

use core::arch::asm;
use core::ffi::c_void;
use core::mem::{align_of, offset_of, size_of, transmute};
use core::ptr::null_mut;
use core::slice;

use promptboot::model_contract::{
    acquired_mask, allocation_status_failure, classified_failure, cleanup_failure, cleanup_plan,
    derive_frequency, file_abi_transition, file_open_transition, first_guard_failure,
    guard_failure, loaded_transition, next_cleanup_bit, primary_survives_cleanup, read_failure,
    record_emission, simple_transition, timer_class, timer_failure, volume_transition,
    ContractFailureKind, ErrorTuple, Mode, ReadDecision, ReadProgress, TimerClass,
};
use promptboot_core::{
    sha256::Sha256, top_logits_8, FrozenTokenizer, InferenceEngine, InferenceError, ModelError,
    ModelView, TopLogit, INDEX_BYTES, KV_BYTES, LOGIT_WORDS, SCRATCH_BYTES,
    TOKENIZER_INDEX_SHA256_HEX,
};

use super::{
    emit_record, establish_fp_state, restore_fp_state, write_fatal_diagnostic, BootServices,
    EfiHandle, EfiStatus, Guid, Record, SystemTable, ALLOCATE_ANY_PAGES, EFI_ERROR_BIT,
    EFI_LOADER_DATA, EFI_OPEN_PROTOCOL_GET_PROTOCOL, EFI_SUCCESS, SAVED_FP_STATE,
};

const MODEL_BYTES: usize = 426_762_944;
const MAX_READ: usize = 1_048_576;
const GENERATION_RESERVE: u32 = 16;
pub(super) const EXPECTED_SHA_HEX: &str = match option_env!("PROMPTBOOT_EXPECTED_MODEL_SHA256") {
    Some(value) => value,
    None => "b0f98ed6e0557ca35e1bced1000c950b3c84414251df65290315a7969981d42d",
};
const SOURCE_SHA: &str = "7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed";
const PROMPT_SHA: &str = "57be9252afa09f8300997803322e86a9044ad274f92f7cfc9f6d1d751d68c52b";

const PROMPT: [u32; 30] = [
    151_644, 8_948, 198, 2_610, 525, 1_207, 16_948, 11, 3_465, 553, 54_364, 14_817, 13, 1_446, 525,
    264, 10_950, 17_847, 13, 151_645, 198, 151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091,
    198,
];

const SIMPLE_FS_GUID: Guid = Guid::new(
    0x964e_5b22,
    0x6459,
    0x11d2,
    [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
);
const LOADED_IMAGE_GUID: Guid = Guid::new(
    0x5b1b_31a1,
    0x9562,
    0x11d2,
    [0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
);

const EFI_BUFFER_TOO_SMALL: usize = EFI_ERROR_BIT | 5;
const EFI_DEVICE_ERROR: usize = EFI_ERROR_BIT | 7;
const EFI_OUT_OF_RESOURCES: usize = EFI_ERROR_BIT | 9;

type OpenVolume =
    unsafe extern "efiapi" fn(*mut SimpleFileSystem, *mut *mut FileProtocol) -> EfiStatus;
type FileOpen = unsafe extern "efiapi" fn(
    *mut FileProtocol,
    *mut *mut FileProtocol,
    *const u16,
    u64,
    u64,
) -> EfiStatus;
type FileClose = unsafe extern "efiapi" fn(*mut FileProtocol) -> EfiStatus;
type FileRead = unsafe extern "efiapi" fn(*mut FileProtocol, *mut usize, *mut c_void) -> EfiStatus;

#[repr(C)]
struct SimpleFileSystem {
    revision: u64,
    open_volume: usize,
}

#[repr(C)]
struct FileProtocol {
    revision: u64,
    open: usize,
    close: usize,
    delete: usize,
    read: usize,
    write: usize,
    get_position: usize,
    set_position: usize,
    get_info: usize,
    set_info: usize,
    flush: usize,
}

#[repr(C)]
struct LoadedImage {
    revision: u32,
    _pad0: u32,
    parent_handle: EfiHandle,
    system_table: *mut c_void,
    device_handle: EfiHandle,
    file_path: *mut c_void,
    reserved: *mut c_void,
    load_options_size: u32,
    _pad1: u32,
    load_options: *mut c_void,
    image_base: *mut c_void,
    image_size: u64,
    image_code_type: u32,
    image_data_type: u32,
    unload: usize,
}

const _: [(); 16] = [(); size_of::<SimpleFileSystem>()];
const _: [(); 8] = [(); align_of::<SimpleFileSystem>()];
const _: [(); 0] = [(); offset_of!(SimpleFileSystem, revision)];
const _: [(); 8] = [(); offset_of!(SimpleFileSystem, open_volume)];
const _: [(); 88] = [(); size_of::<FileProtocol>()];
const _: [(); 8] = [(); align_of::<FileProtocol>()];
const _: [(); 0] = [(); offset_of!(FileProtocol, revision)];
const _: [(); 8] = [(); offset_of!(FileProtocol, open)];
const _: [(); 16] = [(); offset_of!(FileProtocol, close)];
const _: [(); 24] = [(); offset_of!(FileProtocol, delete)];
const _: [(); 32] = [(); offset_of!(FileProtocol, read)];
const _: [(); 40] = [(); offset_of!(FileProtocol, write)];
const _: [(); 48] = [(); offset_of!(FileProtocol, get_position)];
const _: [(); 56] = [(); offset_of!(FileProtocol, set_position)];
const _: [(); 64] = [(); offset_of!(FileProtocol, get_info)];
const _: [(); 72] = [(); offset_of!(FileProtocol, set_info)];
const _: [(); 80] = [(); offset_of!(FileProtocol, flush)];
const _: [(); 96] = [(); size_of::<LoadedImage>()];
const _: [(); 8] = [(); align_of::<LoadedImage>()];
const _: [(); 0] = [(); offset_of!(LoadedImage, revision)];
const _: [(); 8] = [(); offset_of!(LoadedImage, parent_handle)];
const _: [(); 16] = [(); offset_of!(LoadedImage, system_table)];
const _: [(); 24] = [(); offset_of!(LoadedImage, device_handle)];
const _: [(); 32] = [(); offset_of!(LoadedImage, file_path)];
const _: [(); 40] = [(); offset_of!(LoadedImage, reserved)];
const _: [(); 48] = [(); offset_of!(LoadedImage, load_options_size)];
const _: [(); 56] = [(); offset_of!(LoadedImage, load_options)];
const _: [(); 64] = [(); offset_of!(LoadedImage, image_base)];
const _: [(); 72] = [(); offset_of!(LoadedImage, image_size)];
const _: [(); 80] = [(); offset_of!(LoadedImage, image_code_type)];
const _: [(); 84] = [(); offset_of!(LoadedImage, image_data_type)];
const _: [(); 88] = [(); offset_of!(LoadedImage, unload)];
const _: [(); 8] = [(); size_of::<TopLogit>()];
const _: [(); 4] = [(); align_of::<TopLogit>()];

#[derive(Clone, Copy)]
struct RegionSpec {
    name: &'static [u8],
    pages: usize,
    active: usize,
    guard: usize,
}

const REGIONS: [RegionSpec; 6] = [
    RegionSpec {
        name: b"weights",
        pages: 104_191,
        active: MODEL_BYTES,
        guard: 3_392,
    },
    RegionSpec {
        name: b"index",
        pages: 1_025,
        active: INDEX_BYTES,
        guard: 3_072,
    },
    RegionSpec {
        name: b"kv",
        pages: (KV_BYTES + 4_095) / 4_096,
        active: KV_BYTES,
        guard: 0,
    },
    RegionSpec {
        name: b"scratch",
        pages: (SCRATCH_BYTES + 4_095) / 4_096,
        active: SCRATCH_BYTES,
        guard: 0,
    },
    RegionSpec {
        name: b"logits",
        pages: 149,
        active: LOGIT_WORDS * 4,
        guard: 2_560,
    },
    RegionSpec {
        name: b"prompt",
        pages: 1,
        active: 2_048,
        guard: 2_048,
    },
];

fn region_spec(index: usize, mode: &str) -> RegionSpec {
    if index == 5 && mode == "model_repl" {
        RegionSpec {
            name: b"prompt",
            pages: (promptboot::repl_contract::SESSION_BYTES + 4_095) / 4_096,
            active: promptboot::repl_contract::SESSION_BYTES,
            guard: 0,
        }
    } else {
        REGIONS[index]
    }
}

#[derive(Clone, Copy)]
pub(super) struct Failure {
    pub(super) code: &'static [u8],
    pub(super) phase: &'static [u8],
    pub(super) efi: Option<EfiStatus>,
    pub(super) model: Option<ModelError>,
    pub(super) inference: Option<InferenceError>,
    pub(super) region: Option<&'static [u8]>,
}

impl Failure {
    pub(super) const fn simple(code: &'static [u8], phase: &'static [u8]) -> Self {
        Self {
            code,
            phase,
            efi: None,
            model: None,
            inference: None,
            region: None,
        }
    }

    fn contract(tuple: ErrorTuple) -> Self {
        Self {
            code: tuple.code.as_bytes(),
            phase: tuple.phase.as_bytes(),
            efi: tuple.efi.map(|value| value as EfiStatus),
            model: None,
            inference: None,
            region: tuple.region.map(str::as_bytes),
        }
    }
}

impl Failure {
    pub(super) fn model(code: &'static [u8], phase: &'static [u8], error: ModelError) -> Self {
        Self {
            code,
            phase,
            efi: None,
            model: Some(error),
            inference: None,
            region: None,
        }
    }

    pub(super) fn inference(
        code: &'static [u8],
        phase: &'static [u8],
        error: InferenceError,
    ) -> Self {
        Self {
            code,
            phase,
            efi: None,
            model: None,
            inference: Some(error),
            region: None,
        }
    }

    pub(super) fn efi(code: &'static [u8], phase: &'static [u8], status: EfiStatus) -> Self {
        Self {
            code,
            phase,
            efi: Some(status),
            model: None,
            inference: None,
            region: None,
        }
    }
}

fn classified_model_failure(kind: ContractFailureKind, error: ModelError) -> Failure {
    let tuple = classified_failure(
        kind,
        [
            error.status as u64,
            error.domain as u64,
            error.index as u64,
            error.offset,
            error.needed,
            error.available,
            error.detail,
            0,
        ],
    );
    Failure {
        code: tuple.code.as_bytes(),
        phase: tuple.phase.as_bytes(),
        efi: None,
        model: Some(error),
        inference: None,
        region: None,
    }
}

fn classified_inference_failure(kind: ContractFailureKind, error: InferenceError) -> Failure {
    let tuple = classified_failure(
        kind,
        [
            error.status as u64,
            error.domain as u64,
            error.layer as u64,
            error.position as u64,
            error.tensor_id as u64,
            error.needed,
            error.available,
            error.detail,
        ],
    );
    Failure {
        code: tuple.code.as_bytes(),
        phase: tuple.phase.as_bytes(),
        efi: None,
        model: None,
        inference: Some(error),
        region: None,
    }
}

struct Session {
    boot: *mut BootServices,
    image: EfiHandle,
    device: EfiHandle,
    loaded_open: bool,
    simple_open: bool,
    root: *mut FileProtocol,
    file: *mut FileProtocol,
    root_close_valid: bool,
    file_close_valid: bool,
    addresses: [u64; 6],
}

impl Session {
    const fn new(boot: *mut BootServices, image: EfiHandle) -> Self {
        Self {
            boot,
            image,
            device: null_mut(),
            loaded_open: false,
            simple_open: false,
            root: null_mut(),
            file: null_mut(),
            root_close_valid: false,
            file_close_valid: false,
            addresses: [0; 6],
        }
    }

    fn acquired_mask(&self) -> u32 {
        acquired_mask(
            !self.file.is_null(),
            !self.root.is_null(),
            self.simple_open,
            self.loaded_open,
            self.addresses.map(|address| address != 0),
        )
    }

    unsafe fn cleanup(&mut self) -> (u32, u32, Option<(&'static [u8], EfiStatus)>) {
        let attempted = self.acquired_mask();
        let mut callable = attempted;
        let mut failed = 0u32;
        let mut first = None;
        let mut pending = attempted;
        while let Some(bit_index) = next_cleanup_bit(pending) {
            let bit = 1u32 << bit_index;
            pending &= !bit;
            let (is_callable, status, region) = match bit_index {
                0 => {
                    let valid = self.file_close_valid;
                    let status = if valid {
                        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                        (transmute::<usize, FileClose>((*self.file).close))(self.file)
                    } else {
                        EFI_SUCCESS
                    };
                    self.file = null_mut();
                    (valid, status, b"none" as &'static [u8])
                }
                1 => {
                    let valid = self.root_close_valid;
                    let status = if valid {
                        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                        (transmute::<usize, FileClose>((*self.root).close))(self.root)
                    } else {
                        EFI_SUCCESS
                    };
                    self.root = null_mut();
                    (valid, status, b"none" as &'static [u8])
                }
                2 => {
                    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                    let status = ((*self.boot).close_protocol)(
                        self.device,
                        &SIMPLE_FS_GUID,
                        self.image,
                        null_mut(),
                    );
                    self.simple_open = false;
                    (true, status, b"none" as &'static [u8])
                }
                3 => {
                    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                    let status = ((*self.boot).close_protocol)(
                        self.image,
                        &LOADED_IMAGE_GUID,
                        self.image,
                        null_mut(),
                    );
                    self.loaded_open = false;
                    (true, status, b"none" as &'static [u8])
                }
                4..=9 => {
                    let index = 9 - bit_index as usize;
                    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                    let status =
                        ((*self.boot).free_pages)(self.addresses[index], REGIONS[index].pages);
                    self.addresses[index] = 0;
                    (true, status, REGIONS[index].name)
                }
                _ => unreachable!(),
            };
            if !is_callable {
                callable &= !bit;
            } else if status != EFI_SUCCESS {
                failed |= bit;
                if first.is_none() {
                    first = Some((region, status));
                }
            }
        }
        let derived = cleanup_plan(attempted, callable, failed);
        (derived.attempted, derived.ok, first)
    }

    unsafe fn close_storage(&mut self) -> Result<(), Failure> {
        if !self.file.is_null() {
            if !self.file_close_valid {
                return Err(Failure::simple(b"MODEL_FILE_ABI", b"close"));
            }
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            let status = (transmute::<usize, FileClose>((*self.file).close))(self.file);
            if status != EFI_SUCCESS {
                return Err(Failure::efi(b"MODEL_FILE_CLOSE", b"close", status));
            }
            self.file = null_mut();
        }
        if !self.root.is_null() {
            if !self.root_close_valid {
                return Err(Failure::simple(b"MODEL_ROOT_ABI", b"close"));
            }
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            let status = (transmute::<usize, FileClose>((*self.root).close))(self.root);
            if status != EFI_SUCCESS {
                return Err(Failure::efi(b"MODEL_ROOT_CLOSE", b"close", status));
            }
            self.root = null_mut();
        }
        if self.simple_open {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            let status =
                ((*self.boot).close_protocol)(self.device, &SIMPLE_FS_GUID, self.image, null_mut());
            if status != EFI_SUCCESS {
                return Err(Failure::efi(b"MODEL_SIMPLE_CLOSE", b"close", status));
            }
            self.simple_open = false;
        }
        if self.loaded_open {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            let status = ((*self.boot).close_protocol)(
                self.image,
                &LOADED_IMAGE_GUID,
                self.image,
                null_mut(),
            );
            if status != EFI_SUCCESS {
                return Err(Failure::efi(b"MODEL_LOADED_CLOSE", b"close", status));
            }
            self.loaded_open = false;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Timing {
    starts: [u64; 3],
    ends: [u64; 3],
    aux_values: [u32; 3],
    deltas: [u64; 3],
    pub(super) hz: u64,
    aux: u32,
    pub(super) timed: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CalibrationEvidence {
    starts: [u64; 3],
    ends: [u64; 3],
    aux_values: [u32; 3],
}

static mut CALIBRATION_EVIDENCE: CalibrationEvidence = CalibrationEvidence {
    starts: [0; 3],
    ends: [0; 3],
    aux_values: [0; 3],
};

pub(super) unsafe fn run(
    image: EfiHandle,
    system_table: *mut SystemTable,
    boot: *mut BootServices,
    mode: &str,
    build_id: &str,
) -> EfiStatus {
    let timing = match calibrate(boot, mode) {
        Ok(value) => value,
        Err(failure) => {
            emit_failure(failure, 0, 0);
            return EFI_DEVICE_ERROR;
        }
    };
    let mut session = Session::new(boot, image);
    let outcome = execute(&mut session, system_table, mode, build_id, timing);
    let record_failure = matches!(&outcome, Err(failure) if failure.code == b"MODEL_RECORD");
    let (attempted, ok, cleanup_error) = session.cleanup();
    let primary_selection = primary_survives_cleanup(
        outcome.as_ref().err().map(|_| 1u8),
        cleanup_error.map(|_| 2u8),
    );
    let status = match outcome {
        Err(failure) => {
            emit_failure(failure, attempted, ok);
            EFI_DEVICE_ERROR
        }
        Ok(()) if primary_selection == Some(2) => {
            let (_, status) = cleanup_error.unwrap();
            let bit = next_cleanup_bit(attempted & !ok).unwrap_or(0);
            let mut failure = Failure::contract(cleanup_failure(bit, None));
            failure.efi = Some(status);
            emit_failure(failure, attempted, ok);
            EFI_DEVICE_ERROR
        }
        Ok(()) => {
            let mut record = Record::new();
            record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_CLEANUP_COMPLETE attempted=");
            push_hex(&mut record, attempted as u64, 8);
            record.push(b" ok=");
            push_hex(&mut record, ok as u64, 8);
            record.push(b" first_error=none\r\n");
            if record.overflowed() {
                emit_failure(Failure::simple(b"MODEL_RECORD", b"record"), attempted, ok);
                return EFI_DEVICE_ERROR;
            }
            if !emit_record(record.as_bytes()) {
                return EFI_DEVICE_ERROR;
            }
            let mut complete = Record::new();
            complete.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_TARGET_COMPLETE mode=");
            complete.push(mode.as_bytes());
            complete.push(b" build_id=");
            complete.push(build_id.as_bytes());
            complete.push(b"\r\n");
            if complete.overflowed() {
                emit_failure(Failure::simple(b"MODEL_RECORD", b"record"), attempted, ok);
                return EFI_DEVICE_ERROR;
            }
            if !emit_record(complete.as_bytes()) {
                return EFI_DEVICE_ERROR;
            }
            EFI_SUCCESS
        }
    };
    // Keep the persistent application resident long enough for both physical
    // evidence writes to be captured before firmware can reconnect the image.
    if !record_failure {
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let _ = ((*boot).stall)(2_000_000);
    }
    status
}

unsafe fn execute(
    session: &mut Session,
    system_table: *mut SystemTable,
    mode: &str,
    build_id: &str,
    calibration: Timing,
) -> Result<(), Failure> {
    let mut interface: *mut c_void = null_mut();
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*session.boot).open_protocol)(
        session.image,
        &LOADED_IMAGE_GUID,
        &mut interface,
        session.image,
        null_mut(),
        EFI_OPEN_PROTOCOL_GET_PROTOCOL,
    );
    session.loaded_open = status == EFI_SUCCESS;
    let (revision, device_present) = if interface.is_null() {
        (0, false)
    } else {
        let loaded = &*interface.cast::<LoadedImage>();
        (loaded.revision, !loaded.device_handle.is_null())
    };
    if let Some(tuple) = loaded_transition(
        status as u64,
        !interface.is_null(),
        revision,
        device_present,
    ) {
        return Err(Failure::contract(tuple));
    }
    let loaded = &*interface.cast::<LoadedImage>();
    session.device = loaded.device_handle;

    interface = null_mut();
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*session.boot).open_protocol)(
        session.device,
        &SIMPLE_FS_GUID,
        &mut interface,
        session.image,
        null_mut(),
        EFI_OPEN_PROTOCOL_GET_PROTOCOL,
    );
    session.simple_open = status == EFI_SUCCESS;
    let (revision, open_volume) = if interface.is_null() {
        (0, false)
    } else {
        let simple = &*interface.cast::<SimpleFileSystem>();
        (simple.revision, simple.open_volume != 0)
    };
    if let Some(tuple) =
        simple_transition(status as u64, !interface.is_null(), revision, open_volume)
    {
        return Err(Failure::contract(tuple));
    }
    let simple = &*interface.cast::<SimpleFileSystem>();
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status =
        (transmute::<usize, OpenVolume>(simple.open_volume))(interface.cast(), &mut session.root);
    if let Some(tuple) = volume_transition(status as u64, !session.root.is_null()) {
        return Err(Failure::contract(tuple));
    }
    session.root_close_valid = (*session.root).close != 0;
    validate_file_abi(session.root, b"root_abi")?;

    const PATH: [u16; 11] = [92, 77, 79, 68, 69, 76, 46, 80, 66, 84, 0];
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = (transmute::<usize, FileOpen>((*session.root).open))(
        session.root,
        &mut session.file,
        PATH.as_ptr(),
        1,
        0,
    );
    if let Some(tuple) = file_open_transition(status as u64, !session.file.is_null()) {
        return Err(Failure::contract(tuple));
    }
    session.file_close_valid = (*session.file).close != 0;
    validate_file_abi(session.file, b"file_abi")?;

    let mut started = Record::new();
    started.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_LOAD_STARTED mode=");
    started.push(mode.as_bytes());
    started.push(b" build_id=");
    started.push(build_id.as_bytes());
    started.push(b" path=\\MODEL.PBT model_bytes=426762944\r\n");
    emit_built(&started)?;

    for index in 0..REGIONS.len() {
        let spec = region_spec(index, mode);
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let status = ((*session.boot).allocate_pages)(
            ALLOCATE_ANY_PAGES,
            EFI_LOADER_DATA,
            spec.pages,
            &mut session.addresses[index],
        );
        if status != EFI_SUCCESS {
            return Err(Failure::contract(allocation_status_failure(
                index,
                status as u64,
            )));
        }
        if session.addresses[index] == 0 || session.addresses[index] & 63 != 0 {
            return Err(Failure::contract(allocation_status_failure(
                index,
                EFI_OUT_OF_RESOURCES as u64,
            )));
        }
        if spec.guard != 0 {
            let committed = spec.pages * 4096;
            let guard_start = if index == 5 {
                spec.active - spec.guard
            } else {
                spec.active
            };
            slice::from_raw_parts_mut(
                (session.addresses[index] as *mut u8).add(guard_start),
                committed - guard_start,
            )
            .fill(0xa5);
        }
        emit_region(b"allocated", spec, session.addresses[index], 0, 0)?;
    }
    ensure_nonoverlap(&session.addresses)?;

    let weights = slice::from_raw_parts_mut(session.addresses[0] as *mut u8, MODEL_BYTES);
    let mut progress = ReadProgress::new(MODEL_BYTES);
    let mut model_sha256 = Sha256::new();
    let mut load_start = 0u64;
    while progress.offset < MODEL_BYTES {
        let old_offset = progress.offset;
        let requested = core::cmp::min(MAX_READ, MODEL_BYTES - progress.offset);
        let mut count = requested;
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        if progress.offset == 0 {
            load_start = timing_start(calibration);
        }
        let status = (transmute::<usize, FileRead>((*session.file).read))(
            session.file,
            &mut count,
            weights.as_mut_ptr().add(progress.offset).cast(),
        );
        let decision = progress.apply(requested, count, status as u64, false);
        match decision {
            ReadDecision::Advance(count) => {
                model_sha256.update(&weights[old_offset..old_offset + count]);
            }
            _ => {
                return Err(Failure::contract(
                    read_failure(decision, status as u64).unwrap(),
                ));
            }
        }
    }
    let mut probe = 0u8;
    let mut probe_bytes = 1usize;
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = (transmute::<usize, FileRead>((*session.file).read))(
        session.file,
        &mut probe_bytes,
        (&mut probe as *mut u8).cast(),
    );
    let decision = progress.apply(1, probe_bytes, status as u64, true);
    if !matches!(decision, ReadDecision::Complete) {
        return Err(Failure::contract(
            read_failure(decision, status as u64).unwrap(),
        ));
    }
    let load_ticks = timing_elapsed(calibration, load_start)?;
    emit_checked(b"PROMPTBOOT_EVENT v=1 event=MODEL_READ_COMPLETE bytes=426762944 max_chunk=1048576 eof_probe_bytes=0\r\n")?;

    establish_fp_state();
    let verify_start = timing_start(calibration);
    let actual_sha256 = model_sha256.finish();
    if let Some(error) =
        ModelError::full_hash_mismatch(&parse_sha(EXPECTED_SHA_HEX), &actual_sha256)
    {
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        return Err(classified_model_failure(
            ContractFailureKind::ModelChecksum,
            error,
        ));
    }
    let model = match ModelView::open_authenticated(weights) {
        Ok(value) => value,
        Err(error) => {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            return Err(classified_model_failure(
                ContractFailureKind::ModelIdentity,
                error,
            ));
        }
    };
    let verify_ticks = timing_elapsed(calibration, verify_start)?;
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    emit_record_verified()?;

    let index = slice::from_raw_parts_mut(session.addresses[1] as *mut u8, INDEX_BYTES);
    index.fill(0);
    establish_fp_state();
    let index_start = timing_start(calibration);
    let tokenizer = match FrozenTokenizer::build(&model, index) {
        Ok(value) => value,
        Err(error) => {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            return Err(classified_model_failure(
                ContractFailureKind::ModelIndex,
                error,
            ));
        }
    };
    let index_ticks = timing_elapsed(calibration, index_start)?;
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let _ = tokenizer.usage();
    if mode == "model_repl" {
        emit_arenas_ready(mode, b"none")?;
        session.close_storage()?;
        let result = super::model_repl::run(
            session.image,
            system_table,
            session.boot,
            &model,
            &tokenizer,
            session.addresses,
            calibration,
        );
        drop(tokenizer);
        drop(model);
        return result;
    }
    emit_arenas_ready(mode, b"a5")?;

    let mut init_ticks = 0;
    let mut prefill_ticks = 0;
    let mut first_token_ticks = 0;
    let mut kv_current = 0;
    let mut kv_high = 0;
    let mut scratch_high = 0;
    if mode == "model_first_token" {
        let kv = slice::from_raw_parts_mut(session.addresses[2] as *mut u8, KV_BYTES);
        let scratch = slice::from_raw_parts_mut(session.addresses[3] as *mut u8, SCRATCH_BYTES);
        let logits = slice::from_raw_parts_mut(session.addresses[4] as *mut u32, LOGIT_WORDS);
        let prompt_bytes = slice::from_raw_parts_mut(session.addresses[5] as *mut u8, 2_048);
        prompt_bytes.fill(0xa5);
        core::ptr::copy_nonoverlapping(
            PROMPT.as_ptr().cast::<u8>(),
            prompt_bytes.as_mut_ptr(),
            120,
        );
        let prompt_tokens = slice::from_raw_parts(session.addresses[5] as *const u32, PROMPT.len());
        establish_fp_state();
        let init_start = timing_start(calibration);
        let mut engine = match InferenceEngine::build(&model, kv, scratch) {
            Ok(value) => value,
            Err(error) => {
                restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                return Err(classified_inference_failure(
                    ContractFailureKind::Engine,
                    error,
                ));
            }
        };
        init_ticks = timing_elapsed(calibration, init_start)?;
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        emit_checked(b"PROMPTBOOT_EVENT v=1 event=MODEL_PREFILL_STARTED prompt_tokens=30 prompt_sha256=57be9252afa09f8300997803322e86a9044ad274f92f7cfc9f6d1d751d68c52b reserve=16\r\n")?;
        establish_fp_state();
        let prefill_start = timing_start(calibration);
        let step = match engine.prefill(prompt_tokens, GENERATION_RESERVE, logits) {
            Ok(value) => value,
            Err(error) => {
                restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
                return Err(classified_inference_failure(
                    ContractFailureKind::Prefill,
                    error,
                ));
            }
        };
        prefill_ticks = timing_elapsed(calibration, prefill_start)?;
        let mut top = [TopLogit {
            token: 0,
            logit_bits: 0,
        }; 8];
        if let Err(error) = top_logits_8(logits, &mut top) {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            return Err(classified_inference_failure(
                ContractFailureKind::Top8,
                error,
            ));
        }
        first_token_ticks = timing_elapsed(calibration, load_start)?;
        let usage = engine.usage();
        kv_current = usage.kv.current;
        kv_high = usage.kv.high_water;
        scratch_high = usage.scratch.high_water;
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        emit_first_token(
            step.position,
            step.selected_token,
            step.selected_logit_bits,
            step.eos,
            &top,
            kv_current,
        )?;
        drop(engine);
    }
    drop(tokenizer);
    drop(model);
    verify_guards(session, mode)?;

    for (index, spec) in REGIONS.iter().copied().enumerate() {
        let (current, high) = match index {
            0 => (MODEL_BYTES as u64, MODEL_BYTES as u64),
            1 => (INDEX_BYTES as u64, INDEX_BYTES as u64),
            2 => (kv_current, kv_high),
            3 => (0, scratch_high),
            4 if mode == "model_first_token" => {
                ((LOGIT_WORDS * 4) as u64, (LOGIT_WORDS * 4) as u64)
            }
            5 if mode == "model_first_token" => (120, 120),
            _ => (0, 0),
        };
        emit_region(b"final", spec, session.addresses[index], current, high)?;
    }
    emit_timing(
        calibration,
        load_ticks,
        verify_ticks,
        index_ticks,
        init_ticks,
        prefill_ticks,
        first_token_ticks,
    )?;
    Ok(())
}

unsafe fn validate_file_abi(file: *mut FileProtocol, phase: &'static [u8]) -> Result<(), Failure> {
    match file_abi_transition(
        phase == b"root_abi",
        (*file).revision,
        (*file).open != 0,
        (*file).close != 0,
        (*file).read != 0,
    ) {
        Some(tuple) => Err(Failure::contract(tuple)),
        None => Ok(()),
    }
}

fn ensure_nonoverlap(addresses: &[u64; 6]) -> Result<(), Failure> {
    for left in 0..6 {
        let left_end = addresses[left]
            .checked_add((REGIONS[left].pages * 4096) as u64)
            .ok_or(Failure::simple(b"MODEL_OOM", b"allocate"))?;
        for right in left + 1..6 {
            let right_end = addresses[right]
                .checked_add((REGIONS[right].pages * 4096) as u64)
                .ok_or(Failure::simple(b"MODEL_OOM", b"allocate"))?;
            if addresses[left] < right_end && addresses[right] < left_end {
                return Err(Failure {
                    code: b"MODEL_OOM",
                    phase: b"allocate",
                    efi: Some(EFI_OUT_OF_RESOURCES),
                    model: None,
                    inference: None,
                    region: Some(REGIONS[right].name),
                });
            }
        }
    }
    Ok(())
}

unsafe fn verify_guards(session: &Session, mode: &str) -> Result<(), Failure> {
    let tail_ok = |index: usize| {
        let spec = REGIONS[index];
        slice::from_raw_parts(
            (session.addresses[index] as *const u8).add(spec.active),
            spec.pages * 4096 - spec.active,
        )
        .iter()
        .all(|value| *value == 0xa5)
    };
    let prompt_written = if mode == "model_first_token" { 120 } else { 0 };
    let prompt_slack = slice::from_raw_parts(
        (session.addresses[5] as *const u8).add(prompt_written),
        REGIONS[5].active - prompt_written,
    );
    let valid = [
        tail_ok(0),
        tail_ok(1),
        tail_ok(4),
        prompt_slack.iter().all(|value| *value == 0xa5),
        tail_ok(5),
    ];
    if let Some(failure) = first_guard_failure(valid) {
        return Err(Failure::contract(guard_failure(failure)));
    }
    Ok(())
}

unsafe fn calibrate(boot: *mut BootServices, mode: &str) -> Result<Timing, Failure> {
    let highest = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    let rdtscp =
        highest >= 0x8000_0001 && core::arch::x86_64::__cpuid(0x8000_0001).edx & (1 << 27) != 0;
    if !rdtscp {
        return Err(Failure::contract(timer_failure(None)));
    }
    let invariant =
        highest >= 0x8000_0007 && core::arch::x86_64::__cpuid(0x8000_0007).edx & (1 << 8) != 0;
    let compile_mode = Mode::parse(mode).ok_or(Failure::contract(timer_failure(None)))?;
    match timer_class(compile_mode, rdtscp, invariant) {
        Ok(TimerClass::UntimedNonInvariant) => {
            core::ptr::write_volatile(
                core::ptr::addr_of_mut!(CALIBRATION_EVIDENCE),
                CalibrationEvidence {
                    starts: [0; 3],
                    ends: [0; 3],
                    aux_values: [0; 3],
                },
            );
            return Ok(Timing {
                starts: [0; 3],
                ends: [0; 3],
                aux_values: [0; 3],
                deltas: [0; 3],
                hz: 0,
                aux: 0,
                timed: false,
            });
        }
        Err(()) => return Err(Failure::contract(timer_failure(None))),
        Ok(TimerClass::TimedInvariant) => {}
    }
    let mut deltas = [0u64; 3];
    let mut starts = [0u64; 3];
    let mut ends = [0u64; 3];
    let mut aux_values = [0u32; 3];
    let mut expected_aux = None;
    for index in 0..3 {
        let before = rdtsc_start();
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let status = ((*boot).stall)(100_000);
        let (after, after_aux) = rdtscp_read();
        if status != EFI_SUCCESS
            || after <= before
            || expected_aux.is_some_and(|value| value != after_aux)
        {
            return Err(Failure::contract(timer_failure(if status == EFI_SUCCESS {
                None
            } else {
                Some(status as u64)
            })));
        }
        expected_aux = Some(after_aux);
        starts[index] = before;
        ends[index] = after;
        aux_values[index] = after_aux;
        deltas[index] = after - before;
    }
    let hz =
        derive_frequency(deltas, aux_values).map_err(|_| Failure::contract(timer_failure(None)))?;
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(CALIBRATION_EVIDENCE),
        CalibrationEvidence {
            starts,
            ends,
            aux_values,
        },
    );
    Ok(Timing {
        starts,
        ends,
        aux_values,
        deltas,
        hz,
        aux: expected_aux.unwrap_or(0),
        timed: true,
    })
}

#[inline]
unsafe fn rdtsc_start() -> u64 {
    let low: u32;
    let high: u32;
    asm!("lfence", "rdtsc", out("eax") low, out("edx") high, options(nostack));
    ((high as u64) << 32) | low as u64
}

#[inline]
unsafe fn rdtscp_read() -> (u64, u32) {
    let low: u32;
    let high: u32;
    let aux: u32;
    asm!("rdtscp", "lfence", out("eax") low, out("edx") high, out("ecx") aux, options(nostack));
    (((high as u64) << 32) | low as u64, aux)
}

#[inline]
unsafe fn rdtsc_end(expected_aux: u32) -> Result<u64, Failure> {
    let (value, aux) = rdtscp_read();
    if aux != expected_aux {
        Err(Failure::contract(timer_failure(None)))
    } else {
        Ok(value)
    }
}

#[inline]
pub(super) unsafe fn timing_start(timing: Timing) -> u64 {
    if timing.timed {
        rdtsc_start()
    } else {
        0
    }
}

pub(super) unsafe fn timing_end(timing: Timing) -> Result<u64, Failure> {
    if timing.timed {
        rdtsc_end(timing.aux)
    } else {
        Ok(0)
    }
}

#[inline]
unsafe fn timing_elapsed(timing: Timing, start: u64) -> Result<u64, Failure> {
    if timing.timed {
        let end = rdtsc_end(timing.aux)?;
        end.checked_sub(start)
            .filter(|delta| *delta != 0)
            .ok_or(Failure::contract(timer_failure(None)))
    } else {
        Ok(0)
    }
}

fn emit_record_verified() -> Result<(), Failure> {
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_VERIFIED format=PBTQW25 version=1 sha256=");
    record.push(EXPECTED_SHA_HEX.as_bytes());
    record.push(b" source_sha256=");
    record.push(SOURCE_SHA.as_bytes());
    record.push(b" tensors=291 vocab=151936\r\n");
    unsafe { emit_built(&record) }
}

fn emit_arenas_ready(mode: &str, prompt_slack: &[u8]) -> Result<(), Failure> {
    let mut pages = 0usize;
    for index in 0..REGIONS.len() {
        pages += region_spec(index, mode).pages;
    }
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_ARENAS_READY regions=6 pages=");
    record.push_decimal(pages as u64);
    record.push(b" committed=");
    record.push_decimal((pages * 4_096) as u64);
    record.push(b" aligned=1 nonoverlap=1 canaries=ready prompt_slack=");
    record.push(prompt_slack);
    record.push(b" index_sha256=");
    record.push(TOKENIZER_INDEX_SHA256_HEX);
    record.push(b"\r\n");
    unsafe { emit_built(&record) }
}

fn emit_region(
    phase: &[u8],
    spec: RegionSpec,
    base: u64,
    current: u64,
    high: u64,
) -> Result<(), Failure> {
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_REGION phase=");
    record.push(phase);
    record.push(b" name=");
    record.push(spec.name);
    record.push(b" base=");
    push_hex(&mut record, base, 16);
    record.push(b" requested=");
    record.push_decimal(spec.active as u64);
    record.push(b" committed=");
    record.push_decimal((spec.pages * 4096) as u64);
    record.push(b" current=");
    record.push_decimal(current);
    record.push(b" high_water=");
    record.push_decimal(high);
    record.push(b" guard_bytes=");
    record.push_decimal(spec.guard as u64);
    record.push(b" guard=");
    record.push(if spec.guard == 0 { b"none" } else { b"a5" });
    record.push(b"\r\n");
    unsafe { emit_built(&record) }
}

fn emit_first_token(
    position: u32,
    selected: u32,
    bits: u32,
    eos: u32,
    top: &[TopLogit; 8],
    kv: u64,
) -> Result<(), Failure> {
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_FIRST_TOKEN position=");
    record.push_decimal(position as u64);
    record.push(b" selected_id=");
    record.push_decimal(selected as u64);
    record.push(b" selected_logit_bits=");
    push_hex(&mut record, bits as u64, 8);
    record.push(b" eos=");
    record.push_decimal(eos as u64);
    record.push(b" top8=");
    for (index, item) in top.iter().enumerate() {
        if index != 0 {
            record.push(b",");
        }
        record.push_decimal(item.token as u64);
        record.push(b":");
        push_hex(&mut record, item.logit_bits as u64, 8);
    }
    record.push(b" kv_current=");
    record.push_decimal(kv);
    record.push(b"\r\n");
    unsafe { emit_built(&record) }
}

fn emit_timing(
    timing: Timing,
    load: u64,
    verify: u64,
    index: u64,
    init: u64,
    prefill: u64,
    token: u64,
) -> Result<(), Failure> {
    debug_assert!(
        !timing.timed
            || timing
                .starts
                .iter()
                .zip(timing.ends.iter())
                .all(|(start, end)| start < end)
    );
    debug_assert!(!timing.timed || timing.aux_values.iter().all(|value| *value == timing.aux));
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MODEL_TIMING calibration_us=");
    record.push(if timing.timed { b"100000" } else { b"0" });
    record.push(b" delta0=");
    record.push_decimal(timing.deltas[0]);
    record.push(b" delta1=");
    record.push_decimal(timing.deltas[1]);
    record.push(b" delta2=");
    record.push_decimal(timing.deltas[2]);
    record.push(b" tsc_hz=");
    record.push_decimal(timing.hz);
    record.push(b" load_ticks=");
    record.push_decimal(load);
    record.push(b" verify_ticks=");
    record.push_decimal(verify);
    record.push(b" index_ticks=");
    record.push_decimal(index);
    record.push(b" init_ticks=");
    record.push_decimal(init);
    record.push(b" prefill_ticks=");
    record.push_decimal(prefill);
    record.push(b" first_token_ticks=");
    record.push_decimal(token);
    record.push(b"\r\n");
    unsafe { emit_built(&record) }
}

unsafe fn emit_built(record: &Record) -> Result<(), Failure> {
    match record_emission(record) {
        Ok(bytes) => emit_checked(bytes),
        Err(tuple) => Err(Failure::contract(tuple)),
    }
}

unsafe fn emit_checked(bytes: &[u8]) -> Result<(), Failure> {
    if emit_record(bytes) {
        Ok(())
    } else {
        Err(Failure::simple(b"MODEL_RECORD", b"record"))
    }
}

fn emit_failure(failure: Failure, attempted: u32, ok: u32) {
    unsafe { write_fatal_diagnostic(failure.code) };
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=FATAL code=");
    record.push(failure.code);
    record.push(b" phase=");
    record.push(failure.phase);
    record.push(b" efi_status=");
    if let Some(value) = failure.efi {
        push_hex(&mut record, value as u64, 16);
    } else {
        record.push(b"none");
    }
    record.push(b" model_error=");
    if let Some(error) = failure.model {
        for (at, value) in [
            error.status as u64,
            error.domain as u64,
            error.index as u64,
            error.offset,
            error.needed,
            error.available,
            error.detail,
        ]
        .iter()
        .enumerate()
        {
            if at != 0 {
                record.push(b",");
            }
            record.push_decimal(*value);
        }
    } else {
        record.push(b"none");
    }
    record.push(b" inference_error=");
    if let Some(error) = failure.inference {
        for (at, value) in [
            error.status as u64,
            error.domain as u64,
            error.layer as u64,
            error.position as u64,
            error.tensor_id as u64,
            error.needed,
            error.available,
            error.detail,
        ]
        .iter()
        .enumerate()
        {
            if at != 0 {
                record.push(b",");
            }
            record.push_decimal(*value);
        }
    } else {
        record.push(b"none");
    }
    record.push(b" region=");
    record.push(failure.region.unwrap_or(b"none"));
    record.push(b" cleanup_attempted=");
    push_hex(&mut record, attempted as u64, 8);
    record.push(b" cleanup_ok=");
    push_hex(&mut record, ok as u64, 8);
    record.push(b"\r\n");
    unsafe { emit_record(record.as_bytes()) };
}

fn push_hex(record: &mut Record, value: u64, width: usize) {
    let digits = b"0123456789abcdef";
    let mut at = width;
    while at != 0 {
        at -= 1;
        record.push(&[digits[((value >> (at * 4)) & 15) as usize]]);
    }
}

const fn hex_digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => 0xff,
    }
}

const fn parse_sha(value: &str) -> [u8; 32] {
    let input = value.as_bytes();
    let mut output = [0u8; 32];
    let mut index = 0;
    while index < 32 {
        let high = hex_digit(input[index * 2]);
        let low = hex_digit(input[index * 2 + 1]);
        output[index] = (high << 4) | low;
        index += 1;
    }
    output
}

const _: [u8; 32] = parse_sha("b0f98ed6e0557ca35e1bced1000c950b3c84414251df65290315a7969981d42d");
const _: &str = PROMPT_SHA;
const _: usize = EFI_BUFFER_TOO_SMALL;
