#![no_std]
#![no_main]

use core::arch::asm;
use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::panic::PanicInfo;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use promptboot::console_contract::{
    clear_result, read_result, reset_result, validate_bindings, wait_result,
};
use promptboot::console_history::ConsoleHistory;
use promptboot::editor::{Editor, Flow, Output};
use promptboot::model_contract::{record_emission, ModelRecord as Record, MODEL_RECORD_FATAL};
use promptboot::status_bar::{
    bounded_status_geometry, is_event_boundary_line, is_serial_evidence_line, loading_line,
    native_scrolls_for_output, render_status_row, write_content_with_wrap,
};

mod model_repl;
mod model_target;
mod mp_inference;

type EfiStatus = usize;
type EfiHandle = *mut c_void;
type EfiEvent = *mut c_void;
type EfiPhysicalAddress = u64;

const EFI_SUCCESS: EfiStatus = 0;
const EFI_ERROR_BIT: EfiStatus = 1usize << (usize::BITS - 1);
const EFI_DEVICE_ERROR: EfiStatus = EFI_ERROR_BIT | 7;
const EFI_BUFFER_TOO_SMALL: EfiStatus = EFI_ERROR_BIT | 5;
const EFI_OPEN_PROTOCOL_GET_PROTOCOL: u32 = 0x0000_0002;
const BY_PROTOCOL: u32 = 2;
const ALLOCATE_ANY_PAGES: u32 = 0;
const EFI_LOADER_DATA: u32 = 2;
const EFI_CONVENTIONAL_MEMORY: u32 = 7;
const EVT_TIMER: u32 = 0x8000_0000;
const TPL_APPLICATION: usize = 4;

const SERIAL_IO_GUID: Guid = Guid::new(
    0xbb25_cf6f,
    0xf1d4,
    0x11d2,
    [0x9a, 0x0c, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0xfd],
);
const SIMPLE_TEXT_INPUT_EX_GUID: Guid = Guid::new(
    0xdd9e_7534,
    0x7762,
    0x4698,
    [0x8c, 0x14, 0xf5, 0x85, 0x17, 0xa6, 0x25, 0xaa],
);
const LOADED_IMAGE_GUID: Guid = Guid::new(
    0x5b1b_31a1,
    0x9562,
    0x11d2,
    [0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
);

const BUILD_ID: &str = env!("PROMPTBOOT_BUILD_ID");
const SELF_TEST: &str = env!("PROMPTBOOT_SELF_TEST");
const COMPILE_MODE: &str = match option_env!("PROMPTBOOT_MODE") {
    Some(value) => value,
    None => "echo_repl",
};
const PANIC_ASCII: &[u8] = b"PROMPTBOOT_EVENT v=1 event=FATAL code=PANIC\r\n";
const PANIC_UTF16: [u16; 18] = ascii_utf16_terminated(b"promptboot: panic\r\n");

static CONOUT_SINK: AtomicPtr<SimpleTextOutput> = AtomicPtr::new(null_mut());
static SERIAL_SINK: AtomicPtr<SerialIo> = AtomicPtr::new(null_mut());
static EVENT_DISPLAY_ENABLED: AtomicBool = AtomicBool::new(false);
static PANIC_EMITTED: AtomicBool = AtomicBool::new(false);
static FATAL_EMITTED: AtomicBool = AtomicBool::new(false);
static RECORD_FAILED: AtomicBool = AtomicBool::new(false);
static FP_STATE_READY: AtomicBool = AtomicBool::new(false);
static mut SAVED_FP_STATE: FpState = FpState::zero();

struct ConsoleHistoryCell(UnsafeCell<ConsoleHistory>);

// The firmware entry point and input loop access this cell synchronously on one CPU.
unsafe impl Sync for ConsoleHistoryCell {}

static CONSOLE_HISTORY: ConsoleHistoryCell =
    ConsoleHistoryCell(UnsafeCell::new(ConsoleHistory::new()));

#[derive(Clone, Copy)]
struct StatusConsole {
    active: bool,
    columns: usize,
}

struct StatusConsoleCell(UnsafeCell<StatusConsole>);

// Status and content rendering are serialized on the boot CPU.
unsafe impl Sync for StatusConsoleCell {}

static STATUS_CONSOLE: StatusConsoleCell = StatusConsoleCell(UnsafeCell::new(StatusConsole {
    active: false,
    columns: 0,
}));

struct StatusRowCell(UnsafeCell<[u16; promptboot::console_history::MAX_COLUMNS - 1]>);

// Status and content rendering are serialized on the boot CPU.
unsafe impl Sync for StatusRowCell {}

static STATUS_ROW: StatusRowCell = StatusRowCell(UnsafeCell::new(
    [b' ' as u16; promptboot::console_history::MAX_COLUMNS - 1],
));

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl Guid {
    const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

#[repr(C)]
struct TableHeader {
    signature: u64,
    revision: u32,
    header_size: u32,
    crc32: u32,
    reserved: u32,
}

#[repr(C)]
struct SystemTable {
    header: TableHeader,
    firmware_vendor: *mut u16,
    firmware_revision: u32,
    _pad0: u32,
    console_in_handle: EfiHandle,
    con_in: *mut SimpleTextInput,
    console_out_handle: EfiHandle,
    con_out: *mut SimpleTextOutput,
    standard_error_handle: EfiHandle,
    std_err: *mut SimpleTextOutput,
    runtime_services: *mut c_void,
    boot_services: *mut BootServices,
    number_of_table_entries: usize,
    configuration_table: *mut c_void,
}

type TextReset = unsafe extern "efiapi" fn(*mut SimpleTextOutput, bool) -> EfiStatus;
type TextOutputString = unsafe extern "efiapi" fn(*mut SimpleTextOutput, *const u16) -> EfiStatus;
type TextQueryMode =
    unsafe extern "efiapi" fn(*mut SimpleTextOutput, usize, *mut usize, *mut usize) -> EfiStatus;
type TextClearScreen = unsafe extern "efiapi" fn(*mut SimpleTextOutput) -> EfiStatus;
type TextEnableCursor = unsafe extern "efiapi" fn(*mut SimpleTextOutput, bool) -> EfiStatus;
type TextSetAttribute = unsafe extern "efiapi" fn(*mut SimpleTextOutput, usize) -> EfiStatus;
type TextSetCursorPosition =
    unsafe extern "efiapi" fn(*mut SimpleTextOutput, usize, usize) -> EfiStatus;

#[repr(C)]
struct SimpleTextOutput {
    reset: TextReset,
    output_string: TextOutputString,
    test_string: usize,
    query_mode: TextQueryMode,
    set_mode: usize,
    set_attribute: usize,
    clear_screen: TextClearScreen,
    set_cursor_position: usize,
    enable_cursor: usize,
    mode: *mut SimpleTextOutputMode,
}

#[repr(C)]
struct SimpleTextOutputMode {
    max_mode: i32,
    mode: i32,
    attribute: i32,
    cursor_column: i32,
    cursor_row: i32,
    cursor_visible: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EfiInputKey {
    scan_code: u16,
    unicode_char: u16,
}

#[repr(C)]
struct EfiKeyState {
    key_shift_state: u32,
    key_toggle_state: u8,
}

#[repr(C)]
struct EfiKeyData {
    key: EfiInputKey,
    key_state: EfiKeyState,
}

type InputReset = unsafe extern "efiapi" fn(*mut SimpleTextInput, bool) -> EfiStatus;
type ReadKeyStroke = unsafe extern "efiapi" fn(*mut SimpleTextInput, *mut EfiInputKey) -> EfiStatus;
type WaitForEvent = unsafe extern "efiapi" fn(usize, *const EfiEvent, *mut usize) -> EfiStatus;
type CreateEvent = unsafe extern "efiapi" fn(
    u32,
    usize,
    *const c_void,
    *const c_void,
    *mut EfiEvent,
) -> EfiStatus;
type SetTimer = unsafe extern "efiapi" fn(EfiEvent, u32, u64) -> EfiStatus;
type CloseEvent = unsafe extern "efiapi" fn(EfiEvent) -> EfiStatus;
type CheckEvent = unsafe extern "efiapi" fn(EfiEvent) -> EfiStatus;

#[repr(C)]
struct SimpleTextInput {
    reset: usize,
    read_key_stroke: usize,
    wait_for_key: EfiEvent,
}

type ReadKeyStrokeEx =
    unsafe extern "efiapi" fn(*mut SimpleTextInputEx, *mut EfiKeyData) -> EfiStatus;

#[repr(C)]
struct SimpleTextInputEx {
    reset: usize,
    read_key_stroke_ex: usize,
    wait_for_key_ex: EfiEvent,
    set_state: usize,
    register_key_notify: usize,
    unregister_key_notify: usize,
}

type AllocatePages =
    unsafe extern "efiapi" fn(u32, u32, usize, *mut EfiPhysicalAddress) -> EfiStatus;
type FreePages = unsafe extern "efiapi" fn(EfiPhysicalAddress, usize) -> EfiStatus;
type FreePool = unsafe extern "efiapi" fn(*mut c_void) -> EfiStatus;
type GetMemoryMap = unsafe extern "efiapi" fn(
    *mut usize,
    *mut MemoryDescriptor,
    *mut usize,
    *mut usize,
    *mut u32,
) -> EfiStatus;
type SetWatchdogTimer = unsafe extern "efiapi" fn(usize, u64, usize, *const u16) -> EfiStatus;
type Stall = unsafe extern "efiapi" fn(usize) -> EfiStatus;
type OpenProtocol = unsafe extern "efiapi" fn(
    EfiHandle,
    *const Guid,
    *mut *mut c_void,
    EfiHandle,
    EfiHandle,
    u32,
) -> EfiStatus;
type CloseProtocol =
    unsafe extern "efiapi" fn(EfiHandle, *const Guid, EfiHandle, EfiHandle) -> EfiStatus;
type LocateHandleBuffer = unsafe extern "efiapi" fn(
    u32,
    *const Guid,
    *mut c_void,
    *mut usize,
    *mut *mut EfiHandle,
) -> EfiStatus;

#[repr(C)]
struct BootServices {
    header: TableHeader,
    raise_tpl: usize,
    restore_tpl: usize,
    allocate_pages: AllocatePages,
    free_pages: FreePages,
    get_memory_map: GetMemoryMap,
    allocate_pool: usize,
    free_pool: FreePool,
    create_event: CreateEvent,
    set_timer: SetTimer,
    wait_for_event: usize,
    signal_event: usize,
    close_event: CloseEvent,
    check_event: CheckEvent,
    install_protocol_interface: usize,
    reinstall_protocol_interface: usize,
    uninstall_protocol_interface: usize,
    handle_protocol: usize,
    reserved: usize,
    register_protocol_notify: usize,
    locate_handle: usize,
    locate_device_path: usize,
    install_configuration_table: usize,
    load_image: usize,
    start_image: usize,
    exit: usize,
    unload_image: usize,
    exit_boot_services: usize,
    get_next_monotonic_count: usize,
    stall: Stall,
    set_watchdog_timer: SetWatchdogTimer,
    connect_controller: usize,
    disconnect_controller: usize,
    open_protocol: OpenProtocol,
    close_protocol: CloseProtocol,
    open_protocol_information: usize,
    protocols_per_handle: usize,
    locate_handle_buffer: LocateHandleBuffer,
    locate_protocol: usize,
    install_multiple_protocol_interfaces: usize,
    uninstall_multiple_protocol_interfaces: usize,
    calculate_crc32: usize,
    copy_mem: usize,
    set_mem: usize,
    create_event_ex: usize,
}

#[repr(C)]
struct MemoryDescriptor {
    memory_type: u32,
    _pad: u32,
    physical_start: u64,
    virtual_start: u64,
    number_of_pages: u64,
    attribute: u64,
}

type SerialReset = unsafe extern "efiapi" fn(*mut SerialIo) -> EfiStatus;
type SerialSetAttributes =
    unsafe extern "efiapi" fn(*mut SerialIo, u64, u32, u32, u32, u8, u32) -> EfiStatus;
type SerialWrite = unsafe extern "efiapi" fn(*mut SerialIo, *mut usize, *const c_void) -> EfiStatus;

#[repr(C)]
struct SerialIo {
    revision: u32,
    _pad: u32,
    reset: SerialReset,
    set_attributes: SerialSetAttributes,
    set_control: usize,
    get_control: usize,
    write: SerialWrite,
    read: usize,
    mode: *mut c_void,
}

#[repr(C)]
struct LoadedImage {
    revision: u32,
    _pad0: u32,
    parent_handle: EfiHandle,
    system_table: *mut SystemTable,
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

#[derive(Clone, Copy)]
#[repr(C)]
struct FpState {
    cr0: u64,
    cr4: u64,
    x87_control: u16,
    _pad: u16,
    mxcsr: u32,
}

impl FpState {
    const fn zero() -> Self {
        Self {
            cr0: 0,
            cr4: 0,
            x87_control: 0,
            _pad: 0,
            mxcsr: 0,
        }
    }
}

const fn ascii_utf16_terminated<const N: usize>(bytes: &[u8]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut index = 0;
    while index < bytes.len() && index + 1 < N {
        out[index] = bytes[index] as u16;
        index += 1;
    }
    out
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    if PANIC_EMITTED.swap(true, Ordering::SeqCst) {
        loop {
            core::hint::spin_loop();
        }
    }
    unsafe {
        if FP_STATE_READY.load(Ordering::Acquire) {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        }
        let conout = CONOUT_SINK.load(Ordering::Acquire);
        if !conout.is_null() {
            let _ = ((*conout).output_string)(conout, PANIC_UTF16.as_ptr());
        }
        let serial = SERIAL_SINK.load(Ordering::Acquire);
        if !serial.is_null() {
            let mut length = PANIC_ASCII.len();
            let _ = ((*serial).write)(serial, &mut length, PANIC_ASCII.as_ptr().cast());
        }
    }
    loop {
        core::hint::spin_loop();
    }
}

#[no_mangle]
extern "efiapi" fn efi_main(image_handle: EfiHandle, system_table: *mut SystemTable) -> EfiStatus {
    EVENT_DISPLAY_ENABLED.store(false, Ordering::Release);
    if system_table.is_null() {
        unsafe {
            fatal(b"SYSTEM_TABLE");
        }
    }
    unsafe {
        let conout = (*system_table).con_out;
        CONOUT_SINK.store(conout, Ordering::Release);
        let boot = (*system_table).boot_services;
        if boot.is_null() {
            fatal(b"BOOT_SERVICES");
        }

        if let Some(serial) = configure_serial(boot, image_handle) {
            SERIAL_SINK.store(serial, Ordering::Release);
        }

        if conout.is_null() {
            fatal(b"CONSOLE_OUTPUT_MISSING");
        }
        if let Err(error) = clear_result(((*conout).clear_screen)(conout)) {
            fatal(error.code());
        }
        let (physical_columns, physical_rows) = match initialize_console_history(conout) {
            Ok(geometry) => geometry,
            Err(_) => fatal(b"CONSOLE_MODE"),
        };

        if ((*boot).set_watchdog_timer)(0, 0, 0, null()) != EFI_SUCCESS {
            fatal(b"WATCHDOG_DISABLE");
        }

        let conventional_bytes = match measure_conventional_memory(boot) {
            Some(value) => value,
            None => fatal(b"MEMORY_MAP"),
        };

        let features = core::arch::x86_64::__cpuid(1).edx;
        let required = (1u32 << 24) | (1u32 << 25) | (1u32 << 26);
        if features & required != required {
            fatal(b"CPU_SSE2_UNAVAILABLE");
        }

        SAVED_FP_STATE = capture_fp_state();
        FP_STATE_READY.store(true, Ordering::Release);
        establish_fp_state();
        let inputs = [0x3fc0_0000u32, 0xc000_0000u32, 0x3e80_0000u32];
        let mut fp32_bits = 0u32;
        fp32_sse2::smoke(inputs.as_ptr(), &mut fp32_bits);
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));

        if has_load_option(boot, image_handle, b"--self-test-panic") || SELF_TEST == "panic" {
            panic!("self-test");
        }
        if has_load_option(boot, image_handle, b"--self-test-fatal") || SELF_TEST == "fatal" {
            fatal(b"SELF_TEST");
        }
        if fp32_bits != 0xc030_0000 {
            fatal(b"FP_SMOKE");
        }

        let mut started = Record::new();
        started.push(b"PROMPTBOOT_EVENT v=1 event=STARTED build_id=");
        started.push(BUILD_ID.as_bytes());
        started.push(b" firmware=persistent_boot_services console=uefi_simple_text evidence=");
        if SERIAL_SINK.load(Ordering::Acquire).is_null() {
            started.push(b"none");
        } else {
            started.push(b"uefi_serial_io_com1_115200_8n1");
        }
        started.push(b" uefi_conventional_bytes=");
        started.push_decimal(conventional_bytes);
        started.push(b" sse2=ready watchdog=disabled\r\n");
        if !emit_record(validated_record_bytes(&started)) {
            fatal(b"EVIDENCE_RECORD");
        }
        if !emit_record(b"PROMPTBOOT_EVENT v=1 event=BOOT_SMOKE_PASS fp32=c0300000\r\n") {
            fatal(b"EVIDENCE_RECORD");
        }
        if COMPILE_MODE != "echo_repl" {
            if COMPILE_MODE == "model_repl" {
                if let Err(code) = initialize_model_status(physical_columns, physical_rows) {
                    fatal(code);
                }
            }
            let status = model_target::run(
                image_handle,
                system_table,
                boot,
                COMPILE_MODE,
                BUILD_ID,
                conventional_bytes,
            );
            if status != EFI_SUCCESS {
                halt_forever();
            }
            return status;
        }
        return run_repl(system_table, boot);
    }
}

struct ReplOutput;

impl Output for ReplOutput {
    fn live_ascii(&mut self, byte: u8) {
        unsafe {
            write_conout_ascii(&[byte]);
        }
    }

    fn erase_last(&mut self) {
        unsafe {
            write_conout_ascii(b"\x08 \x08");
        }
    }

    fn accepted(&mut self, prompt_index: u64, line: &[u8]) {
        unsafe {
            write_conout_ascii(b"\r\n");
            write_conout_ascii(b"echo> ");
            write_conout_ascii(line);
            write_conout_ascii(b"\r\n");
            let mut record = Record::new();
            record.push(b"PROMPTBOOT_EVENT v=1 event=INPUT_ACCEPTED prompt_index=");
            record.push_decimal(prompt_index);
            record.push(b" bytes=");
            record.push_decimal(line.len() as u64);
            record.push(b" mode=echo_only\r\n");
            emit_record(validated_record_bytes(&record));
        }
    }

    fn rejected(&mut self, prompt_index: u64) {
        unsafe {
            write_conout_ascii(b"\r\n");
            let mut record = Record::new();
            record.push(b"PROMPTBOOT_EVENT v=1 event=INPUT_REJECTED prompt_index=");
            record.push_decimal(prompt_index);
            record.push(b" code=TOO_LONG limit=512\r\n");
            emit_record(validated_record_bytes(&record));
            write_conout_ascii(b"error: prompt exceeds 512 bytes; line discarded\r\n");
        }
    }

    fn prompt(&mut self, prompt_index: u64) {
        unsafe {
            let mut record = Record::new();
            record.push(b"PROMPTBOOT_EVENT v=1 event=PROMPT_READY prompt_index=");
            record.push_decimal(prompt_index);
            record.push(b" limit=512\r\n");
            emit_record(validated_record_bytes(&record));
            write_conout_ascii(b"promptboot> ");
        }
    }
}

unsafe fn run_repl(system_table: *mut SystemTable, boot: *mut BootServices) -> EfiStatus {
    let con_in = (*system_table).con_in;
    let (reset_entry, read_entry, wait_for_key) = if con_in.is_null() {
        (0, 0, null_mut())
    } else {
        (
            (*con_in).reset,
            (*con_in).read_key_stroke,
            (*con_in).wait_for_key,
        )
    };
    if let Err(error) = validate_bindings(
        con_in as usize,
        reset_entry,
        read_entry,
        wait_for_key as usize,
        (*boot).wait_for_event,
    ) {
        fatal(error.code());
    }

    let reset: InputReset = core::mem::transmute(reset_entry);
    if let Err(error) = reset_result(reset(con_in, false)) {
        fatal(error.code());
    }
    let read_key: ReadKeyStroke = core::mem::transmute(read_entry);
    let wait: WaitForEvent = core::mem::transmute((*boot).wait_for_event);
    let event = [wait_for_key];
    let mut editor = Editor::new();
    let mut output = ReplOutput;
    output.prompt(editor.prompt_index());
    if RECORD_FAILED.load(Ordering::Acquire) {
        fatal(b"EVIDENCE_RECORD");
    }

    loop {
        let mut index = usize::MAX;
        if let Err(error) = wait_result(wait(1, event.as_ptr(), &mut index), index) {
            write_conout_ascii(b"\r\n");
            fatal(error.code());
        }
        loop {
            let mut key = EfiInputKey {
                scan_code: 0,
                unicode_char: 0,
            };
            let status = read_key(con_in, &mut key);
            match read_result(status) {
                Ok(true) => {
                    if editor.process_key(key.unicode_char, &mut output)
                        == Flow::PromptIndexOverflow
                    {
                        write_conout_ascii(b"\r\n");
                        fatal(b"PROMPT_INDEX");
                    }
                    if RECORD_FAILED.load(Ordering::Acquire) {
                        fatal(b"EVIDENCE_RECORD");
                    }
                }
                Ok(false) => break,
                Err(error) => {
                    write_conout_ascii(b"\r\n");
                    fatal(error.code());
                }
            }
        }
    }
}

unsafe fn configure_serial(
    boot: *mut BootServices,
    image_handle: EfiHandle,
) -> Option<*mut SerialIo> {
    let mut count = 0usize;
    let mut handles: *mut EfiHandle = null_mut();
    let status = ((*boot).locate_handle_buffer)(
        BY_PROTOCOL,
        &SERIAL_IO_GUID,
        null_mut(),
        &mut count,
        &mut handles,
    );
    if status != EFI_SUCCESS || handles.is_null() {
        return None;
    }
    let mut selected = None;
    let mut index = 0;
    while index < count {
        let handle = *handles.add(index);
        let mut interface: *mut c_void = null_mut();
        let open_status = ((*boot).open_protocol)(
            handle,
            &SERIAL_IO_GUID,
            &mut interface,
            image_handle,
            null_mut(),
            EFI_OPEN_PROTOCOL_GET_PROTOCOL,
        );
        if open_status == EFI_SUCCESS {
            if !interface.is_null() {
                let serial = interface.cast::<SerialIo>();
                if ((*serial).set_attributes)(serial, 115_200, 0, 0, 1, 8, 1) == EFI_SUCCESS {
                    selected = Some(serial);
                    break;
                }
            }
            let _ = ((*boot).close_protocol)(handle, &SERIAL_IO_GUID, image_handle, null_mut());
        }
        index += 1;
    }
    let _ = ((*boot).free_pool)(handles.cast());
    selected
}

unsafe fn measure_conventional_memory(boot: *mut BootServices) -> Option<u64> {
    let mut map_size = 0usize;
    let mut map_key = 0usize;
    let mut descriptor_size = 0usize;
    let mut descriptor_version = 0u32;
    let first = ((*boot).get_memory_map)(
        &mut map_size,
        null_mut(),
        &mut map_key,
        &mut descriptor_size,
        &mut descriptor_version,
    );
    if first != EFI_BUFFER_TOO_SMALL || descriptor_size < core::mem::size_of::<MemoryDescriptor>() {
        return None;
    }
    map_size = map_size.checked_add(descriptor_size.checked_mul(8)?)?;
    let pages = map_size.checked_add(4095)? / 4096;
    let mut address = 0u64;
    if ((*boot).allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, pages, &mut address)
        != EFI_SUCCESS
    {
        return None;
    }
    let map = address as *mut MemoryDescriptor;
    let status = ((*boot).get_memory_map)(
        &mut map_size,
        map,
        &mut map_key,
        &mut descriptor_size,
        &mut descriptor_version,
    );
    if status != EFI_SUCCESS {
        let _ = ((*boot).free_pages)(address, pages);
        return None;
    }
    let mut conventional_pages = 0u64;
    let mut offset = 0usize;
    while offset + core::mem::size_of::<MemoryDescriptor>() <= map_size {
        let descriptor = (address as usize + offset) as *const MemoryDescriptor;
        if (*descriptor).memory_type == EFI_CONVENTIONAL_MEMORY {
            conventional_pages = conventional_pages.saturating_add((*descriptor).number_of_pages);
        }
        offset += descriptor_size;
    }
    if ((*boot).free_pages)(address, pages) != EFI_SUCCESS {
        return None;
    }
    Some(
        conventional_pages
            .saturating_add(pages as u64)
            .saturating_mul(4096),
    )
}

unsafe fn has_load_option(
    boot: *mut BootServices,
    image_handle: EfiHandle,
    expected: &[u8],
) -> bool {
    let mut interface: *mut c_void = null_mut();
    let open_status = ((*boot).open_protocol)(
        image_handle,
        &LOADED_IMAGE_GUID,
        &mut interface,
        image_handle,
        null_mut(),
        EFI_OPEN_PROTOCOL_GET_PROTOCOL,
    );
    if open_status != EFI_SUCCESS {
        return false;
    }
    if interface.is_null() {
        let _ =
            ((*boot).close_protocol)(image_handle, &LOADED_IMAGE_GUID, image_handle, null_mut());
        return false;
    }
    let loaded = &*interface.cast::<LoadedImage>();
    let mut result = false;
    if !loaded.load_options.is_null() && loaded.load_options_size >= 2 {
        let units = loaded.load_options_size as usize / 2;
        let option = loaded.load_options.cast::<u16>();
        if units >= expected.len() {
            let mut start = 0usize;
            while start + expected.len() <= units {
                let mut matched = true;
                let mut index = 0;
                while index < expected.len() {
                    if *option.add(start + index) != expected[index] as u16 {
                        matched = false;
                        break;
                    }
                    index += 1;
                }
                if matched {
                    result = true;
                    break;
                }
                start += 1;
            }
        }
    }
    let _ = ((*boot).close_protocol)(image_handle, &LOADED_IMAGE_GUID, image_handle, null_mut());
    result
}

fn halt_forever() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

unsafe fn fatal(code: &[u8]) -> ! {
    if !FATAL_EMITTED.swap(true, Ordering::SeqCst) {
        write_fatal_diagnostic(code);
        let mut record = Record::new();
        record.push(b"PROMPTBOOT_EVENT v=1 event=FATAL code=");
        record.push(code);
        record.push(b"\r\n");
        let bytes = match record_emission(&record) {
            Ok(bytes) => bytes,
            Err(_) => MODEL_RECORD_FATAL,
        };
        if !SERIAL_SINK.load(Ordering::Acquire).is_null() {
            emit_record(bytes);
        }
    }
    halt_forever()
}

fn validated_record_bytes(record: &Record) -> &[u8] {
    match record_emission(record) {
        Ok(bytes) => bytes,
        Err(_) => MODEL_RECORD_FATAL,
    }
}

unsafe fn emit_record(bytes: &[u8]) -> bool {
    if RECORD_FAILED.load(Ordering::Acquire) {
        return false;
    }
    let fallback_fatal = bytes == MODEL_RECORD_FATAL;
    if bytes.len() > 1023 || !bytes.ends_with(b"\r\n") || bytes.iter().any(|byte| !byte.is_ascii())
    {
        RECORD_FAILED.store(true, Ordering::Release);
        return false;
    }
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    let display_enabled = EVENT_DISPLAY_ENABLED.load(Ordering::Acquire);
    if display_enabled {
        if conout.is_null() {
            RECORD_FAILED.store(true, Ordering::Release);
            return false;
        }
        let mut utf16 = [0u16; 1024];
        for (index, byte) in bytes.iter().copied().enumerate() {
            utf16[index] = byte as u16;
        }
        if FP_STATE_READY.load(Ordering::Acquire) {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        }
        if console_output_utf16(&utf16[..bytes.len()]).is_err() {
            RECORD_FAILED.store(true, Ordering::Release);
            return false;
        }
    }
    let serial = SERIAL_SINK.load(Ordering::Acquire);
    if !serial.is_null() {
        let mut length = bytes.len();
        if FP_STATE_READY.load(Ordering::Acquire) {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        }
        let status = ((*serial).write)(serial, &mut length, bytes.as_ptr().cast());
        if status != EFI_SUCCESS || length != bytes.len() {
            RECORD_FAILED.store(true, Ordering::Release);
            return false;
        }
        if display_enabled {
            // Display writes break their serial prefix deliberately, so the evidence
            // pair is emitted explicitly and remains adjacent.
            length = bytes.len();
            let status = ((*serial).write)(serial, &mut length, bytes.as_ptr().cast());
            if status != EFI_SUCCESS || length != bytes.len() {
                RECORD_FAILED.store(true, Ordering::Release);
                return false;
            }
        }
    }
    if fallback_fatal {
        RECORD_FAILED.store(true, Ordering::Release);
        false
    } else {
        true
    }
}

unsafe fn emit_record_checked(bytes: &[u8]) -> Result<(), EfiStatus> {
    if RECORD_FAILED.load(Ordering::Acquire) {
        return Err(EFI_DEVICE_ERROR);
    }
    if bytes.len() > 1023 || !bytes.ends_with(b"\r\n") {
        RECORD_FAILED.store(true, Ordering::Release);
        return Err(EFI_DEVICE_ERROR);
    }
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    let serial = SERIAL_SINK.load(Ordering::Acquire);
    let display_enabled = EVENT_DISPLAY_ENABLED.load(Ordering::Acquire);
    if display_enabled && conout.is_null() {
        RECORD_FAILED.store(true, Ordering::Release);
        return Err(EFI_DEVICE_ERROR);
    }
    let mut utf16 = [0u16; 1024];
    for (at, byte) in bytes.iter().copied().enumerate() {
        if !byte.is_ascii() {
            RECORD_FAILED.store(true, Ordering::Release);
            return Err(EFI_DEVICE_ERROR);
        }
        utf16[at] = byte as u16;
    }
    if display_enabled {
        if let Err(status) = console_output_utf16(&utf16[..bytes.len()]) {
            RECORD_FAILED.store(true, Ordering::Release);
            return Err(status);
        }
    }
    if !serial.is_null() {
        let mut length = bytes.len();
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let status = ((*serial).write)(serial, &mut length, bytes.as_ptr().cast());
        if status != EFI_SUCCESS || length != bytes.len() {
            RECORD_FAILED.store(true, Ordering::Release);
            return Err(if status == EFI_SUCCESS {
                EFI_DEVICE_ERROR
            } else {
                status
            });
        }
        if display_enabled {
            // Display writes break their serial prefix deliberately, so the evidence
            // pair is emitted explicitly and remains adjacent.
            length = bytes.len();
            let status = ((*serial).write)(serial, &mut length, bytes.as_ptr().cast());
            if status != EFI_SUCCESS || length != bytes.len() {
                RECORD_FAILED.store(true, Ordering::Release);
                return Err(if status == EFI_SUCCESS {
                    EFI_DEVICE_ERROR
                } else {
                    status
                });
            }
        }
    }
    Ok(())
}

unsafe fn write_serial_checked(bytes: &[u8]) -> Result<(), EfiStatus> {
    let serial = SERIAL_SINK.load(Ordering::Acquire);
    if serial.is_null() {
        return Ok(());
    }
    let mut length = bytes.len();
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*serial).write)(serial, &mut length, bytes.as_ptr().cast());
    if status != EFI_SUCCESS || length != bytes.len() {
        Err(if status == EFI_SUCCESS {
            EFI_DEVICE_ERROR
        } else {
            status
        })
    } else {
        Ok(())
    }
}

fn toggle_event_display() -> bool {
    let enabled = !EVENT_DISPLAY_ENABLED.load(Ordering::Acquire);
    EVENT_DISPLAY_ENABLED.store(enabled, Ordering::Release);
    enabled
}

unsafe fn initialize_console_history(
    conout: *mut SimpleTextOutput,
) -> Result<(usize, usize), EfiStatus> {
    if (*conout).mode.is_null() || (*(*conout).mode).mode < 0 {
        return Err(EFI_DEVICE_ERROR);
    }
    let mut columns = 0usize;
    let mut rows = 0usize;
    let status = ((*conout).query_mode)(
        conout,
        (*(*conout).mode).mode as usize,
        &mut columns,
        &mut rows,
    );
    if status != EFI_SUCCESS || columns == 0 || rows == 0 {
        return Err(if status == EFI_SUCCESS {
            EFI_DEVICE_ERROR
        } else {
            status
        });
    }
    (*CONSOLE_HISTORY.0.get()).configure(columns, rows);
    enable_console_cursor()?;
    Ok((columns, rows))
}

unsafe fn initialize_model_status(
    physical_columns: usize,
    physical_rows: usize,
) -> Result<(), &'static [u8]> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null()
        || (*conout).set_attribute == 0
        || (*conout).set_cursor_position == 0
    {
        return Err(b"STATUS_BINDING");
    }
    let (columns, rows) = bounded_status_geometry(physical_columns, physical_rows)
        .ok_or(b"STATUS_GEOMETRY" as &'static [u8])?;
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    if ((*conout).clear_screen)(conout) != EFI_SUCCESS {
        return Err(b"STATUS_RENDER");
    }
    (*CONSOLE_HISTORY.0.get()).configure(columns - 1, rows - 1);
    *STATUS_CONSOLE.0.get() = StatusConsole {
        active: true,
        columns,
    };
    let mut line = [0u8; promptboot::console_history::MAX_COLUMNS - 1];
    let length = loading_line(0, &mut line[..columns - 1]);
    render_status_ascii(&line[..length]).map_err(|_| b"STATUS_RENDER" as &'static [u8])?;
    set_console_cursor(0, 1).map_err(|_| b"STATUS_RENDER" as &'static [u8])
}

unsafe fn render_status_ascii(bytes: &[u8]) -> Result<(), EfiStatus> {
    let status_console = &*STATUS_CONSOLE.0.get();
    if !status_console.active {
        return Err(EFI_DEVICE_ERROR);
    }
    let safe_cells = status_console.columns - 1;
    let rendered = &mut *STATUS_ROW.0.get();
    rendered.fill(b' ' as u16);
    for (at, byte) in bytes.iter().copied().take(safe_cells).enumerate() {
        rendered[at] = if byte.is_ascii() {
            byte as u16
        } else {
            b'?' as u16
        };
    }
    paint_status_row(&rendered[..safe_cells])
}

unsafe fn restore_status_row() -> Result<(), EfiStatus> {
    let status_console = &*STATUS_CONSOLE.0.get();
    if !status_console.active {
        return Err(EFI_DEVICE_ERROR);
    }
    let safe_cells = status_console.columns - 1;
    let rendered = &*STATUS_ROW.0.get();
    paint_status_row(&rendered[..safe_cells])
}

unsafe fn paint_status_row(units: &[u16]) -> Result<(), EfiStatus> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null()
        || (*conout).mode.is_null()
        || (*conout).set_attribute == 0
        || (*conout).set_cursor_position == 0
    {
        return Err(EFI_DEVICE_ERROR);
    }
    let mode = &*(*conout).mode;
    if mode.cursor_column < 0 || mode.cursor_row < 0 || mode.attribute < 0 {
        return Err(EFI_DEVICE_ERROR);
    }
    let old_attribute = mode.attribute as usize;
    let old_column = mode.cursor_column as usize;
    let old_row = mode.cursor_row as usize;
    let set_attribute: TextSetAttribute = core::mem::transmute((*conout).set_attribute);
    let set_cursor: TextSetCursorPosition = core::mem::transmute((*conout).set_cursor_position);
    render_status_row(
        old_attribute,
        old_column,
        old_row,
        0,
        units,
        |column, row| {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            set_cursor(conout, column, row)
        },
        |attribute| {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
            set_attribute(conout, attribute)
        },
        |units| match raw_conout_utf16(units) {
            Ok(()) => EFI_SUCCESS,
            Err(status) => status,
        },
    )
}

unsafe fn set_console_cursor(column: usize, row: usize) -> Result<(), EfiStatus> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null() || (*conout).set_cursor_position == 0 {
        return Err(EFI_DEVICE_ERROR);
    }
    let set_cursor: TextSetCursorPosition = core::mem::transmute((*conout).set_cursor_position);
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = set_cursor(conout, column, row);
    if status == EFI_SUCCESS {
        Ok(())
    } else {
        Err(status)
    }
}

unsafe fn enable_console_cursor() -> Result<(), EfiStatus> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null() || (*conout).enable_cursor == 0 {
        return Err(EFI_DEVICE_ERROR);
    }
    let enable: TextEnableCursor = core::mem::transmute((*conout).enable_cursor);
    if FP_STATE_READY.load(Ordering::Acquire) {
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    }
    let status = enable(conout, true);
    if status == EFI_SUCCESS {
        Ok(())
    } else {
        Err(status)
    }
}

unsafe fn raw_conout_utf16(units: &[u16]) -> Result<(), EfiStatus> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null() {
        return Err(EFI_DEVICE_ERROR);
    }
    for chunk in units.chunks(256) {
        let mut terminated = [0u16; 257];
        terminated[..chunk.len()].copy_from_slice(chunk);
        if FP_STATE_READY.load(Ordering::Acquire) {
            restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        }
        let status = ((*conout).output_string)(conout, terminated.as_ptr());
        if status != EFI_SUCCESS {
            return Err(status);
        }
    }
    Ok(())
}

unsafe fn raw_content_utf16(
    units: &[u16],
    column: usize,
    width: usize,
) -> Result<usize, EfiStatus> {
    write_content_with_wrap(units, column, width, |rendered| {
        match raw_conout_utf16(rendered) {
            Ok(()) => EFI_SUCCESS,
            Err(status) => status,
        }
    })
}

unsafe fn redraw_console_history() -> Result<(), EfiStatus> {
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null() {
        return Err(EFI_DEVICE_ERROR);
    }
    let history = &*CONSOLE_HISTORY.0.get();
    let status_console = *STATUS_CONSOLE.0.get();
    if !status_console.active {
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let status = ((*conout).clear_screen)(conout);
        if status != EFI_SUCCESS {
            return Err(status);
        }
        let lines = history.viewport_len();
        for row in 0..lines {
            let (line, hard_break) = history.viewport_line(row);
            raw_conout_utf16(line)?;
            if hard_break && row + 1 < lines {
                raw_conout_utf16(&[0x000d, 0x000a])?;
            }
        }
        return Ok(());
    }

    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*conout).clear_screen)(conout);
    if status != EFI_SUCCESS {
        return Err(status);
    }
    let lines = history.viewport_len();
    for row in 0..lines {
        let (line, _) = history.viewport_line(row);
        if line.is_empty() {
            continue;
        }
        set_console_cursor(0, row + 1)?;
        // A cursor operation inside a redrawn record keeps firmware serial
        // mirroring from presenting the redraw as a second structured record.
        if is_serial_evidence_line(line) {
            raw_conout_utf16(&line[..1])?;
            set_console_cursor(1, row + 1)?;
            raw_conout_utf16(&line[1..])?;
        } else {
            raw_conout_utf16(line)?;
        }
    }
    let (column, row) = history.viewport_cursor();
    set_console_cursor(column, row + 1)?;
    restore_status_row()
}

unsafe fn console_output_utf16(units: &[u16]) -> Result<(), EfiStatus> {
    let returned = (*CONSOLE_HISTORY.0.get()).return_to_bottom();
    if returned {
        redraw_console_history()?;
    }
    let history = &mut *CONSOLE_HISTORY.0.get();
    let active = (*STATUS_CONSOLE.0.get()).active;
    if !active {
        history.write(units);
        return raw_conout_utf16(units);
    }
    let (column, row) = history.viewport_cursor();
    let scrolls = native_scrolls_for_output(
        units,
        column,
        row,
        history.width(),
        history.rows(),
    )
    .ok_or(EFI_DEVICE_ERROR)?;
    history.write(units);
    set_console_cursor(column, row + 1)?;
    if is_event_boundary_line(units) || is_serial_evidence_line(units) {
        let next_column = raw_content_utf16(&units[..1], column, history.width())?;
        let conout = CONOUT_SINK.load(Ordering::Acquire);
        let mode = &*(*conout).mode;
        if mode.cursor_column < 0 || mode.cursor_row < 0 {
            return Err(EFI_DEVICE_ERROR);
        }
        set_console_cursor(mode.cursor_column as usize, mode.cursor_row as usize)?;
        raw_content_utf16(&units[1..], next_column, history.width())?;
    } else {
        raw_content_utf16(units, column, history.width())?;
    }
    if scrolls == 0 {
        Ok(())
    } else {
        restore_status_row()
    }
}

unsafe fn scroll_console_page(up: bool) -> Result<(), EfiStatus> {
    let changed = if up {
        (*CONSOLE_HISTORY.0.get()).page_up()
    } else {
        (*CONSOLE_HISTORY.0.get()).page_down()
    };
    if changed {
        redraw_console_history()?;
    }
    Ok(())
}

unsafe fn reset_console_history() -> Result<(), EfiStatus> {
    (*CONSOLE_HISTORY.0.get()).reset();
    if (*STATUS_CONSOLE.0.get()).active {
        redraw_console_history()?;
        return enable_console_cursor();
    }
    let conout = CONOUT_SINK.load(Ordering::Acquire);
    if conout.is_null() {
        return Err(EFI_DEVICE_ERROR);
    }
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*conout).clear_screen)(conout);
    if status == EFI_SUCCESS {
        enable_console_cursor()
    } else {
        Err(status)
    }
}

unsafe fn write_fatal_diagnostic(code: &[u8]) {
    write_conout_ascii(b"promptboot: fatal ");
    write_conout_ascii(code);
    write_conout_ascii(b"\r\n");
}

unsafe fn write_conout_utf16_checked(units: &[u16]) -> Result<(), EfiStatus> {
    if units.len() > 132 || units.iter().any(|unit| *unit == 0) {
        return Err(EFI_DEVICE_ERROR);
    }
    console_output_utf16(units)
}

unsafe fn write_conout_ascii(bytes: &[u8]) {
    let mut utf16 = [0u16; 1024];
    let count = core::cmp::min(bytes.len(), utf16.len());
    let mut index = 0;
    while index < count {
        utf16[index] = bytes[index] as u16;
        index += 1;
    }
    let _ = console_output_utf16(&utf16[..count]);
}

unsafe fn capture_fp_state() -> FpState {
    let cr0: u64;
    let cr4: u64;
    let mut x87_control = 0u16;
    let mut mxcsr = 0u32;
    asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    asm!("fnstcw [{}]", in(reg) &mut x87_control, options(nostack, preserves_flags));
    asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
    FpState {
        cr0,
        cr4,
        x87_control,
        _pad: 0,
        mxcsr,
    }
}

unsafe fn establish_fp_state() {
    let mut cr0: u64;
    let mut cr4: u64;
    asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    cr0 |= (1 << 1) | (1 << 5);
    cr0 &= !((1 << 2) | (1 << 3));
    asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags));
    asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    cr4 |= (1 << 9) | (1 << 10);
    asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));
    asm!("fninit", options(nomem, nostack));
    let x87_control: u16 = 0x037f;
    let mxcsr: u32 = 0x1f80;
    asm!("fldcw [{}]", in(reg) &x87_control, options(nostack));
    asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack));
}

unsafe fn restore_fp_state(state: *const FpState) {
    let state = core::ptr::read(state);
    asm!("fldcw [{}]", in(reg) &state.x87_control, options(nostack));
    asm!("ldmxcsr [{}]", in(reg) &state.mxcsr, options(nostack));
    asm!("mov cr4, {}", in(reg) state.cr4, options(nomem, nostack, preserves_flags));
    asm!("mov cr0, {}", in(reg) state.cr0, options(nomem, nostack, preserves_flags));
}

mod fp32_sse2 {
    use core::arch::asm;

    #[inline(never)]
    #[target_feature(enable = "sse2")]
    pub unsafe fn smoke(input_bits: *const u32, output_bits: *mut u32) -> usize {
        // The official UEFI target keeps soft-float defaults globally.  Keep
        // hardware FP entirely inside this integer/pointer boundary and make
        // the actual scalar SSE operations explicit and volatile.
        let mut saved_xmm0 = [0u8; 16];
        asm!(
            "movdqu xmmword ptr [{saved}], xmm0",
            "movss xmm0, dword ptr [{inputs}]",
            "mulss xmm0, dword ptr [{inputs} + 4]",
            "addss xmm0, dword ptr [{inputs} + 8]",
            "movss dword ptr [{output}], xmm0",
            "movdqu xmm0, xmmword ptr [{saved}]",
            inputs = in(reg) input_bits,
            output = in(reg) output_bits,
            saved = in(reg) saved_xmm0.as_mut_ptr(),
            options(nostack),
        );
        0
    }
}
