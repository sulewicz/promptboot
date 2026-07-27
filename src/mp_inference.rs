use core::arch::{asm, x86_64::__cpuid, x86_64::__cpuid_count};
use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::mem::{size_of, transmute};
use core::ptr::{null_mut, write};
use core::slice;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use promptboot_core::{
    inference_worker_entry, InferenceDispatcher, InferenceWorkerJob,
};
use promptboot::status_bar::{CpuMode, CpuTopology};

use super::{
    emit_record, establish_fp_state, halt_forever, restore_fp_state, write_fatal_diagnostic,
    BootServices, EfiEvent, EfiStatus, Guid, Record, ALLOCATE_ANY_PAGES, EFI_LOADER_DATA,
    EFI_SUCCESS, SAVED_FP_STATE,
};

const MP_SERVICES_GUID: Guid = Guid::new(
    0x3fdd_a605,
    0xa76e,
    0x4f46,
    [0xad, 0x29, 0x12, 0xf4, 0x53, 0x1b, 0x3d, 0x08],
);
const DISPATCH_TIMEOUT_US: usize = 45_000_000;

type LocateProtocol =
    unsafe extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> EfiStatus;
type GetNumberOfProcessors =
    unsafe extern "efiapi" fn(*mut MpServices, *mut usize, *mut usize) -> EfiStatus;
type ApProcedure = unsafe extern "efiapi" fn(*mut c_void);
type StartupAllAps = unsafe extern "efiapi" fn(
    *mut MpServices,
    ApProcedure,
    bool,
    EfiEvent,
    usize,
    *mut c_void,
    *mut *mut usize,
) -> EfiStatus;

#[repr(C)]
struct MpServices {
    get_number_of_processors: GetNumberOfProcessors,
    get_processor_info: usize,
    startup_all_aps: StartupAllAps,
    startup_this_ap: usize,
    switch_bsp: usize,
    enable_disable_ap: usize,
    who_am_i: usize,
}

#[derive(Clone, Copy)]
struct CpuState {
    cr0: u64,
    cr4: u64,
    xcr0: u64,
    x87: u16,
    _pad: u16,
    mxcsr: u32,
}

impl CpuState {
    const ZERO: Self = Self {
        cr0: 0,
        cr4: 0,
        xcr0: 0,
        x87: 0,
        _pad: 0,
        mxcsr: 0,
    };
}

struct ApRecord {
    ready: AtomicU32,
    backend: AtomicU32,
    worker_status: AtomicU32,
    restored: AtomicU32,
    saved: UnsafeCell<CpuState>,
}

unsafe impl Sync for ApRecord {}

impl ApRecord {
    const fn new() -> Self {
        Self {
            ready: AtomicU32::new(0),
            backend: AtomicU32::new(u32::MAX),
            worker_status: AtomicU32::new(u32::MAX),
            restored: AtomicU32::new(0),
            saved: UnsafeCell::new(CpuState::ZERO),
        }
    }
}

struct EntryBarrier {
    arrivals: AtomicUsize,
    generation: AtomicUsize,
    workers: usize,
}

impl EntryBarrier {
    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrivals.fetch_add(1, Ordering::AcqRel) + 1 == self.workers {
            self.arrivals.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == generation {
                core::hint::spin_loop();
            }
        }
    }
}

struct MpCall {
    job: *mut InferenceWorkerJob,
    records: *mut ApRecord,
    slot_masks: *mut AtomicU64,
    slot_mask_words: usize,
    workers: usize,
    next_slot: AtomicUsize,
    callbacks_returned: AtomicUsize,
    rejection: AtomicU32,
    entry: EntryBarrier,
}

pub(super) struct MpInferenceAdapter {
    mp: *mut MpServices,
    records: *mut ApRecord,
    slot_masks: *mut AtomicU64,
    slot_mask_words: usize,
    workers: usize,
}

impl MpInferenceAdapter {
    pub(super) unsafe fn prepare(
        boot: *mut BootServices,
        topology: &mut CpuTopology,
    ) -> Option<Self> {
        let locate: LocateProtocol = transmute((*boot).locate_protocol);
        let mut interface = null_mut();
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        let locate_status = locate(&MP_SERVICES_GUID, null_mut(), &mut interface);
        if locate_status != EFI_SUCCESS || interface.is_null() {
            *topology = CpuTopology::UNKNOWN_SERIAL;
            emit_mode(b"serial", b"protocol_absent", 0, 0);
            return None;
        }
        let mp = interface.cast::<MpServices>();
        let mut total = 0usize;
        let mut enabled = 0usize;
        let status = ((*mp).get_number_of_processors)(mp, &mut total, &mut enabled);
        if status != EFI_SUCCESS {
            dispatch_fatal(status, 0);
        }
        if enabled < 3 {
            *topology = CpuTopology {
                mode: CpuMode::Serial,
                active: 1,
                workers: 0,
                enabled: Some(enabled as u32),
                total: Some(total as u32),
            };
            emit_mode(b"serial", b"fewer_than_two_aps", total, enabled);
            return None;
        }

        let workers = enabled - 1;
        *topology = CpuTopology {
            mode: CpuMode::Mp,
            active: enabled as u32,
            workers: workers as u32,
            enabled: Some(enabled as u32),
            total: Some(total as u32),
        };
        let record_bytes = workers
            .checked_mul(size_of::<ApRecord>())
            .unwrap_or_else(|| dispatch_fatal(0, 20));
        let slot_mask_words = workers.div_ceil(64);
        let mask_bytes = slot_mask_words
            .checked_mul(size_of::<AtomicU64>())
            .unwrap_or_else(|| dispatch_fatal(0, 20));
        let bytes = record_bytes
            .checked_add(mask_bytes)
            .unwrap_or_else(|| dispatch_fatal(0, 20));
        let pages = bytes.div_ceil(4096);
        let mut address = 0u64;
        let status = ((*boot).allocate_pages)(
            ALLOCATE_ANY_PAGES,
            EFI_LOADER_DATA,
            pages,
            &mut address,
        );
        if status != EFI_SUCCESS || address == 0 {
            dispatch_fatal(status, 21);
        }
        let records = address as *mut ApRecord;
        slice::from_raw_parts_mut(address as *mut u8, pages * 4096).fill(0);
        for slot in 0..workers {
            write(records.add(slot), ApRecord::new());
        }
        let slot_masks = (address as *mut u8).add(record_bytes).cast::<AtomicU64>();
        for word in 0..slot_mask_words {
            write(slot_masks.add(word), AtomicU64::new(0));
        }
        emit_mode(b"mp", b"mp_services", total, enabled);
        Some(Self {
            mp,
            records,
            slot_masks,
            slot_mask_words,
            workers,
        })
    }

    pub(super) unsafe fn dispatcher(&mut self) -> InferenceDispatcher<'_> {
        InferenceDispatcher::new(self, dispatch_token)
    }
}

unsafe extern "efiapi" fn dispatch_token(
    context: *mut c_void,
    job: *mut InferenceWorkerJob,
) {
    let adapter = &mut *context.cast::<MpInferenceAdapter>();
    for slot in 0..adapter.workers {
        write(adapter.records.add(slot), ApRecord::new());
    }
    for word in 0..adapter.slot_mask_words {
        (*adapter.slot_masks.add(word)).store(0, Ordering::Relaxed);
    }
    let call = MpCall {
        job,
        records: adapter.records,
        slot_masks: adapter.slot_masks,
        slot_mask_words: adapter.slot_mask_words,
        workers: adapter.workers,
        next_slot: AtomicUsize::new(0),
        callbacks_returned: AtomicUsize::new(0),
        rejection: AtomicU32::new(0),
        entry: EntryBarrier {
            arrivals: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            workers: adapter.workers,
        },
    };

    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let status = ((*adapter.mp).startup_all_aps)(
        adapter.mp,
        ap_entry,
        false,
        null_mut(),
        DISPATCH_TIMEOUT_US,
        (&call as *const MpCall).cast_mut().cast(),
        null_mut(),
    );
    if status != EFI_SUCCESS {
        dispatch_fatal(status, 0);
    }

    establish_fp_state();
    let mut shared = call.rejection.load(Ordering::Acquire);
    if call.next_slot.load(Ordering::Acquire) != adapter.workers
        || call.callbacks_returned.load(Ordering::Acquire) != adapter.workers
        || call.entry.arrivals.load(Ordering::Acquire) != 0
        || call.entry.generation.load(Ordering::Acquire) != 1
    {
        reject(&mut shared, 30);
    }
    for word in 0..call.slot_mask_words {
        let remaining = adapter.workers.saturating_sub(word * 64);
        let expected = if remaining >= 64 {
            u64::MAX
        } else {
            (1u64 << remaining) - 1
        };
        if (*call.slot_masks.add(word)).load(Ordering::Acquire) != expected {
            reject(&mut shared, 30);
        }
    }
    let mut backend = None;
    for slot in 0..adapter.workers {
        let record = &*adapter.records.add(slot);
        if record.restored.load(Ordering::Acquire) != 1 {
            reject(&mut shared, 31);
        }
        let worker_status = record.worker_status.load(Ordering::Acquire);
        if worker_status != 0 {
            reject(&mut shared, worker_status);
        }
        let selected = record.backend.load(Ordering::Acquire);
        if backend.is_some_and(|value| value != selected) {
            reject(&mut shared, 32);
        }
        backend = Some(selected);
    }
    let core_status = (*job).protocol_status();
    reject(&mut shared, core_status);
    if shared != 0 {
        restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
        dispatch_fatal(EFI_SUCCESS, shared);
    }
}

fn reject(shared: &mut u32, status: u32) {
    if *shared == 0 && status != 0 {
        *shared = status;
    }
}

unsafe extern "efiapi" fn ap_entry(argument: *mut c_void) {
    let call = &*argument.cast::<MpCall>();
    let slot = call.next_slot.fetch_add(1, Ordering::AcqRel);
    if slot >= call.workers {
        call.rejection.store(40, Ordering::Release);
        return;
    }
    let mask_word = slot / 64;
    let bit = 1u64 << (slot % 64);
    if (*call.slot_masks.add(mask_word)).fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        call.rejection.store(41, Ordering::Release);
    }
    let record = &*call.records.add(slot);
    let saved = capture_cpu_state();
    write(record.saved.get(), saved);
    establish_ap_state();
    let ready = cpu_state_ready();
    let capable = ready && avx2_capable();
    record
        .ready
        .store(u32::from(ready) | (u32::from(capable) << 1), Ordering::Release);
    call.entry.wait();

    let mut all_ready = true;
    let mut common_avx2 = true;
    for peer in 0..call.workers {
        let state = (*call.records.add(peer)).ready.load(Ordering::Acquire);
        all_ready &= state & 1 != 0;
        common_avx2 &= state & 2 != 0;
    }
    if !all_ready {
        call.rejection.store(42, Ordering::Release);
    }
    let backend = u32::from(common_avx2);
    record.backend.store(backend, Ordering::Release);
    let worker_status = if all_ready && call.rejection.load(Ordering::Acquire) == 0 {
        inference_worker_entry(
            call.job,
            slot as u32,
            call.workers as u32,
            backend,
        )
    } else {
        42
    };
    record
        .worker_status
        .store(worker_status, Ordering::Release);

    restore_cpu_state(saved);
    record.restored.store(
        u32::from(cpu_state_equal(saved, capture_cpu_state())),
        Ordering::Release,
    );
    call.callbacks_returned.fetch_add(1, Ordering::Release);
}

unsafe fn dispatch_fatal(status: EfiStatus, shared: u32) -> ! {
    restore_fp_state(core::ptr::addr_of!(SAVED_FP_STATE));
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=FATAL code=MP_DISPATCH_FATAL efi_status=");
    record.push_decimal(status as u64);
    record.push(b" shared_status=");
    record.push_decimal(shared as u64);
    record.push(b" output_inspected=");
    record.push(b"0");
    record.push(b" retry=0\r\n");
    let _ = emit_record(record.as_bytes());
    write_fatal_diagnostic(b"MP_DISPATCH_FATAL");
    halt_forever()
}

unsafe fn emit_mode(mode: &[u8], reason: &[u8], total: usize, enabled: usize) {
    let mut record = Record::new();
    record.push(b"PROMPTBOOT_EVENT v=1 event=MP_INFERENCE_MODE mode=");
    record.push(mode);
    record.push(b" reason=");
    record.push(reason);
    record.push(b" total=");
    record.push_decimal(total as u64);
    record.push(b" enabled=");
    record.push_decimal(enabled as u64);
    record.push(b"\r\n");
    let _ = emit_record(record.as_bytes());
}

unsafe fn read_xcr0() -> u64 {
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

unsafe fn capture_cpu_state() -> CpuState {
    let cr0: u64;
    let cr4: u64;
    let mut x87 = 0u16;
    let mut mxcsr = 0u32;
    asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    asm!("fnstcw [{}]", in(reg) &mut x87, options(nostack, preserves_flags));
    asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
    CpuState {
        cr0,
        cr4,
        xcr0: if cr4 & (1 << 18) != 0 && __cpuid(1).ecx & (1 << 27) != 0 {
            read_xcr0()
        } else {
            0
        },
        x87,
        _pad: 0,
        mxcsr,
    }
}

unsafe fn establish_ap_state() {
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
    let x87 = 0x037fu16;
    let mxcsr = 0x1f80u32;
    asm!("fldcw [{}]", in(reg) &x87, options(nostack));
    asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack));
}

unsafe fn restore_cpu_state(state: CpuState) {
    asm!("fldcw [{}]", in(reg) &state.x87, options(nostack));
    asm!("ldmxcsr [{}]", in(reg) &state.mxcsr, options(nostack));
    asm!("mov cr4, {}", in(reg) state.cr4, options(nomem, nostack, preserves_flags));
    asm!("mov cr0, {}", in(reg) state.cr0, options(nomem, nostack, preserves_flags));
}

fn cpu_state_equal(left: CpuState, right: CpuState) -> bool {
    left.cr0 == right.cr0
        && left.cr4 == right.cr4
        && left.xcr0 == right.xcr0
        && left.x87 == right.x87
        && left.mxcsr == right.mxcsr
}

unsafe fn cpu_state_ready() -> bool {
    let state = capture_cpu_state();
    state.cr0 & ((1 << 1) | (1 << 5)) == ((1 << 1) | (1 << 5))
        && state.cr0 & ((1 << 2) | (1 << 3)) == 0
        && state.cr4 & ((1 << 9) | (1 << 10)) == ((1 << 9) | (1 << 10))
        && state.x87 == 0x037f
        && state.mxcsr == 0x1f80
}

unsafe fn avx2_capable() -> bool {
    let max = __cpuid(0).eax;
    if max < 7 {
        return false;
    }
    let leaf1 = __cpuid(1).ecx;
    let required = (1 << 26) | (1 << 27) | (1 << 28);
    let state = capture_cpu_state();
    state.cr4 & (1 << 18) != 0
        && leaf1 & required == required
        && read_xcr0() & 0x6 == 0x6
        && __cpuid_count(7, 0).ebx & (1 << 5) != 0
}
