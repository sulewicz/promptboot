use crate::gguf::{SOURCE_BYTES, SOURCE_SHA256};
use crate::pack;
use promptboot_core::sha256::Sha256;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const SECTOR: u64 = 512;
const SECTORS: u64 = 1_048_576;
const SECTORS_PER_CLUSTER: u64 = 16;
const SECTORS_PER_FAT: u64 = 256;
const ROOT_ENTRIES: u64 = 512;
const ROOT_SECTORS: u64 = 32;
const DATA_START: u64 = 545;
const CLUSTER_BYTES: u64 = 8_192;
const IMAGE_BYTES: u64 = 536_870_912;
const MODEL_BYTES: u64 = 426_762_944;
const MODEL_SHA256: &str = "b0f98ed6e0557ca35e1bced1000c950b3c84414251df65290315a7969981d42d";
const MODEL_CLUSTERS: u64 = 52_096;
const VOLUME_ID: u32 = 0x544f4250;
const EVENT_TOPOLOGY: &str = "toggle-v1";
const SOURCE_DATE_EPOCH: &str = "315532800";
const CARGO_VERSION: &str = "cargo 1.97.1 (c980f4866 2026-06-30)";
const RUSTC_VERSION: &str = "rustc 1.97.1 (8bab26f4f 2026-07-14)";

const DISTRIBUTION: [(&str, &[u8; 11]); 8] = [
    ("LICENSE", b"LICENSE    "),
    ("SOURCE.TGZ", b"SOURCE  TGZ"),
    ("LICENSES/3RDPARTY.TXT", b"3RDPARTYTXT"),
    ("LICENSES/QWEN.TXT", b"QWEN    TXT"),
    ("LICENSES/LLAMA.TXT", b"LLAMA   TXT"),
    ("LICENSES/LIBM.TXT", b"LIBM    TXT"),
    ("LICENSES/RUSTCORE.HTM", b"RUSTCOREHTM"),
    ("LICENSES/RUSTCB.TXT", b"RUSTCB  TXT"),
];

const INPUT_PATHS: &[&str] = &[
    ".cargo/config.toml",
    "Cargo.lock",
    "Cargo.toml",
    "assets.lock.json",
    "crates/promptboot-core/Cargo.toml",
    "crates/promptboot-core/src/arena.rs",
    "crates/promptboot-core/src/fp32_sse2.rs",
    "crates/promptboot-core/src/inference.rs",
    "crates/promptboot-core/src/lib.rs",
    "crates/promptboot-core/src/model.rs",
    "crates/promptboot-core/src/sha256.rs",
    "crates/promptboot-core/src/tokenizer.rs",
    "crates/promptboot-tools/Cargo.toml",
    "crates/promptboot-tools/src/assets.rs",
    "crates/promptboot-tools/src/gguf.rs",
    "crates/promptboot-tools/src/image.rs",
    "crates/promptboot-tools/src/inspect.rs",
    "crates/promptboot-tools/src/lib.rs",
    "crates/promptboot-tools/src/main.rs",
    "crates/promptboot-tools/src/pack.rs",
    "crates/promptboot-tools/src/release.rs",
    "fixtures/analytic/rope-table.f32le",
    "fixtures/inference/rope-table.f32le",
    "fixtures/reference/model/hello/prompt_tokens.u32le",
    "rust-toolchain.toml",
    "src/console_contract.rs",
    "src/console_history.rs",
    "src/editor.rs",
    "src/lib.rs",
    "src/main.rs",
    "src/model_contract.rs",
    "src/mp_inference.rs",
    "src/model_repl.rs",
    "src/model_target.rs",
    "src/repl_contract.rs",
    "src/status_bar.rs",
];

#[derive(Clone)]
struct DistributionRow {
    short: &'static [u8; 11],
    path: PathBuf,
    first: u16,
    size: u64,
}

#[derive(Clone)]
struct DirEntry {
    name: [u8; 11],
    attributes: u8,
    first: u16,
    size: u32,
    raw: [u8; 32],
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

fn hash_file(path: &Path) -> Result<[u8; 32], String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    hash_reader(&mut file, 0, length)
}

fn hash_reader(file: &mut File, offset: u64, length: u64) -> Result<[u8; 32], String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek failed: {error}"))?;
    let mut state = Sha256::new();
    let mut remaining = length;
    let mut buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..wanted])
            .map_err(|error| format!("short read at {}: {error}", offset + length - remaining))?;
        state.update(&buffer[..wanted]);
        remaining -= wanted as u64;
    }
    Ok(state.finish())
}

fn canonical(value: &Value) -> Result<Vec<u8>, String> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|error| format!("JSON encoding failed: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn command_version(command: &Path) -> Result<String, String> {
    let output = Command::new(command)
        .arg("--version")
        .output()
        .map_err(|error| format!("cannot run {}: {error}", command.display()))?;
    if !output.status.success() {
        return Err(format!("{} --version failed", command.display()));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| format!("{} --version was not UTF-8", command.display()))
}

fn input_rows(repo: &Path) -> Result<Value, String> {
    let mut paths = INPUT_PATHS.to_vec();
    paths.sort();
    let mut rows = Vec::with_capacity(paths.len());
    for name in paths {
        let path = repo.join(name);
        let bytes = path
            .metadata()
            .map_err(|error| format!("missing build input {name}: {error}"))?
            .len();
        rows.push(json!({"bytes": bytes, "path": name, "sha256": hex(&hash_file(&path)?)}));
    }
    Ok(Value::Array(rows))
}

fn media_contract() -> Value {
    json!({
        "bytes": IMAGE_BYTES,
        "bytes_per_sector": SECTOR,
        "data_start_sector": DATA_START,
        "drive_number": 128,
        "extended_boot_signature": 41,
        "fat_count": 2,
        "filesystem": "FAT16",
        "filesystem_label": "FAT16",
        "heads": 255,
        "hidden_sectors": 0,
        "insertion_order": ["EFI","EFI/BOOT","EFI/BOOT/BOOTX64.EFI","EFI/BOOT/BUILD.JSN","MODEL.PBT"],
        "media_descriptor": 248,
        "nt_flags": 0,
        "reserved_sectors": 1,
        "root_entries": ROOT_ENTRIES,
        "sectors_per_cluster": SECTORS_PER_CLUSTER,
        "sectors_per_fat": SECTORS_PER_FAT,
        "sectors_per_track": 63,
        "timestamp": "1980-01-01T00:00:00Z",
        "total_sectors": SECTORS,
        "total_sectors_16": 0,
        "volume_id": "544f4250",
        "volume_label": "PROMPTBOOT"
    })
}

fn clusters(size: u64) -> Result<u64, String> {
    if size == 0 {
        return Err("files embedded in the image must be nonempty".to_owned());
    }
    size.checked_add(CLUSTER_BYTES - 1)
        .map(|value| value / CLUSTER_BYTES)
        .ok_or_else(|| "cluster count overflow".to_owned())
}

fn cluster_offset(cluster: u16) -> Result<u64, String> {
    if cluster < 2 {
        return Err(format!("invalid data cluster {cluster}"));
    }
    Ok(DATA_START * SECTOR + (u64::from(cluster) - 2) * CLUSTER_BYTES)
}

fn entry(name: &[u8; 11], attributes: u8, first: u16, size: u64) -> Result<[u8; 32], String> {
    let size = u32::try_from(size).map_err(|_| "directory file size exceeds u32".to_owned())?;
    let mut result = [0; 32];
    result[..11].copy_from_slice(name);
    result[11] = attributes;
    result[14..16].copy_from_slice(&0u16.to_le_bytes());
    result[16..18].copy_from_slice(&0x21u16.to_le_bytes());
    result[18..20].copy_from_slice(&0x21u16.to_le_bytes());
    result[22..24].copy_from_slice(&0u16.to_le_bytes());
    result[24..26].copy_from_slice(&0x21u16.to_le_bytes());
    result[26..28].copy_from_slice(&first.to_le_bytes());
    result[28..32].copy_from_slice(&size.to_le_bytes());
    Ok(result)
}

fn write_at(file: &mut File, offset: u64, bytes: &[u8]) -> Result<(), String> {
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.write_all(bytes))
        .map_err(|error| format!("image write at {offset} failed: {error}"))
}

fn copy_into(image: &mut File, source_path: &Path, first: u16, size: u64) -> Result<(), String> {
    if source_path
        .metadata()
        .map_err(|error| error.to_string())?
        .len()
        != size
    {
        return Err(format!("payload size changed: {}", source_path.display()));
    }
    image
        .seek(SeekFrom::Start(cluster_offset(first)?))
        .map_err(|error| format!("cannot seek image payload: {error}"))?;
    let mut source = File::open(source_path)
        .map_err(|error| format!("cannot open {}: {error}", source_path.display()))?;
    std::io::copy(&mut source, image).map_err(|error| format!("payload copy failed: {error}"))?;
    Ok(())
}

fn chain_fat(fat: &mut [u8], first: u16, count: u64) -> Result<(), String> {
    for index in 0..count {
        let cluster = u64::from(first)
            .checked_add(index)
            .ok_or_else(|| "FAT cluster overflow".to_owned())?;
        if cluster >= 65_536 {
            return Err("FAT allocation exceeds table".to_owned());
        }
        let value = if index + 1 == count {
            0xffff
        } else {
            cluster + 1
        };
        let offset = cluster as usize * 2;
        fat[offset..offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
    }
    Ok(())
}

fn distribution_rows(
    root: Option<&Path>,
    mut first: u16,
) -> Result<(Vec<DistributionRow>, u16), String> {
    let Some(root) = root else {
        return Ok((Vec::new(), 0));
    };
    let mut rows = Vec::with_capacity(DISTRIBUTION.len());
    for (index, (name, short)) in DISTRIBUTION.iter().enumerate() {
        if index == 2 {
            first = first
                .checked_add(1)
                .ok_or_else(|| "distribution allocation overflow".to_owned())?;
        }
        let path = root.join(name);
        let size = path
            .metadata()
            .map_err(|error| format!("distribution payload missing {name}: {error}"))?
            .len();
        if size == 0 {
            return Err(format!("distribution payload is empty: {name}"));
        }
        rows.push(DistributionRow {
            short,
            path,
            first,
            size,
        });
        first = first
            .checked_add(
                u16::try_from(clusters(size)?)
                    .map_err(|_| "distribution cluster overflow".to_owned())?,
            )
            .ok_or_else(|| "distribution allocation overflow".to_owned())?;
    }
    let licenses_first = rows[2].first - 1;
    Ok((rows, licenses_first))
}

fn build_fat_image(
    efi: &Path,
    manifest: &Path,
    model: &Path,
    output: &Path,
    distribution_root: Option<&Path>,
) -> Result<(), String> {
    let efi_size = efi.metadata().map_err(|error| error.to_string())?.len();
    let manifest_size = manifest
        .metadata()
        .map_err(|error| error.to_string())?
        .len();
    let model_size = model.metadata().map_err(|error| error.to_string())?.len();
    if model_size != MODEL_BYTES {
        return Err(format!("packed model size {model_size} != {MODEL_BYTES}"));
    }
    let efi_first = 4u16;
    let manifest_first = efi_first
        .checked_add(
            u16::try_from(clusters(efi_size)?).map_err(|_| "EFI allocation overflow".to_owned())?,
        )
        .ok_or_else(|| "EFI allocation overflow".to_owned())?;
    let model_first = manifest_first
        .checked_add(
            u16::try_from(clusters(manifest_size)?)
                .map_err(|_| "manifest allocation overflow".to_owned())?,
        )
        .ok_or_else(|| "manifest allocation overflow".to_owned())?;
    if clusters(model_size)? != MODEL_CLUSTERS {
        return Err("model cluster count changed".to_owned());
    }
    let after_model = model_first
        .checked_add(MODEL_CLUSTERS as u16)
        .ok_or_else(|| "model allocation overflow".to_owned())?;
    let (distribution, licenses_first) = distribution_rows(distribution_root, after_model)?;
    let allocation_end = distribution
        .last()
        .map(|row| u64::from(row.first) + clusters(row.size).unwrap())
        .unwrap_or(u64::from(after_model));
    if allocation_end > 65_503 {
        return Err("image allocation exceeds fixed FAT16 geometry".to_owned());
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create image directory: {error}"))?;
    }
    let mut image = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(output)
        .map_err(|error| format!("cannot create image: {error}"))?;
    image
        .set_len(IMAGE_BYTES)
        .map_err(|error| format!("cannot size image: {error}"))?;

    let mut boot = [0u8; 512];
    boot[..3].copy_from_slice(b"\xeb\x3c\x90");
    boot[3..11].copy_from_slice(b"PRMPTBOT");
    boot[11..13].copy_from_slice(&(SECTOR as u16).to_le_bytes());
    boot[13] = SECTORS_PER_CLUSTER as u8;
    boot[14..16].copy_from_slice(&1u16.to_le_bytes());
    boot[16] = 2;
    boot[17..19].copy_from_slice(&(ROOT_ENTRIES as u16).to_le_bytes());
    boot[21] = 0xf8;
    boot[22..24].copy_from_slice(&(SECTORS_PER_FAT as u16).to_le_bytes());
    boot[24..26].copy_from_slice(&63u16.to_le_bytes());
    boot[26..28].copy_from_slice(&255u16.to_le_bytes());
    boot[32..36].copy_from_slice(&(SECTORS as u32).to_le_bytes());
    boot[36] = 0x80;
    boot[38] = 0x29;
    boot[39..43].copy_from_slice(&VOLUME_ID.to_le_bytes());
    boot[43..54].copy_from_slice(b"PROMPTBOOT ");
    boot[54..62].copy_from_slice(b"FAT16   ");
    boot[510..512].copy_from_slice(b"\x55\xaa");
    write_at(&mut image, 0, &boot)?;

    let mut fat = vec![0u8; (SECTORS_PER_FAT * SECTOR) as usize];
    fat[..8].copy_from_slice(b"\xf8\xff\xff\xff\xff\xff\xff\xff");
    chain_fat(&mut fat, efi_first, clusters(efi_size)?)?;
    chain_fat(&mut fat, manifest_first, clusters(manifest_size)?)?;
    chain_fat(&mut fat, model_first, MODEL_CLUSTERS)?;
    if !distribution.is_empty() {
        let offset = licenses_first as usize * 2;
        fat[offset..offset + 2].copy_from_slice(&0xffffu16.to_le_bytes());
        for row in &distribution {
            chain_fat(&mut fat, row.first, clusters(row.size)?)?;
        }
    }
    write_at(&mut image, SECTOR, &fat)?;
    write_at(&mut image, (1 + SECTORS_PER_FAT) * SECTOR, &fat)?;

    let root = (1 + 2 * SECTORS_PER_FAT) * SECTOR;
    write_at(&mut image, root, &entry(b"PROMPTBOOT ", 0x08, 0, 0)?)?;
    write_at(&mut image, root + 32, &entry(b"EFI        ", 0x10, 2, 0)?)?;
    write_at(
        &mut image,
        root + 64,
        &entry(b"MODEL   PBT", 0x20, model_first, model_size)?,
    )?;
    if !distribution.is_empty() {
        write_at(
            &mut image,
            root + 96,
            &entry(
                distribution[0].short,
                0x20,
                distribution[0].first,
                distribution[0].size,
            )?,
        )?;
        write_at(
            &mut image,
            root + 128,
            &entry(
                distribution[1].short,
                0x20,
                distribution[1].first,
                distribution[1].size,
            )?,
        )?;
        write_at(
            &mut image,
            root + 160,
            &entry(b"LICENSES   ", 0x10, licenses_first, 0)?,
        )?;
    }
    let efi_dir = cluster_offset(2)?;
    for (index, row) in [
        entry(b".          ", 0x10, 2, 0)?,
        entry(b"..         ", 0x10, 0, 0)?,
        entry(b"BOOT       ", 0x10, 3, 0)?,
    ]
    .iter()
    .enumerate()
    {
        write_at(&mut image, efi_dir + index as u64 * 32, row)?;
    }
    let boot_dir = cluster_offset(3)?;
    for (index, row) in [
        entry(b".          ", 0x10, 3, 0)?,
        entry(b"..         ", 0x10, 2, 0)?,
        entry(b"BOOTX64 EFI", 0x20, efi_first, efi_size)?,
        entry(b"BUILD   JSN", 0x20, manifest_first, manifest_size)?,
    ]
    .iter()
    .enumerate()
    {
        write_at(&mut image, boot_dir + index as u64 * 32, row)?;
    }
    copy_into(&mut image, efi, efi_first, efi_size)?;
    copy_into(&mut image, manifest, manifest_first, manifest_size)?;
    copy_into(&mut image, model, model_first, model_size)?;
    if !distribution.is_empty() {
        let licenses_dir = cluster_offset(licenses_first)?;
        write_at(
            &mut image,
            licenses_dir,
            &entry(b".          ", 0x10, licenses_first, 0)?,
        )?;
        write_at(
            &mut image,
            licenses_dir + 32,
            &entry(b"..         ", 0x10, 0, 0)?,
        )?;
        for (index, row) in distribution.iter().skip(2).enumerate() {
            write_at(
                &mut image,
                licenses_dir + (index as u64 + 2) * 32,
                &entry(row.short, 0x20, row.first, row.size)?,
            )?;
        }
        for row in &distribution {
            copy_into(&mut image, &row.path, row.first, row.size)?;
        }
    }
    image
        .sync_all()
        .map_err(|error| format!("cannot sync image: {error}"))?;
    Ok(())
}

fn with_staged_output<T>(
    output: &Path,
    build: impl FnOnce(&Path) -> Result<T, String>,
) -> Result<T, String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("cannot create output parent: {error}"))?;
    if output.exists() {
        if !output.is_dir()
            || output
                .read_dir()
                .map_err(|error| error.to_string())?
                .next()
                .is_some()
        {
            return Err("MODEL_IMAGE_OUTPUT_NOT_EMPTY".to_owned());
        }
    }
    let name = output
        .file_name()
        .ok_or_else(|| "output must name a directory".to_owned())?
        .to_string_lossy();
    let mut staging = None;
    for ordinal in 0..100 {
        let candidate = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), ordinal));
        match fs::create_dir(&candidate) {
            Ok(()) => {
                staging = Some(candidate);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(format!("cannot create output staging directory: {error}")),
        }
    }
    let staging = staging.ok_or_else(|| "cannot allocate output staging directory".to_owned())?;
    let result = build(&staging);
    let value = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    if let Err(error) = fs::rename(&staging, output) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("cannot publish image output: {error}"));
    }
    Ok(value)
}

fn build_staged(
    repo: &Path,
    source: &Path,
    output: &Path,
    target: &Path,
    distribution_root: Option<&Path>,
) -> Result<String, String> {
    let cargo = PathBuf::from(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    let rustc = PathBuf::from(env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
    let cargo_version = command_version(&cargo)?;
    let rustc_version = command_version(&rustc)?;
    if cargo_version != CARGO_VERSION || rustc_version != RUSTC_VERSION {
        return Err(format!(
            "pinned toolchain mismatch cargo={cargo_version:?} rustc={rustc_version:?}"
        ));
    }
    if target.exists()
        && target
            .read_dir()
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
    {
        return Err("MODEL_IMAGE_TARGET_NOT_EMPTY".to_owned());
    }
    fs::create_dir_all(target).map_err(|error| format!("cannot create target: {error}"))?;
    let model = output.join("MODEL.PBT");
    pack::run(source, &model).map_err(|failure| match failure {
        pack::PackFailure::Identity(error) => format!("model identity: {error}"),
        pack::PackFailure::Schema(error) => format!("model schema: {error}"),
        pack::PackFailure::Pack(error) => format!("model pack: {error}"),
    })?;
    if model.metadata().map_err(|error| error.to_string())?.len() != MODEL_BYTES
        || hex(&hash_file(&model)?) != MODEL_SHA256
    {
        return Err("fresh packed model identity mismatch".to_owned());
    }

    let build_inputs = json!({
        "artifact_contract": {
            "format": "PBTQW25-v1",
            "media_bytes": IMAGE_BYTES,
            "model_bytes": MODEL_BYTES,
            "model_sha256": MODEL_SHA256
        },
        "fresh_pack": {"format":"PBTQW25-v1","packer":"promptboot-tools pack-model","verified":true},
        "inputs": input_rows(&repo)?,
        "model_outer_sha256": MODEL_SHA256,
        "profile": "release",
        "prompt_contract": {
            "bytes":330288,
            "context":32768,
            "history":"whole_turn",
            "reserve":1024
        },
        "source_date_epoch": 315532800u64,
        "source_model": {
            "bytes": SOURCE_BYTES,
            "path":"asset-cache/qwen2.5-0.5b-instruct-q4_0.gguf",
            "sha256":hex(&SOURCE_SHA256)
        },
        "target":"x86_64-unknown-uefi",
        "toolchain":{"cargo":cargo_version,"rustc":rustc_version}
    });
    let mut id_bytes = canonical(&build_inputs)?;
    id_bytes.extend(b"model_repl");
    let build_id = hex(&promptboot_core::sha256::digest(&id_bytes));
    let rustflags = format!(
        "-C panic=abort -C force-frame-pointers=yes -C link-arg=/Brepro -C link-arg=/debug:none --remap-path-prefix={}=/src/promptboot",
        repo.display()
    );
    let cargo_output = Command::new(&cargo)
        .args([
            "build",
            "--manifest-path",
            repo.join("Cargo.toml").to_str().unwrap(),
            "--target",
            "x86_64-unknown-uefi",
            "--target-dir",
            target.to_str().unwrap(),
            "--release",
            "--locked",
            "--offline",
        ])
        .env("PROMPTBOOT_BUILD_ID", &build_id)
        .env("PROMPTBOOT_EXPECTED_MODEL_SHA256", MODEL_SHA256)
        .env("PROMPTBOOT_MODE", "model_repl")
        .env("PROMPTBOOT_SELF_TEST", "none")
        .env("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)
        .env("RUSTC", &rustc)
        .env("RUSTFLAGS", rustflags)
        .output()
        .map_err(|error| format!("UEFI cargo build failed to start: {error}"))?;
    if !cargo_output.status.success() {
        return Err(format!("UEFI cargo build failed: {}", cargo_output.status));
    }
    let efi_source = target.join("x86_64-unknown-uefi/release/promptboot.efi");
    let efi = output.join("BOOTX64.EFI");
    fs::copy(&efi_source, &efi).map_err(|error| format!("cannot copy BOOTX64.EFI: {error}"))?;
    let manifest = json!({
        "artifact_class":"production",
        "artifacts":{
            "efi":{"bytes":efi.metadata().map_err(|error| error.to_string())?.len(),"path":"EFI/BOOT/BOOTX64.EFI","sha256":hex(&hash_file(&efi)?)},
            "model":{"bytes":MODEL_BYTES,"format":"PBTQW25-v1","path":"MODEL.PBT","sha256":MODEL_SHA256,"source_gguf_sha256":hex(&SOURCE_SHA256)}
        },
        "build_id":build_id,
        "build_inputs":build_inputs,
        "event_topology":EVENT_TOPOLOGY,
        "media":media_contract(),
        "mode":"model_repl",
        "schema":2
    });
    let manifest_path = output.join("BUILD.JSN");
    fs::write(&manifest_path, canonical(&manifest)?)
        .map_err(|error| format!("cannot write BUILD.JSN: {error}"))?;
    let image = output.join("promptboot.img");
    build_fat_image(&efi, &manifest_path, &model, &image, distribution_root)?;
    let report = inspect_image(&image, &manifest_path, distribution_root)?;
    fs::write(
        output.join("promptboot-media-inspection.json"),
        canonical(&report)?,
    )
    .map_err(|error| format!("cannot write image report: {error}"))?;
    let mut sums = String::new();
    for name in [
        "BOOTX64.EFI",
        "BUILD.JSN",
        "MODEL.PBT",
        "promptboot.img",
        "promptboot-media-inspection.json",
    ] {
        sums.push_str(&format!(
            "{}  {name}\n",
            hex(&hash_file(&output.join(name))?)
        ));
    }
    fs::write(output.join("SHA256SUMS"), sums)
        .map_err(|error| format!("cannot write SHA256SUMS: {error}"))?;
    Ok(build_id)
}

pub fn build_image(
    source: &Path,
    output: &Path,
    target: &Path,
    distribution_root: Option<&Path>,
) -> Result<String, String> {
    let repo = env::current_dir().map_err(|error| error.to_string())?;
    if !repo.join("Cargo.toml").is_file() {
        return Err("build-image must run from the promptboot repository root".to_owned());
    }
    with_staged_output(output, |staging| {
        build_staged(&repo, source, staging, target, distribution_root)
    })
}

fn read_at(file: &mut File, offset: u64, length: usize) -> Result<Vec<u8>, String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("image seek failed: {error}"))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("image read at {offset} failed: {error}"))?;
    Ok(bytes)
}

fn parse_entry(file: &mut File, offset: u64) -> Result<DirEntry, String> {
    let bytes: [u8; 32] = read_at(file, offset, 32)?.try_into().unwrap();
    Ok(DirEntry {
        name: bytes[..11].try_into().unwrap(),
        attributes: bytes[11],
        first: u16::from_le_bytes(bytes[26..28].try_into().unwrap()),
        size: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        raw: bytes,
    })
}

fn require_entry(
    actual: &DirEntry,
    name: &[u8; 11],
    attributes: u8,
    first: u16,
    size: u64,
    label: &str,
) -> Result<(), String> {
    if actual.raw != entry(name, attributes, first, size)? {
        return Err(format!("{label} directory entry mismatch"));
    }
    Ok(())
}

fn require_zero(file: &mut File, start: u64, end: u64, label: &str) -> Result<(), String> {
    if end < start {
        return Err(format!("{label} range is inverted"));
    }
    file.seek(SeekFrom::Start(start))
        .map_err(|error| format!("{label} seek failed: {error}"))?;
    let mut remaining = end - start;
    let mut buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..wanted])
            .map_err(|error| format!("{label} is truncated: {error}"))?;
        if buffer[..wanted].iter().any(|byte| *byte != 0) {
            return Err(format!("nonzero bytes in {label}"));
        }
        remaining -= wanted as u64;
    }
    Ok(())
}

fn range_matches_path(
    image: &mut File,
    offset: u64,
    length: u64,
    path: &Path,
) -> Result<bool, String> {
    if path.metadata().map_err(|error| error.to_string())?.len() != length {
        return Ok(false);
    }
    image
        .seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let mut external =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut remaining = length;
    let mut left = vec![0; 1024 * 1024];
    let mut right = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(left.len() as u64) as usize;
        image
            .read_exact(&mut left[..wanted])
            .map_err(|error| error.to_string())?;
        external
            .read_exact(&mut right[..wanted])
            .map_err(|error| error.to_string())?;
        if left[..wanted] != right[..wanted] {
            return Ok(false);
        }
        remaining -= wanted as u64;
    }
    Ok(true)
}

fn file_chain(fat: &[u8], first: u16, size: u64, label: &str) -> Result<Vec<u16>, String> {
    let count = clusters(size)?;
    let mut chain = Vec::with_capacity(count as usize);
    let mut current = first;
    for index in 0..count {
        if !(2..=u16::MAX).contains(&current) {
            return Err(format!("{label} FAT cluster out of range"));
        }
        chain.push(current);
        let offset = current as usize * 2;
        let following = u16::from_le_bytes(fat[offset..offset + 2].try_into().unwrap());
        let wanted = if index + 1 == count {
            0xffff
        } else {
            current
                .checked_add(1)
                .ok_or_else(|| format!("{label} FAT next cluster overflow"))?
        };
        if following != wanted {
            return Err(format!(
                "{label} FAT chain mismatch cluster={current} actual={following} expected={wanted}"
            ));
        }
        current = following;
    }
    Ok(chain)
}

fn value_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("BUILD.JSN {label} must be an object"))
}

fn exact_keys(object: &Map<String, Value>, keys: &[&str], label: &str) -> Result<(), String> {
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = keys.iter().copied().collect();
    if actual != expected {
        return Err(format!("BUILD.JSN {label} schema mismatch"));
    }
    Ok(())
}

fn contains_absolute_string(value: &Value, repo: &Path) -> bool {
    match value {
        Value::String(text) => text.starts_with('/') || text.contains(&repo.display().to_string()),
        Value::Array(items) => items
            .iter()
            .any(|item| contains_absolute_string(item, repo)),
        Value::Object(items) => items
            .iter()
            .any(|(key, item)| key.starts_with('/') || contains_absolute_string(item, repo)),
        _ => false,
    }
}

fn validate_manifest(value: &Value, external: &[u8], repo: &Path) -> Result<(), String> {
    let manifest = value_object(value, "top-level")?;
    exact_keys(
        manifest,
        &[
            "artifact_class",
            "artifacts",
            "build_id",
            "build_inputs",
            "event_topology",
            "media",
            "mode",
            "schema",
        ],
        "top-level",
    )?;
    if manifest.get("schema").and_then(Value::as_u64) != Some(2)
        || manifest.get("mode").and_then(Value::as_str) != Some("model_repl")
        || manifest.get("artifact_class").and_then(Value::as_str) != Some("production")
        || manifest.get("event_topology").and_then(Value::as_str) != Some(EVENT_TOPOLOGY)
    {
        return Err("BUILD.JSN production contract mismatch".to_owned());
    }
    if canonical(value)? != external {
        return Err("BUILD.JSN is not canonical compact sorted JSON+LF".to_owned());
    }
    if manifest.get("media") != Some(&media_contract()) {
        return Err("BUILD.JSN media/BPB contract mismatch".to_owned());
    }
    let artifacts = value_object(
        manifest
            .get("artifacts")
            .ok_or_else(|| "BUILD.JSN artifacts missing".to_owned())?,
        "artifacts",
    )?;
    exact_keys(artifacts, &["efi", "model"], "artifacts")?;
    let efi = value_object(artifacts.get("efi").unwrap(), "artifacts.efi")?;
    let model = value_object(artifacts.get("model").unwrap(), "artifacts.model")?;
    exact_keys(efi, &["bytes", "path", "sha256"], "artifacts.efi")?;
    exact_keys(
        model,
        &["bytes", "format", "path", "sha256", "source_gguf_sha256"],
        "artifacts.model",
    )?;
    if efi.get("path").and_then(Value::as_str) != Some("EFI/BOOT/BOOTX64.EFI") {
        return Err("BUILD.JSN EFI path mismatch".to_owned());
    }
    if model.get("bytes").and_then(Value::as_u64) != Some(MODEL_BYTES)
        || model.get("format").and_then(Value::as_str) != Some("PBTQW25-v1")
        || model.get("path").and_then(Value::as_str) != Some("MODEL.PBT")
        || model.get("sha256").and_then(Value::as_str) != Some(MODEL_SHA256)
        || model.get("source_gguf_sha256").and_then(Value::as_str) != Some(&hex(&SOURCE_SHA256))
    {
        return Err("BUILD.JSN model identity mismatch".to_owned());
    }
    let build_inputs = manifest
        .get("build_inputs")
        .ok_or_else(|| "BUILD.JSN build_inputs missing".to_owned())?;
    let inputs = value_object(build_inputs, "build_inputs")?;
    exact_keys(
        inputs,
        &[
            "artifact_contract",
            "fresh_pack",
            "inputs",
            "model_outer_sha256",
            "profile",
            "prompt_contract",
            "source_date_epoch",
            "source_model",
            "target",
            "toolchain",
        ],
        "build_inputs",
    )?;
    if inputs.get("inputs") != Some(&input_rows(repo)?)
        || inputs.get("profile").and_then(Value::as_str) != Some("release")
        || inputs.get("target").and_then(Value::as_str) != Some("x86_64-unknown-uefi")
        || inputs.get("source_date_epoch").and_then(Value::as_u64) != Some(315532800)
    {
        return Err("BUILD.JSN exact input closure/target mismatch".to_owned());
    }
    if inputs.get("artifact_contract")
        != Some(&json!({
            "format":"PBTQW25-v1",
            "media_bytes":IMAGE_BYTES,
            "model_bytes":MODEL_BYTES,
            "model_sha256":MODEL_SHA256
        }))
    {
        return Err("BUILD.JSN artifact_contract mismatch".to_owned());
    }
    if inputs.get("fresh_pack")
        != Some(
            &json!({"format":"PBTQW25-v1","packer":"promptboot-tools pack-model","verified":true}),
        )
    {
        return Err("BUILD.JSN fresh_pack mismatch".to_owned());
    }
    if inputs.get("model_outer_sha256").and_then(Value::as_str) != Some(MODEL_SHA256) {
        return Err("BUILD.JSN model_outer_sha256 mismatch".to_owned());
    }
    if inputs.get("prompt_contract")
        != Some(&json!({"bytes":330288,"context":32768,"history":"whole_turn","reserve":1024}))
    {
        return Err("BUILD.JSN prompt_contract mismatch".to_owned());
    }
    if inputs.get("source_model")
        != Some(&json!({
            "bytes":SOURCE_BYTES,
            "path":"asset-cache/qwen2.5-0.5b-instruct-q4_0.gguf",
            "sha256":hex(&SOURCE_SHA256)
        }))
    {
        return Err("BUILD.JSN source_model mismatch".to_owned());
    }
    if inputs.get("toolchain") != Some(&json!({"cargo":CARGO_VERSION,"rustc":RUSTC_VERSION})) {
        return Err("BUILD.JSN toolchain mismatch".to_owned());
    }
    if contains_absolute_string(build_inputs, repo) {
        return Err("BUILD.JSN contains absolute host path".to_owned());
    }
    let mut id_bytes = canonical(build_inputs)?;
    id_bytes.extend(b"model_repl");
    if manifest.get("build_id").and_then(Value::as_str)
        != Some(&hex(&promptboot_core::sha256::digest(&id_bytes)))
    {
        return Err("BUILD.JSN build_id derivation mismatch".to_owned());
    }
    Ok(())
}

fn distribution_external(root: Option<&Path>) -> Result<BTreeMap<&'static str, PathBuf>, String> {
    let Some(root) = root else {
        return Ok(BTreeMap::new());
    };
    let mut result = BTreeMap::new();
    for (name, _) in DISTRIBUTION {
        let direct = root.join(name);
        let path = if direct.is_file() {
            direct
        } else {
            root.join(match name {
                "LICENSES/3RDPARTY.TXT" => "THIRD_PARTY_NOTICES.md",
                "LICENSES/QWEN.TXT" => "LICENSES/QWEN-APACHE-2.0.txt",
                "LICENSES/LLAMA.TXT" => "LICENSES/LLAMA-MIT.txt",
                "LICENSES/LIBM.TXT" => "LICENSES/libm-0.2.11.txt",
                "LICENSES/RUSTCORE.HTM" => "LICENSES/RUST-1.97.1-COPYRIGHT-library.html",
                "LICENSES/RUSTCB.TXT" => "LICENSES/compiler-builtins-0.1.160.txt",
                other => other,
            })
        };
        if !path.is_file() || path.metadata().map_err(|error| error.to_string())?.len() == 0 {
            return Err(format!("distribution payload missing/empty: {name}"));
        }
        result.insert(name, path);
    }
    Ok(result)
}

pub fn inspect_image(
    image_path: &Path,
    manifest_path: &Path,
    distribution_root: Option<&Path>,
) -> Result<Value, String> {
    let repo = env::current_dir().map_err(|error| error.to_string())?;
    let image_size = image_path
        .metadata()
        .map_err(|error| format!("cannot stat image: {error}"))?
        .len();
    if image_size != IMAGE_BYTES {
        return Err(format!("image bytes {image_size} != {IMAGE_BYTES}"));
    }
    let mut image =
        File::open(image_path).map_err(|error| format!("cannot open image: {error}"))?;
    let boot = read_at(&mut image, 0, 512)?;
    let mut expected_boot = [0u8; 512];
    expected_boot[..3].copy_from_slice(b"\xeb\x3c\x90");
    expected_boot[3..11].copy_from_slice(b"PRMPTBOT");
    expected_boot[11..13].copy_from_slice(&(SECTOR as u16).to_le_bytes());
    expected_boot[13] = SECTORS_PER_CLUSTER as u8;
    expected_boot[14..16].copy_from_slice(&1u16.to_le_bytes());
    expected_boot[16] = 2;
    expected_boot[17..19].copy_from_slice(&(ROOT_ENTRIES as u16).to_le_bytes());
    expected_boot[21] = 0xf8;
    expected_boot[22..24].copy_from_slice(&(SECTORS_PER_FAT as u16).to_le_bytes());
    expected_boot[24..26].copy_from_slice(&63u16.to_le_bytes());
    expected_boot[26..28].copy_from_slice(&255u16.to_le_bytes());
    expected_boot[32..36].copy_from_slice(&(SECTORS as u32).to_le_bytes());
    expected_boot[36] = 0x80;
    expected_boot[38] = 0x29;
    expected_boot[39..43].copy_from_slice(&VOLUME_ID.to_le_bytes());
    expected_boot[43..54].copy_from_slice(b"PROMPTBOOT ");
    expected_boot[54..62].copy_from_slice(b"FAT16   ");
    expected_boot[510..512].copy_from_slice(b"\x55\xaa");
    if boot != expected_boot {
        return Err("BPB mismatch".to_owned());
    }
    let fat = read_at(&mut image, SECTOR, (SECTORS_PER_FAT * SECTOR) as usize)?;
    let fat2 = read_at(
        &mut image,
        (1 + SECTORS_PER_FAT) * SECTOR,
        (SECTORS_PER_FAT * SECTOR) as usize,
    )?;
    if fat != fat2 {
        return Err("FAT copies differ".to_owned());
    }
    if fat[..8] != b"\xf8\xff\xff\xff\xff\xff\xff\xff"[..] {
        return Err("reserved/directory FAT entries mismatch".to_owned());
    }
    let root = (1 + 2 * SECTORS_PER_FAT) * SECTOR;
    let volume = parse_entry(&mut image, root)?;
    let efi_root = parse_entry(&mut image, root + 32)?;
    let model_root = parse_entry(&mut image, root + 64)?;
    require_entry(&volume, b"PROMPTBOOT ", 0x08, 0, 0, "volume")?;
    require_entry(&efi_root, b"EFI        ", 0x10, 2, 0, "EFI")?;
    if model_root.name != *b"MODEL   PBT" || model_root.attributes != 0x20 {
        return Err("MODEL.PBT root entry mismatch".to_owned());
    }
    let distribution = distribution_external(distribution_root)?;
    let root_used = if distribution.is_empty() { 96 } else { 192 };
    require_zero(
        &mut image,
        root + root_used,
        root + ROOT_SECTORS * SECTOR,
        "root directory slack",
    )?;

    let efi_dir = cluster_offset(2)?;
    for (index, (name, first)) in [
        (b".          ", 2),
        (b"..         ", 0),
        (b"BOOT       ", 3),
    ]
    .iter()
    .enumerate()
    {
        let actual = parse_entry(&mut image, efi_dir + index as u64 * 32)?;
        require_entry(&actual, name, 0x10, *first, 0, "EFI directory")?;
    }
    require_zero(
        &mut image,
        efi_dir + 96,
        efi_dir + CLUSTER_BYTES,
        "EFI directory slack",
    )?;
    let boot_dir = cluster_offset(3)?;
    let dot = parse_entry(&mut image, boot_dir)?;
    let dotdot = parse_entry(&mut image, boot_dir + 32)?;
    require_entry(&dot, b".          ", 0x10, 3, 0, "BOOT dot")?;
    require_entry(&dotdot, b"..         ", 0x10, 2, 0, "BOOT dotdot")?;
    let efi_entry = parse_entry(&mut image, boot_dir + 64)?;
    let manifest_entry = parse_entry(&mut image, boot_dir + 96)?;
    if efi_entry.name != *b"BOOTX64 EFI"
        || efi_entry.attributes != 0x20
        || manifest_entry.name != *b"BUILD   JSN"
        || manifest_entry.attributes != 0x20
    {
        return Err("BOOT file order mismatch".to_owned());
    }
    require_entry(
        &efi_entry,
        b"BOOTX64 EFI",
        0x20,
        efi_entry.first,
        efi_entry.size as u64,
        "BOOTX64.EFI",
    )?;
    require_entry(
        &manifest_entry,
        b"BUILD   JSN",
        0x20,
        manifest_entry.first,
        manifest_entry.size as u64,
        "BUILD.JSN",
    )?;
    require_zero(
        &mut image,
        boot_dir + 128,
        boot_dir + CLUSTER_BYTES,
        "BOOT directory slack",
    )?;
    let efi_chain = file_chain(&fat, efi_entry.first, efi_entry.size as u64, "BOOTX64.EFI")?;
    let manifest_chain = file_chain(
        &fat,
        manifest_entry.first,
        manifest_entry.size as u64,
        "BUILD.JSN",
    )?;
    if efi_chain.first() != Some(&4)
        || manifest_chain.first()
            != efi_chain
                .last()
                .and_then(|value| value.checked_add(1))
                .as_ref()
    {
        return Err("canonical EFI/BUILD allocation order mismatch".to_owned());
    }
    let efi_offset = cluster_offset(efi_entry.first)?;
    let manifest_offset = cluster_offset(manifest_entry.first)?;
    require_zero(
        &mut image,
        efi_offset + efi_entry.size as u64,
        efi_offset + efi_chain.len() as u64 * CLUSTER_BYTES,
        "EFI file slack",
    )?;
    require_zero(
        &mut image,
        manifest_offset + manifest_entry.size as u64,
        manifest_offset + manifest_chain.len() as u64 * CLUSTER_BYTES,
        "BUILD file slack",
    )?;
    let external_manifest = fs::read(manifest_path)
        .map_err(|error| format!("cannot read external BUILD.JSN: {error}"))?;
    let embedded_manifest = read_at(&mut image, manifest_offset, manifest_entry.size as usize)?;
    if embedded_manifest != external_manifest {
        return Err("external manifest does not match embedded BUILD.JSN".to_owned());
    }
    let manifest: Value = serde_json::from_slice(&external_manifest)
        .map_err(|error| format!("BUILD.JSN invalid: {error}"))?;
    validate_manifest(&manifest, &external_manifest, &repo)?;
    let artifacts = manifest["artifacts"].as_object().unwrap();
    let external_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let external_efi = external_root.join("BOOTX64.EFI");
    let external_model = external_root.join("MODEL.PBT");
    if artifacts["efi"]["bytes"].as_u64() != Some(efi_entry.size as u64)
        || artifacts["efi"]["sha256"].as_str() != Some(&hex(&hash_file(&external_efi)?))
        || !range_matches_path(&mut image, efi_offset, efi_entry.size as u64, &external_efi)?
    {
        return Err("embedded EFI does not match external artifact/BUILD.JSN".to_owned());
    }
    if model_root.size as u64 != MODEL_BYTES
        || model_root.first != manifest_chain.last().unwrap() + 1
        || artifacts["model"]["sha256"].as_str() != Some(MODEL_SHA256)
        || hex(&hash_file(&external_model)?) != MODEL_SHA256
    {
        return Err("MODEL.PBT identity/allocation mismatch".to_owned());
    }
    let model_chain = file_chain(&fat, model_root.first, MODEL_BYTES, "MODEL.PBT")?;
    if model_chain.len() as u64 != MODEL_CLUSTERS {
        return Err("MODEL.PBT cluster count mismatch".to_owned());
    }
    let model_offset = cluster_offset(model_root.first)?;
    if !range_matches_path(&mut image, model_offset, MODEL_BYTES, &external_model)? {
        return Err("embedded MODEL.PBT differs from external artifact".to_owned());
    }
    if hex(&hash_reader(&mut image, model_offset, MODEL_BYTES)?) != MODEL_SHA256 {
        return Err("embedded MODEL.PBT hash mismatch".to_owned());
    }
    require_zero(
        &mut image,
        model_offset + MODEL_BYTES,
        model_offset + model_chain.len() as u64 * CLUSTER_BYTES,
        "model file slack",
    )?;
    let mut allocated: BTreeSet<u16> = [2, 3].into_iter().collect();
    allocated.extend(efi_chain.iter().copied());
    allocated.extend(manifest_chain.iter().copied());
    allocated.extend(model_chain.iter().copied());
    let mut distribution_report = Map::new();
    if distribution.is_empty() {
        if distribution_root.is_some() {
            return Err("distribution root did not resolve payloads".to_owned());
        }
    } else {
        let license_root = parse_entry(&mut image, root + 96)?;
        let source_root = parse_entry(&mut image, root + 128)?;
        let licenses_root = parse_entry(&mut image, root + 160)?;
        let mut entries = BTreeMap::new();
        entries.insert("LICENSE", license_root);
        entries.insert("SOURCE.TGZ", source_root);
        let expected_licenses_first = model_chain.last().unwrap()
            + 1
            + clusters(entries["LICENSE"].size as u64)? as u16
            + clusters(entries["SOURCE.TGZ"].size as u64)? as u16;
        if licenses_root.name != *b"LICENSES   "
            || licenses_root.attributes != 0x10
            || licenses_root.first != expected_licenses_first
        {
            return Err("LICENSES directory allocation mismatch".to_owned());
        }
        allocated.insert(licenses_root.first);
        let licenses_dir = cluster_offset(licenses_root.first)?;
        let current = parse_entry(&mut image, licenses_dir)?;
        let parent = parse_entry(&mut image, licenses_dir + 32)?;
        require_entry(
            &current,
            b".          ",
            0x10,
            licenses_root.first,
            0,
            "LICENSES dot",
        )?;
        require_entry(&parent, b"..         ", 0x10, 0, 0, "LICENSES dotdot")?;
        for (index, (name, short)) in DISTRIBUTION.iter().skip(2).enumerate() {
            let item = parse_entry(&mut image, licenses_dir + (index as u64 + 2) * 32)?;
            if item.name != **short || item.attributes != 0x20 {
                return Err(format!("distribution directory entry mismatch: {name}"));
            }
            entries.insert(name, item);
        }
        require_zero(
            &mut image,
            licenses_dir + (2 + DISTRIBUTION.len() as u64 - 2) * 32,
            licenses_dir + CLUSTER_BYTES,
            "LICENSES directory slack",
        )?;
        let mut next = model_chain.last().unwrap() + 1;
        for (index, (name, short)) in DISTRIBUTION.iter().enumerate() {
            if index == 2 {
                next += 1;
            }
            let item = &entries[name];
            let external = &distribution[name];
            require_entry(item, short, 0x20, next, item.size as u64, name)?;
            let chain = file_chain(&fat, item.first, item.size as u64, name)?;
            let offset = cluster_offset(item.first)?;
            if !range_matches_path(&mut image, offset, item.size as u64, external)? {
                return Err(format!("distribution payload mismatch: {name}"));
            }
            require_zero(
                &mut image,
                offset + item.size as u64,
                offset + chain.len() as u64 * CLUSTER_BYTES,
                &format!("{name} file slack"),
            )?;
            let digest = hex(&hash_reader(&mut image, offset, item.size as u64)?);
            distribution_report
                .insert(name.to_string(), json!({"bytes":item.size,"sha256":digest}));
            allocated.extend(chain.iter().copied());
            next = chain.last().unwrap() + 1;
        }
    }
    for cluster in 4u16..=u16::MAX {
        let offset = cluster as usize * 2;
        let value = u16::from_le_bytes(fat[offset..offset + 2].try_into().unwrap());
        if !allocated.contains(&cluster) && value != 0 {
            return Err(format!("orphan FAT entry cluster={cluster} value={value}"));
        }
    }
    let mut cursor = DATA_START * SECTOR;
    for cluster in &allocated {
        let offset = cluster_offset(*cluster)?;
        if cursor < offset {
            require_zero(&mut image, cursor, offset, "unallocated data")?;
        }
        cursor = cursor.max(offset + CLUSTER_BYTES);
    }
    require_zero(&mut image, cursor, IMAGE_BYTES, "trailing unallocated data")?;
    let mut report = json!({
        "build_jsn_bytes":external_manifest.len(),
        "build_jsn_sha256":hex(&promptboot_core::sha256::digest(&external_manifest)),
        "bytes":IMAGE_BYTES,
        "data_start_sector":DATA_START,
        "efi_bytes":efi_entry.size,
        "efi_sha256":hex(&hash_file(&external_efi)?),
        "filesystem":"FAT16",
        "image_sha256":hex(&hash_file(image_path)?),
        "kind":"positive",
        "model_bytes":MODEL_BYTES,
        "model_clusters":MODEL_CLUSTERS,
        "model_first_cluster":model_root.first,
        "model_sha256":MODEL_SHA256,
        "sectors_per_cluster":SECTORS_PER_CLUSTER,
        "volume_id":"544f4250"
    });
    if !distribution_report.is_empty() {
        report.as_object_mut().unwrap().insert(
            "distribution".to_owned(),
            Value::Object(distribution_report),
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_geometry_and_directory_encoding_match_the_format() {
        assert_eq!(clusters(1).unwrap(), 1);
        assert_eq!(clusters(CLUSTER_BYTES).unwrap(), 1);
        assert_eq!(clusters(CLUSTER_BYTES + 1).unwrap(), 2);
        assert!(clusters(0).is_err());
        assert_eq!(cluster_offset(2).unwrap(), 279_040);
        assert_eq!(cluster_offset(3).unwrap(), 287_232);
        let encoded = entry(b"BOOTX64 EFI", 0x20, 0x3456, 0x789a_bcde).unwrap();
        assert_eq!(&encoded[..12], b"BOOTX64 EFI\x20");
        assert_eq!(
            u16::from_le_bytes(encoded[26..28].try_into().unwrap()),
            0x3456
        );
        assert_eq!(
            u32::from_le_bytes(encoded[28..32].try_into().unwrap()),
            0x789a_bcde
        );
        assert_eq!(
            u16::from_le_bytes(encoded[16..18].try_into().unwrap()),
            0x21
        );
    }

    #[test]
    fn fat_chain_requires_consecutive_clusters_and_exact_eoc() {
        let mut fat = vec![0; (SECTORS_PER_FAT * SECTOR) as usize];
        chain_fat(&mut fat, 4, 2).unwrap();
        assert_eq!(
            file_chain(&fat, 4, CLUSTER_BYTES + 1, "fixture").unwrap(),
            [4, 5]
        );
        for value in [0xffffu16, 6, 4] {
            let mut changed = fat.clone();
            changed[8..10].copy_from_slice(&value.to_le_bytes());
            assert!(file_chain(&changed, 4, CLUSTER_BYTES + 1, "fixture").is_err());
        }
    }

    #[test]
    fn production_input_closure_is_native_and_sorted_by_builder() {
        assert!(INPUT_PATHS.contains(&"crates/promptboot-tools/src/image.rs"));
        assert!(INPUT_PATHS.contains(&"src/console_history.rs"));
        assert!(INPUT_PATHS.contains(&"src/mp_inference.rs"));
        assert!(INPUT_PATHS.contains(&"src/model_repl.rs"));
        assert!(!INPUT_PATHS.iter().any(|path| path.contains("model_fat16")));
        assert!(!INPUT_PATHS
            .iter()
            .any(|path| path.contains("fat16_mechanics")));
    }

    #[test]
    fn staged_publication_preserves_absent_empty_and_existing_outputs_on_failure() {
        let root = env::temp_dir().join(format!(
            "promptboot-image-stage-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        for checkpoint in ["pack", "cargo", "image", "inspect"] {
            let absent = root.join(format!("{checkpoint}-absent"));
            let failure = with_staged_output(&absent, |staging| {
                fs::write(staging.join("partial"), checkpoint).unwrap();
                Err::<(), String>(format!("{checkpoint} failed"))
            });
            assert_eq!(failure.unwrap_err(), format!("{checkpoint} failed"));
            assert!(!absent.exists());

            let empty = root.join(format!("{checkpoint}-empty"));
            fs::create_dir(&empty).unwrap();
            let failure = with_staged_output(&empty, |staging| {
                fs::write(staging.join("partial"), checkpoint).unwrap();
                Err::<(), String>(format!("{checkpoint} failed"))
            });
            assert_eq!(failure.unwrap_err(), format!("{checkpoint} failed"));
            assert!(empty.is_dir());
            assert_eq!(fs::read_dir(&empty).unwrap().count(), 0);
        }
        let existing = root.join("existing");
        fs::create_dir(&existing).unwrap();
        fs::write(existing.join("sentinel"), b"preserve").unwrap();
        assert_eq!(
            with_staged_output(&existing, |_| Ok::<(), String>(())).unwrap_err(),
            "MODEL_IMAGE_OUTPUT_NOT_EMPTY"
        );
        assert_eq!(fs::read(existing.join("sentinel")).unwrap(), b"preserve");
        let published = root.join("published");
        fs::create_dir(&published).unwrap();
        with_staged_output(&published, |staging| {
            fs::write(staging.join("complete"), b"complete").unwrap();
            Ok::<(), String>(())
        })
        .unwrap();
        assert_eq!(fs::read(published.join("complete")).unwrap(), b"complete");
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .filter(|entry| entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with('.'))
                .count(),
            0
        );
        fs::remove_dir_all(root).unwrap();
    }
}
