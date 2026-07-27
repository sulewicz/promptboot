use flate2::read::GzDecoder;
use promptboot_core::sha256::Sha256;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const LOCK_NAME: &str = "assets.lock.json";
const DEFAULT_CACHE: &str = ".cache/model-assets";
static TEMPORARY_ORDINAL: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum AssetKind {
    QwenGguf,
    QwenLicense,
    LlamaArchive,
}

impl AssetKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "qwen_gguf" => Some(Self::QwenGguf),
            "qwen_license" => Some(Self::QwenLicense),
            "llama_archive" => Some(Self::LlamaArchive),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::QwenGguf => "qwen_gguf",
            Self::QwenLicense => "qwen_license",
            Self::LlamaArchive => "llama_archive",
        }
    }

    const fn license(self) -> &'static str {
        match self {
            Self::QwenGguf | Self::QwenLicense => "Apache-2.0",
            Self::LlamaArchive => "MIT",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetCategory {
    Missing,
    Download,
    Identity,
    License,
    Schema,
}

impl AssetCategory {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Download => "download",
            Self::Identity => "identity",
            Self::License => "license",
            Self::Schema => "schema",
        }
    }

    pub const fn exit(self) -> i32 {
        match self {
            Self::Missing => 30,
            Self::Download => 31,
            Self::Identity => 32,
            Self::License => 33,
            Self::Schema => 34,
        }
    }
}

#[derive(Debug)]
pub struct AssetError {
    pub category: AssetCategory,
    pub message: String,
}

impl AssetError {
    fn new(category: AssetCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArchiveLicense {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct Asset {
    pub kind: AssetKind,
    pub license: String,
    pub name: String,
    pub revision: String,
    pub sha256: String,
    pub size: u64,
    pub url: String,
    pub archive_license: Option<ArchiveLicense>,
}

#[derive(Clone, Debug)]
pub struct AssetLock {
    assets: BTreeMap<AssetKind, Asset>,
}

impl AssetLock {
    pub fn get(&self, kind: AssetKind) -> &Asset {
        &self.assets[&kind]
    }

    pub fn ordered(&self) -> impl Iterator<Item = &Asset> {
        [
            AssetKind::QwenGguf,
            AssetKind::QwenLicense,
            AssetKind::LlamaArchive,
        ]
        .into_iter()
        .map(|kind| self.get(kind))
    }
}

fn schema(message: impl Into<String>) -> AssetError {
    AssetError::new(AssetCategory::Schema, message)
}

fn exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), AssetError> {
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    if actual != expected {
        return Err(schema(format!("{label} keys mismatch")));
    }
    Ok(())
}

fn safe_name(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains('/')
        && !value.contains('\\')
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_https(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("https://") else {
        return false;
    };
    if value.len() > 2048
        || !value.is_ascii()
        || value.bytes().any(|byte| !byte.is_ascii_graphic())
        || value.contains(['#', '@', '\\', '[', ']'])
    {
        return false;
    }
    let authority = rest.split(['/', '?']).next().unwrap_or("");
    let (host, port) = match authority.split_once(':') {
        Some((host, port)) if !port.contains(':') => (host, Some(port)),
        Some(_) => return false,
        None => (authority, None),
    };
    if host.is_empty()
        || host.len() > 253
        || !host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && label
                    .as_bytes()
                    .last()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
    {
        return false;
    }
    match port {
        None => true,
        Some(port) => {
            !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|value| value != 0)
        }
    }
}

fn string<'a>(
    object: &'a Map<String, Value>,
    name: &str,
    label: &str,
) -> Result<&'a str, AssetError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| schema(format!("{label}.{name} must be a string")))
}

fn positive_u64(object: &Map<String, Value>, name: &str, label: &str) -> Result<u64, AssetError> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .filter(|value| *value != 0)
        .ok_or_else(|| schema(format!("{label}.{name} must be a positive integer")))
}

pub fn load(repo: &Path) -> Result<AssetLock, AssetError> {
    load_path(&repo.join(LOCK_NAME))
}

pub fn load_path(path: &Path) -> Result<AssetLock, AssetError> {
    let bytes = fs::read(path)
        .map_err(|error| schema(format!("cannot read {}: {error}", path.display())))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| schema(format!("invalid JSON: {error}")))?;
    let top = value
        .as_object()
        .ok_or_else(|| schema("asset lock must be an object"))?;
    exact_keys(top, &["assets"], "asset lock")?;
    let rows = top
        .get("assets")
        .and_then(Value::as_array)
        .ok_or_else(|| schema("asset lock assets must be an array"))?;
    if rows.len() != 3 {
        return Err(schema("asset lock must contain exactly three assets"));
    }
    let mut assets = BTreeMap::new();
    for row in rows {
        let object = row
            .as_object()
            .ok_or_else(|| schema("asset row must be an object"))?;
        let kind_text = string(object, "kind", "asset")?;
        let kind = AssetKind::parse(kind_text)
            .ok_or_else(|| schema(format!("unknown asset kind {kind_text}")))?;
        let keys = if kind == AssetKind::LlamaArchive {
            [
                "archive_license",
                "kind",
                "license",
                "name",
                "revision",
                "sha256",
                "size",
                "url",
            ]
            .as_slice()
        } else {
            [
                "kind", "license", "name", "revision", "sha256", "size", "url",
            ]
            .as_slice()
        };
        exact_keys(object, keys, kind.name())?;
        let name = string(object, "name", kind.name())?.to_owned();
        let revision = string(object, "revision", kind.name())?.to_owned();
        let sha256 = string(object, "sha256", kind.name())?.to_owned();
        let license = string(object, "license", kind.name())?.to_owned();
        let url = string(object, "url", kind.name())?.to_owned();
        let size = positive_u64(object, "size", kind.name())?;
        if !safe_name(&name)
            || !lower_hex(&revision, 40)
            || !lower_hex(&sha256, 64)
            || license != kind.license()
            || !safe_https(&url)
        {
            return Err(schema(format!("invalid {} identity", kind.name())));
        }
        let archive_license = if kind == AssetKind::LlamaArchive {
            let license_object = object
                .get("archive_license")
                .and_then(Value::as_object)
                .ok_or_else(|| schema("llama archive license must be an object"))?;
            exact_keys(
                license_object,
                &["path", "sha256", "size"],
                "llama archive license",
            )?;
            let path = string(license_object, "path", "archive_license")?.to_owned();
            let sha256 = string(license_object, "sha256", "archive_license")?.to_owned();
            let size = positive_u64(license_object, "size", "archive_license")?;
            if path != "LICENSE" || !safe_name(&path) || !lower_hex(&sha256, 64) {
                return Err(schema("invalid llama archive license identity"));
            }
            Some(ArchiveLicense { path, sha256, size })
        } else {
            None
        };
        let asset = Asset {
            kind,
            license,
            name,
            revision,
            sha256,
            size,
            url,
            archive_license,
        };
        if assets.insert(kind, asset).is_some() {
            return Err(schema(format!("duplicate asset kind {}", kind.name())));
        }
    }
    if assets.len() != 3 {
        return Err(schema("asset lock is missing a required kind"));
    }
    Ok(AssetLock { assets })
}

pub fn cache_root(repo: &Path, requested: Option<&Path>) -> PathBuf {
    if let Some(path) = requested {
        return if path.is_absolute() {
            path.to_owned()
        } else {
            repo.join(path)
        };
    }
    if let Some(configured) = env::var_os("PROMPTBOOT_ASSET_DIR") {
        let path = PathBuf::from(configured);
        return if path.is_absolute() {
            path
        } else {
            repo.join(path)
        };
    }
    repo.join(DEFAULT_CACHE)
}

pub fn path(cache: &Path, asset: &Asset) -> PathBuf {
    cache.join(&asset.name)
}

pub fn hash_file(path: &Path) -> Result<String, AssetError> {
    let mut input = File::open(path).map_err(|error| {
        AssetError::new(
            AssetCategory::Missing,
            format!("cannot open {}: {error}", path.display()),
        )
    })?;
    let mut state = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|error| {
            AssetError::new(
                AssetCategory::Identity,
                format!("cannot read {}: {error}", path.display()),
            )
        })?;
        if count == 0 {
            break;
        }
        state.update(&buffer[..count]);
    }
    Ok(hex(&state.finish()))
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

pub fn verify_one(path: &Path, asset: &Asset) -> Result<(), AssetError> {
    let metadata = path.symlink_metadata().map_err(|error| {
        AssetError::new(
            AssetCategory::Missing,
            format!(
                "missing {} at {}: {error}",
                asset.kind.name(),
                path.display()
            ),
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(AssetError::new(
            AssetCategory::Identity,
            format!(
                "{} is not a regular file at {}",
                asset.kind.name(),
                path.display()
            ),
        ));
    }
    if metadata.len() != asset.size {
        return Err(AssetError::new(
            AssetCategory::Identity,
            format!(
                "{} length mismatch expected={} actual={}",
                asset.kind.name(),
                asset.size,
                metadata.len()
            ),
        ));
    }
    let actual = hash_file(path)?;
    if actual != asset.sha256 {
        return Err(AssetError::new(
            AssetCategory::Identity,
            format!(
                "{} SHA-256 mismatch expected={} actual={actual}",
                asset.kind.name(),
                asset.sha256
            ),
        ));
    }
    match asset.kind {
        AssetKind::QwenLicense => {
            let bytes = fs::read(path).map_err(|error| {
                AssetError::new(
                    AssetCategory::License,
                    format!("license read failed: {error}"),
                )
            })?;
            if !bytes
                .windows(b"Apache License".len())
                .any(|window| window == b"Apache License")
                || !bytes
                    .windows(b"Version 2.0, January 2004".len())
                    .any(|window| window == b"Version 2.0, January 2004")
            {
                return Err(AssetError::new(
                    AssetCategory::License,
                    "Qwen license markers mismatch",
                ));
            }
        }
        AssetKind::LlamaArchive => verify_llama_license(path, asset)?,
        AssetKind::QwenGguf => {}
    }
    Ok(())
}

fn verify_llama_license(path: &Path, asset: &Asset) -> Result<(), AssetError> {
    let record = asset
        .archive_license
        .as_ref()
        .ok_or_else(|| schema("llama archive license record missing"))?;
    let root = format!("llama.cpp-{}", asset.revision);
    let expected = format!("{root}/{}", record.path);
    let file = File::open(path).map_err(|error| {
        AssetError::new(
            AssetCategory::Identity,
            format!("cannot open llama archive: {error}"),
        )
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive.entries().map_err(|error| {
        AssetError::new(
            AssetCategory::Identity,
            format!("invalid llama archive: {error}"),
        )
    })?;
    let mut license = None;
    let mut global_header_seen = false;
    for item in entries {
        let mut entry = item.map_err(|error| {
            AssetError::new(
                AssetCategory::Identity,
                format!("invalid llama archive entry: {error}"),
            )
        })?;
        let name = entry
            .path()
            .map_err(|error| {
                AssetError::new(
                    AssetCategory::Identity,
                    format!("invalid archive path: {error}"),
                )
            })?
            .to_string_lossy()
            .into_owned();
        if entry.header().entry_type().is_pax_global_extensions() {
            if name != "pax_global_header" || global_header_seen {
                return Err(AssetError::new(
                    AssetCategory::Identity,
                    "llama archive has invalid global metadata",
                ));
            }
            global_header_seen = true;
            continue;
        }
        if name != root && !name.starts_with(&(root.clone() + "/")) {
            return Err(AssetError::new(
                AssetCategory::Identity,
                "llama archive path escapes pinned revision root",
            ));
        }
        if name == expected {
            if !entry.header().entry_type().is_file() || license.is_some() {
                return Err(AssetError::new(
                    AssetCategory::License,
                    "llama LICENSE is not one regular file",
                ));
            }
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).map_err(|error| {
                AssetError::new(
                    AssetCategory::License,
                    format!("cannot read llama LICENSE: {error}"),
                )
            })?;
            license = Some(bytes);
        }
    }
    let license = license
        .ok_or_else(|| AssetError::new(AssetCategory::License, "llama LICENSE is missing"))?;
    if license.len() as u64 != record.size
        || hex(&promptboot_core::sha256::digest(&license)) != record.sha256
        || !license
            .windows(b"MIT License".len())
            .any(|window| window == b"MIT License")
    {
        return Err(AssetError::new(
            AssetCategory::License,
            "llama LICENSE identity mismatch",
        ));
    }
    Ok(())
}

pub fn verify_all(
    repo: &Path,
    requested_cache: Option<&Path>,
) -> Result<Vec<(AssetKind, PathBuf)>, AssetError> {
    let lock = load(repo)?;
    let cache = cache_root(repo, requested_cache);
    let mut result = Vec::new();
    for asset in lock.ordered() {
        let location = path(&cache, asset);
        verify_one(&location, asset)?;
        result.push((asset.kind, location));
    }
    Ok(result)
}

fn sync_parent(path: &Path) -> Result<(), AssetError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| {
            AssetError::new(
                AssetCategory::Download,
                format!("cannot sync {}: {error}", path.display()),
            )
        })
}

fn unique_siblings(final_path: &Path) -> Result<(PathBuf, PathBuf, PathBuf), AssetError> {
    let parent = final_path.parent().ok_or_else(|| {
        AssetError::new(AssetCategory::Download, "asset cache path has no parent")
    })?;
    for _ in 0..100 {
        let ordinal = TEMPORARY_ORDINAL.fetch_add(1, Ordering::Relaxed);
        let stem = format!(
            ".{}.{}.{}",
            final_path.file_name().unwrap().to_string_lossy(),
            std::process::id(),
            ordinal
        );
        let guard = parent.join(format!("{stem}.guard"));
        match OpenOptions::new().write(true).create_new(true).open(&guard) {
            Ok(_) => {
                return Ok((
                    parent.join(format!("{stem}.download")),
                    parent.join(format!("{stem}.headers")),
                    guard,
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(AssetError::new(
                    AssetCategory::Download,
                    format!("cannot reserve download staging: {error}"),
                ));
            }
        }
    }
    Err(AssetError::new(
        AssetCategory::Download,
        "cannot allocate download staging",
    ))
}

fn qwen_header_matches(path: &Path, revision: &str) -> Result<bool, AssetError> {
    let bytes = fs::read(path).map_err(|error| {
        AssetError::new(
            AssetCategory::Download,
            format!("cannot read curl headers: {error}"),
        )
    })?;
    let text = String::from_utf8_lossy(&bytes);
    let mut found = false;
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("x-repo-commit") {
            found = true;
            if value.trim() != revision {
                return Ok(false);
            }
        }
    }
    Ok(found)
}

pub fn fetch_all(
    repo: &Path,
    requested_cache: Option<&Path>,
) -> Result<Vec<(bool, AssetKind, PathBuf)>, AssetError> {
    let lock = load(repo)?;
    let cache = cache_root(repo, requested_cache);
    fs::create_dir_all(&cache).map_err(|error| {
        AssetError::new(
            AssetCategory::Download,
            format!("cannot create cache {}: {error}", cache.display()),
        )
    })?;
    let mut curl = None;
    let mut result = Vec::new();
    for asset in lock.ordered() {
        let final_path = path(&cache, asset);
        match final_path.symlink_metadata() {
            Ok(_) => {
                verify_one(&final_path, asset)?;
                result.push((false, asset.kind, final_path));
                continue;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(AssetError::new(
                    AssetCategory::Download,
                    format!("cannot inspect {}: {error}", final_path.display()),
                ));
            }
        }
        let curl = if let Some(path) = &curl {
            path
        } else {
            let name = find_executable("curl").ok_or_else(|| {
                AssetError::new(AssetCategory::Download, "curl executable is missing")
            })?;
            let resolved = fs::canonicalize(&name).map_err(|error| {
                AssetError::new(
                    AssetCategory::Download,
                    format!("cannot resolve curl {}: {error}", name.display()),
                )
            })?;
            curl.insert(resolved)
        };
        let (download, headers, guard) = unique_siblings(&final_path)?;
        let outcome = (|| {
            let command = [
                "--location",
                "--fail",
                "--show-error",
                "--silent",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--max-redirs",
                "8",
                "--connect-timeout",
                "30",
                "--max-time",
                "1800",
                "--user-agent",
                "promptboot-asset-fetch/1",
                "--dump-header",
            ];
            let output = Command::new(&curl)
                .args(command)
                .arg(&headers)
                .arg("--output")
                .arg(&download)
                .arg("--url")
                .arg(&asset.url)
                .output()
                .map_err(|error| {
                    AssetError::new(
                        AssetCategory::Download,
                        format!("cannot execute curl: {error}"),
                    )
                })?;
            if !output.status.success() {
                return Err(AssetError::new(
                    AssetCategory::Download,
                    format!(
                        "curl failed for {} exit={}",
                        asset.kind.name(),
                        output.status.code().unwrap_or(-1)
                    ),
                ));
            }
            if matches!(asset.kind, AssetKind::QwenGguf | AssetKind::QwenLicense)
                && !qwen_header_matches(&headers, &asset.revision)?
            {
                return Err(AssetError::new(
                    AssetCategory::Identity,
                    format!("{} x-repo-commit mismatch", asset.kind.name()),
                ));
            }
            verify_one(&download, asset)?;
            File::open(&download)
                .and_then(|file| file.sync_all())
                .map_err(|error| {
                    AssetError::new(
                        AssetCategory::Download,
                        format!("cannot sync download: {error}"),
                    )
                })?;
            fs::rename(&download, &final_path).map_err(|error| {
                AssetError::new(
                    AssetCategory::Download,
                    format!("cannot publish {}: {error}", asset.kind.name()),
                )
            })?;
            sync_parent(&cache)?;
            Ok(())
        })();
        let _ = fs::remove_file(&download);
        let _ = fs::remove_file(&headers);
        let _ = fs::remove_file(&guard);
        outcome?;
        result.push((true, asset.kind, final_path));
    }
    Ok(result)
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

pub fn value(
    repo: &Path,
    requested_cache: Option<&Path>,
    kind: AssetKind,
    field: &str,
) -> Result<String, AssetError> {
    let lock = load(repo)?;
    let cache = cache_root(repo, requested_cache);
    let asset = lock.get(kind);
    let location = path(&cache, asset);
    match field {
        "path" => {
            verify_one(&location, asset)?;
            Ok(location.to_string_lossy().into_owned())
        }
        "name" => Ok(asset.name.clone()),
        "revision" => Ok(asset.revision.clone()),
        "sha256" => Ok(asset.sha256.clone()),
        "size" => Ok(asset.size.to_string()),
        _ => Err(schema(format!("unknown asset field {field}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lock(value: &Value) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "promptboot-assets-test-{}-{}.json",
            std::process::id(),
            TEMPORARY_ORDINAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }

    #[test]
    fn exact_lock_accepts_reordering_and_rejects_unknown_or_unsafe_rows() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let lock = load(&repo).unwrap();
        assert_eq!(
            lock.ordered()
                .map(|asset| asset.kind.name())
                .collect::<Vec<_>>(),
            ["qwen_gguf", "qwen_license", "llama_archive"]
        );
        assert!(safe_name("QWEN-LICENSE"));
        assert!(!safe_name("../LICENSE"));
        assert!(safe_https("https://example.invalid/a"));
        assert!(!safe_https("http://example.invalid/a"));
        assert!(!safe_https("https://user@example.invalid/a"));

        let original: Value =
            serde_json::from_slice(&fs::read(repo.join(LOCK_NAME)).unwrap()).unwrap();
        let mut reordered = original.clone();
        reordered["assets"].as_array_mut().unwrap().reverse();
        let path = write_lock(&reordered);
        assert!(load_path(&path).is_ok());
        fs::remove_file(path).unwrap();

        let mut malformed = Vec::new();
        let mut missing = original.clone();
        missing["assets"].as_array_mut().unwrap().pop();
        malformed.push(missing);
        let mut duplicate = original.clone();
        duplicate["assets"][1] = duplicate["assets"][0].clone();
        malformed.push(duplicate);
        let mut unknown = original.clone();
        unknown["assets"][0]["kind"] = Value::String("other".to_owned());
        malformed.push(unknown);
        let mut unsafe_name = original.clone();
        unsafe_name["assets"][0]["name"] = Value::String("../model.gguf".to_owned());
        malformed.push(unsafe_name);
        let mut insecure_url = original.clone();
        insecure_url["assets"][0]["url"] = Value::String("http://example.invalid/model".to_owned());
        malformed.push(insecure_url);
        let mut extra_key = original;
        extra_key["assets"][0]["unexpected"] = Value::Bool(true);
        malformed.push(extra_key);

        for value in malformed {
            let path = write_lock(&value);
            assert!(load_path(&path).is_err());
            fs::remove_file(path).unwrap();
        }
    }
}
