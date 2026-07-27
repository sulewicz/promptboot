use crate::{assets, image};
use flate2::bufread::GzDecoder;
use promptboot_core::sha256::Sha256;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

const IMAGE_MEMBERS: [&str; 5] = [
    "BOOTX64.EFI",
    "BUILD.JSN",
    "MODEL.PBT",
    "promptboot.img",
    "promptboot-media-inspection.json",
];
const TRACKED_DISTRIBUTION: [(&str, &str); 7] = [
    ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    (
        "LICENSES/QWEN-APACHE-2.0.txt",
        "LICENSES/QWEN-APACHE-2.0.txt",
    ),
    ("LICENSES/LLAMA-MIT.txt", "LICENSES/LLAMA-MIT.txt"),
    ("LICENSES/libm-0.2.11.txt", "LICENSES/libm-0.2.11.txt"),
    (
        "LICENSES/RUST-1.97.1-COPYRIGHT-library.html",
        "LICENSES/RUST-1.97.1-COPYRIGHT-library.html",
    ),
    (
        "LICENSES/compiler-builtins-0.1.160.txt",
        "LICENSES/compiler-builtins-0.1.160.txt",
    ),
];
const FAT_DISTRIBUTION: [(&str, &str); 8] = [
    ("LICENSE", "LICENSE"),
    ("SOURCE.TGZ", "SOURCE.TGZ"),
    ("LICENSES/3RDPARTY.TXT", "THIRD_PARTY_NOTICES.md"),
    ("LICENSES/QWEN.TXT", "LICENSES/QWEN-APACHE-2.0.txt"),
    ("LICENSES/LLAMA.TXT", "LICENSES/LLAMA-MIT.txt"),
    ("LICENSES/LIBM.TXT", "LICENSES/libm-0.2.11.txt"),
    (
        "LICENSES/RUSTCORE.HTM",
        "LICENSES/RUST-1.97.1-COPYRIGHT-library.html",
    ),
    (
        "LICENSES/RUSTCB.TXT",
        "LICENSES/compiler-builtins-0.1.160.txt",
    ),
];
static TEMPORARY_ORDINAL: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseCategory {
    Source,
    Build,
    Output,
    Validation,
}

impl ReleaseCategory {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Build => "build",
            Self::Output => "output",
            Self::Validation => "validation",
        }
    }

    pub const fn exit(self) -> i32 {
        match self {
            Self::Source => 40,
            Self::Build => 41,
            Self::Output => 42,
            Self::Validation => 43,
        }
    }
}

#[derive(Debug)]
pub struct ReleaseError {
    pub category: ReleaseCategory,
    pub message: String,
}

impl ReleaseError {
    fn new(category: ReleaseCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct Source {
    commit: String,
    tree: String,
}

#[derive(Clone, Debug)]
struct ArchiveIdentity {
    bytes: u64,
    prefix: String,
    sha256: String,
}

#[derive(Clone, Debug)]
pub struct ReleaseOutcome {
    pub output: PathBuf,
    pub commit: String,
    pub builds: u8,
    pub reused: bool,
}

fn run_git(repo: &Path, args: &[&str]) -> Result<Output, ReleaseError> {
    Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Source,
                format!("cannot execute git: {error}"),
            )
        })
}

fn git_stdout(repo: &Path, args: &[&str]) -> Result<Vec<u8>, ReleaseError> {
    let output = run_git(repo, args)?;
    if !output.status.success() {
        return Err(ReleaseError::new(
            ReleaseCategory::Source,
            format!("git {} failed", args.join(" ")),
        ));
    }
    Ok(output.stdout)
}

fn source_identity(repo: &Path, require_clean: bool) -> Result<Source, ReleaseError> {
    if require_clean {
        let status = git_stdout(repo, &["status", "--porcelain"])?;
        if !status.is_empty() {
            return Err(ReleaseError::new(
                ReleaseCategory::Source,
                "worktree changes are not releasable",
            ));
        }
    }
    let commit = String::from_utf8(git_stdout(repo, &["rev-parse", "HEAD"])?)
        .map_err(|_| ReleaseError::new(ReleaseCategory::Source, "commit is not ASCII"))?
        .trim()
        .to_owned();
    let tree = String::from_utf8(git_stdout(repo, &["rev-parse", "HEAD^{tree}"])?)
        .map_err(|_| ReleaseError::new(ReleaseCategory::Source, "tree is not ASCII"))?
        .trim()
        .to_owned();
    if !lower_hex(&commit, 40) || !lower_hex(&tree, 40) {
        return Err(ReleaseError::new(
            ReleaseCategory::Source,
            "git source identity is malformed",
        ));
    }
    Ok(Source { commit, tree })
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalize_absolute(path: &Path) -> Result<PathBuf, ReleaseError> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                if !result.pop() {
                    return Err(ReleaseError::new(
                        ReleaseCategory::Output,
                        "output path escapes its root",
                    ));
                }
            }
            Component::Normal(name) => result.push(name),
        }
    }
    Ok(result)
}

fn release_location(repo: &Path, requested: &Path) -> Result<PathBuf, ReleaseError> {
    let joined = if requested.is_absolute() {
        requested.to_owned()
    } else {
        repo.join(requested)
    };
    let output = normalize_absolute(&joined)?;
    let build = normalize_absolute(&repo.join("build"))?;
    let relative = output.strip_prefix(&build).map_err(|_| {
        ReleaseError::new(
            ReleaseCategory::Output,
            "output must be below repository build/",
        )
    })?;
    if relative.as_os_str().is_empty() {
        return Err(ReleaseError::new(
            ReleaseCategory::Output,
            "output must be a non-root descendant of build/",
        ));
    }
    if build
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(ReleaseError::new(
            ReleaseCategory::Output,
            "repository build/ must not be a symbolic link",
        ));
    }
    let mut cursor = build;
    for component in relative.components() {
        cursor.push(component.as_os_str());
        if cursor
            .symlink_metadata()
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(ReleaseError::new(
                ReleaseCategory::Output,
                "output path contains a symbolic link",
            ));
        }
    }
    Ok(output)
}

fn digest(path: &Path) -> Result<String, ReleaseError> {
    let mut input = File::open(path).map_err(|error| {
        ReleaseError::new(
            ReleaseCategory::Validation,
            format!("cannot open {}: {error}", path.display()),
        )
    })?;
    let mut state = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Validation,
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

fn copy_file(source: &Path, destination: &Path) -> Result<(), ReleaseError> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Build,
                format!("cannot create {}: {error}", parent.display()),
            )
        })?;
    }
    fs::copy(source, destination).map_err(|error| {
        ReleaseError::new(
            ReleaseCategory::Build,
            format!(
                "cannot copy {} to {}: {error}",
                source.display(),
                destination.display()
            ),
        )
    })?;
    Ok(())
}

fn write_distribution(repo: &Path, destination: &Path) -> Result<(), ReleaseError> {
    for (source, target) in TRACKED_DISTRIBUTION {
        copy_file(&repo.join(source), &destination.join(target))?;
    }
    Ok(())
}

fn source_archive(
    repo: &Path,
    destination: &Path,
    source: &Source,
) -> Result<ArchiveIdentity, ReleaseError> {
    let prefix = format!("promptboot-{}/", &source.commit[..12]);
    let output = Command::new("git")
        .args([
            OsStr::new("archive"),
            OsStr::new("--format=tar.gz"),
            OsStr::new(&format!("--prefix={prefix}")),
            OsStr::new("-o"),
            destination.as_os_str(),
            OsStr::new(&source.commit),
        ])
        .current_dir(repo)
        .output()
        .map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Build,
                format!("cannot execute git archive: {error}"),
            )
        })?;
    if !output.status.success() {
        return Err(ReleaseError::new(
            ReleaseCategory::Build,
            "git archive failed",
        ));
    }
    Ok(ArchiveIdentity {
        bytes: destination.metadata().map_err(build_io)?.len(),
        prefix,
        sha256: digest(destination)?,
    })
}

fn build_io(error: io::Error) -> ReleaseError {
    ReleaseError::new(ReleaseCategory::Build, error.to_string())
}

fn validation(message: impl Into<String>) -> ReleaseError {
    ReleaseError::new(ReleaseCategory::Validation, message)
}

fn git_tree(
    repo: &Path,
    source: &Source,
) -> Result<BTreeMap<String, (String, String)>, ReleaseError> {
    let output = git_stdout(
        repo,
        &["ls-tree", "-r", "-z", "--full-tree", &source.commit],
    )?;
    let mut result = BTreeMap::new();
    for raw in output
        .split(|byte| *byte == 0)
        .filter(|row| !row.is_empty())
    {
        let tab = raw
            .iter()
            .position(|byte| *byte == b'\t')
            .ok_or_else(|| validation("git tree record is malformed"))?;
        let metadata = std::str::from_utf8(&raw[..tab])
            .map_err(|_| validation("git tree metadata is not ASCII"))?;
        let name = std::str::from_utf8(&raw[tab + 1..])
            .map_err(|_| validation("git path is not UTF-8"))?
            .to_owned();
        let mut fields = metadata.split(' ');
        let mode = fields.next().unwrap_or("").to_owned();
        let kind = fields.next().unwrap_or("");
        let object = fields.next().unwrap_or("").to_owned();
        if fields.next().is_some()
            || kind != "blob"
            || !matches!(mode.as_str(), "100644" | "100755" | "120000")
            || !lower_hex(&object, 40)
            || result.insert(name, (mode, object)).is_some()
        {
            return Err(validation("git source tree contains an unsupported entry"));
        }
    }
    Ok(result)
}

fn verify_source_archive(
    repo: &Path,
    archive_path: &Path,
    expected: &ArchiveIdentity,
    source: &Source,
) -> Result<(), ReleaseError> {
    let metadata = archive_path
        .metadata()
        .map_err(|error| validation(format!("source archive is missing: {error}")))?;
    if metadata.len() != expected.bytes
        || expected.prefix != format!("promptboot-{}/", &source.commit[..12])
        || digest(archive_path)? != expected.sha256
    {
        return Err(validation("release source archive identity mismatch"));
    }
    let tracked = git_tree(repo, source)?;
    let mut valid_directories = BTreeSet::from([expected.prefix.trim_end_matches('/').to_owned()]);
    for name in tracked.keys() {
        let mut parent = Path::new(name).parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            valid_directories.insert(format!("{}{}", expected.prefix, path.to_string_lossy()));
            parent = path.parent();
        }
    }
    let file = File::open(archive_path)
        .map_err(|error| validation(format!("cannot open source archive: {error}")))?;
    let decoder = GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);
    let mut seen = BTreeSet::new();
    let mut archived_directories = BTreeSet::new();
    let mut archived_files = BTreeSet::new();
    let mut global_header_seen = false;
    let entries = archive
        .entries()
        .map_err(|error| validation(format!("invalid source archive: {error}")))?;
    for item in entries {
        let mut entry =
            item.map_err(|error| validation(format!("invalid archive entry: {error}")))?;
        let path = entry
            .path()
            .map_err(|error| validation(format!("invalid archive path: {error}")))?
            .to_string_lossy()
            .into_owned();
        let entry_type = entry.header().entry_type();
        let normalized = if entry_type.is_dir() {
            path.trim_end_matches('/')
        } else {
            &path
        };
        if normalized.is_empty()
            || path.starts_with('/')
            || Path::new(&path)
                .components()
                .any(|component| component == Component::ParentDir)
            || !seen.insert(normalized.to_owned())
        {
            return Err(validation("source archive has an unsafe or duplicate path"));
        }
        if entry_type.is_pax_global_extensions() {
            if path != "pax_global_header" || global_header_seen {
                return Err(validation("source archive has invalid global metadata"));
            }
            global_header_seen = true;
            continue;
        }
        if entry_type.is_dir() {
            if !valid_directories.contains(normalized) {
                return Err(validation("source archive contains an invented directory"));
            }
            archived_directories.insert(normalized.to_owned());
            continue;
        }
        let name = path
            .strip_prefix(&expected.prefix)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| validation(format!("source archive prefix mismatch: {path}")))?;
        let (mode, object) = tracked
            .get(name)
            .ok_or_else(|| validation("source archive contains an extra member"))?;
        archived_files.insert(name.to_owned());
        let blob = git_stdout(repo, &["cat-file", "blob", object])?;
        match mode.as_str() {
            "120000" => {
                if !entry_type.is_symlink()
                    || entry
                        .link_name()
                        .map_err(|error| validation(format!("invalid symlink: {error}")))?
                        .as_deref()
                        .map(|target| target.as_os_str().as_encoded_bytes())
                        != Some(blob.as_slice())
                {
                    return Err(validation("source archive symlink mismatch"));
                }
            }
            "100644" | "100755" => {
                if !entry_type.is_file()
                    || (entry
                        .header()
                        .mode()
                        .map_err(|error| validation(format!("invalid archive mode: {error}")))?
                        & 0o111
                        != 0)
                        != (mode == "100755")
                {
                    return Err(validation("source archive member type/mode mismatch"));
                }
                let mut bytes = Vec::new();
                entry
                    .read_to_end(&mut bytes)
                    .map_err(|error| validation(format!("cannot read archive member: {error}")))?;
                if bytes != blob {
                    return Err(validation("source archive member content mismatch"));
                }
            }
            _ => return Err(validation("unsupported git source mode")),
        }
    }
    let mut decoder = archive.into_inner();
    io::copy(&mut decoder, &mut io::sink())
        .map_err(|error| validation(format!("cannot finish gzip stream: {error}")))?;
    let mut underlying = decoder.into_inner();
    if !underlying
        .fill_buf()
        .map_err(|error| validation(format!("cannot inspect gzip tail: {error}")))?
        .is_empty()
    {
        return Err(validation("source archive has trailing unparsed data"));
    }
    if !global_header_seen
        || archived_directories != valid_directories
        || archived_files != tracked.keys().cloned().collect()
    {
        return Err(validation("source archive is missing tracked members"));
    }
    Ok(())
}

fn expected_release_files() -> BTreeSet<String> {
    IMAGE_MEMBERS
        .into_iter()
        .chain(TRACKED_DISTRIBUTION.into_iter().map(|(_, target)| target))
        .chain(["SOURCE.TGZ", "RUN.md", "release.json", "SHA256SUMS"])
        .map(str::to_owned)
        .collect()
}

fn visit_layout(
    root: &Path,
    directory: &Path,
    files: &mut BTreeSet<String>,
    directories: &mut BTreeSet<String>,
) -> Result<(), ReleaseError> {
    for item in fs::read_dir(directory)
        .map_err(|error| validation(format!("cannot read release directory: {error}")))?
    {
        let entry = item.map_err(|error| validation(format!("invalid release entry: {error}")))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| validation(format!("cannot stat release entry: {error}")))?;
        let name = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        if metadata.file_type().is_symlink() {
            return Err(validation(format!(
                "release must not contain symbolic links: {name}"
            )));
        }
        if metadata.is_dir() {
            directories.insert(name);
            visit_layout(root, &path, files, directories)?;
        } else if metadata.is_file() {
            files.insert(name);
        } else {
            return Err(validation("release contains an unsupported entry"));
        }
    }
    Ok(())
}

fn verify_layout(destination: &Path) -> Result<(), ReleaseError> {
    if !destination.is_dir() {
        return Err(ReleaseError::new(
            ReleaseCategory::Output,
            format!("release directory is missing: {}", destination.display()),
        ));
    }
    let mut files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    visit_layout(destination, destination, &mut files, &mut directories)?;
    if files != expected_release_files() || directories != BTreeSet::from(["LICENSES".to_owned()]) {
        return Err(validation(format!(
            "release membership mismatch files={files:?} directories={directories:?}"
        )));
    }
    Ok(())
}

fn write_checksums(destination: &Path) -> Result<(), ReleaseError> {
    let mut names = expected_release_files();
    names.remove("SHA256SUMS");
    let mut output = String::new();
    for name in names {
        output.push_str(&format!("{}  {name}\n", digest(&destination.join(&name))?));
    }
    fs::write(destination.join("SHA256SUMS"), output).map_err(build_io)
}

fn verify_checksums(destination: &Path) -> Result<(), ReleaseError> {
    let bytes = fs::read(destination.join("SHA256SUMS"))
        .map_err(|error| validation(format!("cannot read SHA256SUMS: {error}")))?;
    if bytes.is_empty() || bytes.last() != Some(&b'\n') {
        return Err(validation("malformed SHA256SUMS"));
    }
    let mut identities = BTreeMap::new();
    for line in bytes[..bytes.len() - 1].split(|byte| *byte == b'\n') {
        let (Some(wanted), Some(separator), Some(name_bytes)) =
            (line.get(..64), line.get(64..66), line.get(66..))
        else {
            return Err(validation("malformed SHA256SUMS"));
        };
        if separator != b"  " {
            return Err(validation("malformed SHA256SUMS"));
        }
        let name =
            std::str::from_utf8(name_bytes).map_err(|_| validation("malformed SHA256SUMS"))?;
        let wanted_text =
            std::str::from_utf8(wanted).map_err(|_| validation("malformed SHA256SUMS"))?;
        if !wanted
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
            || name.is_empty()
            || Path::new(name).is_absolute()
            || Path::new(name)
                .components()
                .any(|component| component == Component::ParentDir)
            || identities
                .insert(name.to_owned(), wanted_text.to_owned())
                .is_some()
        {
            return Err(validation("malformed SHA256SUMS"));
        }
    }
    let mut expected = expected_release_files();
    expected.remove("SHA256SUMS");
    if identities.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err(validation("SHA256SUMS does not cover release exactly"));
    }
    for (name, wanted) in identities {
        if digest(&destination.join(name))? != wanted {
            return Err(validation("release checksum mismatch"));
        }
    }
    Ok(())
}

fn distribution_manifest(destination: &Path) -> Result<Value, ReleaseError> {
    let mut result = Map::new();
    for name in TRACKED_DISTRIBUTION
        .into_iter()
        .map(|(_, target)| target)
        .chain(["SOURCE.TGZ"])
    {
        let path = destination.join(name);
        result.insert(
            name.to_owned(),
            json!({"bytes":path.metadata().map_err(|error| validation(error.to_string()))?.len(),"sha256":digest(&path)?}),
        );
    }
    Ok(Value::Object(result))
}

fn artifact_manifest(destination: &Path) -> Result<Value, ReleaseError> {
    let mut result = Map::new();
    for name in IMAGE_MEMBERS {
        let path = destination.join(name);
        result.insert(
            name.to_owned(),
            json!({"bytes":path.metadata().map_err(|error| validation(error.to_string()))?.len(),"sha256":digest(&path)?}),
        );
    }
    Ok(Value::Object(result))
}

fn archive_identity_value(identity: &ArchiveIdentity) -> Value {
    json!({
        "bytes":identity.bytes,
        "format":"tar.gz",
        "path":"SOURCE.TGZ",
        "prefix":identity.prefix,
        "sha256":identity.sha256
    })
}

fn parse_archive_identity(value: &Value) -> Result<ArchiveIdentity, ReleaseError> {
    let object = value
        .as_object()
        .ok_or_else(|| validation("source_archive must be an object"))?;
    let keys: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if keys != BTreeSet::from(["bytes", "format", "path", "prefix", "sha256"]) {
        return Err(validation("source_archive schema mismatch"));
    }
    if object.get("format").and_then(Value::as_str) != Some("tar.gz")
        || object.get("path").and_then(Value::as_str) != Some("SOURCE.TGZ")
    {
        return Err(validation("source_archive contract mismatch"));
    }
    Ok(ArchiveIdentity {
        bytes: object
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| validation("source_archive bytes missing"))?,
        prefix: object
            .get("prefix")
            .and_then(Value::as_str)
            .ok_or_else(|| validation("source_archive prefix missing"))?
            .to_owned(),
        sha256: object
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|value| lower_hex(value, 64))
            .ok_or_else(|| validation("source_archive sha256 invalid"))?
            .to_owned(),
    })
}

fn verify_tracked_distribution(repo: &Path, destination: &Path) -> Result<(), ReleaseError> {
    for (source, target) in TRACKED_DISTRIBUTION {
        let source = repo.join(source);
        let target = destination.join(target);
        if source
            .metadata()
            .map_err(|error| validation(error.to_string()))?
            .len()
            != target
                .metadata()
                .map_err(|error| validation(error.to_string()))?
                .len()
            || digest(&source)? != digest(&target)?
        {
            return Err(validation("release distribution file mismatch"));
        }
    }
    Ok(())
}

fn verify_release(repo: &Path, destination: &Path, source: &Source) -> Result<(), ReleaseError> {
    verify_layout(destination)?;
    verify_checksums(destination)?;
    let release: Value = serde_json::from_slice(
        &fs::read(destination.join("release.json"))
            .map_err(|error| validation(format!("cannot read release.json: {error}")))?,
    )
    .map_err(|error| validation(format!("invalid release.json: {error}")))?;
    let object = release
        .as_object()
        .ok_or_else(|| validation("release.json must be an object"))?;
    let expected_keys = BTreeSet::from([
        "artifacts",
        "build",
        "distribution",
        "schema",
        "source",
        "source_archive",
    ]);
    if object.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_keys
        || object.get("schema").and_then(Value::as_u64) != Some(1)
        || object.get("source") != Some(&json!({"commit":source.commit,"tree":source.tree}))
    {
        return Err(validation("release.json source/schema mismatch"));
    }
    let build = object
        .get("build")
        .and_then(Value::as_object)
        .ok_or_else(|| validation("release build record missing"))?;
    if build.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from(["build_id", "command", "count"])
        || build.get("count").and_then(Value::as_u64) != Some(1)
        || build.get("command").and_then(Value::as_str) != Some("promptboot-tools release")
    {
        return Err(validation("release build record mismatch"));
    }
    if object.get("artifacts") != Some(&artifact_manifest(destination)?)
        || object.get("distribution") != Some(&distribution_manifest(destination)?)
    {
        return Err(validation(
            "release artifact/distribution identity mismatch",
        ));
    }
    verify_tracked_distribution(repo, destination)?;
    let archive = parse_archive_identity(
        object
            .get("source_archive")
            .ok_or_else(|| validation("source archive identity missing"))?,
    )?;
    verify_source_archive(repo, &destination.join("SOURCE.TGZ"), &archive, source)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(destination.join("BUILD.JSN"))
            .map_err(|error| validation(format!("cannot read BUILD.JSN: {error}")))?,
    )
    .map_err(|error| validation(format!("invalid BUILD.JSN: {error}")))?;
    if build.get("build_id") != manifest.get("build_id") {
        return Err(validation("release build ID mismatch"));
    }
    let report = image::inspect_image(
        &destination.join("promptboot.img"),
        &destination.join("BUILD.JSN"),
        Some(destination),
    )
    .map_err(|error| validation(format!("release image invalid: {error}")))?;
    let mut report_bytes =
        serde_json::to_vec(&report).map_err(|error| validation(error.to_string()))?;
    report_bytes.push(b'\n');
    if fs::read(destination.join("promptboot-media-inspection.json"))
        .map_err(|error| validation(error.to_string()))?
        != report_bytes
        || report.get("build_jsn_sha256").and_then(Value::as_str)
            != Some(&digest(&destination.join("BUILD.JSN"))?)
        || report.get("image_sha256").and_then(Value::as_str)
            != Some(&digest(&destination.join("promptboot.img"))?)
    {
        return Err(validation("release image/report identity mismatch"));
    }
    Ok(())
}

fn unique_directory(
    parent: &Path,
    label: &str,
    category: ReleaseCategory,
) -> Result<PathBuf, ReleaseError> {
    fs::create_dir_all(parent).map_err(|error| {
        ReleaseError::new(
            category,
            format!("cannot create {}: {error}", parent.display()),
        )
    })?;
    for _ in 0..100 {
        let ordinal = TEMPORARY_ORDINAL.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{label}.{}.{}.tmp", std::process::id(), ordinal));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(ReleaseError::new(
                    category,
                    format!("cannot create release staging: {error}"),
                ));
            }
        }
    }
    Err(ReleaseError::new(
        category,
        "cannot allocate release staging directory",
    ))
}

#[derive(Default)]
struct TemporaryDirectories {
    paths: Vec<PathBuf>,
}

impl TemporaryDirectories {
    fn allocate(
        &mut self,
        parent: &Path,
        label: &str,
        category: ReleaseCategory,
    ) -> Result<PathBuf, ReleaseError> {
        let path = unique_directory(parent, label, category)?;
        self.paths.push(path.clone());
        Ok(path)
    }
}

impl Drop for TemporaryDirectories {
    fn drop(&mut self) {
        for path in self.paths.iter().rev() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn sync_tree(path: &Path) -> Result<(), ReleaseError> {
    for entry in fs::read_dir(path).map_err(build_io)? {
        let entry = entry.map_err(build_io)?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(build_io)?;
        if metadata.file_type().is_symlink() {
            return Err(validation("staged release contains a symbolic link"));
        }
        if metadata.is_dir() {
            sync_tree(&entry.path())?;
        } else if metadata.is_file() {
            File::open(entry.path())
                .and_then(|file| file.sync_all())
                .map_err(build_io)?;
        } else {
            return Err(validation("staged release contains unsupported entry"));
        }
    }
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(build_io)
}

fn build_release(
    repo: &Path,
    destination: &Path,
    source: &Source,
    requested_cache: Option<&Path>,
) -> Result<(), ReleaseError> {
    let verified = assets::verify_all(repo, requested_cache).map_err(|error| {
        ReleaseError::new(
            ReleaseCategory::Build,
            format!("asset {}: {}", error.category.name(), error.message),
        )
    })?;
    let qwen = verified
        .iter()
        .find(|(kind, _)| *kind == assets::AssetKind::QwenGguf)
        .map(|(_, path)| path.clone())
        .ok_or_else(|| ReleaseError::new(ReleaseCategory::Build, "Qwen model missing"))?;
    let parent = destination.parent().unwrap();
    let label = destination.file_name().unwrap().to_string_lossy();
    let mut temporary = TemporaryDirectories::default();
    let staging = temporary.allocate(parent, &label, ReleaseCategory::Output)?;
    let work = temporary.allocate(&repo.join("build"), "release-work", ReleaseCategory::Build)?;
    let target = temporary.allocate(
        &repo.join("target"),
        "release-target",
        ReleaseCategory::Build,
    )?;
    let result = (|| {
        let fat = work.join("distribution");
        fs::create_dir(&fat).map_err(build_io)?;
        let archive = source_archive(repo, &fat.join("SOURCE.TGZ"), source)?;
        for (fat_name, outer_name) in FAT_DISTRIBUTION {
            if fat_name == "SOURCE.TGZ" {
                continue;
            }
            let tracked_source = TRACKED_DISTRIBUTION
                .into_iter()
                .find(|(_, target)| *target == outer_name)
                .map(|(source, _)| source)
                .ok_or_else(|| {
                    ReleaseError::new(ReleaseCategory::Build, "distribution mapping missing")
                })?;
            copy_file(&repo.join(tracked_source), &fat.join(fat_name))?;
        }
        let built = work.join("image");
        let image_target = target.join("image");
        image::build_image(&qwen, &built, &image_target, Some(&fat)).map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Build,
                format!("image build failed: {error}"),
            )
        })?;
        for name in IMAGE_MEMBERS {
            copy_file(&built.join(name), &staging.join(name))?;
        }
        copy_file(&fat.join("SOURCE.TGZ"), &staging.join("SOURCE.TGZ"))?;
        write_distribution(repo, &staging)?;
        let manifest: Value =
            serde_json::from_slice(&fs::read(staging.join("BUILD.JSN")).map_err(build_io)?)
                .map_err(|error| {
                    ReleaseError::new(
                        ReleaseCategory::Build,
                        format!("invalid BUILD.JSN: {error}"),
                    )
                })?;
        let build_id = manifest
            .get("build_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ReleaseError::new(ReleaseCategory::Build, "build ID missing"))?;
        let release = json!({
            "artifacts":artifact_manifest(&staging)?,
            "build":{"build_id":build_id,"command":"promptboot-tools release","count":1},
            "distribution":distribution_manifest(&staging)?,
            "schema":1,
            "source":{"commit":source.commit,"tree":source.tree},
            "source_archive":archive_identity_value(&archive)
        });
        let mut bytes = serde_json::to_vec(&release)
            .map_err(|error| ReleaseError::new(ReleaseCategory::Build, error.to_string()))?;
        bytes.push(b'\n');
        fs::write(staging.join("release.json"), bytes).map_err(build_io)?;
        let relative = destination.strip_prefix(repo).map_err(|_| {
            ReleaseError::new(
                ReleaseCategory::Build,
                "release output is outside repository",
            )
        })?;
        fs::write(
            staging.join("RUN.md"),
            format!(
                "# Run\n\nFrom the matching promptboot checkout, run:\n\n```sh\nmake play RELEASE_DIR={}\n```\n\nThe project license, third-party notices, and exact source archive are included beside and inside the image. Keep them together when conveying the release.\n",
                relative.display()
            ),
        )
        .map_err(build_io)?;
        write_checksums(&staging)?;
        verify_release(repo, &staging, source)?;
        sync_tree(&staging)?;
        fs::rename(&staging, destination).map_err(|error| {
            ReleaseError::new(
                ReleaseCategory::Output,
                format!("cannot publish release: {error}"),
            )
        })?;
        File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| {
                ReleaseError::new(
                    ReleaseCategory::Output,
                    format!("cannot sync release parent: {error}"),
                )
            })?;
        Ok(())
    })();
    result
}

pub fn release(
    repo: &Path,
    requested: &Path,
    requested_cache: Option<&Path>,
) -> Result<ReleaseOutcome, ReleaseError> {
    let source = source_identity(repo, true)?;
    let output = release_location(repo, requested)?;
    if output.exists() {
        verify_release(repo, &output, &source)?;
        return Ok(ReleaseOutcome {
            output,
            commit: source.commit,
            builds: 0,
            reused: true,
        });
    }
    build_release(repo, &output, &source, requested_cache)?;
    Ok(ReleaseOutcome {
        output,
        commit: source.commit,
        builds: 1,
        reused: false,
    })
}

pub fn verify(repo: &Path, requested: &Path) -> Result<ReleaseOutcome, ReleaseError> {
    let source = source_identity(repo, true)?;
    let output = release_location(repo, requested)?;
    verify_release(repo, &output, &source)?;
    Ok(ReleaseOutcome {
        output,
        commit: source.commit,
        builds: 0,
        reused: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::read::GzDecoder as ReadGzDecoder;
    use flate2::write::GzEncoder;
    use std::env;
    use std::fs::OpenOptions;
    use std::io::{Cursor, Write};
    use std::os::unix::fs::symlink;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "promptboot-release-test-{}-{}",
                std::process::id(),
                TEMPORARY_ORDINAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn archive_identity(path: &Path, prefix: &str) -> ArchiveIdentity {
        ArchiveIdentity {
            bytes: path.metadata().unwrap().len(),
            prefix: prefix.to_owned(),
            sha256: digest(path).unwrap(),
        }
    }

    fn rewrite_archive(
        source: &Path,
        destination: &Path,
        mutator: impl FnOnce(&mut Vec<(tar::Header, Vec<u8>)>),
    ) {
        let decoder = ReadGzDecoder::new(File::open(source).unwrap());
        let mut archive = tar::Archive::new(decoder);
        let mut rows = Vec::new();
        for item in archive.entries().unwrap() {
            let mut entry = item.unwrap();
            let header = entry.header().clone();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            rows.push((header, bytes));
        }
        mutator(&mut rows);
        let encoder = GzEncoder::new(File::create(destination).unwrap(), Compression::fast());
        let mut output = tar::Builder::new(encoder);
        for (mut header, bytes) in rows {
            header.set_cksum();
            output.append(&header, Cursor::new(bytes)).unwrap();
        }
        output.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn output_is_bounded_below_build() {
        let repo = Path::new("/tmp/promptboot-release-path-test");
        assert_eq!(
            release_location(repo, Path::new("build/release")).unwrap(),
            repo.join("build/release")
        );
        assert!(release_location(repo, Path::new("release")).is_err());
        assert!(release_location(repo, Path::new("build")).is_err());
        assert!(release_location(repo, Path::new("build/../release")).is_err());

        let temporary = TestDirectory::new();
        let outside = temporary.0.join("outside");
        fs::create_dir(&outside).unwrap();
        symlink(&outside, temporary.0.join("build")).unwrap();
        assert!(release_location(&temporary.0, Path::new("build/release")).is_err());
    }

    #[test]
    fn source_archive_accepts_representation_changes_but_rejects_semantic_changes() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = source_identity(&repo, false).unwrap();
        let temporary = TestDirectory::new();
        let original = temporary.0.join("original.tgz");
        let original_identity = source_archive(&repo, &original, &source).unwrap();
        verify_source_archive(&repo, &original, &original_identity, &source).unwrap();

        let represented = temporary.0.join("represented.tgz");
        rewrite_archive(&original, &represented, |rows| {
            rows.reverse();
            for (header, _) in rows {
                let executable = header.mode().unwrap() & 0o111 != 0;
                header.set_mode(if executable { 0o755 } else { 0o600 });
                header.set_uid(7);
                header.set_gid(8);
                header.set_mtime(1);
            }
        });
        let represented_identity = archive_identity(&represented, &original_identity.prefix);
        verify_source_archive(&repo, &represented, &represented_identity, &source).unwrap();

        let changed = temporary.0.join("changed.tgz");
        rewrite_archive(&original, &changed, |rows| {
            let (_, bytes) = rows
                .iter_mut()
                .find(|(header, bytes)| header.entry_type().is_file() && !bytes.is_empty())
                .unwrap();
            bytes[0] ^= 1;
        });
        let changed_identity = archive_identity(&changed, &original_identity.prefix);
        assert!(verify_source_archive(&repo, &changed, &changed_identity, &source).is_err());

        let wrong_mode = temporary.0.join("wrong-mode.tgz");
        rewrite_archive(&original, &wrong_mode, |rows| {
            let (header, _) = rows
                .iter_mut()
                .find(|(header, _)| {
                    header.entry_type().is_file() && header.mode().unwrap() & 0o111 != 0
                })
                .unwrap();
            header.set_mode(0o644);
        });
        let wrong_mode_identity = archive_identity(&wrong_mode, &original_identity.prefix);
        assert!(verify_source_archive(&repo, &wrong_mode, &wrong_mode_identity, &source).is_err());

        let missing_directory = temporary.0.join("missing-directory.tgz");
        rewrite_archive(&original, &missing_directory, |rows| {
            let index = rows
                .iter()
                .position(|(header, _)| header.entry_type().is_dir())
                .unwrap();
            rows.remove(index);
        });
        let missing_directory_identity =
            archive_identity(&missing_directory, &original_identity.prefix);
        assert!(
            verify_source_archive(
                &repo,
                &missing_directory,
                &missing_directory_identity,
                &source,
            )
            .is_err()
        );

        let trailing = temporary.0.join("trailing.tgz");
        fs::copy(&original, &trailing).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&trailing)
            .unwrap()
            .write_all(b"not part of gzip")
            .unwrap();
        let trailing_identity = archive_identity(&trailing, &original_identity.prefix);
        assert!(verify_source_archive(&repo, &trailing, &trailing_identity, &source).is_err());
    }
}
