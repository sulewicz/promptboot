use promptboot_tools::{assets, image as image_tool, inspect, pack, release as release_tool};
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage:\n  promptboot-tools fetch-assets [--cache-dir DIR]\n  promptboot-tools verify-assets [--cache-dir DIR]\n  promptboot-tools asset-value [--cache-dir DIR] --kind KIND --field FIELD\n  promptboot-tools pack-model [--source GGUF] --output PBTQW25\n  promptboot-tools inspect-model --model PBTQW25 [--source GGUF] [--output JSON]\n  promptboot-tools build-image [--cache-dir DIR] --output-dir DIR --target-dir DIR [--distribution-root DIR]\n  promptboot-tools inspect-image --image IMG --manifest BUILD.JSN [--distribution-root DIR]\n  promptboot-tools release --output PATH\n  promptboot-tools verify-release --release PATH"
}

#[derive(Default)]
struct Options {
    source: Option<PathBuf>,
    output: Option<PathBuf>,
    model: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    distribution_root: Option<PathBuf>,
    image: Option<PathBuf>,
    manifest: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    release: Option<PathBuf>,
    kind: Option<PathBuf>,
    field: Option<PathBuf>,
}

fn options(args: &[String], allowed: &[&str]) -> Result<Options, String> {
    let mut result = Options::default();
    let mut index = 0;
    while index < args.len() {
        let name = args[index].as_str();
        if !allowed.contains(&name) {
            return Err(format!("unknown option {name}"));
        }
        index += 1;
        let value = PathBuf::from(
            args.get(index)
                .ok_or_else(|| format!("missing value for {name}"))?,
        );
        let slot = match name {
            "--source" => &mut result.source,
            "--output" => &mut result.output,
            "--model" => &mut result.model,
            "--output-dir" => &mut result.output_dir,
            "--target-dir" => &mut result.target_dir,
            "--distribution-root" => &mut result.distribution_root,
            "--image" => &mut result.image,
            "--manifest" => &mut result.manifest,
            "--cache-dir" => &mut result.cache_dir,
            "--release" => &mut result.release,
            "--kind" => &mut result.kind,
            "--field" => &mut result.field,
            _ => unreachable!(),
        };
        if slot.replace(value).is_some() {
            return Err(format!("duplicate option {name}"));
        }
        index += 1;
    }
    Ok(result)
}

fn pack_command(args: &[String]) -> i32 {
    let values = match options(args, &["--source", "--output"]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let source = match values.source {
        Some(value) => value,
        None => {
            let repo = match env::current_dir() {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("MODEL_PACK_FAILED {error}");
                    return 42;
                }
            };
            match assets::value(&repo, None, assets::AssetKind::QwenGguf, "path") {
                Ok(value) => PathBuf::from(value),
                Err(error) => {
                    eprintln!(
                        "MODEL_PACK_FAILED asset {}: {}",
                        error.category.name(),
                        error.message
                    );
                    return 42;
                }
            }
        }
    };
    let output = match values.output {
        Some(value) => value,
        None => {
            eprintln!("missing --output\n{}", usage());
            return 2;
        }
    };
    match pack::run(&source, &output) {
        Ok(report) => {
            println!(
                "MODEL_INSPECTION_PASS file_sha256={} tensors={} tensor_hash_manifest_sha256={}",
                report.file_sha256, report.tensor_count, report.tensor_hash_manifest_sha256
            );
            println!(
                "MODEL_PACKED path={} bytes={}",
                output.display(),
                report.file_size
            );
            0
        }
        Err(pack::PackFailure::Identity(error)) => {
            eprintln!("MODEL_IDENTITY_FAILED {error}");
            40
        }
        Err(pack::PackFailure::Schema(error)) => {
            eprintln!("MODEL_SCHEMA_FAILED {error}");
            41
        }
        Err(pack::PackFailure::Pack(error)) => {
            eprintln!("MODEL_PACK_FAILED {error}");
            42
        }
    }
}

fn inspect_command(args: &[String]) -> i32 {
    let values = match options(args, &["--model", "--source", "--output"]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let model = match values.model {
        Some(value) => value,
        None => {
            eprintln!("missing --model\n{}", usage());
            return 2;
        }
    };
    let source = match values.source {
        Some(value) => value,
        None => {
            let repo = match env::current_dir() {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("MODEL_INSPECTION_FAILED {error}");
                    return 43;
                }
            };
            match assets::value(&repo, None, assets::AssetKind::QwenGguf, "path") {
                Ok(value) => PathBuf::from(value),
                Err(error) => {
                    eprintln!(
                        "MODEL_INSPECTION_FAILED asset {}: {}",
                        error.category.name(),
                        error.message
                    );
                    return 43;
                }
            }
        }
    };
    let output = values.output;
    match inspect::inspect(&model, &source) {
        Ok(report) => {
            let json = report.json();
            if let Some(path) = output {
                if let Err(error) = inspect::write_atomic(&path, json.as_bytes()) {
                    eprintln!("MODEL_INSPECTION_FAILED {error}");
                    return 43;
                }
            }
            print!("{json}");
            0
        }
        Err(error) => {
            eprintln!("MODEL_INSPECTION_FAILED {error}");
            43
        }
    }
}

fn required(value: Option<PathBuf>, name: &str) -> Result<PathBuf, i32> {
    value.ok_or_else(|| {
        eprintln!("missing {name}\n{}", usage());
        2
    })
}

fn build_image_command(args: &[String]) -> i32 {
    let values = match options(
        args,
        &[
            "--cache-dir",
            "--output-dir",
            "--target-dir",
            "--distribution-root",
        ],
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let Ok(output) = required(values.output_dir, "--output-dir") else {
        return 2;
    };
    let Ok(target) = required(values.target_dir, "--target-dir") else {
        return 2;
    };
    let repo = match env::current_dir() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("MODEL_IMAGE_BUILD_FAILED {error}");
            return 22;
        }
    };
    let source = match assets::value(
        &repo,
        values.cache_dir.as_deref(),
        assets::AssetKind::QwenGguf,
        "path",
    ) {
        Ok(value) => PathBuf::from(value),
        Err(error) => {
            eprintln!(
                "MODEL_IMAGE_BUILD_FAILED asset {}: {}",
                error.category.name(),
                error.message
            );
            return 22;
        }
    };
    match image_tool::build_image(
        &source,
        &output,
        &target,
        values.distribution_root.as_deref(),
    ) {
        Ok(build_id) => {
            println!(
                "MODEL_IMAGE_BUILD_OK mode=model_repl build_id={build_id} output={}",
                output.display()
            );
            0
        }
        Err(error) => {
            eprintln!("MODEL_IMAGE_BUILD_FAILED {error}");
            22
        }
    }
}

fn asset_command(args: &[String], fetch: bool) -> i32 {
    let values = match options(args, &["--cache-dir"]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let repo = match env::current_dir() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("MODEL_ASSET_FAILED category=schema {error}");
            return 34;
        }
    };
    let result = if fetch {
        assets::fetch_all(&repo, values.cache_dir.as_deref())
    } else {
        assets::verify_all(&repo, values.cache_dir.as_deref()).map(|rows| {
            rows.into_iter()
                .map(|(kind, path)| (false, kind, path))
                .collect()
        })
    };
    match result {
        Ok(rows) => {
            for (fetched, kind, path) in rows {
                println!(
                    "MODEL_ASSET_{} kind={} path={}",
                    if fetched { "FETCHED" } else { "VERIFIED" },
                    kind.name(),
                    path.display()
                );
            }
            0
        }
        Err(error) => {
            eprintln!(
                "MODEL_ASSET_FAILED category={} {}",
                error.category.name(),
                error.message
            );
            error.category.exit()
        }
    }
}

fn asset_value_command(args: &[String]) -> i32 {
    let values = match options(args, &["--cache-dir", "--kind", "--field"]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let Some(kind_text) = values
        .kind
        .and_then(|value| value.into_os_string().into_string().ok())
    else {
        eprintln!("missing or invalid --kind\n{}", usage());
        return 2;
    };
    let Some(kind) = assets::AssetKind::parse(&kind_text) else {
        eprintln!("invalid --kind\n{}", usage());
        return 2;
    };
    let Some(field) = values
        .field
        .and_then(|value| value.into_os_string().into_string().ok())
    else {
        eprintln!("missing or invalid --field\n{}", usage());
        return 2;
    };
    if !matches!(
        field.as_str(),
        "path" | "name" | "revision" | "sha256" | "size"
    ) {
        eprintln!("invalid --field\n{}", usage());
        return 2;
    }
    let repo = match env::current_dir() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("MODEL_ASSET_FAILED category=schema {error}");
            return 34;
        }
    };
    match assets::value(&repo, values.cache_dir.as_deref(), kind, &field) {
        Ok(value) => {
            println!("{value}");
            0
        }
        Err(error) => {
            eprintln!(
                "MODEL_ASSET_FAILED category={} {}",
                error.category.name(),
                error.message
            );
            error.category.exit()
        }
    }
}

fn release_command(args: &[String], verify: bool) -> i32 {
    let allowed = if verify {
        ["--release"].as_slice()
    } else {
        ["--output"].as_slice()
    };
    let values = match options(args, allowed) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let requested = if verify {
        match values.release {
            Some(value) => value,
            None => {
                eprintln!("missing --release\n{}", usage());
                return 2;
            }
        }
    } else {
        match values.output {
            Some(value) => value,
            None => {
                eprintln!("missing --output\n{}", usage());
                return 2;
            }
        }
    };
    let repo = match env::current_dir() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("RELEASE_FAILED category=source {error}");
            return 40;
        }
    };
    let result = if verify {
        release_tool::verify(&repo, &requested)
    } else {
        release_tool::release(&repo, &requested, None)
    };
    match result {
        Ok(outcome) => {
            println!(
                "RELEASE_OK output={} commit={} builds={} reused={}",
                outcome.output.display(),
                outcome.commit,
                outcome.builds,
                outcome.reused
            );
            0
        }
        Err(error) => {
            eprintln!(
                "RELEASE_FAILED category={} {}",
                error.category.name(),
                error.message
            );
            error.category.exit()
        }
    }
}

fn inspect_image_command(args: &[String]) -> i32 {
    let values = match options(args, &["--image", "--manifest", "--distribution-root"]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            return 2;
        }
    };
    let Ok(image) = required(values.image, "--image") else {
        return 2;
    };
    let Ok(manifest) = required(values.manifest, "--manifest") else {
        return 2;
    };
    match image_tool::inspect_image(&image, &manifest, values.distribution_root.as_deref()) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(encoded) => {
                println!("{encoded}");
                0
            }
            Err(error) => {
                eprintln!("IMAGE_INVALID report encoding failed: {error}");
                26
            }
        },
        Err(error) => {
            eprintln!("IMAGE_INVALID {error}");
            26
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let status = match args.split_first() {
        Some((command, rest)) if command == "pack-model" => pack_command(rest),
        Some((command, rest)) if command == "inspect-model" => inspect_command(rest),
        Some((command, rest)) if command == "fetch-assets" => asset_command(rest, true),
        Some((command, rest)) if command == "verify-assets" => asset_command(rest, false),
        Some((command, rest)) if command == "asset-value" => asset_value_command(rest),
        Some((command, rest)) if command == "build-image" => build_image_command(rest),
        Some((command, rest)) if command == "inspect-image" => inspect_image_command(rest),
        Some((command, rest)) if command == "release" => release_command(rest, false),
        Some((command, rest)) if command == "verify-release" => release_command(rest, true),
        _ => {
            eprintln!("{}", usage());
            2
        }
    };
    std::process::exit(status);
}
