use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn pack_cli_classifies_identity_failure_without_publishing() {
    let root = std::env::temp_dir().join(format!("promptboot-tools-cli-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir(&root).unwrap();
    let source = root.join("bad.gguf");
    let output = root.join("model.pbtqw25");
    fs::write(&source, b"GGUF-not-the-locked-asset").unwrap();

    for existing in [None, Some(b"existing".as_slice())] {
        if let Some(bytes) = existing {
            fs::write(&output, bytes).unwrap();
        }
        let result = Command::new(env!("CARGO_BIN_EXE_promptboot-tools"))
            .args([
                "pack-model",
                "--source",
                source.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(40));
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(stderr.contains("MODEL_IDENTITY_FAILED"));
        assert!(!stderr.contains("MODEL_SCHEMA_FAILED"));
        assert!(!stderr.contains("MODEL_PACK_FAILED"));
        match existing {
            None => assert!(!output.exists()),
            Some(bytes) => assert_eq!(fs::read(&output).unwrap(), bytes),
        }
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
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn build_image_cli_failures_do_not_publish_partial_directories() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let root = std::env::temp_dir().join(format!(
        "promptboot-tools-image-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir(&root).unwrap();
    let cargo = repo.join(".cache/rust-1.97.1/bin/cargo");
    let rustc = repo.join(".cache/rust-1.97.1/bin/rustc");

    for (failure, existing_empty, cargo_command, rustc_command) in [
        ("cargo", false, PathBuf::from("/bin/false"), rustc.clone()),
        ("rustc", true, cargo.clone(), PathBuf::from("/bin/false")),
    ] {
        let suffix = if existing_empty { "empty" } else { "absent" };
        let output = root.join(format!("{failure}-{suffix}"));
        let target = root.join(format!("{failure}-{suffix}-target"));
        fs::create_dir(&target).unwrap();
        if existing_empty {
            fs::create_dir(&output).unwrap();
        }
        let result = Command::new(env!("CARGO_BIN_EXE_promptboot-tools"))
            .args([
                "build-image",
                "--output-dir",
                output.to_str().unwrap(),
                "--target-dir",
                target.to_str().unwrap(),
            ])
            .current_dir(&repo)
            .env("CARGO", &cargo_command)
            .env("RUSTC", &rustc_command)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(22));
        assert!(
            String::from_utf8(result.stderr)
                .unwrap()
                .contains("MODEL_IMAGE_BUILD_FAILED")
        );
        if existing_empty {
            assert!(output.is_dir());
            assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
        } else {
            assert!(!output.exists());
        }
    }
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
