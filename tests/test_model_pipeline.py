from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests/fixtures/analytic"))
sys.path.insert(0, str(REPO / "tests/fixtures/reference"))

import generate_analytic_fixtures


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AssetLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                "cargo", "build", "--package", "promptboot-tools",
                "--target", "x86_64-unknown-linux-gnu", "--locked", "--offline",
            ],
            cwd=REPO, check=True,
        )
        cls.binary = (
            REPO / "target/x86_64-unknown-linux-gnu/debug/promptboot-tools"
        )

    @staticmethod
    def fixture(root: Path) -> tuple[Path, dict[str, bytes], str]:
        revision = "a" * 40
        qwen = b"small model fixture"
        license_text = b"Apache License\nVersion 2.0, January 2004\n"
        llama_license = b"MIT License\nfixture\n"
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            directory = tarfile.TarInfo(f"llama.cpp-{revision}/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            member = tarfile.TarInfo(f"llama.cpp-{revision}/LICENSE")
            member.size = len(llama_license)
            archive.addfile(member, io.BytesIO(llama_license))
        payloads = {
            "model": qwen,
            "license": license_text,
            "llama": archive_bytes.getvalue(),
        }
        assets = [
            {
                "kind": "llama_archive", "license": "MIT",
                "name": "llama.tar.gz", "revision": revision,
                "sha256": hashlib.sha256(payloads["llama"]).hexdigest(),
                "size": len(payloads["llama"]),
                "url": "https://example.invalid/llama",
                "archive_license": {
                    "path": "LICENSE",
                    "sha256": hashlib.sha256(llama_license).hexdigest(),
                    "size": len(llama_license),
                },
            },
            {
                "kind": "qwen_license", "license": "Apache-2.0",
                "name": "QWEN-LICENSE", "revision": revision,
                "sha256": hashlib.sha256(license_text).hexdigest(),
                "size": len(license_text),
                "url": "https://example.invalid/license",
            },
            {
                "kind": "qwen_gguf", "license": "Apache-2.0",
                "name": "model.gguf", "revision": revision,
                "sha256": hashlib.sha256(qwen).hexdigest(),
                "size": len(qwen),
                "url": "https://example.invalid/model",
            },
        ]
        (root / "assets.lock.json").write_text(
            json.dumps({"assets": assets}), encoding="utf-8"
        )
        payload_root = root / "payloads"
        payload_root.mkdir()
        for name, payload in payloads.items():
            (payload_root / name).write_bytes(payload)
        return payload_root, payloads, revision

    def run_tool(
        self, root: Path, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args], cwd=root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    @staticmethod
    def lock(root: Path) -> dict[str, object]:
        return json.loads((root / "assets.lock.json").read_text(encoding="utf-8"))

    @staticmethod
    def asset(lock: dict[str, object], kind: str) -> dict[str, object]:
        return next(
            row for row in lock["assets"]
            if isinstance(row, dict) and row.get("kind") == kind
        )

    @staticmethod
    def populate_cache(root: Path, payloads: dict[str, bytes]) -> Path:
        cache = root / "cache"
        cache.mkdir(exist_ok=True)
        (cache / "model.gguf").write_bytes(payloads["model"])
        (cache / "QWEN-LICENSE").write_bytes(payloads["license"])
        (cache / "llama.tar.gz").write_bytes(payloads["llama"])
        return cache

    def assert_asset_failure(
        self,
        result: subprocess.CompletedProcess[str],
        category: str,
        status: int,
    ) -> None:
        self.assertEqual(result.returncode, status, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1)
        self.assertTrue(
            result.stderr.startswith(f"MODEL_ASSET_FAILED category={category} "),
            result.stderr,
        )

    def test_fetch_uses_locked_curl_contract_then_reuses_verified_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload_root, payloads, revision = self.fixture(root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            curl = bin_dir / "curl"
            curl.write_text(
                """#!/bin/sh
printf 'CALL\\n' >> "$CURL_LOG"
printf '%s\\n' "$@" >> "$CURL_LOG"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dump-header) headers=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --url) url=$2; shift 2 ;;
    *) shift ;;
  esac
done
: > "$headers"
case "$url" in
  */model) payload=model; printf 'x-repo-commit: %s\\r\\n' "$REVISION" > "$headers" ;;
  */license) payload=license; printf 'x-repo-commit: %s\\r\\nx-repo-commit: %s\\r\\n' "$REVISION" "$REVISION" > "$headers" ;;
  */llama) payload=llama ;;
  *) exit 9 ;;
esac
cp "$PAYLOAD_ROOT/$payload" "$output"
""",
                encoding="ascii",
            )
            curl.chmod(0o755)
            log = root / "curl.log"
            environment = {
                **os.environ,
                "CURL_LOG": str(log),
                "PROMPTBOOT_ASSET_DIR": str(root / "cache"),
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "PAYLOAD_ROOT": str(payload_root),
                "REVISION": revision,
            }
            result = self.run_tool(root, "fetch-assets", env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("MODEL_ASSET_FETCHED"), 3)
            for name, payload in payloads.items():
                cached = {
                    "model": "model.gguf",
                    "license": "QWEN-LICENSE",
                    "llama": "llama.tar.gz",
                }[name]
                self.assertEqual((root / "cache" / cached).read_bytes(), payload)
            rows = log.read_text(encoding="utf-8").splitlines()
            calls: list[list[str]] = []
            for row in rows:
                if row == "CALL":
                    calls.append([])
                else:
                    calls[-1].append(row)
            self.assertEqual(len(calls), 3)
            staged_paths: set[str] = set()
            for call, url in zip(
                calls,
                (
                    "https://example.invalid/model",
                    "https://example.invalid/license",
                    "https://example.invalid/llama",
                ),
                strict=True,
            ):
                self.assertEqual(
                    call[:17],
                    [
                        "--location", "--fail", "--show-error", "--silent",
                        "--proto", "=https", "--proto-redir", "=https",
                        "--max-redirs", "8", "--connect-timeout", "30",
                        "--max-time", "1800", "--user-agent",
                        "promptboot-asset-fetch/1", "--dump-header",
                    ],
                )
                self.assertEqual(call[18], "--output")
                self.assertEqual(call[20:], ["--url", url])
                header, download = Path(call[17]), Path(call[19])
                self.assertEqual(header.parent, root / "cache")
                self.assertEqual(download.parent, root / "cache")
                self.assertTrue(header.name.endswith(".headers"))
                self.assertTrue(download.name.endswith(".download"))
                staged_paths.update((str(header), str(download)))
            self.assertEqual(len(staged_paths), 6)
            before = log.read_bytes()
            no_tools = root / "no-tools"
            no_tools.mkdir()
            offline = {**environment, "PATH": str(no_tools)}
            reused = self.run_tool(root, "fetch-assets", env=offline)
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertEqual(reused.stdout.count("MODEL_ASSET_VERIFIED"), 3)
            self.assertEqual(log.read_bytes(), before)
            verified = self.run_tool(root, "verify-assets", env=offline)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            value = self.run_tool(
                root,
                "asset-value", "--kind", "qwen_gguf", "--field", "path",
                env=offline,
            )
            self.assertEqual(value.returncode, 0, value.stderr)
            self.assertEqual(value.stdout.strip(), str(root / "cache/model.gguf"))
            self.assertEqual(
                [path for path in (root / "cache").iterdir() if path.name.startswith(".")],
                [],
            )

    def test_verify_and_value_are_offline_and_fail_with_typed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            cache = root / "cache"
            cache.mkdir()
            sentinel = root / "curl"
            sentinel.write_text(
                "#!/bin/sh\n: > \"$SENTINEL\"\nexit 99\n", encoding="ascii"
            )
            sentinel.chmod(0o755)
            touched = root / "invoked"
            environment = {
                **os.environ,
                "PROMPTBOOT_ASSET_DIR": str(cache),
                "PATH": str(root),
                "SENTINEL": str(touched),
            }
            missing = self.run_tool(root, "verify-assets", env=environment)
            self.assertEqual(missing.returncode, 30)
            self.assertIn("MODEL_ASSET_FAILED category=missing", missing.stderr)
            self.assertFalse(touched.exists())
            bad_usage = self.run_tool(root, "asset-value", "--kind", "bad", "--field", "path")
            self.assertEqual(bad_usage.returncode, 2)

    def test_every_asset_failure_category_has_one_typed_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, payloads, _ = self.fixture(root)
            empty_path = root / "empty-path"
            empty_path.mkdir()
            base_environment = {
                **os.environ,
                "PROMPTBOOT_ASSET_DIR": str(root / "cache"),
                "PATH": str(empty_path),
            }
            missing = self.run_tool(root, "verify-assets", env=base_environment)
            self.assert_asset_failure(missing, "missing", 30)

            download = self.run_tool(root, "fetch-assets", env=base_environment)
            self.assert_asset_failure(download, "download", 31)

            cache = self.populate_cache(root, payloads)
            (cache / "model.gguf").write_bytes(b"x" * len(payloads["model"]))
            identity = self.run_tool(root, "verify-assets", env=base_environment)
            self.assert_asset_failure(identity, "identity", 32)

            (cache / "model.gguf").write_bytes(payloads["model"])
            bad_license = b"not an approved license marker\n"
            (cache / "QWEN-LICENSE").write_bytes(bad_license)
            lock = self.lock(root)
            license_row = self.asset(lock, "qwen_license")
            license_row["size"] = len(bad_license)
            license_row["sha256"] = hashlib.sha256(bad_license).hexdigest()
            (root / "assets.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            license_failure = self.run_tool(root, "verify-assets", env=base_environment)
            self.assert_asset_failure(license_failure, "license", 33)

            lock["unexpected"] = True
            (root / "assets.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            schema = self.run_tool(root, "verify-assets", env=base_environment)
            self.assert_asset_failure(schema, "schema", 34)

    def test_https_lock_validation_is_narrow_and_public(self) -> None:
        invalid = (
            " https://example.invalid/model",
            "https:// example.invalid/model",
            "https:///model",
            "https://:443/model",
            "https://[::1/model",
            "https://[::1]/model",
            "https://user@example.invalid/model",
            "https://example.invalid/#fragment",
            "https://example.invalid:0/model",
            "https://example.invalid:65536/model",
            "https://-bad.example/model",
            "https://bad-.example/model",
            "https://example..invalid/model",
        )
        for url in invalid:
            with self.subTest(url=url), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                lock = self.lock(root)
                self.asset(lock, "qwen_gguf")["url"] = url
                (root / "assets.lock.json").write_text(
                    json.dumps(lock), encoding="utf-8"
                )
                result = self.run_tool(
                    root, "verify-assets", env={**os.environ, "PATH": str(root)}
                )
                self.assert_asset_failure(result, "schema", 34)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            lock = self.lock(root)
            self.asset(lock, "qwen_gguf")["url"] = (
                "https://example.invalid:443/model?download=true"
            )
            (root / "assets.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            result = self.run_tool(
                root,
                "verify-assets",
                "--cache-dir",
                str(root / "empty"),
                env={**os.environ, "PATH": str(root)},
            )
            self.assert_asset_failure(result, "missing", 30)

    def test_fake_curl_failures_are_typed_and_leave_no_staging(self) -> None:
        cases = (
            ("exit", "download", 31),
            ("header", "identity", 32),
            ("identity", "identity", 32),
            ("license", "license", 33),
        )
        for mode, category, status in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload_root, payloads, revision = self.fixture(root)
                if mode == "identity":
                    (payload_root / "model").write_bytes(
                        b"x" * len(payloads["model"])
                    )
                if mode == "license":
                    bad_license = b"wrong license content\n"
                    (payload_root / "license").write_bytes(bad_license)
                    lock = self.lock(root)
                    row = self.asset(lock, "qwen_license")
                    row["size"] = len(bad_license)
                    row["sha256"] = hashlib.sha256(bad_license).hexdigest()
                    (root / "assets.lock.json").write_text(
                        json.dumps(lock), encoding="utf-8"
                    )
                bin_dir = root / "bin"
                bin_dir.mkdir()
                curl = bin_dir / "curl"
                curl.write_text(
                    """#!/bin/sh
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dump-header) headers=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --url) url=$2; shift 2 ;;
    *) shift ;;
  esac
done
if [ "$MODE" = exit ]; then
  printf partial > "$output"
  exit 7
fi
: > "$headers"
case "$url" in
  */model) payload=model ;;
  */license) payload=license ;;
  */llama) payload=llama ;;
  *) exit 9 ;;
esac
if [ "$MODE" = header ] && [ "$payload" = model ]; then
  printf 'x-repo-commit: %s\\r\\nx-repo-commit: wrong\\r\\n' "$REVISION" > "$headers"
elif [ "$payload" != llama ]; then
  printf 'x-repo-commit: %s\\r\\n' "$REVISION" > "$headers"
fi
/bin/cp "$PAYLOAD_ROOT/$payload" "$output"
""",
                    encoding="ascii",
                )
                curl.chmod(0o755)
                environment = {
                    **os.environ,
                    "MODE": mode,
                    "PROMPTBOOT_ASSET_DIR": str(root / "cache"),
                    "PATH": str(bin_dir),
                    "PAYLOAD_ROOT": str(payload_root),
                    "REVISION": revision,
                }
                result = self.run_tool(root, "fetch-assets", env=environment)
                self.assert_asset_failure(result, category, status)
                self.assertEqual(
                    [
                        path for path in (root / "cache").iterdir()
                        if path.name.startswith(".")
                    ],
                    [],
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            lock = self.lock(root)
            self.asset(lock, "qwen_gguf")["url"] = "http://example.invalid/model"
            (root / "assets.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            curl = root / "curl"
            curl.write_text(
                "#!/bin/sh\n: > \"$CALLED\"\nexit 99\n", encoding="ascii"
            )
            curl.chmod(0o755)
            called = root / "called"
            result = self.run_tool(
                root,
                "fetch-assets",
                env={**os.environ, "CALLED": str(called), "PATH": str(root)},
            )
            self.assert_asset_failure(result, "schema", 34)
            self.assertFalse(called.exists())


class AnalyticFixtureTests(unittest.TestCase):
    def test_committed_analytic_oracle_regenerates_byte_identically(self) -> None:
        data, blob = generate_analytic_fixtures.materialize()
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        self.assertEqual(encoded, (REPO / "fixtures/analytic/primitives.json").read_text(encoding="ascii"))
        self.assertEqual(blob, (REPO / "fixtures/analytic/primitives.f32le").read_bytes())
        names = {item["name"] for item in data["fixtures"]}
        self.assertEqual(names, {
            "f32_bias_residual", "q4_0_dequant_dot_matvec", "q8_0_dequant_dot_matvec",
            "rmsnorm", "rope", "softmax", "gqa_attention_kv_append_reuse",
            "silu_swiglu", "argmax_lowest_id_tie",
        })
        self.assertEqual(data["expected_f32le"]["sha256"], hashlib.sha256(blob).hexdigest())


class ReferenceFixtureTests(unittest.TestCase):
    @staticmethod
    def u32le(path: Path) -> list[int]:
        data = path.read_bytes()
        if len(data) % 4:
            raise AssertionError(f"unaligned u32 fixture {path}")
        return list(struct.unpack(f"<{len(data) // 4}I", data))

    def test_three_public_api_fixtures_match_their_reference_outputs(self) -> None:
        root = REPO / "fixtures/reference/model"
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        self.assertEqual(
            manifest["model_sha256"],
            "7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed",
        )
        self.assertEqual(
            [(item["name"], item["user_message"]) for item in manifest["cases"]],
            [("hello", "Hello"), ("arithmetic", "What is 2+2?"), ("color", "Name one color.")],
        )
        for case in manifest["cases"]:
            case_dir = root / case["name"]
            prompt = (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                "<|im_end|>\n<|im_start|>user\n" + case["user_message"] +
                "<|im_end|>\n<|im_start|>assistant\n"
            ).encode()
            self.assertEqual((case_dir / "prompt.txt").read_bytes(), prompt)
            for name, digest in case["files"].items():
                self.assertEqual(sha256_file(case_dir / name), digest)
            tokens = self.u32le(case_dir / "prompt_tokens.u32le")
            continuation = self.u32le(case_dir / "continuation.u32le")
            self.assertEqual(len(tokens), case["prompt_tokens"])
            self.assertEqual(len(continuation), case["continuation_tokens"])
            self.assertLessEqual(len(continuation), 16)
            if len(continuation) < 16:
                self.assertEqual(continuation[-1], 151645)
            logits_bytes = (case_dir / "prompt_final_logits.f32le").read_bytes()
            self.assertEqual(len(logits_bytes), 151936 * 4)
            logits = struct.unpack("<151936f", logits_bytes)
            steps_path = case_dir / "steps.json"
            steps = json.loads(steps_path.read_text(encoding="ascii"))["steps"]
            self.assertEqual([step["selected_id"] for step in steps], continuation)
            for index, step in enumerate(steps):
                self.assertEqual(step["selected_id"], step["top8"][0]["id"])
                self.assertEqual(step["selected_logit_bits"], step["top8"][0]["logit_bits"])
                ranked = [
                    (struct.unpack("<f", struct.pack("<I", int(item["logit_bits"], 16)))[0], item["id"])
                    for item in step["top8"]
                ]
                self.assertEqual(ranked, sorted(ranked, key=lambda item: (-item[0], item[1])))
                if index == 0:
                    self.assertEqual(
                        struct.unpack("<I", struct.pack("<f", logits[step["selected_id"]]))[0],
                        int(step["selected_logit_bits"], 16),
                    )
            provenance_path = case_dir / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="ascii"))
            self.assertEqual(provenance["model_sha256"], manifest["model_sha256"])
            self.assertEqual(provenance["model_size"], 428_730_208)
            self.assertEqual(
                provenance["llama_revision"],
                "571d0d540df04f25298d0e159e520d9fc62ed121",
            )
            self.assertEqual(
                provenance["llama_archive_sha256"],
                "26a60bd05d7d078d44b2d67babc5d21ab2365da1cf3fe66b30368f4ffc7d78ad",
            )
            for owner in ("generator", "extractor"):
                identity = provenance[owner]
                self.assertEqual(identity["version"], 1)
                self.assertEqual(
                    sha256_file(REPO / identity["path"]),
                    identity["sha256"],
                )
            self.assertEqual(
                provenance["emitted_file_sha256"],
                {
                    name: digest
                    for name, digest in case["files"].items()
                    if name != "provenance.json"
                },
            )
            self.assertEqual(provenance["fixture"]["user_message"], case["user_message"])
            self.assertEqual(provenance["extractor_parameters"]["threads"], 1)
            self.assertEqual(provenance["extractor_parameters"]["kv_type_k"], "GGML_TYPE_F32")
            self.assertEqual(
                provenance["extractor_parameters"]["prompt_construction"],
                "segmented_trusted_markers",
            )
            self.assertEqual(
                provenance["extractor_parameters"]["ordinary_segment_parse_special"], 0
            )
            self.assertEqual(
                provenance["extractor_parameters"]["trusted_marker_ids"],
                {"im_end": 151645, "im_start": 151644},
            )
if __name__ == "__main__":
    unittest.main()
