from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
IMAGE_FILES = (
    "BOOTX64.EFI", "BUILD.JSN", "MODEL.PBT", "promptboot.img",
    "promptboot-media-inspection.json",
)
DISTRIBUTION_FILES = (
    "LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/QWEN-APACHE-2.0.txt",
    "LICENSES/LLAMA-MIT.txt", "LICENSES/libm-0.2.11.txt",
    "LICENSES/RUST-1.97.1-COPYRIGHT-library.html",
    "LICENSES/compiler-builtins-0.1.160.txt",
)
RELEASE_FILES = set(IMAGE_FILES + DISTRIBUTION_FILES + (
    "SOURCE.TGZ", "RUN.md", "release.json", "SHA256SUMS",
))


def load_script(name: str):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseCliTests(unittest.TestCase):
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
    def git_repo(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "tracked").write_text("tracked\n", encoding="ascii")
        subprocess.run(["git", "add", "tracked"], cwd=root, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test",
                "-c", "user.email=test@example.invalid",
                "commit", "-qm", "fixture",
            ],
            cwd=root, check=True,
        )

    def run_tool(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args], cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def identity(cls, path: Path) -> dict[str, object]:
        return {"bytes": path.stat().st_size, "sha256": cls.digest(path)}

    @classmethod
    def write_checksums(cls, release: Path) -> None:
        (release / "SHA256SUMS").write_text(
            "".join(
                f"{cls.digest(release / name)}  {name}\n"
                for name in sorted(RELEASE_FILES - {"SHA256SUMS"})
            ),
            encoding="ascii",
        )

    @classmethod
    def synthetic_release(cls, root: Path) -> Path:
        (root / ".gitignore").write_text("build/\ntarget/\ncache/\n", encoding="ascii")
        for name in DISTRIBUTION_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {name}\n", encoding="ascii")
        (root / "tracked").write_text("tracked source\n", encoding="ascii")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test",
                "-c", "user.email=test@example.invalid",
                "commit", "-qm", "fixture",
            ],
            cwd=root, check=True,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip()
        release = root / "build/release"
        (release / "LICENSES").mkdir(parents=True)
        for name in IMAGE_FILES:
            payload = b'{"build_id":"fixture"}\n' if name == "BUILD.JSN" else name.encode()
            (release / name).write_bytes(payload)
        for name in DISTRIBUTION_FILES:
            (release / name).write_bytes((root / name).read_bytes())
        (release / "RUN.md").write_text("fixture run\n", encoding="ascii")
        prefix = f"promptboot-{commit[:12]}/"
        subprocess.run(
            [
                "git", "archive", "--format=tar.gz", f"--prefix={prefix}",
                "-o", release / "SOURCE.TGZ", commit,
            ],
            cwd=root, check=True,
        )
        release_json = {
            "artifacts": {
                name: cls.identity(release / name) for name in IMAGE_FILES
            },
            "build": {
                "build_id": "fixture",
                "command": "promptboot-tools release",
                "count": 1,
            },
            "distribution": {
                name: cls.identity(release / name)
                for name in (*DISTRIBUTION_FILES, "SOURCE.TGZ")
            },
            "schema": 1,
            "source": {"commit": commit, "tree": tree},
            "source_archive": {
                **cls.identity(release / "SOURCE.TGZ"),
                "format": "tar.gz",
                "path": "SOURCE.TGZ",
                "prefix": prefix,
            },
        }
        (release / "release.json").write_text(
            json.dumps(release_json, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        cls.write_checksums(release)
        return release

    @classmethod
    def refresh_archive_identities(cls, release: Path) -> None:
        payload = json.loads((release / "release.json").read_text(encoding="ascii"))
        identity = cls.identity(release / "SOURCE.TGZ")
        payload["distribution"]["SOURCE.TGZ"] = identity
        payload["source_archive"]["bytes"] = identity["bytes"]
        payload["source_archive"]["sha256"] = identity["sha256"]
        (release / "release.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        cls.write_checksums(release)

    @staticmethod
    def rewrite_archive(
        source: Path,
        destination: Path,
        mutate,
    ) -> None:
        with tarfile.open(source, "r:gz") as archive:
            rows = [
                (
                    copy.copy(member),
                    archive.extractfile(member).read() if member.isfile() else None,
                )
                for member in archive.getmembers()
            ]
        mutate(rows)
        with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            global_header = tarfile.TarInfo("pax_global_header")
            global_header.type = tarfile.XGLTYPE
            global_bytes = b"19 comment=fixture\n"
            global_header.size = len(global_bytes)
            archive.addfile(global_header, io.BytesIO(global_bytes))
            for member, data in rows:
                archive.addfile(
                    member, None if data is None else io.BytesIO(data)
                )

    def assert_release_validation(
        self, root: Path, release: Path
    ) -> subprocess.CompletedProcess[str]:
        result = self.run_tool(
            root, "verify-release", "--release", str(release.relative_to(root))
        )
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1)
        self.assertTrue(
            result.stderr.startswith("RELEASE_FAILED category=validation "),
            result.stderr,
        )
        self.assertNotIn("panicked", result.stderr)
        return result

    def test_dirty_source_is_rejected_before_output_or_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.git_repo(root)
            (root / "tracked").write_text("dirty\n", encoding="ascii")
            result = self.run_tool(root, "release", "--output", "build/release")
            self.assertEqual(result.returncode, 40)
            self.assertIn("RELEASE_FAILED category=source", result.stderr)
            self.assertFalse((root / "build").exists())

    def test_release_output_is_bounded_below_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.git_repo(root)
            result = self.run_tool(root, "release", "--output", "release")
            self.assertEqual(result.returncode, 42)
            self.assertIn("RELEASE_FAILED category=output", result.stderr)

    def test_release_commands_have_usage_status_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self.run_tool(root, "release")
            unknown = self.run_tool(
                root, "verify-release", "--release", "build/x", "--extra", "x"
            )
            self.assertEqual(missing.returncode, 2)
            self.assertEqual(unknown.returncode, 2)

    def test_malformed_utf8_checksum_is_one_validation_failure_not_a_panic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = self.synthetic_release(root)
            (release / "SHA256SUMS").write_bytes(
                b"a" * 63 + "é".encode("utf-8") + b"  RUN.md\n"
            )
            result = self.assert_release_validation(root, release)
            self.assertIn("malformed SHA256SUMS", result.stderr)

    def test_public_release_layout_and_corruption_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = self.synthetic_release(root)
            base = self.assert_release_validation(root, release)
            self.assertIn("release image invalid", base.stderr)

            (release / "unexpected").write_bytes(b"x")
            extra = self.assert_release_validation(root, release)
            self.assertIn("release membership mismatch", extra.stderr)
            (release / "unexpected").unlink()

            (release / "RUN.md").write_bytes(b"corrupt")
            corrupt = self.assert_release_validation(root, release)
            self.assertIn("release checksum mismatch", corrupt.stderr)

    def test_public_archive_semantic_mutations_are_rejected(self) -> None:
        def regular(rows):
            return next(
                index for index, (member, _) in enumerate(rows) if member.isfile()
            )

        def duplicate(rows):
            index = regular(rows)
            member, data = rows[index]
            rows.append((copy.copy(member), data))

        def unsafe(rows):
            index = regular(rows)
            rows[index][0].name = rows[index][0].name.split("/", 1)[0] + "/../escape"

        def wrong_prefix(rows):
            rows[regular(rows)][0].name = "wrong-prefix/file"

        def missing(rows):
            rows.pop(regular(rows))

        def extra(rows):
            prefix = rows[0][0].name.split("/", 1)[0]
            member = tarfile.TarInfo(f"{prefix}/extra")
            member.size = 1
            rows.append((member, b"x"))

        def typed(kind, link=""):
            def mutate(rows):
                index = regular(rows)
                member, _ = rows[index]
                member.type = kind
                member.linkname = link
                member.size = 0
                rows[index] = (member, None)
            return mutate

        cases = (
            ("duplicate", duplicate),
            ("unsafe", unsafe),
            ("wrong-prefix", wrong_prefix),
            ("missing", missing),
            ("extra", extra),
            ("wrong-type", typed(tarfile.DIRTYPE)),
            ("symlink", typed(tarfile.SYMTYPE, "tracked")),
            ("hardlink", typed(tarfile.LNKTYPE, "tracked")),
            ("device", typed(tarfile.CHRTYPE)),
            ("fifo", typed(tarfile.FIFOTYPE)),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                release = self.synthetic_release(root)
                original = release / "ORIGINAL.TGZ"
                original.write_bytes((release / "SOURCE.TGZ").read_bytes())
                self.rewrite_archive(original, release / "SOURCE.TGZ", mutate)
                original.unlink()
                self.refresh_archive_identities(release)
                result = self.assert_release_validation(root, release)
                self.assertIn("source archive", result.stderr)

    def test_target_allocation_failure_cleans_partial_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "a" * 40
            model = b"model"
            qwen_license = b"Apache License\nVersion 2.0, January 2004\n"
            llama_license = b"MIT License\nfixture\n"
            encoded = io.BytesIO()
            with tarfile.open(fileobj=encoded, mode="w:gz") as archive:
                directory = tarfile.TarInfo(f"llama.cpp-{revision}/")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                member = tarfile.TarInfo(f"llama.cpp-{revision}/LICENSE")
                member.size = len(llama_license)
                archive.addfile(member, io.BytesIO(llama_license))
            llama = encoded.getvalue()
            assets = (
                {
                    "kind": "qwen_gguf", "license": "Apache-2.0",
                    "name": "model.gguf", "revision": revision,
                    "sha256": hashlib.sha256(model).hexdigest(),
                    "size": len(model), "url": "https://example.invalid/model",
                },
                {
                    "kind": "qwen_license", "license": "Apache-2.0",
                    "name": "QWEN-LICENSE", "revision": revision,
                    "sha256": hashlib.sha256(qwen_license).hexdigest(),
                    "size": len(qwen_license),
                    "url": "https://example.invalid/license",
                },
                {
                    "archive_license": {
                        "path": "LICENSE",
                        "sha256": hashlib.sha256(llama_license).hexdigest(),
                        "size": len(llama_license),
                    },
                    "kind": "llama_archive", "license": "MIT",
                    "name": "llama.tar.gz", "revision": revision,
                    "sha256": hashlib.sha256(llama).hexdigest(),
                    "size": len(llama), "url": "https://example.invalid/llama",
                },
            )
            (root / ".gitignore").write_text(
                "build/\ntarget\ncache/\n", encoding="ascii"
            )
            (root / "assets.lock.json").write_text(
                json.dumps({"assets": assets}), encoding="utf-8"
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "model.gguf").write_bytes(model)
            (cache / "QWEN-LICENSE").write_bytes(qwen_license)
            (cache / "llama.tar.gz").write_bytes(llama)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture",
                ],
                cwd=root, check=True,
            )
            (root / "target").write_bytes(b"not a directory")
            tools = cache / "tools"
            tools.mkdir()
            os.symlink("/usr/bin/git", tools / "git")
            called = cache / "curl-called"
            curl = tools / "curl"
            curl.write_text(
                "#!/bin/sh\n: > \"$CURL_CALLED\"\nexit 99\n", encoding="ascii"
            )
            curl.chmod(0o755)
            result = subprocess.run(
                [self.binary, "release", "--output", "build/release"],
                cwd=root,
                env={
                    **os.environ,
                    "CURL_CALLED": str(called),
                    "PROMPTBOOT_ASSET_DIR": str(cache),
                    "PATH": str(tools),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 41, result.stderr)
            self.assertEqual(len(result.stderr.splitlines()), 1)
            self.assertTrue(
                result.stderr.startswith("RELEASE_FAILED category=build ")
            )
            self.assertFalse((root / "build/release").exists())
            self.assertEqual(list((root / "build").iterdir()), [])
            self.assertFalse(called.exists())


class LaunchScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.play = load_script("play")
        cls.usb = load_script("write_usb")

    def test_launchers_delegate_release_once_without_parsing_output(self) -> None:
        for module in (self.play, self.usb):
            with self.subTest(module=module.__name__), mock.patch.object(
                module.subprocess, "run"
            ) as run:
                resolved = module.ensure_release(Path("build/example"))
                self.assertEqual(resolved, (REPO / "build/example").resolve())
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertEqual(
                    command[-4:], ["--", "release", "--output", "build/example"]
                )
                self.assertEqual(run.call_args.kwargs, {"cwd": REPO, "check": True})

    def test_usb_rejects_non_absolute_or_incomplete_inventory_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            self.usb.normalized_node({"path": "sda", "size": 1, "ro": 0, "maj:min": "8:0"})
        with self.assertRaisesRegex(ValueError, "invalid size"):
            self.usb.normalized_node(
                {"path": "/dev/sda", "size": 0, "ro": 0, "maj:min": "8:0"}
            )
        with self.assertRaisesRegex(ValueError, "read-only state"):
            self.usb.normalized_node(
                {"path": "/dev/sda", "size": 1, "ro": "maybe", "maj:min": "8:0"}
            )

    def test_usb_inspection_rejects_real_non_block_devices_before_lsblk(self) -> None:
        with tempfile.NamedTemporaryFile() as ordinary:
            with self.assertRaisesRegex(ValueError, "not a block device"):
                self.usb.inspect_device(Path(ordinary.name))
        with self.assertRaisesRegex(ValueError, "not a block device"):
            self.usb.inspect_device(Path("/dev/null"))


if __name__ == "__main__":
    unittest.main()
