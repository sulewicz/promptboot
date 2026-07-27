#!/usr/bin/env python3
"""Record the source identities used by the model reference extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
TOOLS = [
    "cargo", "run", "--manifest-path", str(REPO / "Cargo.toml"),
    "--package", "promptboot-tools", "--target", "x86_64-unknown-linux-gnu",
    "--release", "--locked", "--offline", "--quiet", "--",
]


def asset_value(kind: str, field: str) -> str:
    return subprocess.run(
        [*TOOLS, "asset-value", "--kind", kind, "--field", field],
        cwd=REPO, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.binary.is_file():
            raise RuntimeError(f"missing reference extractor {args.binary}")
        extractor = REPO / "tests/fixtures/reference/reference_extract.cpp"
        report = {
            "binary_sha256": sha256_file(args.binary),
            "extractor_sha256": sha256_file(extractor),
            "llama_archive_sha256": asset_value("llama_archive", "sha256"),
            "llama_revision": asset_value("llama_archive", "revision"),
            "result": "PASS",
        }
        encoded = compact(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="ascii")
        os.replace(temporary, args.output)
        sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"REFERENCE_BUILD_AUDIT_FAILED {error}", file=sys.stderr)
        return 52


if __name__ == "__main__":
    raise SystemExit(main())
