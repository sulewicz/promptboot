#!/usr/bin/env python3
"""Audit the pinned llama.cpp REPL extractor and regenerate fixed oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((REPO / "fixtures/reference/repl/provenance.json").read_text(encoding="ascii"))
    source = REPO / provenance["extractor"]
    model = Path(asset_value("qwen_gguf", "path"))
    if sha256(source) != provenance["extractor_sha256"]:
        raise SystemExit("REPL_REFERENCE_AUDIT_FAILED extractor identity")
    if not args.binary.is_file() or not model.is_file():
        raise SystemExit("REPL_REFERENCE_AUDIT_FAILED prerequisite")
    if sha256(model) != provenance["source_gguf_sha256"]:
        raise SystemExit("REPL_REFERENCE_AUDIT_FAILED source GGUF identity")
    with tempfile.TemporaryDirectory(prefix="repl-reference-audit-") as temporary:
        directory = Path(temporary)
        base_report = directory / "base.json"
        subprocess.run([
            REPO / "tests/fixtures/reference/audit_reference_build.py",
            "--binary", args.binary, "--output", base_report,
        ], check=True, stdout=subprocess.DEVNULL)
        counts = {}
        for length, expected in ((450, 479), (451, 480), (452, 481)):
            result = subprocess.run(
                [args.binary, "--token-count", model, "0" * length],
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout.strip()
            if result != f"REPL_TOKEN_COUNT tokens={expected}":
                raise SystemExit(f"REPL_REFERENCE_AUDIT_FAILED threshold={length}")
            counts[str(length)] = expected
        cases = (
            ("max32.json", "max32", "color", None),
            ("eos-turn2.json", "eos_turn2", "Name one color.", "Name another color."),
            ("fresh480.json", "fresh480", "0" * 451, None),
        )
        regenerated = {}
        for name, case, first, second in cases:
            output = directory / name
            command = [args.binary, model, output, case, first]
            if second is not None:
                command.append(second)
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tracked = REPO / "fixtures/reference/repl" / name
            if output.read_bytes() != tracked.read_bytes():
                raise SystemExit(f"REPL_REFERENCE_AUDIT_FAILED regeneration={name}")
            regenerated[name] = sha256(output)
        report = {
            "base_build_audit": json.loads(base_report.read_text(encoding="ascii")),
            "binary_sha256": sha256(args.binary),
            "extractor_sha256": sha256(source),
            "llama_revision": provenance["llama_revision"],
            "source_gguf_sha256": sha256(model),
            "regenerated": regenerated,
            "result": "PASS",
            "schema": 1,
            "threshold_token_counts": counts,
        }
        args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("REPL_REFERENCE_BUILD_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
