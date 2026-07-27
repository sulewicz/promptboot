#!/usr/bin/python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = [
    "cargo", "run", "--manifest-path", str(ROOT / "Cargo.toml"),
    "--package", "promptboot-tools", "--target", "x86_64-unknown-linux-gnu",
    "--release", "--locked", "--offline", "--quiet", "--",
]


def asset_value(kind: str, field: str) -> str:
    return subprocess.run(
        [*TOOLS, "asset-value", "--kind", kind, "--field", field],
        cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"INFERENCE_AUDIT_FAILED {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provenance_path = ROOT / "fixtures/inference/kernels/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    llama_revision = asset_value("llama_archive", "revision")
    require(provenance["schema"] == 1 and
            provenance["source_revision"] == llama_revision,
            "oracle provenance revision")
    require(set(provenance) == {
        "helper", "inputs", "oracle_boundaries", "outputs", "schema",
        "source_archive_sha256", "source_files", "source_revision",
    }, "oracle provenance schema")
    require(
        provenance["source_archive_sha256"]
        == asset_value("llama_archive", "sha256"),
        "oracle source archive identity",
    )

    outputs = provenance["outputs"]
    require(len(outputs) == 9, "oracle output exact count")
    verified_outputs: dict[str, dict[str, object]] = {}
    for name, identity in sorted(outputs.items()):
        path = ROOT / "fixtures/inference/kernels" / name
        require(path.is_file(), f"missing oracle {name}")
        require(path.stat().st_size == identity["bytes"], f"oracle size {name}")
        actual = digest(path)
        require(actual == identity["sha256"], f"oracle hash {name}")
        verified_outputs[name] = {"bytes": path.stat().st_size, "sha256": actual}
    helper = ROOT / provenance["helper"]["path"]
    require(digest(helper) == provenance["helper"]["sha256"], "oracle helper hash")
    require(provenance["helper"]["promptboot_core_linked"] is False,
            "oracle must not link candidate core")
    require(provenance["helper"]["two_builds_byte_identical"] is True and
            provenance["helper"]["two_output_runs_byte_identical"] is True,
            "oracle reproducibility")

    report = {
        "schema": 1,
        "llama_revision": llama_revision,
        "oracle_helper_sha256": digest(helper),
        "oracle_outputs": verified_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"INFERENCE_AUDIT_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
