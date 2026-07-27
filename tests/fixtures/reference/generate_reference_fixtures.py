#!/usr/bin/env python3
"""Run the pinned public llama API extractor for three model fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]

CASES = [
    ("hello", "Hello"),
    ("arithmetic", "What is 2+2?"),
    ("color", "Name one color."),
]

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


def encode_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def count_u32(path: Path) -> int:
    size = path.stat().st_size
    if size % 4:
        raise RuntimeError(f"{path.name} is not u32le aligned")
    return size // 4


def run(output: Path) -> None:
    subprocess.run([*TOOLS, "verify-assets"], cwd=REPO, check=True)
    model = Path(asset_value("qwen_gguf", "path"))
    llama_archive_sha256 = asset_value("llama_archive", "sha256")
    llama_revision = asset_value("llama_archive", "revision")
    model_sha256 = asset_value("qwen_gguf", "sha256")
    model_size = int(asset_value("qwen_gguf", "size"))
    work_root = REPO / ".cache/llama-reference"
    binary = work_root / "bin/reference_extract"
    if not binary.is_file():
        raise RuntimeError("reference build missing; run scripts/build_model_reference.sh --clean")
    generator_path = REPO / "tests/fixtures/reference/generate_reference_fixtures.py"
    extractor_path = REPO / "tests/fixtures/reference/reference_extract.cpp"
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        case_manifest = []
        for slug, message in CASES:
            case_dir = temporary / slug
            case_dir.mkdir()
            result = subprocess.run(
                [str(binary), str(model), message, str(case_dir)],
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, "GGML_NUMA": "0"},
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"extractor failed for {slug} exit={result.returncode}: {result.stderr.strip()}"
            )
            steps_path = case_dir / "steps.json"
            steps = json.loads(steps_path.read_text(encoding="ascii"))
            expected_files = [
                "continuation.u32le", "prompt.txt", "prompt_final_logits.f32le",
                "prompt_tokens.u32le", "steps.json",
            ]
            if sorted(path.name for path in case_dir.iterdir()) != expected_files:
                raise RuntimeError(f"extractor emitted unexpected files for {slug}")
            if (case_dir / "prompt_final_logits.f32le").stat().st_size != 151936 * 4:
                raise RuntimeError(f"prompt-final logits length mismatch for {slug}")
            continuation_count = count_u32(case_dir / "continuation.u32le")
            if not 1 <= continuation_count <= 16 or len(steps.get("steps", [])) != continuation_count:
                raise RuntimeError(f"continuation/step count mismatch for {slug}")
            hashes = {name: sha256_file(case_dir / name) for name in expected_files}
            prompt_hash = hashes["prompt.txt"]
            provenance = {
                "emitted_file_sha256": hashes,
                "extractor": {
                    "path": str(extractor_path.relative_to(REPO)),
                    "sha256": sha256_file(extractor_path),
                    "version": 1,
                },
                "extractor_parameters": {
                    "batch": 512,
                    "context": 512,
                    "continuation_limit": 16,
                    "flash_attention": "disabled",
                    "greedy_tie": "lowest_token_id",
                    "kv_type_k": "GGML_TYPE_F32",
                    "kv_type_v": "GGML_TYPE_F32",
                    "n_gpu_layers": 0,
                    "offload_kqv": 0,
                    "ordinary_segment_parse_special": 0,
                    "prompt_construction": "segmented_trusted_markers",
                    "threads": 1,
                    "trusted_marker_ids": {"im_end": 151645, "im_start": 151644},
                },
                "fixture": {
                    "continuation_tokens": continuation_count,
                    "name": slug,
                    "prompt_sha256": prompt_hash,
                    "prompt_tokens": count_u32(case_dir / "prompt_tokens.u32le"),
                    "user_message": message,
                },
                "generator": {
                    "path": str(generator_path.relative_to(REPO)),
                    "sha256": sha256_file(generator_path),
                    "version": 1,
                },
                "llama_archive_sha256": llama_archive_sha256,
                "llama_revision": llama_revision,
                "model_sha256": model_sha256,
                "model_size": model_size,
                "schema": 1,
                "template_sha256": "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4",
            }
            provenance_path = case_dir / "provenance.json"
            provenance_path.write_text(encode_json(provenance), encoding="ascii")
            case_manifest.append({
                "continuation_tokens": continuation_count,
                "files": {
                    name: sha256_file(case_dir / name)
                    for name in expected_files + ["provenance.json"]
                },
                "name": slug,
                "prompt_tokens": count_u32(case_dir / "prompt_tokens.u32le"),
                "user_message": message,
            })
        manifest = {
            "cases": case_manifest,
            "model_sha256": model_sha256,
            "result": "PASS",
            "schema": 1,
        }
        (temporary / "manifest.json").write_text(encode_json(manifest), encoding="ascii")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
        print(f"REFERENCE_FIXTURES_PASS output={args.output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"REFERENCE_FIXTURES_FAILED {error}", file=sys.stderr)
        return 53


if __name__ == "__main__":
    raise SystemExit(main())
