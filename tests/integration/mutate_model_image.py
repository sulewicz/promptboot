#!/usr/bin/env python3
"""Independent black-box corruption checks for the production FAT image."""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def clone(source: Path, destination: Path, *, try_reflink: bool = True) -> None:
    # SEEK_DATA/SEEK_HOLE operate on the file descriptor offset. Buffered file
    # objects maintain a second logical offset and become inconsistent when the
    # descriptor is moved directly.
    with source.open("rb", buffering=0) as input_stream, destination.open(
        "w+b", buffering=0
    ) as output_stream:
        if try_reflink:
            try:
                fcntl.ioctl(output_stream.fileno(), 0x40049409, input_stream.fileno())
                return
            except OSError as error:
                if error.errno not in (
                    errno.EINVAL,
                    errno.ENOTTY,
                    errno.EOPNOTSUPP,
                    errno.EXDEV,
                ):
                    raise
        size = source.stat().st_size
        output_stream.truncate(size)
        offset = 0
        while offset < size:
            try:
                data = os.lseek(input_stream.fileno(), offset, os.SEEK_DATA)
            except OSError as error:
                if error.errno == errno.ENXIO:
                    break
                if error.errno in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                    input_stream.seek(0)
                    output_stream.seek(0)
                    while chunk := input_stream.read(1024 * 1024):
                        if any(chunk):
                            output_stream.write(chunk)
                        else:
                            output_stream.seek(len(chunk), os.SEEK_CUR)
                    return
                raise
            if data >= size:
                break
            try:
                hole = min(
                    os.lseek(input_stream.fileno(), data, os.SEEK_HOLE),
                    size,
                )
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                    raise
                input_stream.seek(data)
                output_stream.seek(data)
                remaining = size - data
                while remaining:
                    chunk = input_stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise OSError("unexpected end of sparse source")
                    if any(chunk):
                        output_stream.write(chunk)
                    else:
                        output_stream.seek(len(chunk), os.SEEK_CUR)
                    remaining -= len(chunk)
                return
            if hole <= data:
                raise OSError("invalid sparse extent")
            input_stream.seek(data)
            output_stream.seek(data)
            remaining = hole - data
            while remaining:
                chunk = input_stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError("unexpected end of sparse source")
                output_stream.write(chunk)
                remaining -= len(chunk)
            offset = hole


def check_cross_filesystem_sparse_copy(root: Path) -> None:
    source = root / "sparse-copy-source"
    with source.open("w+b") as stream:
        stream.truncate(1024 * 1024)
        stream.write(b"start")
        stream.seek(700_000)
        stream.write(b"middle")
    destination_parent = None
    for candidate in (Path("/tmp"), Path("/dev/shm")):
        if candidate.is_dir() and os.stat(candidate).st_dev != os.stat(root).st_dev:
            destination_parent = candidate
            break
    if destination_parent is None:
        raise AssertionError("cross-filesystem sparse-copy destination unavailable")
    with tempfile.TemporaryDirectory(
        prefix="promptboot-sparse-copy.",
        dir=destination_parent,
    ) as temporary:
        destination = Path(temporary) / "copy"
        clone(source, destination, try_reflink=False)
        if source.read_bytes() != destination.read_bytes():
            raise AssertionError("cross-filesystem sparse copy changed bytes")
        if destination.stat().st_blocks * 512 >= destination.stat().st_size // 2:
            raise AssertionError("cross-filesystem sparse copy materialized holes")
    source.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--distribution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    check_cross_filesystem_sparse_copy(args.output)
    source = args.build / "promptboot.img"
    manifest = args.build / "BUILD.JSN"
    bootx64 = args.build / "BOOTX64.EFI"
    model = args.build / "MODEL.PBT"
    for path in (source, manifest, bootx64, model, args.tool):
        if not path.is_file():
            raise SystemExit(f"MUTATOR_PREREQ_MISSING {path}")

    with source.open("rb") as stream:
        bpb = stream.read(512)
        sector = u16(bpb, 11)
        sectors_per_cluster = bpb[13]
        reserved = u16(bpb, 14)
        fat_count = bpb[16]
        root_entries = u16(bpb, 17)
        sectors_per_fat = u16(bpb, 22)
        total_sectors = u32(bpb, 32)
        root = (reserved + fat_count * sectors_per_fat) * sector
        root_sectors = (root_entries * 32 + sector - 1) // sector
        data_start = (reserved + fat_count * sectors_per_fat + root_sectors) * sector

        def entry(offset: int) -> tuple[bytes, int, int, int]:
            stream.seek(offset)
            raw = stream.read(32)
            return raw[:11], raw[11], u16(raw, 26), u32(raw, 28)

        model_entry = entry(root + 64)
        efi_dir = data_start
        boot_dir = data_start + sectors_per_cluster * sector
        efi_entry = entry(boot_dir + 64)
        manifest_entry = entry(boot_dir + 96)
        license_entry = entry(root + 96)

    cluster_bytes = sectors_per_cluster * sector

    def cluster_offset(cluster: int) -> int:
        return data_start + (cluster - 2) * cluster_bytes

    fat1 = reserved * sector
    fat2 = (reserved + sectors_per_fat) * sector
    image_bytes = total_sectors * sector
    manifest_offset = cluster_offset(manifest_entry[2])
    original_manifest = manifest.read_bytes()
    scratch = args.output / "scratch"
    scratch.mkdir()
    changed_image = scratch / "promptboot.img"
    changed_manifest = scratch / "BUILD.JSN"
    (scratch / "BOOTX64.EFI").symlink_to(bootx64.resolve())
    (scratch / "MODEL.PBT").symlink_to(model.resolve())
    passed: list[str] = []

    def rejected(
        name: str,
        edits: list[tuple[int, bytes]],
        expected: str,
        *,
        external_manifest: bytes | None = None,
        truncate: bool = False,
    ) -> None:
        clone(source, changed_image)
        with changed_image.open("r+b") as stream:
            for offset, payload in edits:
                stream.seek(offset)
                before = stream.read(len(payload))
                if before == payload:
                    raise AssertionError(f"{name} mutation is a no-op at {offset}")
                stream.seek(offset)
                stream.write(payload)
            if truncate:
                stream.truncate(image_bytes - 1)
        changed_manifest.write_bytes(external_manifest or original_manifest)
        result = subprocess.run(
            [
                args.tool,
                "inspect-image",
                "--image",
                changed_image,
                "--manifest",
                changed_manifest,
                "--distribution-root",
                args.distribution_root,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0 or "IMAGE_INVALID" not in result.stderr:
            raise AssertionError(f"{name} was accepted: {result.stdout} {result.stderr}")
        if expected not in result.stderr or "Traceback" in result.stderr:
            raise AssertionError(f"{name} wrong failure: {result.stderr}")
        changed_image.unlink()
        changed_manifest.unlink()
        passed.append(name)

    rejected("length", [], "image bytes", truncate=True)
    rejected("bpb", [(11, b"\x01")], "BPB mismatch")
    rejected("fat-copies", [(fat2 + 10, b"\x01")], "FAT copies differ")
    rejected(
        "early-eoc",
        [
            (fat1 + model_entry[2] * 2, b"\xff\xff"),
            (fat2 + model_entry[2] * 2, b"\xff\xff"),
        ],
        "FAT chain mismatch",
    )
    skip = struct.pack("<H", model_entry[2] + 2)
    rejected(
        "nonascending-chain",
        [(fat1 + model_entry[2] * 2, skip), (fat2 + model_entry[2] * 2, skip)],
        "FAT chain mismatch",
    )
    rejected(
        "shared-allocation",
        [(root + 64 + 26, struct.pack("<H", manifest_entry[2]))],
        "allocation",
    )
    rejected(
        "orphan",
        [
            (fat1 + 65_000 * 2, b"\xff\xff"),
            (fat2 + 65_000 * 2, b"\xff\xff"),
        ],
        "orphan FAT entry",
    )
    rejected("root-directory", [(root + 192, b"x")], "root directory slack")
    rejected("efi-directory", [(efi_dir + 96, b"x")], "EFI directory slack")
    rejected("boot-directory", [(boot_dir + 128, b"x")], "BOOT directory slack")
    rejected(
        "file-slack",
        [(cluster_offset(model_entry[2]) + model_entry[3], b"x")],
        "model file slack",
    )
    rejected(
        "efi-payload",
        [(cluster_offset(efi_entry[2]), b"N")],
        "embedded EFI",
    )
    rejected(
        "model-payload",
        [(cluster_offset(model_entry[2]) + 3_980_480, b"9")],
        "embedded MODEL.PBT",
    )
    rejected(
        "distribution",
        [(cluster_offset(license_entry[2]), b"N")],
        "distribution payload mismatch",
    )
    for name, before, after, expected in (
        ("manifest-schema", b'"schema":2', b'"schema":3', "production contract"),
        ("topology", b'"event_topology":"toggle-v1"', b'"event_topology":"toggle-v2"', "production contract"),
        (
            "efi-path",
            b'"path":"EFI/BOOT/BOOTX64.EFI"',
            b'"path":"EFI/BOOT/BOOTX63.EFI"',
            "EFI path mismatch",
        ),
        (
            "artifact-contract",
            b'"artifact_contract":{"format":"PBTQW25-v1"',
            b'"artifact_contract":{"format":"PBTQW25-v2"',
            "artifact_contract mismatch",
        ),
        (
            "fresh-pack",
            b'"fresh_pack":{"format":"PBTQW25-v1"',
            b'"fresh_pack":{"format":"PBTQW25-v2"',
            "fresh_pack mismatch",
        ),
        (
            "model-outer-sha",
            b'"model_outer_sha256":"b0f98ed6',
            b'"model_outer_sha256":"0b073952',
            "model_outer_sha256 mismatch",
        ),
        (
            "prompt-contract",
            b'"prompt_contract":{"bytes":330288,"context":32768,"history":"whole_turn","reserve":1024}',
            b'"prompt_contract":{"bytes":330288,"context":32768,"history":"whole_turn","reserve":1025}',
            "prompt_contract mismatch",
        ),
        (
            "source-model",
            b'"source_model":{"bytes":428730208,',
            b'"source_model":{"bytes":428730209,',
            "source_model mismatch",
        ),
        (
            "toolchain",
            b'"rustc":"rustc 1.97.1 (8bab26f4f 2026-07-14)"',
            b'"rustc":"rustc 1.97.2 (8bab26f4f 2026-07-14)"',
            "toolchain mismatch",
        ),
    ):
        changed = original_manifest.replace(before, after, 1)
        if changed == original_manifest:
            raise AssertionError(f"{name} marker missing")
        rejected(
            name,
            [(manifest_offset, changed)],
            expected,
            external_manifest=changed,
        )
    marker = b'"build_id":"'
    position = original_manifest.index(marker) + len(marker)
    changed = bytearray(original_manifest)
    changed[position] = ord("0") if changed[position] != ord("0") else ord("1")
    rejected(
        "build-id",
        [(manifest_offset, bytes(changed))],
        "build_id derivation",
        external_manifest=bytes(changed),
    )
    (args.output / "results.txt").write_text(
        "PASS sparse-copy-cross-filesystem\n"
        + "".join(f"PASS {name}\n" for name in passed),
        encoding="ascii",
    )
    print(
        "MODEL_IMAGE_MUTATIONS_PASS "
        f"mutations={len(passed)} sparse_copy=cross-filesystem output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
