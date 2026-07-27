#!/usr/bin/env python3
"""Safely write a verified promptboot release image to a whole USB disk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
TOOLS = [
    "cargo", "run", "--manifest-path", str(REPO / "Cargo.toml"),
    "--package", "promptboot-tools", "--target", "x86_64-unknown-linux-gnu",
    "--release", "--locked", "--offline", "--quiet", "--",
]


def ensure_release(output: Path) -> Path:
    subprocess.run(
        [*TOOLS, "release", "--output", str(output)],
        cwd=REPO, check=True,
    )
    return output if output.is_absolute() else (REPO / output).resolve()


LSBLK_COLUMNS = (
    "NAME,KNAME,PATH,TYPE,RO,MOUNTPOINTS,MODEL,SIZE,TRAN,PKNAME,"
    "SERIAL,WWN,MAJ:MIN"
)


def flatten(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for node in nodes:
        result.append(node)
        children = node.get("children", [])
        if isinstance(children, list):
            result.extend(flatten(children))
    return result


def active_mounts(node: dict[str, object]) -> list[str]:
    mounts = node.get("mountpoints")
    if mounts is None:
        mount = node.get("mountpoint")
        mounts = [] if mount is None else [mount]
    if not isinstance(mounts, list):
        mounts = [mounts]
    return [str(value) for value in mounts if value]


def normalized_text(value: object) -> str:
    return str(value or "").strip()


def normalized_node(node: dict[str, object]) -> dict[str, object]:
    node_path = normalized_text(node.get("path"))
    if not node_path or not Path(node_path).is_absolute():
        raise ValueError("lsblk node is missing an absolute path")
    try:
        size = int(node.get("size", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"lsblk node has invalid size: {node_path}") from error
    if size <= 0:
        raise ValueError(f"lsblk node has invalid size: {node_path}")
    read_only = str(node.get("ro", "")).lower()
    if read_only not in ("0", "1", "false", "true"):
        raise ValueError(f"lsblk node has invalid read-only state: {node_path}")
    major_minor = normalized_text(node.get("maj:min"))
    if not major_minor:
        raise ValueError(f"lsblk node is missing MAJ:MIN: {node_path}")
    return {
        "kname": normalized_text(node.get("kname")),
        "maj:min": major_minor,
        "mountpoints": sorted(set(active_mounts(node))),
        "parent": normalized_text(node.get("pkname")),
        "path": str(Path(node_path).resolve()),
        "read_only": read_only in ("1", "true"),
        "serial": normalized_text(node.get("serial")),
        "size": size,
        "type": normalized_text(node.get("type")).lower(),
        "wwn": normalized_text(node.get("wwn")),
    }


def device_snapshot(
    resolved: Path,
    device: dict[str, object],
    nodes: list[dict[str, object]],
    all_nodes: list[dict[str, object]],
) -> dict[str, object]:
    topology = sorted(
        (normalized_node(node) for node in nodes),
        key=lambda node: str(node["path"]),
    )
    root = normalized_node(device)
    if root not in topology:
        raise ValueError("lsblk topology does not contain the selected root")
    if any(not node["kname"] or not node["type"] for node in topology):
        raise ValueError("lsblk topology has an incomplete node identity")
    if any(node != root and not node["parent"] for node in topology):
        raise ValueError("lsblk topology has an incomplete parent relation")
    serial = str(root["serial"])
    wwn = str(root["wwn"])
    stable_field, stable_value = ("wwn", wwn) if wwn else ("serial", serial)
    if not stable_value:
        raise ValueError("USB device has no stable SERIAL or WWN identity")
    peers = [
        node
        for node in all_nodes
        if normalized_text(node.get("type")).lower() == "disk"
        and normalized_text(node.get(stable_field)) == stable_value
    ]
    if len(peers) != 1:
        raise ValueError(f"ambiguous USB {stable_field.upper()} identity")
    actual_major_minor = (
        f"{os.major(resolved.stat().st_rdev)}:{os.minor(resolved.stat().st_rdev)}"
    )
    if root["maj:min"] != actual_major_minor:
        raise ValueError("lsblk MAJ:MIN does not match the selected device node")
    return {
        "root": {
            **root,
            "model": normalized_text(device.get("model")),
            "transport": normalized_text(device.get("tran")).lower(),
        },
        "stable_identity": {
            "field": stable_field,
            "value": stable_value,
        },
        "topology": topology,
    }


def inspect_device(
    requested: Path,
) -> tuple[Path, dict[str, object], list[dict[str, object]], dict[str, object]]:
    resolved = requested.resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISBLK(mode):
        raise ValueError(f"not a block device: {resolved}")
    result = subprocess.run(
        [
            "lsblk", "--json", "--bytes",
            "--output", LSBLK_COLUMNS,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    roots = payload.get("blockdevices")
    if not isinstance(roots, list):
        raise ValueError("lsblk returned no block-device inventory")
    all_nodes = flatten(roots)
    matches = [
        node for node in all_nodes
        if isinstance(node.get("path"), str)
        and Path(str(node["path"])).resolve() == resolved
    ]
    if len(matches) != 1:
        raise ValueError(f"ambiguous lsblk identity for {resolved}")
    device = matches[0]
    if device.get("type") != "disk" or device.get("pkname"):
        raise ValueError(f"whole-disk device required: {resolved}")
    if str(device.get("tran") or "").lower() != "usb":
        raise ValueError(f"USB transport required: {resolved}")
    if str(device.get("ro", "")).lower() not in ("0", "false"):
        raise ValueError(f"read-only device: {resolved}")
    nodes = flatten([device])
    mounts = [
        f"{node.get('path')}: {mount}"
        for node in nodes
        for mount in active_mounts(node)
    ]
    if mounts:
        raise ValueError("mounted device or descendant: " + ", ".join(mounts))
    swaps = {
        Path(line.split()[0]).resolve()
        for line in Path("/proc/swaps").read_text(encoding="ascii").splitlines()[1:]
        if line.split()
    }
    node_paths = {
        Path(str(node["path"])).resolve()
        for node in nodes
        if isinstance(node.get("path"), str)
    }
    if swaps & node_paths:
        raise ValueError(f"swap is active on {resolved} or a descendant")
    for node in nodes:
        kname = node.get("kname")
        if isinstance(kname, str):
            holders = Path("/sys/class/block") / kname / "holders"
            if holders.is_dir() and any(holders.iterdir()):
                raise ValueError(f"device is in use through {kname} holders")
    snapshot = device_snapshot(resolved, device, nodes, all_nodes)
    return resolved, device, nodes, snapshot


def confirm_device(resolved: Path) -> bool:
    confirmation = input(f"Type the exact resolved path to erase {resolved}: ")
    return confirmation == str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    args = parser.parse_args()
    try:
        release = ensure_release(args.release)
        resolved, device, nodes, snapshot = inspect_device(args.device)
        image = release / "promptboot.img"
        image_size = image.stat().st_size
        device_size = int(device.get("size", 0))
        if image_size > device_size:
            raise ValueError(
                f"image is larger than device: image={image_size} device={device_size}"
            )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"USB_FAILED {error}", file=sys.stderr)
        return 1

    children = [
        f"{node.get('path')} ({node.get('type')}, {node.get('size')} bytes)"
        for node in nodes
        if node is not device
    ]
    print(f"Resolved path: {resolved}")
    print(f"Model: {str(device.get('model') or '').strip() or 'unknown'}")
    print(f"Size: {device_size} bytes")
    print(f"Transport: {device.get('tran') or 'unknown'}")
    print(f"Serial: {device.get('serial') or 'unavailable'}")
    print(f"WWN: {device.get('wwn') or 'unavailable'}")
    print(f"MAJ:MIN: {device.get('maj:min')}")
    print("Children: " + (", ".join(children) if children else "none"))
    print(
        "Topology: "
        + json.dumps(snapshot["topology"], sort_keys=True, separators=(",", ":"))
    )
    print(f"Image: {image} ({image_size} bytes)")
    if not confirm_device(resolved):
        print("USB_ABORTED confirmation did not match")
        return 1

    try:
        subprocess.run(["sudo", "-v"], check=True)
        final_resolved, _final_device, _final_nodes, final_snapshot = inspect_device(
            resolved
        )
        if final_resolved != resolved or final_snapshot != snapshot:
            raise ValueError("device identity or topology changed after confirmation")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"USB_FAILED post-confirmation safety check: {error}", file=sys.stderr)
        return 1

    commands = (
        [
            "sudo", "dd", f"if={image}", f"of={resolved}",
            "bs=4M", "conv=fsync", "status=progress",
        ],
        ["sudo", "cmp", "-n", str(image_size), str(image), str(resolved)],
        ["sudo", "eject", str(resolved)],
    )
    try:
        for command in commands:
            subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"USB_FAILED {error}", file=sys.stderr)
        return 1
    print(f"USB_OK device={resolved} bytes={image_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
