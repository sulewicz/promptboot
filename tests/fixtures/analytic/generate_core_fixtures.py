#!/usr/bin/env python3
"""Generate the independently specified RoPE table and expf edge oracle."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import struct

REPO = Path(__file__).resolve().parents[3]
ROPE = REPO / "fixtures/analytic/rope-table.f32le"
EXP = REPO / "fixtures/analytic/expf-edge.u32le"
ROPE_SHA256 = "7fe306d751340de8e5e1d6a44efc5dd34f9513a2917452dd235e7ef4eca14acc"
EXP_SHA256 = "a86e1c3256729d5ca7a94b19a41ca312a3007f18df04d8fce90618378831a58d"
EXP_PAIRS = (
    (0x00000000, 0x3F800000), (0x80000000, 0x3F800000),
    (0x38000000, 0x3F800100), (0xB8000000, 0x3F7FFE00),
    (0x3EB17218, 0x3FB504F3), (0xBEB17218, 0x3F3504F3),
    (0x3F851592, 0x403504F3), (0xBF851592, 0x3EB504F3),
    (0x40000000, 0x40EC7326), (0xC0000000, 0x3E0A9555),
    (0x42AEAC50, 0x7E80001A), (0xC2AEAC50, 0x007FFFE6),
    (0x42B17217, 0x7F7FFF84), (0x42B17218, 0x7F800000),
    (0xC2CFF1B4, 0x00000001), (0xC2CFF1B5, 0x00000000),
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rope_bytes() -> bytes:
    result = bytearray()
    theta = f32(1_000_000.0)
    for position in range(32_768):
        for pair in range(32):
            exponent = f32(f32(2 * pair) / f32(64))
            angle = f32(position / math.pow(theta, exponent))
            result.extend(struct.pack("<f", f32(math.cos(angle))))
            result.extend(struct.pack("<f", f32(math.sin(angle))))
    return bytes(result)


def exp_bytes() -> bytes:
    return b"".join(struct.pack("<II", left, right) for left, right in EXP_PAIRS)


def verify(data: bytes, length: int, digest: str, name: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != length or actual != digest:
        raise RuntimeError(f"{name} mismatch length={len(data)} sha256={actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = (
        (ROPE, rope_bytes(), 8_388_608, ROPE_SHA256),
        (EXP, exp_bytes(), 128, EXP_SHA256),
    )
    for path, data, length, digest in outputs:
        verify(data, length, digest, path.name)
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise RuntimeError(f"CORE_FIXTURE_MISMATCH {path.relative_to(REPO)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f"CORE_FIXTURE_PASS path={path.relative_to(REPO)} bytes={length} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
