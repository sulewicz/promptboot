#!/usr/bin/env python3
"""Generate the frozen llama.cpp-compatible Qwen2 inference RoPE table."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import pathlib
import struct
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "fixtures/inference/rope-table.f32le"
EXPECTED_SHA256 = "cd75fcc63f7514055daf75917521bfe4612ce6417419de5ce77ca766473c01c5"
LIBM = pathlib.Path("/usr/lib64/libm.so.6")
LIBM_SHA256 = "ad29671950bd805b801d732c0977864cfc5dd8e72789ae18c33e0423c222dc09"


def libc_math() -> tuple[object, object, object]:
    if not LIBM.is_file() or hashlib.sha256(LIBM.read_bytes()).hexdigest() != LIBM_SHA256:
        raise RuntimeError("pinned libm identity mismatch")
    library = ctypes.CDLL(str(LIBM))
    functions = []
    for name in ("powf", "cosf", "sinf"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_float, ctypes.c_float] if name == "powf" else [ctypes.c_float]
        function.restype = ctypes.c_float
        functions.append(function)
    return tuple(functions)


def table_bytes() -> bytes:
    powf, cosf, sinf = libc_math()
    theta_scale = powf(ctypes.c_float(1_000_000.0), ctypes.c_float(-2.0 / 64.0))
    output = bytearray()
    for position in range(32_768):
        theta = ctypes.c_float(float(position)).value
        for _pair in range(32):
            output.extend(struct.pack("<ff", cosf(theta), sinf(theta)))
            theta = ctypes.c_float(theta * theta_scale).value
    assert len(output) == 8_388_608
    return bytes(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the committed fixture")
    arguments = parser.parse_args()
    generated = table_bytes()
    digest = hashlib.sha256(generated).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"INFERENCE_ROPE_GENERATION_FAIL sha256={digest}", file=sys.stderr)
        return 1
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != generated:
            print("INFERENCE_ROPE_CHECK_FAIL committed fixture differs", file=sys.stderr)
            return 1
        print(f"INFERENCE_ROPE_CHECK_PASS bytes={len(generated)} sha256={digest}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(generated)
    print(f"INFERENCE_ROPE_GENERATE_PASS bytes={len(generated)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
