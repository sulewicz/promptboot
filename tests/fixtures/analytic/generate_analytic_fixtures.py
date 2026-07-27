#!/usr/bin/env python3
"""Independent stdlib-only f32 analytic oracle for all inference primitives."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import hashlib

REPO = Path(__file__).resolve().parents[3]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', f32(value)))[0]:08x}"


def bits_list(values) -> list[str]:
    return [bits(value) for value in values]


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def mul(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def dot(left: list[float], right: list[float]) -> float:
    total = f32(0.0)
    for left_value, right_value in zip(left, right):
        total = add(total, mul(left_value, right_value))
    return total


def q4_block(scale: float, quants: list[int]) -> bytes:
    if len(quants) != 32 or any(value < -8 or value > 7 for value in quants):
        raise ValueError("Q4_0 requires 32 signed nibbles")
    packed = bytearray(struct.pack("<e", scale))
    for index in range(16):
        packed.append((quants[index] + 8) | ((quants[index + 16] + 8) << 4))
    return bytes(packed)


def q4_decode(block: bytes) -> list[float]:
    scale = f32(struct.unpack_from("<e", block, 0)[0])
    values = []
    for byte in block[2:18]:
        values.append(mul(scale, (byte & 0xF) - 8))
    for byte in block[2:18]:
        values.append(mul(scale, (byte >> 4) - 8))
    return values


def q8_block(scale: float, quants: list[int]) -> bytes:
    if len(quants) != 32 or any(value < -128 or value > 127 for value in quants):
        raise ValueError("Q8_0 requires 32 signed bytes")
    return struct.pack("<e", scale) + struct.pack("<32b", *quants)


def q8_decode(block: bytes) -> list[float]:
    scale = f32(struct.unpack_from("<e", block, 0)[0])
    return [mul(scale, value) for value in struct.unpack_from("<32b", block, 2)]


def residual_fixture() -> dict:
    source = [1.25, -2.0, 0.5, 16.0]
    bias = [0.25, 0.75, -1.5, -0.125]
    residual = [-0.5, 1.0, 2.0, -15.0]
    expected = [add(add(x, b), r) for x, b, r in zip(source, bias, residual)]
    return {
        "dimensions": [4],
        "expected_f32le": bits_list(expected),
        "inputs_f32le": {"bias": bits_list(bias), "residual": bits_list(residual), "source": bits_list(source)},
        "name": "f32_bias_residual",
        "operation_order": "out[i]=f32(f32(source[i]+bias[i])+residual[i]); i ascending",
    }


def quant_fixture(kind: str) -> dict:
    if kind == "q4_0":
        first_q = [((index * 5 + 3) % 16) - 8 for index in range(32)]
        second_q = [7 - (index % 16) for index in range(32)]
        first = q4_block(0.5, first_q)
        second = q4_block(-0.25, second_q)
        decode = q4_decode
    else:
        first_q = [((index * 11 + 7) % 63) - 31 for index in range(32)]
        second_q = [31 - ((index * 9) % 63) for index in range(32)]
        first = q8_block(0.125, first_q)
        second = q8_block(-0.0625, second_q)
        decode = q8_decode
    vector = [f32(((index * 7) % 19 - 9) / 8.0) for index in range(32)]
    first_values = decode(first)
    second_values = decode(second)
    return {
        "dimensions": [2, 32],
        "expected_dequant_f32le": [bits_list(first_values), bits_list(second_values)],
        "expected_dot_f32le": bits(dot(first_values, vector)),
        "expected_matvec_f32le": bits_list([dot(first_values, vector), dot(second_values, vector)]),
        "inputs": {"blocks_hex": [first.hex(), second.hex()], "vector_f32le": bits_list(vector)},
        "name": f"{kind}_dequant_dot_matvec",
        "operation_order": "decode block elements in index order; each product rounded f32; accumulator rounded f32 after every ascending-index add",
    }


def rmsnorm_fixture() -> dict:
    source = [1.0, -2.0, 3.0, -4.0]
    weight = [0.5, 1.5, -0.75, 2.0]
    epsilon = f32(1.0e-6)
    squares = f32(0.0)
    for value in source:
        squares = add(squares, mul(value, value))
    mean = f32(squares / len(source))
    inverse = f32(1.0 / math.sqrt(add(mean, epsilon)))
    expected = [mul(mul(value, inverse), scale) for value, scale in zip(source, weight)]
    return {
        "dimensions": [4],
        "expected_f32le": bits_list(expected),
        "inputs_f32le": {"epsilon": bits(epsilon), "source": bits_list(source), "weight": bits_list(weight)},
        "name": "rmsnorm",
        "operation_order": "ascending f32 square-sum; f32 divide by width; f32 add epsilon; correctly-rounded sqrt then f32 reciprocal; f32 source*inverse then f32 *weight",
    }


def rope_fixture() -> dict:
    source = [1.0, 2.0, -3.0, 4.0]
    position = 7
    theta = 1_000_000.0
    output = [f32(0.0)] * 4
    half = 2
    for pair in range(2):
        exponent = f32((2.0 * pair) / 4.0)
        angle = f32(position / math.pow(theta, exponent))
        cosine = f32(math.cos(angle))
        sine = f32(math.sin(angle))
        left, right = source[pair], source[pair + half]
        output[pair] = add(mul(left, cosine), -mul(right, sine))
        output[pair + half] = add(mul(left, sine), mul(right, cosine))
    return {
        "dimensions": [2, 2],
        "expected_f32le": bits_list(output),
        "inputs_f32le": {"source": bits_list(source), "theta": bits(theta)},
        "name": "rope",
        "operation_order": "Qwen/NeoX half-split pairs (i,i+head_dim/2); exponent=f32(2*i/head_dim); angle=f32(position/pow(theta,exponent)); sin/cos rounded f32; products and ordered sums rounded f32",
        "position": position,
    }


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in values]
    total = f32(0.0)
    for value in exponentials:
        total = add(total, value)
    return [f32(value / total) for value in exponentials]


def softmax_fixture() -> dict:
    source = [1.0, -2.0, 3.0, 3.0]
    return {
        "dimensions": [4],
        "expected_f32le": bits_list(softmax(source)),
        "inputs_f32le": {"source": bits_list(source)},
        "name": "softmax",
        "operation_order": "subtract scalar max; exp then round f32; ascending f32 sum; ascending f32 divide",
    }


def attention_fixture() -> dict:
    queries = [[1.0, 0.5], [-0.25, 2.0]]
    keys = [[0.5, -1.0], [1.5, 0.25]]
    values = [[2.0, -0.5], [0.25, 3.0]]
    scale = f32(1.0 / math.sqrt(2.0))
    heads = []
    for query in queries:
        scores = [mul(dot(query, key), scale) for key in keys]
        probabilities = softmax(scores)
        result = []
        for column in range(2):
            total = f32(0.0)
            for row in range(2):
                total = add(total, mul(probabilities[row], values[row][column]))
            result.append(total)
        heads.append({"output_f32le": bits_list(result), "probabilities_f32le": bits_list(probabilities), "scores_f32le": bits_list(scores)})
    return {
        "dimensions": {"head_dim": 2, "kv_heads": 1, "positions": 2, "query_heads": 2},
        "expected_heads": heads,
        "inputs_f32le": {
            "appended_keys": [bits_list(row) for row in keys],
            "appended_values": [bits_list(row) for row in values],
            "queries": [bits_list(row) for row in queries],
        },
        "name": "gqa_attention_kv_append_reuse",
        "operation_order": "append positions 0 then 1 to one KV head; both query heads reuse that cache; ascending f32 dot; f32 scale; stable softmax; ascending-position f32 weighted sum",
    }


def swiglu_fixture() -> dict:
    gate = [-3.0, -0.5, 0.0, 2.0]
    up = [0.25, -2.0, 4.0, 1.5]
    silu = [mul(value, f32(1.0 / (1.0 + math.exp(-value)))) for value in gate]
    expected = [mul(left, right) for left, right in zip(silu, up)]
    return {
        "dimensions": [4],
        "expected_f32le": bits_list(expected),
        "expected_silu_f32le": bits_list(silu),
        "inputs_f32le": {"gate": bits_list(gate), "up": bits_list(up)},
        "name": "silu_swiglu",
        "operation_order": "sigmoid=round_f32(1/(1+exp(-x))); silu=f32(x*sigmoid); swiglu=f32(silu*up); i ascending",
    }


def argmax_fixture() -> dict:
    logits = [1.0, 4.0, 4.0, -2.0, 4.0]
    selected = 0
    for index in range(1, len(logits)):
        if logits[index] > logits[selected]:
            selected = index
    return {
        "dimensions": [5],
        "expected_token_id": selected,
        "inputs_f32le": {"logits": bits_list(logits)},
        "name": "argmax_lowest_id_tie",
        "operation_order": "scan IDs ascending and replace selection only for strictly greater f32 logit",
    }


def generate() -> dict:
    return {
        "fixtures": [
            residual_fixture(), quant_fixture("q4_0"), quant_fixture("q8_0"),
            rmsnorm_fixture(), rope_fixture(), softmax_fixture(), attention_fixture(),
            swiglu_fixture(), argmax_fixture(),
        ],
        "format": "all JSON f32 values are eight lowercase hexadecimal IEEE-754 binary32 bit strings; binary fixture files, where present, are little-endian",
        "generator": "tests/fixtures/analytic/generate_analytic_fixtures.py",
        "oracle": "Python standard library only; no llama.cpp or target-core call",
        "schema": 1,
        "tolerance_max_abs": "0x3727c5ac",
    }


def materialize() -> tuple[dict, bytes]:
    data = generate()
    stream = bytearray()

    def append_expected(value) -> int:
        count = 0
        if isinstance(value, str) and len(value) == 8 and all(character in "0123456789abcdef" for character in value):
            stream.extend(struct.pack("<I", int(value, 16)))
            return 1
        if isinstance(value, list):
            for item in value:
                count += append_expected(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                count += append_expected(value[key])
        return count

    for fixture in data["fixtures"]:
        offset = len(stream)
        count = 0
        for key in sorted(fixture):
            if key.startswith("expected") and key != "expected_token_id":
                count += append_expected(fixture[key])
        fixture["expected_stream"] = {"count": count, "offset": offset}
    blob = bytes(stream)
    data["expected_f32le"] = {
        "bytes": len(blob),
        "path": "primitives.f32le",
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    return data, blob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "fixtures/analytic/primitives.json")
    args = parser.parse_args()
    data, expected_blob = materialize()
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="ascii")
    os.replace(temporary, args.output)
    binary_output = args.output.with_name("primitives.f32le")
    binary_temporary = binary_output.with_name(f".{binary_output.name}.{os.getpid()}.tmp")
    binary_temporary.write_bytes(expected_blob)
    os.replace(binary_temporary, binary_output)
    print(f"ANALYTIC_FIXTURES_WRITTEN path={args.output} bytes={len(encoded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
