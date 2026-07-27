#!/usr/bin/env python3
"""Run the production model_repl image with real QMP keyboard input."""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import re

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from host_check import HostError, check  # noqa: E402
from event_topology import (  # noqa: E402
    POLLING_PREFIX,
    STRICT_FINAL,
    TOGGLE_V1,
    parse_records as parse_topology,
)
from qmp_session import QmpSession  # noqa: E402

PREFIX = b"PROMPTBOOT_EVENT v=1 "

_REQUIRED_FIELD_ROWS = {
    "STARTED": ("event", "build_id", "firmware", "console", "evidence", "uefi_conventional_bytes", "sse2", "watchdog"),
    "BOOT_SMOKE_PASS": ("event", "fp32"),
    "MODEL_LOAD_STARTED": ("event", "mode", "build_id", "path", "model_bytes"),
    "MODEL_REGION": ("event", "phase", "name", "base", "requested", "committed", "current", "high_water", "guard_bytes", "guard"),
    "MODEL_READ_COMPLETE": ("event", "bytes", "max_chunk", "eof_probe_bytes"),
    "MODEL_VERIFIED": ("event", "format", "version", "sha256", "source_sha256", "tensors", "vocab"),
    "MODEL_ARENAS_READY": ("event", "regions", "pages", "committed", "aligned", "nonoverlap", "canaries", "prompt_slack", "index_sha256"),
    "MODEL_REPL_READY": ("event", "mode", "model_sha256", "index_sha256", "context", "max_new_tokens", "prompt_bytes", "history", "timing_class", "sampling", "sampling_seed", "sampling_seed_source", "interrupt_input"),
    "PROMPT_READY": ("event", "prompt_index", "input_limit", "history_turns", "history_tokens", "reserve", "sampling_draws"),
    "INPUT_ACCEPTED": ("event", "prompt_index", "bytes", "accepted_tsc"),
    "INPUT_REJECTED": ("event", "prompt_index", "code", "limit"),
    "HISTORY_RESET": ("event", "prompt_index", "prior_turns", "prior_tokens", "reason"),
    "CONTEXT_REJECTED": ("event", "prompt_index", "user_tokens", "fresh_prompt_tokens", "limit", "reserve", "code"),
    "PROMPT_TOKENIZED": ("event", "prompt_index", "user_tokens", "history_tokens", "prompt_tokens", "reserve", "reset"),
    "GENERATION_STARTED": ("event", "prompt_index", "accepted_tsc", "prompt_tokens", "limit", "cached_tokens", "start_tsc"),
    "TOKEN": ("event", "prompt_index", "token_index", "id", "kind", "piece_bytes", "utf16_units", "infer_start_tsc", "infer_end_tsc", "output_start_tsc", "output_end_tsc"),
    "GENERATION_COMPLETE": ("event", "prompt_index", "reason", "generated", "visible_tokens", "visible_utf16_units", "committed", "history_turns", "history_tokens", "start_tsc", "end_tsc"),
    "GENERATION_FAILED": ("event", "prompt_index", "code", "phase", "partial", "recoverable", "generated", "model_status", "inference_status", "efi_status"),
    "MODEL_CLEANUP_COMPLETE": ("event", "attempted", "ok", "first_error"),
    "MODEL_TARGET_COMPLETE": ("event", "mode", "build_id"),
}
REQUIRED_FIELDS = {
    event: frozenset(fields) for event, fields in _REQUIRED_FIELD_ROWS.items()
}
FATAL_REQUIRED_FIELDS = frozenset(("event", "code"))

UNSHIFTED_QCODES = {
    " ": "spc",
    "'": "apostrophe",
    ",": "comma",
    "-": "minus",
    ".": "dot",
    "/": "slash",
    ";": "semicolon",
    "=": "equal",
    "[": "bracket_left",
    "\\": "backslash",
    "]": "bracket_right",
    "`": "grave_accent",
}

SHIFTED_QCODES = {
    "!": "1",
    '"': "apostrophe",
    "#": "3",
    "$": "4",
    "%": "5",
    "&": "7",
    "(": "9",
    ")": "0",
    "*": "8",
    "+": "equal",
    ":": "semicolon",
    "<": "comma",
    ">": "dot",
    "?": "slash",
    "@": "2",
    "^": "6",
    "_": "minus",
    "{": "bracket_left",
    "|": "backslash",
    "}": "bracket_right",
    "~": "grave_accent",
}


def qcode_for_ascii(character: str) -> tuple[str, bool]:
    if len(character) != 1 or not 0x20 <= ord(character) <= 0x7e:
        raise ValueError(f"not printable ASCII: {character!r}")
    if "a" <= character <= "z" or "0" <= character <= "9":
        return character, False
    if "A" <= character <= "Z":
        return character.lower(), True
    if character in UNSHIFTED_QCODES:
        return UNSHIFTED_QCODES[character], False
    if character in SHIFTED_QCODES:
        return SHIFTED_QCODES[character], True
    raise ValueError(f"unmapped printable ASCII: {character!r}")


def field_items(record: bytes) -> list[tuple[str, str]]:
    if not record.startswith(PREFIX) or not record.endswith(b"\r\n"):
        raise ValueError("malformed event record framing")
    text = record[:-2].decode("ascii")
    result = []
    seen = set()
    for item in text.split(" ")[2:]:
        if "=" not in item:
            raise ValueError(f"record field {item!r}")
        key, value = item.split("=", 1)
        if (
            not key
            or not value
            or not key[0].islower()
            or not all(character.islower() or character.isdigit() or character == "_" for character in key)
        ):
            raise ValueError(f"malformed record field {item!r}")
        if key in seen:
            raise ValueError(f"duplicate field {key}")
        seen.add(key)
        result.append((key, value))
    return result


def fields(record: bytes) -> dict[str, str]:
    return dict(field_items(record))


def record_matches(
    record: bytes,
    event: str,
    **expected_fields: object,
) -> bool:
    values = fields(record)
    return values.get("event") == event and all(
        values.get(key) == str(expected)
        for key, expected in expected_fields.items()
    )


def matching_records(
    records: list[bytes] | tuple[bytes, ...],
    event: str,
    **expected_fields: object,
) -> list[bytes]:
    return [
        record
        for record in records
        if record_matches(record, event, **expected_fields)
    ]


def without_ranges(
    payload: bytes,
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bytes:
    result = bytearray()
    cursor = 0
    for start, end in ranges:
        if start < cursor or end < start or end > len(payload):
            raise ValueError("invalid serial exclusion range")
        result.extend(payload[cursor:start])
        cursor = end
    result.extend(payload[cursor:])
    return bytes(result)


def validate_record_schemas(records: list[bytes]) -> None:
    for index, record in enumerate(records):
        items = field_items(record)
        values = dict(items)
        if "event" not in values:
            raise ValueError(f"missing event field record={index}")
        event = values["event"]
        required = (
            FATAL_REQUIRED_FIELDS if event == "FATAL" else REQUIRED_FIELDS.get(event)
        )
        if required is None:
            raise ValueError(f"unknown event {event}")
        actual = set(values)
        if not required.issubset(actual):
            raise ValueError(
                f"record schema event={event} missing={sorted(required - actual)}"
            )


def human_stream_before(
    payload: bytes,
    records: list[bytes],
    prompt_index: int,
    boundary: bytes | None,
) -> bytes:
    started = next(
        record
        for record in records
        if record_matches(
            record,
            "GENERATION_STARTED",
            prompt_index=prompt_index,
        )
    )
    first = payload.find(started)
    if first < 0:
        raise ValueError(f"generation-start record turn={prompt_index}")
    second = payload.find(started, first + len(started))
    begin = (
        second + len(started)
        if second == first + len(started)
        else first + len(started)
    )
    end = len(payload) if boundary is None else payload.find(boundary, begin)
    if end < begin:
        raise ValueError(f"human stream boundary turn={prompt_index}")
    return payload[begin:end]


def human_stream(payload: bytes, records: list[bytes], prompt_index: int) -> bytes:
    token = next(
        record
        for record in records
        if record_matches(
            record,
            "TOKEN",
            prompt_index=prompt_index,
            token_index=0,
        )
    )
    return human_stream_before(payload, records, prompt_index, token)


def validate_oracle(
    records: list[bytes],
    payload: bytes,
    oracle_path: Path,
    accel: str,
    host_turns: list[dict],
) -> dict:
    oracle = json.loads(oracle_path.read_text(encoding="ascii"))
    turns = oracle["turns"]
    report = []
    for ordinal, expected in enumerate(turns, 1):
        prompt_index = int(expected.get("prompt_index", ordinal))
        selected = [
            fields(record)
            for record in matching_records(
                records, "TOKEN", prompt_index=prompt_index
            )
        ]
        expected_tokens = expected["generated"]
        if len(selected) != len(expected_tokens):
            raise ValueError(f"oracle token count turn={prompt_index}")
        for token_index, (actual, wanted) in enumerate(zip(selected, expected_tokens)):
            if actual["token_index"] != str(token_index) or actual["id"] != str(wanted["id"]) or actual["kind"] != wanted["kind"]:
                raise ValueError(f"oracle token mismatch turn={prompt_index} token={token_index}")
            if int(actual["piece_bytes"]) != len(bytes.fromhex(wanted["piece_hex"])):
                raise ValueError(f"oracle piece length turn={prompt_index} token={token_index}")
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        expected_stream = bytearray()
        for token_index, (actual, wanted) in enumerate(zip(selected, expected_tokens)):
            piece = bytes.fromhex(wanted["piece_hex"])
            if wanted["kind"] == "TEXT":
                expected_stream.extend(piece)
                text = decoder.decode(piece, final=False)
                expected_units = len(text.encode("utf-16-le")) // 2
            else:
                expected_units = 0
            if int(actual["utf16_units"]) != expected_units:
                raise ValueError(f"oracle UTF-16 units turn={prompt_index} token={token_index}")
        decoder.decode(b"", final=True)
        expected_stream.extend(b"\r\n")
        actual_stream = human_stream(payload, records, prompt_index)
        if actual_stream != bytes(expected_stream):
            raise ValueError(f"live human stream turn={prompt_index}")
        tokenized = [
            fields(record)
            for record in matching_records(
                records, "PROMPT_TOKENIZED", prompt_index=prompt_index
            )
        ]
        complete = [
            fields(record)
            for record in matching_records(
                records, "GENERATION_COMPLETE", prompt_index=prompt_index
            )
        ]
        accepted = [
            fields(record)
            for record in matching_records(
                records, "INPUT_ACCEPTED", prompt_index=prompt_index
            )
        ]
        if len(tokenized) != 1 or int(tokenized[0]["prompt_tokens"]) != len(expected["prompt_tokens"]):
            raise ValueError(f"oracle prompt count turn={prompt_index}")
        if len(complete) != 1 or complete[0]["reason"] != expected["reason"] or int(complete[0]["generated"]) != len(expected_tokens):
            raise ValueError(f"oracle termination turn={prompt_index}")
        if len(accepted) != 1:
            raise ValueError(f"accepted count turn={prompt_index}")
        timing_keys = ("infer_start_tsc", "infer_end_tsc", "output_start_tsc", "output_end_tsc")
        values = [[int(item[key], 16) for key in timing_keys] for item in selected]
        if accel == "tcg":
            if any(any(value != 0 for value in row) for row in values):
                raise ValueError("TCG target timings are not canonical zero")
            metrics = {"timing_class": "untimed_noninvariant_tsc"}
        else:
            for at, row in enumerate(values):
                if not (row[0] < row[1] <= row[2] <= row[3]):
                    raise ValueError(f"target timing interval turn={prompt_index} token={at}")
                if at + 1 < len(values) and not row[3] < values[at + 1][0]:
                    raise ValueError(f"target streaming order turn={prompt_index} token={at}")
            first_visible = next((row for row, item in zip(values, selected) if int(item["utf16_units"]) != 0), None)
            if first_visible is None:
                raise ValueError("visible token fixture required")
            accepted_tsc = int(accepted[0]["accepted_tsc"], 16)
            generation = complete[0]
            host_wall = host_turns[prompt_index - 1]["complete_host_monotonic"] - host_turns[prompt_index - 1]["accepted_host_monotonic"]
            metrics = {
                "timing_class": "timed_invariant_tsc",
                "host_complete_seconds": host_wall,
                "first_visible_elapsed_tsc": first_visible[3] - accepted_tsc,
                "target_elapsed_tsc": int(generation["end_tsc"], 16) - accepted_tsc,
            }
        report.append({"prompt_index": prompt_index, "tokens": len(selected), "reason": expected["reason"], **metrics})
    return {"oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(), "turns": report}


def wait_record(
    serial: Path,
    event: str,
    deadline: float,
    process: subprocess.Popen,
    topology: str,
    ignored_ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    **expected_fields: object,
) -> float:
    while time.monotonic() < deadline:
        parsed = parse_topology(
            without_ranges(serial.read_bytes(), ignored_ranges),
            topology,
            POLLING_PREFIX,
        )
        if matching_records(
            parsed.logical_records, event, **expected_fields
        ):
            return time.monotonic()
        if process.poll() is not None:
            raise RuntimeError(f"QEMU exited rc={process.returncode}")
        time.sleep(0.01)
    detail = " ".join(
        f"{key}={value}" for key, value in expected_fields.items()
    )
    raise TimeoutError(f"event={event} {detail}".rstrip())


def wait_model_ready(
    serial: Path,
    deadline: float,
    process: subprocess.Popen,
    topology: str,
) -> float:
    load_started = None
    while time.monotonic() < deadline:
        parsed = parse_topology(serial.read_bytes(), topology, POLLING_PREFIX)
        now = time.monotonic()
        if load_started is None and matching_records(
            parsed.logical_records, "MODEL_LOAD_STARTED"
        ):
            load_started = now
        if load_started is not None and matching_records(
            parsed.logical_records, "MODEL_REPL_READY"
        ):
            return now - load_started
        if process.poll() is not None:
            raise RuntimeError(f"QEMU exited rc={process.returncode}")
        time.sleep(0.01)
    raise TimeoutError("event=MODEL_REPL_READY")


RUN_OUTPUTS = (
    "com1.log",
    "OVMF_VARS.fd",
    "qmp.sock",
    "qemu-command.json",
    "process-tree.txt",
    "failure.txt",
    "firmware-console.ppm",
    "scroll-bottom.ppm",
    "scroll-page-up.ppm",
    "scroll-page-down.ppm",
    "qemu-stderr.log",
    "qmp-transcript.json",
    "outcome.json",
)


def prepare_evidence_directory(path: Path) -> Path:
    evidence = path.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    for name in RUN_OUTPUTS:
        output = evidence / name
        if output.is_dir():
            raise ValueError(f"run output is a directory: {output}")
        output.unlink(missing_ok=True)
    return evidence


def load_bound_inputs(
    esp: Path,
    manifest_path: Path,
    *,
    inspector=None,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("ascii"))
    distribution_root = esp.parent if (esp.parent / "release.json").is_file() else None
    if inspector is None:
        command = [
            "cargo", "run", "--manifest-path", REPO / "Cargo.toml",
            "--package", "promptboot-tools", "--target", "x86_64-unknown-linux-gnu",
            "--release", "--locked", "--offline", "--quiet", "--",
            "inspect-image", "--image", esp, "--manifest", manifest_path,
        ]
        if distribution_root is not None:
            command.extend(("--distribution-root", distribution_root))
        result = subprocess.run(
            command, cwd=REPO, check=True, text=True, stdout=subprocess.PIPE
        )
        image_report = json.loads(result.stdout)
    else:
        image_report = inspector(esp, manifest_path, distribution_root)
    if image_report.get("kind") != "positive":
        raise ValueError("image inspection is not the production positive kind")
    if image_report.get("build_jsn_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("external manifest does not match embedded BUILD.JSN")
    return manifest, image_report


def inject(
    session: QmpSession,
    deadline: float,
    qcode: str,
    shifted: bool = False,
    control: bool = False,
) -> float:
    events = []
    if control:
        events.append({"type": "key", "data": {"down": True, "key": {"type": "qcode", "data": "ctrl"}}})
    if shifted:
        events.append({"type": "key", "data": {"down": True, "key": {"type": "qcode", "data": "shift"}}})
    events.extend([
        {"type": "key", "data": {"down": True, "key": {"type": "qcode", "data": qcode}}},
        {"type": "key", "data": {"down": False, "key": {"type": "qcode", "data": qcode}}},
    ])
    if shifted:
        events.append({"type": "key", "data": {"down": False, "key": {"type": "qcode", "data": "shift"}}})
    if control:
        events.append({"type": "key", "data": {"down": False, "key": {"type": "qcode", "data": "ctrl"}}})
    if "return" not in session.command(
        "input-send-event", deadline, {"events": events}
    ):
        raise RuntimeError(f"key injection failed {qcode}")
    return time.monotonic()


def wait_screen_hash(
    session: QmpSession,
    deadline: float,
    path: Path,
    expected: str,
    equal: bool,
) -> str:
    screen_deadline = min(deadline, time.monotonic() + 5.0)
    while time.monotonic() < screen_deadline:
        if "return" not in session.command(
            "screendump", deadline, {"filename": str(path)}
        ):
            raise RuntimeError("screendump failed")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (digest == expected) == equal:
            return digest
        time.sleep(0.01)
    relation = "match" if equal else "change from"
    raise TimeoutError(f"firmware screen did not {relation} expected state")


def wait_stable_screen(
    session: QmpSession,
    deadline: float,
    path: Path,
) -> str:
    screen_deadline = min(deadline, time.monotonic() + 5.0)
    previous = ""
    while time.monotonic() < screen_deadline:
        if "return" not in session.command(
            "screendump", deadline, {"filename": str(path)}
        ):
            raise RuntimeError("screendump failed")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == previous:
            return digest
        previous = digest
        time.sleep(0.05)
    raise TimeoutError("firmware screen did not settle")


def terminate_qemu(
    session: QmpSession | None,
    process: subprocess.Popen,
    deadline: float,
    shutdown: dict[str, bool],
) -> None:
    quit_error = None
    if process.poll() is None and session is not None and not shutdown["quit_acknowledged"]:
        try:
            session.command("quit", min(deadline, time.monotonic() + 2.0))
        except Exception as error:
            quit_error = error
        else:
            shutdown["quit_acknowledged"] = True

    if process.poll() is None:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"QEMU exited rc={process.returncode}")
    if session is not None and not shutdown["quit_acknowledged"]:
        if quit_error is not None:
            raise RuntimeError("QMP quit was not acknowledged") from quit_error
        raise RuntimeError("QMP quit was not acknowledged")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esp", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accel", choices=("kvm", "tcg"), required=True)
    parser.add_argument("--prompt", default="color")
    parser.add_argument("--second-prompt")
    parser.add_argument("--third-prompt")
    parser.add_argument("--expect-context-rejected", action="append", type=int, default=[])
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--event-toggle-scenario", action="store_true")
    args = parser.parse_args()
    prompts = [args.prompt]
    if args.second_prompt is not None:
        prompts.append(args.second_prompt)
    if args.third_prompt is not None:
        if args.second_prompt is None:
            parser.error("--third-prompt requires --second-prompt")
        prompts.append(args.third_prompt)
    rejected_prompts = set(args.expect_context_rejected)
    if any(index < 1 or index > len(prompts) for index in rejected_prompts):
        parser.error("--expect-context-rejected index is outside supplied prompts")
    if any(not prompt or any(not 0x20 <= ord(character) <= 0x7e for character in prompt) for prompt in prompts):
        parser.error("diagnostic prompts must be nonempty printable ASCII")
    if args.event_toggle_scenario and (len(prompts) != 3 or rejected_prompts):
        parser.error("--event-toggle-scenario requires exactly three real prompts and no rejected prompt")
    try:
        manifest, _image_report = load_bound_inputs(args.esp, args.manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"IMAGE_INPUT_INVALID {error}", file=sys.stderr)
        return 22
    if (
        manifest.get("mode") != "model_repl"
        or manifest.get("artifact_class") != "production"
        or manifest.get("event_topology") != TOGGLE_V1
    ):
        print("IMAGE_INPUT_INVALID model_repl production manifest required", file=sys.stderr)
        return 22
    try:
        host = check(REPO)
    except HostError as error:
        print(f"{error.kind} {error.detail}", file=sys.stderr)
        return error.exit_code
    if args.accel == "kvm" and not os.access("/dev/kvm", os.R_OK | os.W_OK):
        print("HOST_PREREQ_MISSING /dev/kvm", file=sys.stderr)
        return 20
    try:
        evidence = prepare_evidence_directory(args.evidence)
    except (OSError, ValueError) as error:
        print(f"EVIDENCE_PATH_INVALID {error}", file=sys.stderr)
        return 22
    serial = evidence / "com1.log"
    serial.write_bytes(b"")
    vars_copy = evidence / "OVMF_VARS.fd"
    shutil.copyfile(host["ovmf_vars"], vars_copy)
    qmp_path = evidence / "qmp.sock"
    qemu = shutil.which("qemu-system-x86_64")
    if qemu is None:
        return 20
    cpu = "host,invtsc=on" if args.accel == "kvm" else "max"
    machine = "q35,kernel_irqchip=split" if args.accel == "kvm" else "q35"
    argv = [
        "/usr/bin/taskset", "-c", "0", qemu, "-machine", machine, "-accel", args.accel,
        "-cpu", cpu, "-m", "2048", "-smp", "1", "-nodefaults", "-no-reboot", "-nic", "none",
        "-display", "none", "-monitor", "none", "-device", "VGA",
        "-drive", f"if=pflash,unit=0,format=raw,readonly=on,file={host['ovmf_code']}",
        "-drive", f"if=pflash,unit=1,format=raw,file={vars_copy}",
        "-drive", f"if=ide,format=raw,snapshot=on,file={args.esp.resolve()}",
        "-chardev", f"file,id=com1,path={serial}", "-device", "isa-serial,chardev=com1",
        "-qmp", f"unix:{qmp_path},server=on,wait=off", "-boot", "order=c,menu=off,strict=on",
    ]
    (evidence / "qemu-command.json").write_text(json.dumps(argv, indent=2) + "\n", encoding="ascii")
    process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    start = time.monotonic()
    deadline = start + args.timeout
    transcript: list[dict] = []
    session = None
    result = {}
    shutdown = {"quit_acknowledged": False}
    try:
        model_ready_seconds = wait_model_ready(
            serial,
            deadline,
            process,
            manifest["event_topology"],
        )
        session = QmpSession.connect(qmp_path, deadline)
        process_tree = subprocess.check_output(["/usr/bin/ps", "-eo", "pid,ppid,comm,args"], text=True)
        if re.search(r"(^|/)repl_reference_extract(?:\s|$)|llama-(?:cli|server)", process_tree, re.MULTILINE):
            raise ValueError("host inference process coexists with QEMU")
        (evidence / "process-tree.txt").write_text(process_tree, encoding="utf-8")
        turn_results = []
        ignored_serial_ranges: list[tuple[int, int]] = []
        actions = (
            [("help", index, "/help") for index in range(1, 6)]
            + [
                ("toggle", 6, "/events"),
                ("prompt", 7, prompts[0]),
                ("new", 8, "/new"),
                ("prompt", 9, prompts[1]),
                ("prompt", 10, prompts[2]),
                ("interrupt", 11, "What can you do?"),
                ("toggle", 12, "/events"),
            ]
            if args.event_toggle_scenario
            else [("prompt", index, prompt) for index, prompt in enumerate(prompts, 1)]
        )
        event_display_enabled = False
        topology = manifest["event_topology"]
        for action, prompt_index, prompt in actions:
            wait_record(
                serial,
                "PROMPT_READY",
                deadline,
                process,
                topology,
                ignored_serial_ranges,
                prompt_index=prompt_index,
            )
            injections = []
            for character in prompt:
                serial_before = serial.stat().st_size
                qcode, shifted = qcode_for_ascii(character)
                injections.append({
                    "character": character,
                    "monotonic": inject(session, deadline, qcode, shifted),
                })
                while serial.stat().st_size <= serial_before:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"key consumption {character!r}")
                    if process.poll() is not None:
                        raise RuntimeError(f"QEMU exited consuming {character!r}")
                    time.sleep(0.001)
            accepted_host = inject(session, deadline, "ret")
            if action == "interrupt":
                wait_record(
                    serial,
                    "GENERATION_STARTED",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index,
                )
                inject(session, deadline, "c", control=True)
            if action in ("toggle", "help", "new"):
                if action == "toggle":
                    event_display_enabled = not event_display_enabled
                ready_host = wait_record(
                    serial,
                    "PROMPT_READY",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index + 1,
                )
                turn_results.append({
                    "accepted_host_monotonic": accepted_host,
                    "complete_host_monotonic": None,
                    "next_prompt_host_monotonic": ready_host,
                    "outcome": (
                        "HELP"
                        if action == "help"
                        else "NEW_SESSION"
                        if action == "new"
                        else "EVENTS_ON"
                        if event_display_enabled
                        else "EVENTS_OFF"
                    ),
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "terminal_host_monotonic": ready_host,
                    "wall_seconds": ready_host - accepted_host,
                })
                continue
            if prompt_index in rejected_prompts:
                complete_host = None
                terminal_host = wait_record(
                    serial,
                    "CONTEXT_REJECTED",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index,
                )
                outcome = "CONTEXT_REJECTED"
                ready_host = wait_record(
                    serial,
                    "PROMPT_READY",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index + 1,
                )
            else:
                complete_host = wait_record(
                    serial,
                    "GENERATION_COMPLETE",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index,
                )
                terminal_host = complete_host
                outcome = "GENERATION_COMPLETE"
                ready_host = wait_record(
                    serial,
                    "PROMPT_READY",
                    deadline,
                    process,
                    topology,
                    ignored_serial_ranges,
                    prompt_index=prompt_index + 1,
                )
            if args.event_toggle_scenario and prompt_index == 7:
                scroll_bottom = evidence / "scroll-bottom.ppm"
                bottom_hash = wait_stable_screen(
                    session,
                    deadline,
                    scroll_bottom,
                )
                scroll_serial_start = serial.stat().st_size
                inject(session, deadline, "pgup")
                wait_screen_hash(
                    session,
                    deadline,
                    evidence / "scroll-page-up.ppm",
                    bottom_hash,
                    False,
                )
                inject(session, deadline, "pgdn")
                wait_screen_hash(
                    session,
                    deadline,
                    evidence / "scroll-page-down.ppm",
                    bottom_hash,
                    True,
                )
                ignored_serial_ranges.append(
                    (scroll_serial_start, serial.stat().st_size)
                )
            turn_results.append({
                "accepted_host_monotonic": accepted_host,
                "complete_host_monotonic": complete_host,
                "next_prompt_host_monotonic": ready_host,
                "outcome": outcome,
                "prompt": prompt,
                "prompt_index": prompt_index,
                "terminal_host_monotonic": terminal_host,
                "wall_seconds": (ready_host if ready_host is not None else terminal_host) - accepted_host,
            })
        screenshot = evidence / "firmware-console.ppm"
        if "return" not in session.command(
            "screendump", deadline, {"filename": str(screenshot)}
        ):
            raise RuntimeError("screendump failed")
        payload = without_ranges(serial.read_bytes(), ignored_serial_ranges)
        parsed_topology = parse_topology(payload, topology, STRICT_FINAL)
        physical = parsed_topology.physical_records
        records = list(parsed_topology.logical_records)
        validate_record_schemas(records)
        names = [fields(record)["event"] for record in records]
        token_count = sum(name == "TOKEN" for name in names)
        expected_complete = len(prompts) - len(rejected_prompts) + int(args.event_toggle_scenario)
        if token_count == 0 or names.count("GENERATION_COMPLETE") != expected_complete or names.count("CONTEXT_REJECTED") != len(rejected_prompts) or names[-1] != "PROMPT_READY":
            raise ValueError(f"incomplete REPL evidence {names}")
        repl_ready = matching_records(records, "MODEL_REPL_READY")
        if len(repl_ready) != 1:
            raise ValueError("MODEL_REPL_READY count")
        repl_fields = fields(repl_ready[0])
        if (
            repl_fields["sampling"] != "temperature_0p7_top_k_20_top_p_0p8_repetition_penalty_1p1"
            or not re.fullmatch(r"[0-9a-f]{16}", repl_fields["sampling_seed"])
            or repl_fields["sampling_seed"] == "0000000000000000"
            or repl_fields["sampling_seed_source"] not in ("rdrand", "fixed_fallback")
            or repl_fields["interrupt_input"] != "uefi_simple_text_input_ex"
        ):
            raise ValueError("sampling policy evidence")
        tokenized = {
            int(fields(record)["prompt_index"]): fields(record)
            for record in matching_records(records, "PROMPT_TOKENIZED")
        }
        started = {
            int(fields(record)["prompt_index"]): fields(record)
            for record in matching_records(records, "GENERATION_STARTED")
        }
        for prompt_index, started_fields in started.items():
            history_tokens = int(tokenized[prompt_index]["history_tokens"])
            expected_cached = history_tokens - 1 if history_tokens else 0
            if int(started_fields["cached_tokens"]) not in (0, expected_cached):
                raise ValueError(f"generation cache contract prompt={prompt_index}")
        if args.event_toggle_scenario:
            for command_index in (1, 2, 3, 4, 5, 6, 8, 12):
                if any(
                    matching_records(
                        records,
                        event,
                        prompt_index=command_index,
                    )
                    for event in (
                        "INPUT_ACCEPTED",
                        "PROMPT_TOKENIZED",
                        "GENERATION_STARTED",
                        "GENERATION_COMPLETE",
                        "TOKEN",
                    )
                ):
                    raise ValueError(f"local command became model turn {command_index}")
            prompt_fields = {
                int(fields(record)["prompt_index"]): fields(record)
                for record in matching_records(records, "PROMPT_READY")
            }
            sampling_draws = {
                index: int(value["sampling_draws"])
                for index, value in prompt_fields.items()
            }
            sampled_tokens = {
                prompt_index: len(
                    matching_records(records, "TOKEN", prompt_index=prompt_index)
                )
                for prompt_index in (7, 9, 10)
            }
            third_complete = matching_records(
                records,
                "GENERATION_COMPLETE",
                prompt_index=10,
            )
            if len(third_complete) != 1:
                raise ValueError("third prompt completion contract")
            turns_after_third = 1 + int(fields(third_complete[0])["committed"])
            if (
                any(
                    prompt_fields[index]["history_turns"] != "0"
                    for index in range(1, 8)
                )
                or prompt_fields[8]["history_turns"] != "1"
                or prompt_fields[9]["history_turns"] != "0"
                or prompt_fields[10]["history_turns"] != "1"
                or int(prompt_fields[11]["history_turns"]) != turns_after_third
                or int(prompt_fields[12]["history_turns"]) != turns_after_third
                or int(prompt_fields[13]["history_turns"]) != turns_after_third
                or int(started[10]["cached_tokens"])
                != int(tokenized[10]["history_tokens"]) - 1
                or any(sampling_draws[index] != 0 for index in range(1, 8))
                or sampling_draws[8] == 0
                or sampling_draws[8] != sampled_tokens[7]
                or sampling_draws[9] != sampling_draws[8]
                or sampling_draws[10] <= sampling_draws[9]
                or sampling_draws[10] - sampling_draws[9] != sampled_tokens[9]
                or sampling_draws[11] <= sampling_draws[10]
                or sampling_draws[11] - sampling_draws[10] != sampled_tokens[10]
                or sampling_draws[12] not in (sampling_draws[11], sampling_draws[11] + 1)
                or sampling_draws[13] != sampling_draws[12]
                or payload.count(b"events: on\r\n") != 1
                or payload.count(b"events: off\r\n") != 1
                or payload.count(b"new session\r\n") != 1
                or payload.count(b"Ctrl-C stops generation; /help lists commands.\r\n") != 1
                or payload.count(b"^C\r\n") != 1
                or payload.count(
                    b"commands:\r\n"
                    b"/events - toggle structured event display\r\n"
                    b"/help - show this help\r\n"
                    b"/new - clear the session and scrollback\r\n"
                    b"Ctrl-C - stop generation\r\n"
                    b"Page Up/Page Down - scroll output\r\n"
                )
                != 5
            ):
                raise ValueError("toggle status/history contract")
            interrupted = [
                fields(record)
                for record in matching_records(
                    records,
                    "GENERATION_COMPLETE",
                    prompt_index=11,
                )
            ]
            if (
                len(interrupted) != 1
                or interrupted[0]["reason"] != "INTERRUPTED"
                or interrupted[0]["committed"] != "0"
                or int(interrupted[0]["history_turns"]) != turns_after_third
            ):
                raise ValueError("Ctrl-C interruption contract")
        result = {
            "console_redraw_serial_ranges": [
                {"bytes": end - start, "end": end, "start": start}
                for start, end in ignored_serial_ranges
            ],
            "logical_events": names,
            "model_ready_seconds": model_ready_seconds,
            "physical_records": len(physical),
            "token_records": token_count,
            "turns": turn_results,
        }
        if args.oracle is not None:
            result["oracle"] = validate_oracle(
                records,
                payload,
                args.oracle,
                args.accel,
                turn_results,
            )
        terminate_qemu(session, process, deadline, shutdown)
    except Exception as error:
        (evidence / "failure.txt").write_text(str(error) + "\n", encoding="utf-8")
        try:
            terminate_qemu(session, process, deadline, shutdown)
        except Exception:
            pass
        print(f"MODEL_REPL_QEMU_FAILED {error}", file=sys.stderr)
        return 26
    finally:
        if session is not None:
            transcript = session.trace
            session.close()
        stderr = process.stderr.read() if process.stderr else b""
        (evidence / "qemu-stderr.log").write_bytes(stderr)
        (evidence / "qmp-transcript.json").write_text(json.dumps(transcript, sort_keys=True, indent=2) + "\n", encoding="ascii")
    payload = serial.read_bytes()
    if (evidence / "qemu-stderr.log").stat().st_size != 0:
        print("MODEL_REPL_QEMU_FAILED nonempty QEMU stderr", file=sys.stderr)
        return 26
    result.update({
        "accel": args.accel,
        "artifact_class": "production",
        "build_id": manifest["build_id"],
        "esp_sha256": hashlib.sha256(args.esp.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model_sha256": manifest["artifacts"]["model"]["sha256"],
        "serial_bytes": len(payload),
        "serial_sha256": hashlib.sha256(payload).hexdigest(),
        "utc_end": datetime.now(timezone.utc).isoformat(),
    })
    (evidence / "outcome.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="ascii")
    print(f"MODEL_REPL_QEMU_PASS accel={args.accel} tokens={result['token_records']} evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
