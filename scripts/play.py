#!/usr/bin/env python3
"""Launch the verified production REPL interactively in QEMU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from host_check import HostError, check as check_host
from qmp_session import QmpSession

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


class StopSignal(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def request_stop(signum: int, _frame: object) -> None:
    raise StopSignal(signum)


def stop_qemu(process: subprocess.Popen, qmp_path: Path, trace_path: Path) -> None:
    if process.poll() is not None:
        return
    session = None
    try:
        deadline = time.monotonic() + 2
        session = QmpSession.connect(qmp_path, deadline)
        session.command("quit", deadline)
        trace_path.write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in session.trace
            ),
            encoding="utf-8",
        )
    except (OSError, TimeoutError, ValueError):
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    finally:
        if session is not None:
            session.close()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--accel", choices=("kvm", "tcg"), default="kvm")
    args = parser.parse_args()

    try:
        release = ensure_release(args.release)
        host = check_host(REPO)
        if args.accel == "kvm" and not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise ValueError("KVM requested but /dev/kvm is unavailable; use ACCEL=tcg")
    except (HostError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PLAY_FAILED {error}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = REPO / "build/play" / f"{stamp}-{os.getpid()}"
    evidence.mkdir(parents=True)
    serial = evidence / "com1.log"
    stderr_path = evidence / "qemu-stderr.log"
    command_path = evidence / "qemu-command.json"
    outcome_path = evidence / "outcome.json"
    serial.write_bytes(b"")
    interrupted = False
    stop_signal = None
    status = 1

    with tempfile.TemporaryDirectory(prefix="promptboot-play.") as temporary:
        runtime = Path(temporary)
        vars_copy = runtime / "OVMF_VARS.fd"
        qmp_path = runtime / "qmp.sock"
        shutil.copyfile(host["ovmf_vars"], vars_copy)
        qemu = shutil.which("qemu-system-x86_64")
        if qemu is None:
            print("PLAY_FAILED qemu-system-x86_64 is unavailable", file=sys.stderr)
            return 1
        machine = "q35,kernel_irqchip=split" if args.accel == "kvm" else "q35"
        cpu = "host,invtsc=on" if args.accel == "kvm" else "max"
        command = [
            qemu,
            "-name", "promptboot",
            "-machine", machine,
            "-accel", args.accel,
            "-cpu", cpu,
            "-m", "2048",
            "-smp", "1",
            "-nodefaults",
            "-no-reboot",
            "-nic", "none",
            "-display", "gtk",
            "-monitor", "none",
            "-device", "VGA",
            "-drive", f"if=pflash,unit=0,format=raw,readonly=on,file={host['ovmf_code']}",
            "-drive", f"if=pflash,unit=1,format=raw,file={vars_copy}",
            "-drive", f"if=ide,format=raw,snapshot=on,file={release / 'promptboot.img'}",
            "-chardev", f"file,id=com1,path={serial}",
            "-device", "isa-serial,chardev=com1",
            "-qmp", f"unix:{qmp_path},server=on,wait=off",
            "-boot", "order=c,menu=off,strict=on",
        ]
        command_path.write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"PLAY_START release={release} "
            f"accel={args.accel} evidence={evidence}"
        )
        with stderr_path.open("wb") as stderr:
            previous_handlers = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGTERM, signal.SIGHUP)
            }
            process = None
            try:
                for signum in previous_handlers:
                    signal.signal(signum, request_stop)
                process = subprocess.Popen(command, stderr=stderr)
                status = process.wait()
            except KeyboardInterrupt:
                interrupted = True
            except StopSignal as requested:
                stop_signal = requested.signum
            finally:
                for signum in previous_handlers:
                    signal.signal(signum, signal.SIG_IGN)
                if process is not None:
                    stop_qemu(
                        process,
                        qmp_path,
                        evidence / "qmp-shutdown.jsonl",
                    )
                    status = process.returncode

    try:
        outcome_path.write_text(
            json.dumps(
                {
                    "accel": args.accel,
                    "interrupted": interrupted,
                    "qemu_exit_status": status,
                    "release": str(release),
                    "termination_signal": (
                        signal.Signals(stop_signal).name
                        if stop_signal is not None
                        else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if interrupted:
        print(f"PLAY_STOPPED evidence={evidence}")
        return 0
    if stop_signal is not None:
        name = signal.Signals(stop_signal).name
        print(f"PLAY_STOPPED signal={name} evidence={evidence}")
        return 128 + stop_signal
    if status != 0:
        print(
            f"PLAY_FAILED qemu_exit_status={status} evidence={evidence}",
            file=sys.stderr,
        )
        return 1
    print(f"PLAY_OK evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
