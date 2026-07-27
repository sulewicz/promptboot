#!/usr/bin/env python3
"""Locate the QEMU and OVMF inputs needed to boot promptboot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MISSING = 20
MISMATCH = 21

OVMF_PAIRS = (
    # Fedora/RHEL
    (Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"), Path("/usr/share/edk2/ovmf/OVMF_VARS.fd")),
    (Path("/usr/share/edk2/ovmf/OVMF_CODE.secboot.fd"), Path("/usr/share/edk2/ovmf/OVMF_VARS.fd")),
    # Debian/Ubuntu
    (Path("/usr/share/OVMF/OVMF_CODE_4M.fd"), Path("/usr/share/OVMF/OVMF_VARS_4M.fd")),
    (Path("/usr/share/OVMF/OVMF_CODE.fd"), Path("/usr/share/OVMF/OVMF_VARS.fd")),
)


class HostError(Exception):
    def __init__(self, kind: str, detail: str, exit_code: int) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.exit_code = exit_code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str, expected_sha: str | None = None) -> None:
    if not path.is_file():
        raise HostError("HOST_PREREQ_MISSING", f"{label}={path}", MISSING)
    if expected_sha is not None:
        actual = sha256(path)
        if actual != expected_sha:
            raise HostError(
                "HOST_PREREQ_MISMATCH",
                f"{label}={path} expected={expected_sha} actual={actual}",
                MISMATCH,
            )


def discover_firmware(
    code_path: Path | None = None,
    vars_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    candidates: tuple[tuple[Path, Path], ...] = OVMF_PAIRS,
) -> tuple[Path, Path]:
    environment = os.environ if environ is None else environ
    code_override = code_path or (
        Path(environment["OVMF_CODE"]) if environment.get("OVMF_CODE") else None
    )
    vars_override = vars_path or (
        Path(environment["OVMF_VARS"]) if environment.get("OVMF_VARS") else None
    )
    if (code_override is None) != (vars_override is None):
        raise HostError(
            "HOST_PREREQ_MISSING",
            "OVMF_CODE and OVMF_VARS must be provided together",
            MISSING,
        )
    if code_override is not None and vars_override is not None:
        require_file(code_override, "ovmf_code")
        require_file(vars_override, "ovmf_vars")
        return code_override.resolve(), vars_override.resolve()
    for code, variables in candidates:
        if code.is_file() and variables.is_file():
            return code.resolve(), variables.resolve()
    searched = ", ".join(f"{code} + {variables}" for code, variables in candidates)
    raise HostError(
        "HOST_PREREQ_MISSING",
        f"OVMF firmware not found; set OVMF_CODE and OVMF_VARS (searched {searched})",
        MISSING,
    )


def check(
    _repo: Path,
    *,
    firmware_only: bool = False,
    code_path: Path | None = None,
    vars_path: Path | None = None,
    code_sha: str | None = None,
    vars_sha: str | None = None,
) -> dict[str, str]:
    code, variables = discover_firmware(code_path, vars_path)
    require_file(code, "ovmf_code", code_sha)
    require_file(variables, "ovmf_vars", vars_sha)
    result = {
        "ovmf_code": str(code),
        "ovmf_code_sha256": sha256(code),
        "ovmf_vars": str(variables),
        "ovmf_vars_sha256": sha256(variables),
    }
    if firmware_only:
        return result

    qemu = shutil.which("qemu-system-x86_64")
    if qemu is None:
        raise HostError("HOST_PREREQ_MISSING", "command=qemu-system-x86_64", MISSING)
    try:
        version = subprocess.run(
            [qemu, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise HostError("HOST_PREREQ_MISMATCH", f"qemu unusable: {error}", MISMATCH) from error
    result["qemu"] = qemu
    result["qemu_version"] = version
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--firmware-only", action="store_true")
    parser.add_argument("--code", type=Path)
    parser.add_argument("--vars", dest="vars_path", type=Path)
    parser.add_argument("--code-sha")
    parser.add_argument("--vars-sha")
    arguments = parser.parse_args()
    try:
        result = check(
            arguments.repo.resolve(),
            firmware_only=arguments.firmware_only,
            code_path=arguments.code,
            vars_path=arguments.vars_path,
            code_sha=arguments.code_sha,
            vars_sha=arguments.vars_sha,
        )
    except HostError as error:
        print(f"{error.kind} {error.detail}", file=sys.stderr)
        return error.exit_code
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
