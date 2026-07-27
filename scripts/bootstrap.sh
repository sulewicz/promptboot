#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if ! command -v rustup >/dev/null 2>&1; then
    printf '%s\n' "SETUP_PREREQ_MISSING rustup; install rustup from https://rustup.rs" >&2
    exit 20
fi

rustup toolchain install 1.97.1 --profile minimal --target x86_64-unknown-uefi
(
    cd "${repo_root}"
    cargo fetch --locked
)

printf '%s\n' "SETUP_RUST_OK $(
    cd "${repo_root}"
    rustc --version
)"
