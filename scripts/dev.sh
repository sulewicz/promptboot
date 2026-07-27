#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for command in cargo rustc python3; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        printf '%s\n' "DEV_PREREQ_MISSING command=${command}; run make setup" >&2
        exit 20
    fi
done

python3 -m unittest discover -s "${repo_root}/tests" -q
cargo test \
    --manifest-path "${repo_root}/Cargo.toml" \
    --workspace --lib \
    --target x86_64-unknown-linux-gnu \
    --locked -q
cargo test \
    --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-tools --test cli \
    --target x86_64-unknown-linux-gnu \
    --locked -q

printf '%s\n' "DEV_CHECK_PASS"
