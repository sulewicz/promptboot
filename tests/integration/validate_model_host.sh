#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"${repo_root}/scripts/dev.sh"

model="${repo_root}/build/model/qwen2.5-0.5b-instruct.pbtqw25"
if [[ ! -f "${model}" ]]; then
    printf '%s\n' "HOST_VALIDATION_PREREQ_MISSING ${model}" >&2
    exit 20
fi
PROMPTBOOT_TEST_MODEL="${model}" cargo test \
    --manifest-path "${repo_root}/Cargo.toml" \
    -p promptboot-core --lib \
    --target x86_64-unknown-linux-gnu --release --locked \
    -- --ignored --test-threads=1
printf '%s\n' "MODEL_HOST_VALIDATION_PASS"
