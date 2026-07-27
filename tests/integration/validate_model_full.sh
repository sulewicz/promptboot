#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${1:-"${repo_root}/build/full-validation"}
"${repo_root}/tests/integration/validate_model_host.sh"
"${repo_root}/tests/integration/validate_model_deterministic.sh" "${root}"
"${repo_root}/tests/integration/validate_model_kvm.sh" "${root}/a" "${root}/kvm"
"${repo_root}/tests/integration/validate_model_tcg.sh" "${root}/a" "${root}/tcg"
printf '%s\n' "MODEL_FULL_VALIDATION_PASS output=${root}"
