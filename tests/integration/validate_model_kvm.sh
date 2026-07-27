#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
image=${1:-"${repo_root}/build/release"}
evidence=${2:-"${repo_root}/build/validation-kvm"}
python3 -I "${repo_root}/tests/integration/repl/run_model_repl_qemu.py" \
    --esp "${image}/promptboot.img" --manifest "${image}/BUILD.JSN" \
    --accel kvm --prompt 'Name one color.' --second-prompt 'Name one color.' \
    --third-prompt 'awesome' \
    --event-toggle-scenario --timeout 300 \
    --evidence "${evidence}"
