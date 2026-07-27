#!/usr/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
revision=$(cargo run --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-tools --target x86_64-unknown-linux-gnu \
    --release --locked --offline --quiet -- asset-value \
    --kind llama_archive --field revision)
source_root="${repo_root}/.cache/llama-reference/source/llama.cpp-${revision}"
build_root="${repo_root}/.cache/llama-reference/build/ggml/src"
fixtures="${repo_root}/fixtures/inference/kernels"
work=$(mktemp -d "${repo_root}/build/inference-oracle.XXXXXX")
cleanup() {
    rm -rf -- "${work}"
}
trap cleanup EXIT

cargo run --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-tools --target x86_64-unknown-linux-gnu \
    --release --locked --offline --quiet -- verify-assets >/dev/null
for path in \
    "${source_root}/CMakeLists.txt" \
    "${source_root}/ggml/include/ggml.h" \
    "${source_root}/ggml/include/ggml-cpu.h" \
    "${source_root}/ggml/src/ggml-cpu/ops.cpp" \
    "${build_root}/libggml-cpu.a" \
    "${build_root}/libggml-base.a"; do
    [[ -f ${path} ]] || {
        printf '%s\n' "INFERENCE_ORACLE_INPUT_MISSING path=${path}" >&2
        exit 65
    }
done

cxx=${CXX:-c++}
for command in "${cxx}" cmp diff grep mkdir nm python3; do
    command -v "${command}" >/dev/null 2>&1 || {
        printf '%s\n' "INFERENCE_ORACLE_TOOL_MISSING command=${command}" >&2
        exit 65
    }
done

compile_commands="${repo_root}/.cache/llama-reference/build/compile_commands.json"
python3 - "${compile_commands}" <<'PY'
import json
import sys

commands = json.load(open(sys.argv[1], encoding="utf-8"))
text = "\n".join(item.get("command", "") for item in commands)
for forbidden in ("-march=native", "-msse4", "-mavx", "-mfma", "-mf16c", "-mbmi", "-mamx"):
    if forbidden in text:
        raise SystemExit(f"INFERENCE_ORACLE_FORBIDDEN_FLAG {forbidden}")
PY

flags=(
    -std=c++17 -O2 -msse2 -mno-avx -ffp-contract=off
    -I"${source_root}/ggml/include"
    -I"${source_root}/ggml/src"
    -I"${source_root}/ggml/src/ggml-cpu"
)
for build in first second; do
    "${cxx}" "${flags[@]}" "${repo_root}/tests/fixtures/inference/inference_kernel_oracle.cpp" \
        "${build_root}/libggml-cpu.a" "${build_root}/libggml-base.a" \
        -lpthread -lm -o "${work}/oracle-${build}"
done
cmp "${work}/oracle-first" "${work}/oracle-second"
if nm -C "${work}/oracle-first" | grep -E 'promptboot|InferenceEngine'; then
    printf '%s\n' "INFERENCE_ORACLE_CANDIDATE_LINK_FAILED" >&2
    exit 66
fi

for run in first second; do
    mkdir "${work}/run-${run}"
    "${work}/oracle-first" "${work}/run-${run}" "${repo_root}/fixtures/inference/rope-table.f32le"
done
diff -qr "${work}/run-first" "${work}/run-second"
diff -qr "${fixtures}" "${work}/run-first" \
    --exclude provenance.json
python3 -I "${repo_root}/tests/fixtures/inference/audit_inference_core.py" \
    --output "${work}/audit.json"
compiler_line=$("${cxx}" --version)
compiler_line=${compiler_line%%$'\n'*}
printf '%s\n' "INFERENCE_ORACLE_PASS revision=${revision} compiler=${compiler_line} mxcsr=0x1f80"
