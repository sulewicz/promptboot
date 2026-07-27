#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
asset_value() {
    cargo run --manifest-path "${repo_root}/Cargo.toml" \
        --package promptboot-tools --target x86_64-unknown-linux-gnu \
        --release --locked --offline --quiet -- asset-value \
        --kind "$1" --field "$2"
}
archive=$(asset_value llama_archive path)
work_root="${repo_root}/.cache/llama-reference"
source_parent="${work_root}/source"
revision=$(asset_value llama_archive revision)
source_dir="${source_parent}/llama.cpp-${revision}"
build_dir="${work_root}/build"
binary_dir="${work_root}/bin"
binary="${binary_dir}/reference_extract"
repl_binary="${binary_dir}/repl_reference_extract"
report="${work_root}/reference-build.json"
repl_report="${work_root}/repl-reference-build.json"
mode=build
completed=0

cleanup_failed_build() {
    local status=$?
    if [[ ${completed} -ne 1 ]]; then
        rm -rf -- "${source_parent}" "${build_dir}" "${binary_dir}"
        rm -f -- "${report}" "${repl_report}"
        printf '%s\n' "REFERENCE_BUILD_FAILED exit=${status}" >&2
    fi
}
trap cleanup_failed_build EXIT

if [[ $# -gt 1 ]]; then
    printf '%s\n' "REFERENCE_USAGE expected: $0 [--clean|--prepare-only]" >&2
    exit 50
fi
case ${1:-} in
    ""|--clean) mode=build ;;
    --prepare-only) mode=prepare ;;
    *)
        printf '%s\n' "REFERENCE_USAGE expected: $0 [--clean|--prepare-only]" >&2
        exit 50
        ;;
esac

cargo run --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-tools --target x86_64-unknown-linux-gnu \
    --release --locked --offline --quiet -- verify-assets >/dev/null
# Every invocation discards source and object state before reconstructing from
# the already identity-verified archive. There is no reusable extracted tree.
rm -rf -- "${source_parent}" "${build_dir}" "${binary_dir}"
rm -f -- "${report}" "${repl_report}"
mkdir -p "${source_parent}" "${build_dir}" "${binary_dir}"
tar --extract --gzip --file "${archive}" --directory "${source_parent}" --no-same-owner --no-same-permissions
if [[ ! -f ${source_dir}/CMakeLists.txt || ! -f ${source_dir}/include/llama.h ]]; then
    printf '%s\n' "REFERENCE_SOURCE_IDENTITY_FAILED pinned archive root is incomplete" >&2
    exit 56
fi
if [[ ${mode} == prepare ]]; then
    completed=1
    printf '%s\n' "REFERENCE_SOURCE_PREPARE_PASS source=${source_dir}"
    exit 0
fi

cmake_options=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_FLAGS_RELEASE=-O2\ -DNDEBUG\ -ffp-contract=off
    -DCMAKE_CXX_FLAGS_RELEASE=-O2\ -DNDEBUG\ -ffp-contract=off
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_OPENMP=OFF
)
SOURCE_DATE_EPOCH=315532800 GIT_CEILING_DIRECTORIES="${source_parent}" cmake \
    -S "${source_dir}" -B "${build_dir}" "${cmake_options[@]}"
cmake --build "${build_dir}" --target llama --parallel 1

cxx=${CXX:-c++}
"${cxx}" -std=c++17 -O2 -DNDEBUG -ffp-contract=off \
    -I"${source_dir}/include" -I"${source_dir}/ggml/include" \
    "${repo_root}/tests/fixtures/reference/reference_extract.cpp" \
    -Wl,--start-group \
    "${build_dir}/src/libllama.a" \
    "${build_dir}/ggml/src/libggml.a" \
    "${build_dir}/ggml/src/libggml-cpu.a" \
    "${build_dir}/ggml/src/libggml-base.a" \
    -Wl,--end-group -pthread -ldl -lm \
    -o "${binary}"

"${cxx}" -std=c++17 -O2 -DNDEBUG -ffp-contract=off \
    -I"${source_dir}/include" -I"${source_dir}/ggml/include" \
    "${repo_root}/tests/fixtures/reference/repl_reference_extract.cpp" \
    -Wl,--start-group \
    "${build_dir}/src/libllama.a" \
    "${build_dir}/ggml/src/libggml.a" \
    "${build_dir}/ggml/src/libggml-cpu.a" \
    "${build_dir}/ggml/src/libggml-base.a" \
    -Wl,--end-group -pthread -ldl -lm \
    -o "${repl_binary}"

"${repo_root}/tests/fixtures/reference/audit_reference_build.py" \
    --binary "${binary}" \
    --output "${report}" >/dev/null
"${repo_root}/tests/fixtures/reference/audit_repl_reference_build.py" \
    --binary "${repl_binary}" --output "${repl_report}" >/dev/null
completed=1
printf '%s\n' "REFERENCE_BUILD_PASS binary=${binary} report=${report}"
