#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model="${MODEL_OUTPUT:-${repo_root}/build/model/qwen2.5-0.5b-instruct.pbtqw25}"
evidence="${1:-${repo_root}/build/host-benchmark}"

if [[ ! -f "${model}" ]]; then
    printf '%s\n' "HOST_BENCHMARK_PREREQ_MISSING ${model}" >&2
    exit 20
fi
if ! command -v taskset >/dev/null 2>&1; then
    printf '%s\n' "HOST_BENCHMARK_AFFINITY_UNAVAILABLE taskset" >&2
    exit 21
fi

allowed_list=$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)
selected="${CPU:-${allowed_list%%,*}}"
selected="${selected%%-*}"
if ! [[ "${selected}" =~ ^[0-9]+$ ]] || ! taskset -c "${selected}" true; then
    printf '%s\n' "HOST_BENCHMARK_AFFINITY_REJECTED selected=${selected} allowed=${allowed_list}" >&2
    exit 22
fi

physical_count=$(lscpu -p=SOCKET,CORE | awk -F, '!/^#/ { seen[$1 "," $2]=1 } END { print length(seen) }')
logical_count=$(getconf _NPROCESSORS_CONF)
online_count=$(getconf _NPROCESSORS_ONLN)
affinity_count=$(nproc)

mkdir -p "${evidence}"
PROMPTBOOT_BENCH_CPU="${selected}" \
PROMPTBOOT_HOST_PHYSICAL_COUNT="${physical_count}" \
PROMPTBOOT_HOST_LOGICAL_COUNT="${logical_count}" \
PROMPTBOOT_HOST_ONLINE_COUNT="${online_count}" \
PROMPTBOOT_HOST_AFFINITY_COUNT="${affinity_count}" \
taskset -c "${selected}" cargo run \
    --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-core \
    --example validate_inference \
    --target x86_64-unknown-linux-gnu \
    --release --locked --offline -- \
    "${model}" \
    "${repo_root}/fixtures/reference/model" \
    "${repo_root}/fixtures/inference/exact-host-oracle.txt" \
    "${evidence}"
