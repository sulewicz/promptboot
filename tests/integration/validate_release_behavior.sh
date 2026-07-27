#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=${1:-"${repo_root}/build/release-behavior"}
failure_output=${output}-failed
if [[ -e ${output} || -e ${failure_output} ]]; then
    printf '%s\n' "RELEASE_BEHAVIOR_OUTPUT_EXISTS output=${output}" >&2
    exit 22
fi
cd "${repo_root}"

cargo build --manifest-path "${repo_root}/Cargo.toml" \
    --package promptboot-tools --target x86_64-unknown-linux-gnu \
    --release --locked --offline --quiet
tool="${repo_root}/target/x86_64-unknown-linux-gnu/release/promptboot-tools"
temporary=$(mktemp -d "${TMPDIR:-/tmp}/promptboot-release-behavior.XXXXXX")
cleanup() {
    rm -rf -- "${temporary}"
}
trap cleanup EXIT
cat > "${temporary}/curl" <<'EOF'
#!/usr/bin/env sh
: > "${CURL_CALLED}"
exit 99
EOF
chmod +x "${temporary}/curl"
export CURL_CALLED="${temporary}/curl-called"
export PATH="${temporary}:${PATH}"

expect_marker() {
    local expected=$1
    shift
    local output_text
    output_text=$("$@")
    if [[ $(printf '%s\n' "${output_text}" | wc -l) -ne 1 ||
          ${output_text} != RELEASE_OK*"${expected}"* ]]; then
        printf '%s\n' "RELEASE_BEHAVIOR_BAD_MARKER ${output_text}" >&2
        exit 1
    fi
}

expect_marker "builds=1 reused=false" \
    "${tool}" release --output "${output}"
expect_marker "builds=0 reused=true" \
    "${tool}" release --output "${output}"
expect_marker "builds=0 reused=true" \
    "${tool}" verify-release --release "${output}"
if [[ -e ${CURL_CALLED} ]]; then
    printf '%s\n' "RELEASE_BEHAVIOR_NETWORK_ATTEMPT" >&2
    exit 1
fi

run_copy="${temporary}/RUN.md"
cp "${output}/RUN.md" "${run_copy}"
printf '%s\n' corrupt > "${output}/RUN.md"
set +e
failure_text=$("${tool}" release --output "${output}" 2>&1)
failure_status=$?
set -e
if [[ ${failure_status} -ne 43 ||
      $(printf '%s\n' "${failure_text}" | wc -l) -ne 1 ||
      ${failure_text} != RELEASE_FAILED\ category=validation* ||
      $(cat "${output}/RUN.md") != corrupt ]]; then
    printf '%s\n' "RELEASE_BEHAVIOR_CORRUPTION_NOT_PRESERVED ${failure_text}" >&2
    exit 1
fi
cp "${run_copy}" "${output}/RUN.md"
expect_marker "builds=0 reused=true" \
    "${tool}" verify-release --release "${output}"

set +e
failure_text=$(CARGO=/bin/false "${tool}" release --output "${failure_output}" 2>&1)
failure_status=$?
set -e
if [[ ${failure_status} -ne 41 ||
      $(printf '%s\n' "${failure_text}" | wc -l) -ne 1 ||
      ${failure_text} != RELEASE_FAILED\ category=build* ||
      -e ${failure_output} ]]; then
    printf '%s\n' "RELEASE_BEHAVIOR_ATOMIC_FAILURE ${failure_text}" >&2
    exit 1
fi
if find "$(dirname "${failure_output}")" -maxdepth 1 \
    -name ".$(basename "${failure_output}").*.tmp" -print -quit | grep -q .; then
    printf '%s\n' "RELEASE_BEHAVIOR_STAGING_LEAK" >&2
    exit 1
fi
if [[ -e ${CURL_CALLED} ]]; then
    printf '%s\n' "RELEASE_BEHAVIOR_NETWORK_ATTEMPT" >&2
    exit 1
fi
printf '%s\n' "RELEASE_BEHAVIOR_PASS output=${output}"
