#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${1:-"${repo_root}/build/deterministic"}
if [[ -e "${root}" ]]; then
    printf '%s\n' "DETERMINISTIC_OUTPUT_EXISTS ${root}" >&2
    exit 22
fi
mkdir -p "${root}"
distribution="${root}/distribution"
mkdir -p "${distribution}/LICENSES"
cp "${repo_root}/LICENSE" "${distribution}/LICENSE"
cp "${repo_root}/THIRD_PARTY_NOTICES.md" "${distribution}/LICENSES/3RDPARTY.TXT"
cp "${repo_root}/LICENSES/QWEN-APACHE-2.0.txt" "${distribution}/LICENSES/QWEN.TXT"
cp "${repo_root}/LICENSES/LLAMA-MIT.txt" "${distribution}/LICENSES/LLAMA.TXT"
cp "${repo_root}/LICENSES/libm-0.2.11.txt" "${distribution}/LICENSES/LIBM.TXT"
cp "${repo_root}/LICENSES/RUST-1.97.1-COPYRIGHT-library.html" "${distribution}/LICENSES/RUSTCORE.HTM"
cp "${repo_root}/LICENSES/compiler-builtins-0.1.160.txt" "${distribution}/LICENSES/RUSTCB.TXT"
printf '%s\n' 'deterministic source archive fixture' > "${distribution}/SOURCE.TGZ"
for side in a b; do
    cargo run --manifest-path "${repo_root}/Cargo.toml" \
        --package promptboot-tools --target x86_64-unknown-linux-gnu \
        --release --locked --offline --quiet -- build-image \
        --output-dir "${root}/${side}" \
        --target-dir "${root}/target-${side}" \
        --distribution-root "${distribution}"
done
for name in BOOTX64.EFI BUILD.JSN MODEL.PBT promptboot.img promptboot-media-inspection.json; do
    cmp "${root}/a/${name}" "${root}/b/${name}"
done
python3 -I "${repo_root}/tests/integration/mutate_model_image.py" \
    --build "${root}/a" \
    --tool "${repo_root}/target/x86_64-unknown-linux-gnu/release/promptboot-tools" \
    --distribution-root "${distribution}" \
    --output "${root}/mutations"
printf '%s\n' "MODEL_DETERMINISTIC_VALIDATION_PASS output=${root}"
