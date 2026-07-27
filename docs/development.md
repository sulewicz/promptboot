# Development

## Prerequisites

Install Git, Make, Python 3, and [rustup](https://rustup.rs). Building the llama.cpp reference also needs CMake and a C++ compiler. Running promptboot needs `qemu-system-x86_64` and OVMF firmware; KVM is optional.

Run this once after cloning:

```sh
make setup
```

This installs the Rust version selected by `rust-toolchain.toml`, fetches Cargo dependencies, and downloads the Qwen model, its license, and the pinned llama.cpp source archive. These inputs are pinned in `assets.lock.json` and verified before use. `make setup` is the only normal command that downloads anything.

Check an existing asset cache without using the network:

```sh
make verify-assets
```

## Everyday development

Run the Python and Rust library tests:

```sh
make dev
```

Use `make help` for the complete command list.

## Language boundary

Rust owns the target/runtime and the contracts for shipped inputs, artifacts, images, and releases. Python is limited to thin host OS, process, and device orchestration or implementation-independent test drivers, corruptors, and oracles. Make is the documented human workflow.

Model packing and reference generation are normally handled by the release and validation targets. The Make targets expose the native Rust packer and inspector directly when working on the container format:

```sh
make pack-model
make inspect-model

./scripts/build_model_reference.sh --clean
./tests/fixtures/reference/generate_reference_fixtures.py --output build/reference
```

Override `MODEL_OUTPUT` or `INSPECTION_OUTPUT` when using non-default paths. The model source always comes from the verified asset lock.

Choose additional validation based on what changed:

| Change | Command |
| --- | --- |
| Host inference | `make validate-host` |
| Single-core host performance and exact-output gate | `make benchmark-host EVIDENCE_DIR=build/perf-baseline` |
| Model packing, EFI, or FAT image | `make validate-deterministic` |
| Firmware behavior under KVM | `make validate-kvm` |
| Portable emulation | `make validate-tcg` |
| Several of these areas | `make validate-full` |

`make benchmark-host EVIDENCE_DIR=build/perf-baseline` selects the first logical CPU in the caller's affinity mask, pins the complete run to it, and writes `build/perf-baseline-host/run-report.json`. Set `CPU=<logical-id>` to choose another allowed CPU. The command fails before inference if it cannot establish single-CPU affinity.

## Release and play

Create a release from a clean worktree and the verified local cache:

```sh
make release
```

The default output directory is `build/release-<commit>`. Set `RELEASE_DIR=build/<name>` to use another directory below `build/`.

Start the release with KVM and a GTK window:

```sh
make play
```

Use software emulation when KVM is unavailable:

```sh
make play ACCEL=tcg
```

The interactive firmware console requires a graphical session with QEMU's GTK display support. QEMU is found through `PATH`. OVMF is discovered in common system locations. If discovery fails, provide both firmware files explicitly:

```sh
OVMF_CODE=/path/to/OVMF_CODE.fd \
OVMF_VARS=/path/to/OVMF_VARS.fd \
make play
```

## USB boot

First identify the whole USB disk carefully. Then write the selected release:

```sh
make usb DEVICE=/dev/sdX
```

There is no default device. The command rejects mounted or in-use disks, displays the resolved device, and asks you to type its exact path before writing. It verifies the result and ejects the disk.

A conveyed release or USB image must retain the bundled project license, third-party notices, and corresponding source archive.

## Common failures

- Missing Rust tools: install rustup, then run `make setup`.
- Missing or corrupt model inputs: run `make verify-assets`; run `make setup` to fetch missing inputs.
- A release rejects local changes: commit or discard them, then retry.
- KVM is unavailable: use `ACCEL=tcg`.
- OVMF is not found: install your distribution's OVMF package or set `OVMF_CODE` and `OVMF_VARS`.
- GTK cannot open a window: run `make play` from a graphical session with QEMU's GTK display support.
