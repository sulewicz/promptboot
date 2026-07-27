HEAD := $(shell git rev-parse --short=12 HEAD)

RELEASE_DIR ?= build/release-$(HEAD)
EVIDENCE_DIR ?= build/validation-$(HEAD)
ACCEL ?= kvm
MODEL_OUTPUT ?= build/model/qwen2.5-0.5b-instruct.pbtqw25
INSPECTION_OUTPUT ?= build/model/inspection.json

.PHONY: help setup verify-assets dev pack-model inspect-model release play validate-host validate-deterministic validate-kvm validate-tcg validate-full usb

help:
	@printf '%s\n' \
	  'promptboot commands' \
	  '' \
	  '  make setup                    Install the Rust toolchain and fetch pinned model assets' \
	  '  make verify-assets             Verify cached model assets without network access' \
	  '  make dev                       Run the normal Python and Rust development checks' \
	  '  make pack-model                Pack the pinned GGUF with the native Rust tool' \
	  '  make inspect-model             Independently inspect the packed model' \
	  '  make release                   Build or verify the current-HEAD release' \
	  '  make play                      Run the interactive production REPL in QEMU' \
	  '  make validate-host             Validate host-side inference' \
	  '  make validate-deterministic    Build and compare two model images' \
	  '  make validate-kvm              Validate the selected release with KVM' \
	  '  make validate-tcg              Validate the selected release with TCG' \
	  '  make validate-full             Run host, deterministic, KVM, and TCG validation' \
	  '  make usb DEVICE=/dev/sdX       Confirm, write, verify, and eject a whole USB disk' \
	  '' \
	  'Variables:' \
	  '  RELEASE_DIR=<path>             Release below build/ (default: build/release-<HEAD>)' \
	  '  EVIDENCE_DIR=<path>            Validation output prefix (default: build/validation-<HEAD>)' \
	  '  ACCEL=kvm|tcg                  QEMU accelerator for play (default: kvm)' \
	  '  DEVICE=/dev/...                Required whole-disk target for usb'

setup:
	@./scripts/bootstrap.sh
	@cargo run --package promptboot-tools --target x86_64-unknown-linux-gnu \
	  --release --locked --offline --quiet -- fetch-assets

verify-assets:
	@cargo run --package promptboot-tools --target x86_64-unknown-linux-gnu \
	  --release --locked --offline --quiet -- verify-assets

dev:
	@./scripts/dev.sh

pack-model:
	@cargo run --package promptboot-tools --target x86_64-unknown-linux-gnu \
	  --release --locked --offline -- pack-model \
	  --output "$(MODEL_OUTPUT)"

inspect-model:
	@cargo run --package promptboot-tools --target x86_64-unknown-linux-gnu \
	  --release --locked --offline -- inspect-model \
	  --model "$(MODEL_OUTPUT)" --output "$(INSPECTION_OUTPUT)"

release:
	@cargo run --package promptboot-tools --target x86_64-unknown-linux-gnu \
	  --release --locked --offline --quiet -- release --output "$(RELEASE_DIR)"

play:
	@./scripts/play.py --release "$(RELEASE_DIR)" --accel "$(ACCEL)"

validate-host:
	@./tests/integration/validate_model_host.sh

validate-deterministic:
	@./tests/integration/validate_model_deterministic.sh "$(EVIDENCE_DIR)-deterministic"

validate-kvm: release
	@./tests/integration/validate_model_kvm.sh "$(RELEASE_DIR)" "$(EVIDENCE_DIR)-kvm"

validate-tcg: release
	@./tests/integration/validate_model_tcg.sh "$(RELEASE_DIR)" "$(EVIDENCE_DIR)-tcg"

validate-full:
	@./tests/integration/validate_model_full.sh "$(EVIDENCE_DIR)-full"

usb:
	@if [ -z "$(strip $(DEVICE))" ]; then \
	  printf '%s\n' 'DEVICE is required; use make usb DEVICE=/dev/sdX' >&2; \
	  exit 2; \
	fi
	@./scripts/write_usb.py --release "$(RELEASE_DIR)" --device "$(DEVICE)"
