PYTHON ?= python3
SEED ?= 2026

.PHONY: build install test reproduce check
build:
	cargo build --release --locked

install:
	cargo install --path . --force --locked

test:
	cargo fmt --check
	cargo test --release --locked
	cargo clippy --all-targets --locked -- -D warnings

reproduce: install
	SEED=$(SEED) $(PYTHON) scripts/reproduce.py

# Use the independent checker supplied alongside the learning sheet.
check:
	@test -n "$(CHECKER)" || (echo "Pass CHECKER=/path/to/week4/checker/check" >&2; exit 2)
	SEED=$(SEED) $(PYTHON) "$(CHECKER)" .
