.PHONY: reproduce figures setup

reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts

figures:
	cargo run --manifest-path md/Cargo.toml --example field > field.csv
	.venv/bin/python scripts/plot_field.py field.csv --out field.png
	cargo run --manifest-path md/Cargo.toml --example dimer > dimer.csv
	.venv/bin/python scripts/plot_dimer.py dimer.csv --out dimer.png

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt
