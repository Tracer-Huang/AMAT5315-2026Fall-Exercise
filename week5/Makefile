PYTHON ?= .venv/bin/python
.PHONY: build test check figures reproduce viewer
build:
	./scripts/cargo.sh build --release --locked --offline
test:
	./scripts/cargo.sh test --release --locked --offline
	$(PYTHON) -m pytest tests -q
check:
	$(PYTHON) vendor/course-check .
figures:
	MPLCONFIGDIR=/tmp/week5-mpl $(PYTHON) scripts/ad.py
	MPLCONFIGDIR=/tmp/week5-mpl $(PYTHON) scripts/figures.py
reproduce:
	$(PYTHON) scripts/reproduce.py
viewer:
	$(PYTHON) -m http.server 8765 --bind 127.0.0.1
