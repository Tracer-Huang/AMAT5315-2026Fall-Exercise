#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WEEK3_PYTHON="${WEEK3_PYTHON:-$PWD/.venv/bin/python}"
cargo build --release --locked --bins
cargo test --locked
"$WEEK3_PYTHON" scripts/run_protocol.py
"$WEEK3_PYTHON" scripts/analysis.py
./target/release/descending 64 2000 5000 2026 runs/descending
"$WEEK3_PYTHON" scripts/descending.py
"$WEEK3_PYTHON" scripts/viewer.py
"$WEEK3_PYTHON" -m pytest tests/test_stats.py tests/test_course.py -q
"$WEEK3_PYTHON" scripts/check.py
