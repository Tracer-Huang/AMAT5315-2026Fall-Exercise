# Week 5 — Automatic differentiation and checkpointing

Local implementation and evidence for the 2026-09-30 learning sheet. The three strict-audit plotting/printed-evidence findings are fixed and regression-tested; all numerical gates pass. See the latest section of `AUDIT.md`. **Submission version authorized by the user on 2026-09-30.** Open `review.html` for the Chinese visual overview and `STUDY_NOTES.zh.md` for explanations. `AUDIT.md` distinguishes completed local work from student/account actions.

## What was built

- Hand-written node-wise forward and reverse JAX passes for Lennard-Jones energy, float64, with analytic and finite-difference comparisons and actual recorded JAX graphs.
- A Rust `seismic` executable with `forward`, `born`, and `adjoint` modes. The five-point damped acoustic update is differentiated by real Enzyme timestep JVP/VJP calls in an isolated `no_std` kernel. There is no handwritten production tangent or adjoint.
- Full-history reversal and the handout's exact recursive Treeverse schedule. The initial state has its own permanent slot; `--checkpoints 5` permits six saved states. Complete state = previous pressure + current pressure. Replay never adds to the image; only `grad` does.
- Reflector and Marmousi runs, original course checker, action audits, PNG evidence and local viewer exports. Saved-state byte counts exclude workspace, adjoints, model, data and Enzyme scratch.

## Setup and build

Run commands from `week5/`. Python 3.14 and the pinned requirements are used here:

```sh
uv sync --locked
./scripts/cargo.sh build --release --locked
./scripts/cargo.sh install --path . --locked
```

If `uv` is unavailable, bootstrap a local environment with `python3 -m venv .venv`, then `.venv/bin/python -m pip install uv` and `.venv/bin/uv sync --locked`. Alternatively `pip install -r requirements.lock.txt` reproduces the measured Python environment. `pyproject.toml` and `uv.lock` are the course-requested dependency records.

Normal Rust setup: `rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme`. This Mac did not have rustup, so the official rustc, cargo, rust-std and enzyme-preview archives were SHA-256 verified and installed **only under `.toolchains/nightly-2026-09-05/`**. `scripts/cargo.sh` detects that local installation and supplies its Rust compiler and dynamic-library path; otherwise it uses rustup's pinned toolchain. Global Homebrew Rust was not changed. Official archive URLs/hashes are recorded in `artifacts/provenance.json`.

Enzyme's flag is set only in `build.rs` for `src/kernel.rs`; not globally. `--enzyme-smoke` prints `[8.0, 12.0, 12.0]`. Compiling the kernel without `-Zautodiff=Enable` fails as expected (negative-control log retained).

## Inputs and original materials

Original PDFs, resource/input ZIPs, extracted resources, viewer URL and readings are at `/Users/joshua/Downloads/amat5315/week5/`, matching the Week 4 organization. The inputs are local and ignored by Git. To recreate them:

```sh
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip week5-inputs.zip
```

Restore `MARMOUSI-LICENSE` from the resource archive if needed. Compare each required input's SHA-256 with `artifacts/provenance.json` before reproducing; the copied reflector and Marmousi inputs were verified byte-identical to the dedicated input archive. `seismic.design.toml` and `vendor/viewer.html` are unchanged course versions.

## Run the experiments

```sh
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$b" --out "artifacts/checkpoint-$b"
done
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image
make figures
make test
make check
```

Existing nonempty simulation output directories are rejected. For a complete additional numerical run without altering this evidence, use `.venv/bin/python scripts/reproduce.py --artifacts /tmp/week5-new-run` with a new empty destination. This performs the numerical runs and plots; the two course-viewer PNG exports are the explicit UI step below.

## View wavefields and export the required frames

Run `make viewer`, then open:

- `http://127.0.0.1:8765/vendor/viewer.html?src=/artifacts/forward/wavefield.npy`
- `http://127.0.0.1:8765/vendor/viewer.html?src=/artifacts/adjoint/wavefield.npy`

Alternatively open `vendor/viewer.html` directly and load `wavefield.npy` plus `run.json` together. No upload is needed. Select forward frame 48/80 (step 144, 2.88 s) or adjoint frame 36/80 (step 132, 2.64 s), click **Save PNG**, and save to each run's `wavefield.png`. These operations and both playback directions were verified in the actual viewer.

## Measured acceptance results

| Quantity | Actual | Requirement |
|---|---:|---:|
| Forward AD maximum error | 2.1316e-14 | <1e-12 |
| Reverse AD maximum error | 1.4211e-14 | <1e-12 |
| Finite difference maximum error | 4.9499e-9 | Greater than both AD errors |
| All forward traces L2 | 11.5747695036 | Relative error to 11.574770 <1e-4 |
| Pulse displacement | 1.272792206 km | Within 0.141421 km of 1.296 km |
| Transpose relative error | 1.9912e-16 | <1e-9 |
| Reflector depth peak | 2.1 km | Within 0.1 km of 2.1 km |
| Checkpoint image relative L2 errors | 0, 0, 0, 0 | <1e-9 |
| Peak saved states, budgets 1/3/5/10 | 2 / 4 / 6 / 11 | Budget + initial state |
| Forward calls per shot | 28680 / 1695 / 990 / 642 | Published schedule values |
| Marmousi image L2 | 6.70377406038e-4 | Relative error <1e-4 |
| Marmousi peak saved-state bytes | 20,788,320 | Six complete states |

The image is `J^T J m`, not an exact recovery of the input perturbation. Finite bandwidth and survey illumination limit resolution; shallow artifact arcs and weak deep layers remain visible on the fixed amplitude scale. No depth-dependent display gain is used.

## Tests and evidence conventions

Five Rust tests cover the damped update, genuine Enzyme smoke, independent finite-difference/transpose derivatives, checkpoint invariants across 264 small configurations, and published work counts. Seven Python tests cover hand AD, sampled analytic derivatives, required CLI arguments forward reference/no-overwrite behavior, recording-time consistency, measured full-history baselines and required printed metrics. Initial failing tests are preserved with `*-red.txt`; later results show the implemented behavior. Supplemental derivative validation was added after the initial kernel implementation, so the entire test suite is not described as test-first.

`inputs/`, all `.npy`, toolchain, venv and build outputs are ignored. Evidence PNG/JSON/text remains eligible for a later reviewed commit; each such artifact is under 5 MB. The original course checker is preserved in `vendor/course-check` and was executed without weakening thresholds. The inventory below maps every retained evidence file to its producing operation; diagnostic logs are historical receipts, not additional numerical outputs.

## Complete evidence inventory

| File | Producing command or operation |
|---|---|
| `artifacts/ad/derivatives.json` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/errors.json` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/grad-graph.png` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/grad-graph.txt` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/graph.png` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/graph.txt` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad/modes.png` | `.venv/bin/python scripts/ad.py` |
| `artifacts/ad-tests-green.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/ad-tests-red.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/ad.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/adjoint/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/adjoint/image.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/adjoint/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/adjoint/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/adjoint/wavefield.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/adjoint/wavefield.png` | Course viewer: load paired files, select specified step, Save PNG |
| `artifacts/adjoint.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/born/born_data.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/born/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/born/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/born.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/build.txt` | `scripts/cargo.sh build --release --offline` (historical initial log) |
| `artifacts/checkpoint-1/actions-0.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1/actions-1.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1/actions-2.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-1.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/checkpoint-10/actions-0.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10/actions-1.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10/actions-2.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-10.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/checkpoint-3/actions-0.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3/actions-1.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3/actions-2.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-3.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/checkpoint-5/actions-0.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5/actions-1.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5/actions-2.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/checkpoint-5.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/checkpoint-actions.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/checkpoint-audit.json` | `.venv/bin/python scripts/figures.py` |
| `artifacts/checkpoint-work.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/cli-tests-green.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/cli-tests-red.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/course-check.txt` | `.venv/bin/python vendor/course-check .` |
| `artifacts/enzyme-negative-control.txt` | Compile src/kernel.rs with pinned rustc, omitting -Zautodiff=Enable (expected failure) |
| `artifacts/enzyme-smoke.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/enzyme-test-red.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/figure-checks.txt` | `scripts/figures.py` stdout |
| `artifacts/forward/gathers.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/forward/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/forward/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/forward/traces.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/forward/wave-speed.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/forward/wavefield.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/forward/wavefield.png` | Course viewer: load paired files, select specified step, Save PNG |
| `artifacts/forward.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/inputs.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/install.txt` | `scripts/cargo.sh install --path . --locked --offline` |
| `artifacts/marmousi-born/born_data.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-born/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-born/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-born.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/marmousi-checks.json` | `.venv/bin/python scripts/figures.py` |
| `artifacts/marmousi-image/actions-0.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-1.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-2.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-3.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-4.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-5.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-6.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-7.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/actions-8.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/image.npy` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/result.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image/run.json` | Corresponding `seismic` command in Run the experiments |
| `artifacts/marmousi-image.stdout.txt` | Corresponding simulation stdout (`ad.stdout.txt`: scripts/ad.py) |
| `artifacts/marmousi.png` | `.venv/bin/python scripts/figures.py` |
| `artifacts/provenance.json` | Download SHA-256/byte comparison and actual viewer-export receipt |
| `artifacts/python-tests.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/reflector-checks.json` | `.venv/bin/python scripts/figures.py` |
| `artifacts/rust-tests-green.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/rust-tests-red.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/rust-tests.txt` | Named Rust/Python test or smoke command; -red is the preserved preimplementation failure |
| `artifacts/uv-lock.txt` | `uv lock` |

## Strict follow-up audit receipts

- `audit/2026-09-30-strict/course-check.txt`: `.venv/bin/python vendor/course-check .`, re-executed against the preserved outputs.
- `audit/2026-09-30-strict/independent.json`: installed-command reruns, an independent NumPy trajectory, centered trajectory finite differences at h=1e-3/1e-4, and a seed-5315 random-weight transpose check; method and scratch output paths are documented in AUDIT.md.

## Closed strict-audit findings and submission preparation

The three strict-audit findings are closed; see the current verdict at the top of AUDIT.md. Old plots and scripts are preserved under `audit/2026-09-30-remediation/before/`. They are recovery copies, not submission evidence.

| Receipt | Producing command |
|---|---|
| `audit/2026-09-30-remediation/tests-red.txt` | `python -m pytest tests/test_figure_evidence.py -q`, before fixes |
| `audit/2026-09-30-remediation/tests-green.txt` | Same regression tests after fixes |
| `audit/2026-09-30-remediation/python-tests.txt` | `.venv/bin/python -m pytest tests -q` |
| `audit/2026-09-30-remediation/course-check.txt` | `.venv/bin/python vendor/course-check .` |

This submission includes the Week 5 source, tests, scripts, project/lock files, design contract, license, documentation, course helper files and non-NPY artifacts. Exclude `inputs/`, `.npy`, virtual environments, toolchains, build outputs and historical `audit/**/before/` recovery copies. Unrelated Week 4 work is excluded. The user authorized the Week 5 commit and push on 2026-09-30.
