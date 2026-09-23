# Week 4 — two-dimensional incompressible flow

A Rust crate with `field` and `fluid` binaries solves the periodic vorticity equation using Fourier pseudospectral derivatives, a coordinate-wise two-thirds dealiasing mask and classical RK4. Particles use periodic bilinear interpolation and the same RK4 stages. This directory contains the Week 4 deliverables and only the source files needed to test and regenerate them. The Challenge and separately labelled Extensions are outside the required final checklist.

## Setup from a clean clone

Run from `week4/`. The supplied course checker is kept outside this repository in the Week 4 resources package.

```sh
cargo install --path . --locked
export PATH="$HOME/.cargo/bin:$PATH"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
npx playwright install chromium
cargo test --release --locked
```

`field.design.toml` and `fluid.design.toml` match the supplied references byte-for-byte. `field` writes one JSON object to stdout; `fluid` reads it from stdin. All required physical parameters are explicit. Raw runs go under ignored `artifacts/`; use a clean clone or a fresh output directory for complete regeneration.

## Two simulations

```sh
mkdir -p evidence
field taylor-green --n 64 | fluid --nu 0.1 --dt 0.01 --t-end 1 \
  --every 0.1 --out artifacts/taylor-green | tee evidence/taylor-green.txt
field random --n 128 --seed 2026 --k-min 2 --k-max 6 | fluid \
  --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random \
  | tee evidence/decay.txt
```

The first recording has 11 frames; the second has 101. `run.json` records the parameters and `fields.jsonl` stores `t`, `step`, `u`, `v`, `omega` as row-major arrays with six decimal places. At `t=1`, Taylor–Green energy is `0.167580`. For seed 2026, random energy falls from `0.5` to `0.295129` and enstrophy from `6.634685` to `0.879173`.

After the random run, add the 4000 tracers as the sheet specifies:

```sh
field random --n 128 --seed 2026 --k-min 2 --k-max 6 | fluid \
  --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --tracers 4000 \
  --tracer-seed 7 --tracer-every 2 --out artifacts/random > /dev/null
```

The repeated run verifies that every saved field matches the existing recording before adding particle positions. It preserves the original field and backs up its first `run.json` as `run.before-tracers.json`. Changed parameters or fields are rejected. The particle file has 51 frames and each position is in `[0,2π)`.

## Evidence generation, in order

Run these commands after the two simulations. Each row names **every file committed under `evidence/`**; intermediate JSON/PNG previews stay local.

| Committed evidence file | Command that produces it |
|---|---|
| `taylor-green.txt` | Taylor–Green pipeline above |
| `decay.txt` | Random pipeline above, before adding tracers |
| `physics.txt` | `.venv/bin/python scripts/analyze.py` |
| `tracers.txt` | `.venv/bin/python scripts/histogram.py` |
| `spectrum.html`, `budget.html` | `.venv/bin/python scripts/analyze.py` then `.venv/bin/python scripts/plots.py` |
| `convergence.html` | `.venv/bin/python scripts/refinement.py` then `.venv/bin/python scripts/plots.py` |
| `viewer-taylor-green.png`, `viewer-t0.png`, `viewer-t2.png`, `viewer-t5.png`, `viewer-t10.png`, `viewer-tracers-t10.png` | `node scripts/export_viewer.cjs` after the two viewer recordings below are made |

Before exporting viewer images:

```sh
.venv/bin/python scripts/viewer_copy.py
cp artifacts/random/tracers.jsonl tracers.jsonl
.venv/bin/python scripts/histogram.py
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/refinement.py
.venv/bin/python scripts/plots.py
node scripts/export_viewer.cjs
```

`viewer_copy.py` samples every second point of the actual `N=128` simulation and writes `fields.jsonl` at `64×64` with `t`, `step`, `omega` only; it does **not** run a second `N=64` simulation. `tracers.jsonl` is copied from the 4000 real trajectories. The unmodified supplied viewer is included as `viewer.html` so its Save PNG button can generate the six stamped images locally. The reduced viewer file cannot show energy; use the full `u,v` recording for energy. `scripts/export_viewer.cjs` uses Playwright; its unstaged `viewer-export.json` log and PNG previews are local diagnostics.

`analyze.py` recomputes energy, enstrophy, curl and divergence from the saved velocities. It compares the random endpoint with pure diffusion and integrates the energy budget by the trapezoid rule. `refinement.py` uses the installed `field` and `fluid` binaries for the following `t=2` study: grid `N=32,64,128` at `dt=.01` against `N=256,dt=.0025`; time steps `.02,.0125,.01` at `N=128` against a same-grid `dt=.0025` reference. The initial modes and phases are identical across resolutions. Error is `sqrt(sum((ω−ω_ref)²)/sum(ω_ref²))` at coincident points; the temporal slope is fitted on log-log axes. `plots.py` embeds the measured source rows and generating command in each standalone HTML chart.

The supplied course checker can be run from this directory as `SEED=2026 python /path/to/week4-resources/week4/checker/check .`. It recomputes all physical gates from the raw saved fields. On the specified runs its six checks pass: consistency `2.102412e-5`, divergence `1.349601e-5`, Taylor–Green field `7.038587e-7`, Taylor–Green energy `9.565108e-8`, random energy budget `3.014893e-5`, transfer `0.569218`. Grid error reductions are `4.457×` and `17.112×` (required `≥3×`, `≥10×`); RK4 slope is `4.042427` (required `3.7–4.3`); tracer `χ²/255` is `0.953224` (required `<2`). These measurements use seed 2026 and may change for other seeds.

In the course viewer, inspect `t=0,2,5,10`, the Taylor–Green checkerboard decay and the particle overlay at `t=10`. The six included PNGs were generated using that viewer. The course's student-operated terminal and browser verification remains for the student to perform; generated evidence does not impersonate it.
