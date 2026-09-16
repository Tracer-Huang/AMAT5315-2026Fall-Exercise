# Week 3 — Monte Carlo simulation

Rust simulation of the periodic, zero-field square-lattice Ising model, J=1. This directory contains Parts 1–4 and the two non-neural extensions in the learning sheet. `ising.design.toml` matches the supplied contract section for section.

## Install and reproduce

From a clean clone, with Rust/Cargo, Python 3.14, Node/npm (`npx`) and Chrome installed:

```sh
cd week3
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
cargo install --path . --locked --quiet
bash scripts/reproduce.sh
```

The script builds the release binaries, runs the exact protocols below, regenerates every evidence file, runs tests and checks the raw rows. It requires internet access to download the unmodified course viewer and pinned Playwright CLI 0.1.20. It takes several minutes and writes raw runs locally. Use a clean directory to preserve earlier runs: the course CLI overwrites same-named outputs in its selected folder.

`runs/`, `artifacts/`, Python caches and Rust builds stay local. Each committed file is below 5 MB. `evidence/` contains exactly the twelve requested PNGs plus `peaks.txt` and `errors.txt`, requested in Parts 2 and 3. Auxiliary statistics and verification receipts are generated under ignored `runs/analysis/`.

## Simulation commands and their order

`ising` uses one ChaCha8 random stream per run, carried through the ascending ramp. Each temperature starts from the previous final lattice; only the first starts all-up. One Metropolis sweep is L² uniform site proposals **with replacement**, including rejections. One Wolff step is one cluster flip; every move is measured, irrespective of cluster size. `M` is signed mean spin and `E` is energy per site. `--every` controls only lattice frames, never the measurement frequency.

The following blocks document the exact runs executed by `run_protocol.py`; after `reproduce.sh`, they need not be run again. Use `run_protocol.py` for execution so the timing receipts required by the runtime extension are recorded.

Part 1:

```sh
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/a
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/b
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2027 --out runs/c
diff runs/a/series.jsonl runs/b/series.jsonl
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

Part 2:

```sh
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

Part 4:

```sh
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

`scripts/run_protocol.py` executes all those commands with the release binary and additionally captures `stdout.tsv` and `timing.json` in each run folder. The reproduction script uses it so the runtime extension has measured timings. When coarse and window grids overlap, analysis uses the longer window trajectory; it does not concatenate the two chains. There are 27 distinct temperatures per size.

## Analysis and every evidence file

After all ramps finish:

```sh
.venv/bin/python scripts/analysis.py
.venv/bin/python scripts/viewer.py
```

The first command runs all numerical analyses in dependency order. The viewer script loads the local recording in the unmodified course viewer, exports the three stamped images using its Save PNG button and checks playback advances.

| File in `evidence/` | Command that produces it |
|---|---|
| `boltzmann.png` | `.venv/bin/python scripts/peaks.py` |
| `magnetization.png` | `.venv/bin/python scripts/peaks.py` |
| `susceptibility.png` | `.venv/bin/python scripts/peaks.py` |
| `peaks.txt` | `.venv/bin/python scripts/peaks.py` |
| `trace.png` | `.venv/bin/python scripts/errors.py` |
| `acf-binning.png` | `.venv/bin/python scripts/errors.py` |
| `tau.png` | `.venv/bin/python scripts/errors.py` |
| `errors.txt` | `.venv/bin/python scripts/errors.py` |
| `chi-bootstrap.png` | `.venv/bin/python scripts/analysis.py` |
| `magnetization-compare.png` | `.venv/bin/python scripts/compare.py`, after `analysis.py` |
| `tau-compare.png` | `.venv/bin/python scripts/compare.py`, after `analysis.py` |
| `viewer-T1.8.png` | `.venv/bin/python scripts/viewer.py` |
| `viewer-T2.3.png` | `.venv/bin/python scripts/viewer.py` |
| `viewer-T3.0.png` | `.venv/bin/python scripts/viewer.py` |

### Definitions and uncertainty

Susceptibility follows the sheet: `chi=L²*(mean(M²)-mean(|M|)²)/T`. Fit a quadratic to the five grid points surrounding each maximum and use `Tc=2*Tpeak(64)-Tpeak(32)`. The integrated autocorrelation time is `1/2 + sum(rho(lag))`, with the sheet's self-consistent cutoff `lag > 6*tau`. The 50-block error is computed from block means; it is provisional until a block-length plateau is established.

The circular moving-block bootstrap preserves the original sample count, including a final partial block; sizes and temperatures are resampled separately. It uses 500 replicates at block lengths 2000/4000/8000, seeds 53152026 (Metropolis) and 53152027 (Wolff). The peak neighborhood is selected again in each replicate. Invalid curvature or an out-of-window vertex is counted as a failed fit. The plotted shading is the min/max envelope of valid bootstrap parabolas, not a calibrated confidence band. Errors are called stable only when their range is at most 10% of their mean across the three block lengths.

## Results to check and explain

- T1.8: mean |M|=0.956668, acceptance=0.041506. T3.0: mean |M|=0.045956, acceptance=0.460432. Repeated seed 2026 files are identical; seed 2027 gives 0.957291 and a different series.
- The total-energy histogram log-ratio slope is about 0.01053, compared with `1/3.0-1/3.1=0.0107527`. Short correlated histograms fluctuate; the largest dense-bin residual is about 0.54. This checks relative Boltzmann weights over the overlap, not all configuration probabilities.
- Metropolis: fitted peaks 2.349623 (L32) and 2.310820 (L64), giving Tc=2.272017, about 0.125% from the exact 2.26919. The lowest-temperature means are 0.986233 and 0.986474.
- Metropolis Tc bootstrap SEs are 0.007525/0.007660/0.007229. Wolff gives Tc=2.274424 (0.231% deviation), with SEs 0.000667/0.000635/0.000605. These Tc sampling errors are stable; all recorded peak fits are valid.
- At L64/T2.3, mean |M| is 0.424489 (Metropolis) and 0.435769 (Wolff), separated by 0.588 combined SEs using block=4000. **Agreement is provisional**: the mean-magnetization errors remain block-length sensitive. A small difference alone does not resolve that problem.
- Work-normalized tau is approximately 520.75 for Metropolis and 1.131 for Wolff at T2.3: about 460 times less spin-update work. This is not a wall-clock speedup.

More samples reduce sampling error, but do not remove finite-size corrections or the bias of a five-point parabola. Two sizes cannot test the assumed 1/L scaling. The critical-region short traces and ten frames per temperature are illustrations, not equilibrium estimates. All ten frames at each requested temperature show the expected progression from a dominant domain to large fluctuating regions to small hot domains; individual signed magnetizations can differ substantially.

## Non-neural extensions

Cooling from random spins uses a separate executable, preserving the supplied `ising` interface:

```sh
./target/release/descending 64 2000 5000 2026 runs/descending
.venv/bin/python scripts/descending.py
```

It runs T3.5→1.5 in steps of −0.05, discards 2000 sweeps and measures 5000 at each temperature, and records ten frames per temperature in `runs/descending/spins.jsonl`. The lowest provisionally agreeing temperature is 1.5 for this seed. It does not become trapped in a low-temperature striped state; differences near T2.4/2.45 and block-sensitive errors prevent claiming annealing direction is generally harmless. A descending anneal can encounter metastability, which motivates the sheet's fixed ascending, all-up protocol.

For the run-length extension, the course L64/T2.3 chain has tau≈520.75, so 100000 recorded sweeps represent approximately **96 independent |M| samples**. Keeping tau and variance fixed, matching the present naive error requires about **1.0415e8 measured sweeps**. The original release run measured approximately 10170 sweeps/s including output, giving **about 2.84 hours**, excluding additional warmup. This is an estimate, not an additional simulation; current hardware load can change it. Regenerated timing and effective-sample results are in `runs/analysis/effective-samples.json`.

## Tests and public recording

```sh
cargo test --locked
.venv/bin/python -m pytest tests/test_stats.py tests/test_course.py -q
.venv/bin/python scripts/check.py
```

The checker validates raw row counts, fixed parameters, temperature order, observable bounds, peak estimates and uncertainty labels. It is an independent implementation of the sheet's checks, not an unavailable instructor grading script. Passing its numerical checks does not certify all uncertainties resolved.

[Raw recording](https://raw.githubusercontent.com/Tracer-Huang/AMAT5315-2026Fall-Exercise/main/week3/spins.jsonl). Open the course viewer, load this URL, choose each temperature and use Copy link. The complete links should load in a private window without login:

- [T=1.8](https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2FTracer-Huang%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=1.8)
- [T=2.3](https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2FTracer-Huang%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=2.3)
- [T=3.0](https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2FTracer-Huang%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=3.0)

Automated public-link check, using actual Copy link and fresh browser contexts:

```sh
.venv/bin/python scripts/viewer.py --verify-public https://raw.githubusercontent.com/Tracer-Huang/AMAT5315-2026Fall-Exercise/main/week3/spins.jsonl
```

At the next session, show the committed figures/recording and explain the limitations above. The sheet requires no separate submission portal.
