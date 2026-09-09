# Week 2: Agentic coding with Rust

This is the same course repository as Week 1. All commands below start in `week2/` unless stated otherwise.

## Current status

Parts 1-4 are implemented and development-tested; 14 release tests pass. The student subsequently delegated the verification/timing checklist back to the agent. Agent-executed command receipts will be recorded honestly; they are not represented as student-run checks. Part 5 baseline measurements are next. See [MANUAL-CHECKS.md](MANUAL-CHECKS.md) for reproducible commands. Nothing marked pending below is claimed to have passed.

## Setup

Rust and Cargo are required. Plotting and video dependencies are isolated:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the versions used for the supplied figures. It is a dependency record, not a physics acceptance result.

## Part 1: Rust ABC

At commit `0e8b72a`, `cargo run` prints `Hello, world!` and the library has one passing test. The executable and test call the same `greeting` function. Development red/green/run transcripts are in `evidence/part1-*.txt`.

The owning struct is responsible for particle arrays. A shared borrow permits reading while a mutable borrow permits exclusive updates. The compiler checks types and borrowing; physical tests must check the update rule.

## Part 2: a tested force

`md::pair::energy` implements U(r)=4(r^-12-r^-6). `md::pair::force` independently implements F(r)=24(2r^-12-r^-6)/r. The force is positive for repulsion. Both functions require positive finite separations.

The well-depth test fixes the scale at U(2^(1/6))=-1. A central-difference energy derivative tests the analytic force at seven distances on both sides of the minimum, with h=1e-5 and tolerance 1e-6*max(1,abs(F)). The derivative is the test's independent calculation, not the force implementation.

Red commit: `9bfa536`; energy implementation: `4ab83a6`; force implementation and all-green pair tests: `06240f6`.

Reproduce `field.png` using the actual Rust functions:

```sh
cargo run --manifest-path md/Cargo.toml --example field > field.csv
.venv/bin/python scripts/plot_field.py field.csv --out field.png
```

![Pair energy and force](field.png)

Arrows point outwards inside r0 and inwards outside; the attractive negative-energy ring surrounds the repulsive core. All arrows are normalized to the same displayed length (0.17 distance units) and show direction only, not force magnitude. White arrows remain visible inside the repulsive core, dark arrows make the weak outer attraction readable, and a small arrow-free band leaves the zero-force circle visible. The energy heatmap uses a 241×241 grid exported by the Rust functions.

## Part 3: from force to motion

`System` owns position, velocity, and acceleration vectors. `total_energy(&self)` only reads them. `Integrator::step(&self, system: &mut System, dt)` borrows the update rule for reading and the particle state exclusively for updates. The generic `advance` driver accepts either `ForwardEuler` or `VelocityVerlet`, so the experiment changes only the integrator value.

Euler updates position from the old velocity and velocity from the old acceleration. Verlet applies a half kick, drift, one force evaluation at the new positions, and a second half kick, retaining the new accelerations. `println!` is a formatting/output macro; `assert!` checks test conditions; `#[test]` is the attribute that registers tests.

For initial positions (0,0),(1.2,0), zero velocities, and dt=0.01, development measurements gave Euler's final relative energy error 1.931776 after 500 steps; Verlet's maximum absolute relative error was 0.000325048. The 5000-step Verlet maximum was 0.000325049, below 1e-3. Small bounded energy oscillations do not mean the numerical trajectory is exact.

```sh
cargo run --manifest-path md/Cargo.toml --example dimer > dimer.csv
.venv/bin/python scripts/plot_dimer.py dimer.csv --out dimer.png
```

![Dimer integrator comparison](dimer.png)

The plot matches the PDF's display sampling: every 2 integration steps in the left panel and every 10 in the right. All 5000 integration steps remain in the CSV and conservation tests. Connecting every step makes the dense periodic waveform look different from the PDF's coarser, aliased display; it is the same trajectory. An optional full-resolution view is reproducible with `scripts/plot_dimer.py dimer.csv --full-resolution --out evidence/dimer-full-resolution.png` using the environment's Python.

The PDF comparison is documented in `evidence/dimer-pdf-comparison.json`. Its right-hand vector path contains 500 points; converting those points back using the printed axes and comparing with the Rust CSV at steps 10,20,...,5000 gives agreement at approximately 1e-9 in relative energy. The caption's “within +/-3e-4” is approximate: the actual full-series bound is 3.2505e-4, while the sheet's required test threshold is 1e-3. No energies were clipped or rescaled to force the caption's rounded number.

The test and example both route through `advance(&impl Integrator, ...)`, with no integrator-specific experimental setup. Raw development test output is in `evidence/part3-green.txt`. Student verification remains separate.

## Part 4: equilibrium CLI

`md run` implements the PDF's default contract: N=100, rho=0.8, T=0.5, dt=0.01, 2000 equilibration steps, 10000 production steps, sample every 50, seed 2026, velocity-Verlet. The initial triangular lattice uses periodic minimum-image separations and the energy-shifted cutoff rc=2.5. The force is unchanged inside the cutoff and zero outside; it is not force-shifted. The isolated dimer retains open boundaries and the plain LJ potential.

Random velocities use the pinned rand_distr Normal sampler and ChaCha8Rng seeded with 2026. They have independent Gaussian components, their centre-of-mass velocity is removed, and velocities are rescaled initially and every 50 equilibration steps using T_thermo=2K/(2N-2). Ordinary production never rescales velocities. Its measured T_speed=<v^2>/2 differs from T_thermo by the finite-size factor (N-1)/N.

```sh
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out fluid.mp4
```

These commands are for the student's independent check. The default trajectory has 200 frames, steps 50 through 10000. run.json records every required parameter; traj.jsonl records wrapped positions, velocities, times and energies with full JSON floating-point precision. Repeated runs preserve previous run.json/traj.jsonl under the output directory's history/ before installing a completed new run. Failed simulations do not replace the active trajectory.

`md check` validates metadata and frame structure, recomputes forces/energy from positions with the naive path, recomputes kinetic energy and speeds from velocities, and only then cross-checks stored energies. It exits nonzero on malformed files or failed physics. The fixed acceptance bounds are the sheet's unheated T=0.5 contract: early/late 10% mean-energy drift <2e-3; abs(T_speed-0.5)<0.05; chi2/22<2 using 24 equal-predicted-probability bins. They are not a general acceptance test for cold or heated trajectories.

Development values: drift 9.82785e-5, T_speed 0.47705724, chi2/22 0.70185455. An independent NumPy calculation reproduced these quantities and found maximum energy discrepancy 5.12e-13. See `evidence/development-review-parts1-4.md` for its scope and the fixed input-validation finding. These are development results, not student-submitted evidence.

`md video` validates the raw trajectory and renders positions beside g(r), using recent 20-frame averages, one output frame per saved frame. The Python helper is embedded in the installed binary. It uses `MD_PYTHON` if set, otherwise the build workspace's `.venv/bin/python`, then python3. The encoder is `MD_FFMPEG`, ffmpeg on PATH, or the imageio-ffmpeg packaged executable. Default videos use 20 fps and enforce the <2 MB bound.

The unchanged supplied viewer is available at `support/week2-viewer.html`; drop both output files into it. The published heating page is pending Part 5.

## Timing

Measured before the cell list, at source commit efda417, three runs per program. Each run includes startup, 2000 equilibration + 10000 production steps, and trajectory output. NumPy 2.5.3 uses the unchanged supplied seed-42 program; Rust follows the PDF seed-2026 contract. These are comparable workloads, not identical random trajectories.

| Program | Median (s) | Range: min-max (s) |
| --- | ---: | ---: |
| Supplied NumPy week2-sim.py (seed 42) | 3.051785 | 3.047359-3.058270 |
| Rust debug (seed 2026) | 1.273987 | 1.255990-1.279784 |
| Rust release, naive (seed 2026) | 0.091495 | 0.091437-0.091592 |

The release median is below one third of debug. On this machine release Rust also beat the supplied NumPy baseline; this is measured here, not assumed universally. Full commands, all nine runs, environment and raw output locations are in `evidence/baseline-timings.json`. Reproduce before changing the force implementation with `python3 scripts/benchmark.py baseline --python /tmp/venv/bin/python --out evidence/baseline-new.json`; compile debug/release first.

## Profile

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 96.3303 | 0.110848 |
| Cell list | pending implementation | pending |

The naive profile contains 109 weighted samples, 105 with `md::periodic::forces` in their stack (inclusive share). This confirms the >90% force hot spot before optimization. Elapsed time is the sampled process lifetime, excluding samply startup. Actual samply data, its symbol sidecar and the reproducible summary are in `evidence/profile-naive*`; the Firefox Profiler screenshot is still pending browser access.

```sh
~/.cargo/bin/samply record --save-only --unstable-presymbolicate --output evidence/profile-naive.json.gz md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
python3 scripts/profile_summary.py evidence/profile-naive.json.gz --symbols evidence/profile-naive.json.syms.json --out evidence/profile-naive-summary.json
~/.cargo/bin/samply load evidence/profile-naive.json.gz
```

## Benchmark

The N=100,400,1600 naive/cells measurements and scaling.png follow the baseline profile and cell-list implementation. No speedup is claimed yet.

## Pages and recording

Pending the 400-atom, 200-frame heating experiment. GitHub Pages and the <=2-minute one-take student narration have not yet been published. The recording will be a GitHub release attachment, not a git-tracked file.

## Design and workflow

The student approved the Part 3-5 design and ordered plan on 2026-09-09. They are under `../docs/superpowers/`. The four repository skills come from [obra/superpowers](https://github.com/obra/superpowers), installed with the Codex skill-installer. All four entrypoints were byte-compared against revision `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`; the upstream MIT license is retained in `.agents/skills/SUPERPOWERS-LICENSE` at repository root. Existing repository and host instructions remain controlling.

## Sources

- Week 2 learning sheet, pages 1-20, supplied by the student.
- [The Rust Programming Language](https://doc.rust-lang.org/book/): sections 4.2, 10.2, 11.1.
- [Course trajectory viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week2-viewer.html).
- Supplied NumPy baseline: `/Users/joshua/Downloads/week2-sim.py`, unchanged, seed 42. The Rust contract requires seed 2026; this seed difference will be disclosed in timing comparisons.
