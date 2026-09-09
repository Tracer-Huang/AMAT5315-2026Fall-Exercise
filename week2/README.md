# Week 2: Agentic coding with Rust

This is the same course repository as Week 1. All commands below start in `week2/` unless stated otherwise.

## Current status

Work is in progress. Development evidence is separate from the student's independent grey-box checks. Nothing marked pending below is claimed to have passed.

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

## Design and workflow

The student approved the Part 3-5 design and ordered plan on 2026-09-09. They are under `../docs/superpowers/`. The four repository skills come from [obra/superpowers](https://github.com/obra/superpowers), installed with the Codex skill-installer. All four entrypoints were byte-compared against revision `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`; the upstream MIT license is retained in `.agents/skills/SUPERPOWERS-LICENSE` at repository root. Existing repository and host instructions remain controlling.

## Sources

- Week 2 learning sheet, pages 1-20, supplied by the student.
- [The Rust Programming Language](https://doc.rust-lang.org/book/): sections 4.2, 10.2, 11.1.
- [Course trajectory viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week2-viewer.html).
- Supplied NumPy baseline: `/Users/joshua/Downloads/week2-sim.py`, unchanged, seed 42. The Rust contract requires seed 2026; this seed difference will be disclosed in timing comparisons.
