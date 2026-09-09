# Verification and reproduction commands

The student initially chose to run grey-box commands manually after Computer Use refused terminal access, then explicitly delegated this checklist back to the agent ("你替我完成"). The agent now executes available terminal commands and records their actual outputs. These are NOT represented as student-run checks. This file remains the reproducible command guide.

Current scope: Parts 1-5, including corrected field/dimer figures, measured cell-list scaling, and heating/cold/hot data. All 21 release tests and the default physics check passed in a fresh GitHub clone. The actual command receipts and raw measurements are in evidence/. The commands below allow independent repetition; their output will vary with machine and timing noise.

Before any bare `md` command, start in this checkout's week2 directory and install its executable:

```sh
cargo install --path md --locked --force
export PATH="$HOME/.cargo/bin:$PATH"
command -v md
md --version
```

The release profile retains debug symbols for samply. Install the plotting/video environment using the README's setup commands if it is not already present.

## 1. Current Part 2-4 checks

In your own terminal:

```sh
cd /Users/joshua/Downloads/amat5315/Repositorie/AMAT5315-2026Fall-Exercise/week2
cargo test --manifest-path md/Cargo.toml --release
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out fluid.mp4
```

Expected: all tests pass; the check prints three measured values and PASS; fluid.mp4 is below 2 MB and contains 200 frames showing atoms and g(r). Do not treat the development numbers in the README as your own measurements.

Open `support/week2-viewer.html` locally and drag both `artifacts/traj.jsonl` and `artifacts/run.json` into it. Inspect its four panels. Open field.png and dimer.png too. The Part 2 arrows show direction at equal displayed length; the Part 3 main figure uses the PDF's display sampling, while its raw measurements and tests retain all integration steps.

Read the tests and code as requested by the PDF: independent energy derivative, same-trait dimer experiment, total internal force, cutoff continuity just inside rc, CLI file output, and raw-frame recomputation in check. Be ready to explain System ownership, shared/mutable borrows, Integrator, and one macro.

Please report whether the commands passed and provide the three numbers printed by your physics check. Save any terminal evidence you want retained under `evidence/student-...`.

## 2. Baseline timings before optimization (PDF Part 5)

The supplied `/Users/joshua/Downloads/week2-sim.py` has been copied unchanged to this directory for reproducibility. Its seed is 42; the Rust contract's seed is 2026. Both use N=100 and 2000+10000 steps. The isolated NumPy working directory below prevents its fixed `artifacts/` output from overwriting your Rust acceptance trajectory.

Run the setup in your terminal:

```sh
cd /Users/joshua/Downloads/amat5315/Repositorie/AMAT5315-2026Fall-Exercise/week2
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install numpy
cargo build --manifest-path md/Cargo.toml
command -v md
md --version
mkdir -p /tmp/amat5315-week2-numpy-baseline
```

Run each of these timing commands **three times** and record each elapsed/real time:

```sh
(cd /tmp/amat5315-week2-numpy-baseline && time /tmp/venv/bin/python /Users/joshua/Downloads/amat5315/Repositorie/AMAT5315-2026Fall-Exercise/week2/week2-sim.py)
time ./md/target/debug/md run --force naive --out /tmp/md-debug
time md run --force naive --out /tmp/md-release
```

Send the nine elapsed times as text, for example:

```text
NumPy: [first, second, third] seconds
Rust debug: [first, second, third] seconds
Rust release: [first, second, third] seconds
```

Record the NumPy version too: `/tmp/venv/bin/python -c 'import numpy; print(numpy.__version__)'`.

## 3. Naive sampling profile (required before the cell list)

```sh
cargo install samply --locked
~/.cargo/bin/samply record md run --force naive --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-naive
```

In Firefox Profiler's Call Tree, locate `md::periodic::forces`. Record its inclusive sample percentage (the function plus its calls) and the profiled run's elapsed time. Save the screenshot as `week2/profile-naive.png` and send the two numbers as text. The required reference target is a force share above 90%; report the actual number even if it differs.

If `samply` is not on PATH after its Cargo installation, invoke `~/.cargo/bin/samply record ...` or add `~/.cargo/bin` to that terminal's PATH. Do not substitute wall-clock timing for a sampling profile or fill in reference-machine percentages.

The original pre-optimization measurements are already recorded in the README, from source efda417. Current-source reruns must explicitly select naive because cells is now the default. To inspect the optimized profile with the same workload, run:

```sh
~/.cargo/bin/samply record md run --force cells --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
python3 scripts/benchmark.py scaling --out evidence/scaling-new.json
```

The saved original profile files and symbol sidecars can be reopened without rerunning the simulation. Record actual sample percentages and elapsed times; do not use the source machine's numbers as new measurements.

## 4. Heating, public page, and fresh-clone acceptance

The complete heating/cold/hot reproduction commands are in the README's Part 5 section. The public page is https://tracer-huang.github.io/AMAT5315-2026Fall-Exercise/ . Check its four synchronized panels, 400 atoms, 200 frames, and loss of distant g(r) peaks as temperature rises. A fresh clone can verify the core without any pre-existing installed md:

```sh
WK2_CHECK_DIR=$(mktemp -d /tmp/amat5315-final-check-XXXXXX)
gh repo clone Tracer-Huang/AMAT5315-2026Fall-Exercise "$WK2_CHECK_DIR/exercise"
cd "$WK2_CHECK_DIR/exercise/week2"
cargo test --manifest-path md/Cargo.toml --release
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
```

The final independent review is recorded in REVIEW.md. The <=2-minute recording must show the check reaching PASS and explain cold/hot g(r); its media file belongs on a GitHub release rather than in git. The narration outline is in RECORDING-SCRIPT.md.

## Optional: reproduce the exact one-test Part 1 checkpoint

The current crate includes later tests. To see Part 1's exact `1 passed` result without reverting this working directory, use a disposable local clone:

```sh
WK2_PART1_DIR=$(mktemp -d /tmp/amat5315-part1-XXXXXX)
git clone /Users/joshua/Downloads/amat5315/Repositorie/AMAT5315-2026Fall-Exercise "$WK2_PART1_DIR/exercise"
git -C "$WK2_PART1_DIR/exercise" switch --detach 0e8b72a
cd "$WK2_PART1_DIR/exercise/week2/md"
cargo run
cargo test --lib
```

Expected: Hello, world!, and exactly one passing library test. Return to the main repository's week2 directory for subsequent work.
