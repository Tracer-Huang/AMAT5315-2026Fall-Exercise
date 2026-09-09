# Student-run grey-box checks

The student initially chose to run grey-box commands manually after Computer Use refused terminal access, then explicitly delegated this checklist back to the agent ("你替我完成"). The agent now executes available terminal commands and records their actual outputs. These are NOT represented as student-run checks. This file remains the reproducible command guide.

Current ready scope: Parts 1-4, including corrected field/dimer figures. Part 5 optimization waits for the baseline timings and naive sampling profile below. `md` is installed as a release command with debug symbols.

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
time ./md/target/debug/md run --out /tmp/md-debug
time md run --out /tmp/md-release
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
samply record md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```

In Firefox Profiler's Call Tree, locate `md::periodic::forces`. Record its inclusive sample percentage (the function plus its calls) and the profiled run's elapsed time. Save the screenshot as `week2/profile-naive.png` and send the two numbers as text. The required reference target is a force share above 90%; report the actual number even if it differs.

If `samply` is not on PATH after its Cargo installation, invoke `~/.cargo/bin/samply record ...` or add `~/.cargo/bin` to that terminal's PATH. Do not substitute wall-clock timing for a sampling profile or fill in reference-machine percentages.

The agent will inspect these results, populate the Timing/Profile tables, then implement and test the cell list and heating schedule. The post-optimization profile, heating runs, Pages inspection, fresh-clone acceptance, and narrated recording come after that.

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
