"""Exact learning-sheet runs; timings include JSONL output. Run from week3/."""

import json
import pathlib
import shutil
import subprocess
import time

BIN = "./target/release/ising"


def run(name, update, l, start, stop, step, discard, measure, seed, every=0):
    out = pathlib.Path(name)
    out.mkdir(parents=True, exist_ok=True)
    args = [
        BIN,
        "--update",
        update,
        "--l",
        str(l),
        "--t-from",
        str(start),
        "--t-to",
        str(stop),
        "--t-step",
        str(step),
        "--discard",
        str(discard),
        "--measure",
        str(measure),
        "--seed",
        str(seed),
        "--every",
        str(every),
        "--out",
        name,
    ]
    tic = time.perf_counter()
    with open(out / "stdout.tsv", "w") as f:
        subprocess.run(args, stdout=f, check=True)
    receipt = {"command": args, "wall_seconds": time.perf_counter() - tic}
    (out / "timing.json").write_text(json.dumps(receipt, indent=2))
    print(name, receipt["wall_seconds"], flush=True)


if __name__ == "__main__":
    for t in [1.8, 3.0, 3.1]:
        run(f"runs/T{t}", "metropolis", 64, t, t, 0.1, 2000, 2000, 2026)
    for name, seed in [("a", 2026), ("b", 2026), ("c", 2027)]:
        run("runs/" + name, "metropolis", 64, 1.8, 1.8, 0.1, 2000, 2000, seed)
    assert (
        pathlib.Path("runs/a/series.jsonl").read_bytes()
        == pathlib.Path("runs/b/series.jsonl").read_bytes()
    )
    assert (
        pathlib.Path("runs/a/series.jsonl").read_bytes()
        != pathlib.Path("runs/c/series.jsonl").read_bytes()
    )
    run("runs/ramp", "metropolis", 64, 1.5, 3.5, 0.05, 2000, 200, 2026, 20)
    shutil.copyfile("runs/ramp/spins.jsonl", "spins.jsonl")
    for l, seed in [(32, 1042), (64, 42)]:
        run(f"artifacts/coarse-l{l}", "metropolis", l, 1.5, 3.5, 0.1, 2000, 5000, seed)
        run(f"artifacts/window-l{l}", "metropolis", l, 2, 2.6, 0.05, 2000, 100000, seed)
    for l, seed in [(32, 1042), (64, 42)]:
        run(f"artifacts/wolff-l{l}", "wolff", l, 2, 2.6, 0.05, 20000, 100000, seed)
