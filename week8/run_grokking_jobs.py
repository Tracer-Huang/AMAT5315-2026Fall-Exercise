"""Run each required grokking experiment in its own process and retain its log."""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ([f"main_s{i}" for i in range(5)] + [f"wd0_s{i}" for i in range(3)]
         + [f"frac05_s{i}" for i in range(3)] + ["random_s0"])


def worker(name):
    sys.path.insert(0, str(ROOT / "grokking"))
    import run_study

    matches = [overrides for group in run_study.GROUPS.values()
               for n, overrides in group if n == name]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous experiment: {name}")
    run_study.run(name, matches[0])


def launch(name):
    log = ROOT / "artifacts" / "logs" / f"grok-{name}.log"
    cache = ROOT / "artifacts" / "grokking-runs" / f"{name}.npz"
    if cache.exists():
        print(f"cached: {name}", flush=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    print(f"starting: {name}", flush=True)
    with log.open("x") as output:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name],
                       cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    print(f"finished: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=NAMES)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(launch, NAMES))
