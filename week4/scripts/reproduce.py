"""Generate every required Week 4 output from source; preserve existing raw runs."""
import os
import subprocess
import sys
from pathlib import Path

WEEK4 = Path(__file__).resolve().parents[1]
ARTIFACTS = WEEK4 / "artifacts"
EVIDENCE = WEEK4 / "evidence"


def run(command, *, input_bytes=None, stdout_path=None):
    print("RUN", " ".join(map(str, command)), flush=True)
    if stdout_path is None:
        subprocess.run(command, input=input_bytes, cwd=WEEK4, check=True)
    else:
        with stdout_path.open("wb") as stream:
            subprocess.run(command, input=input_bytes, cwd=WEEK4, stdout=stream, check=True)


def main():
    for name in ["taylor-green", "random", "order", "unstable", "scan", "sensitivity"]:
        if (ARTIFACTS / name).exists():
            raise SystemExit(f"Refusing to replace existing raw results: {ARTIFACTS / name}")
    ARTIFACTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    initial = subprocess.check_output(["field", "taylor-green", "--n", "64"], cwd=WEEK4)
    (ARTIFACTS / "taylor-green-initial.json").write_bytes(initial)
    run(["fluid", "--method", "rk4", "--nu", "0.1", "--dt", "0.01", "--t-end", "1",
         "--every", "0.1", "--out", "artifacts/taylor-green"],
        input_bytes=initial, stdout_path=ARTIFACTS / "taylor-green.tsv")
    exact = subprocess.check_output(["field", "taylor-green", "--n", "64", "--nu", "0.1", "--t", "1"], cwd=WEEK4)
    (ARTIFACTS / "taylor-green" / "exact-t1.json").write_bytes(exact)
    for script in ["compare_derivatives.py", "plot_line_stability.py", "plot_line_accuracy.py", "plot_taylor_green.py",
                   "run_part3.py", "plot_part3.py", "run_part4.py", "plot_part4.py"]:
        run([sys.executable, "scripts/" + script])
    names = {p.name for p in EVIDENCE.iterdir() if p.is_file()}
    required = {"line-stability.png", "line-accuracy.png", "taylor-green.png", "blowup.png", "sensitivity.png",
                "random.png", "order.png", "convergence.png", "convergence.json"}
    if not required.issubset(names):
        raise RuntimeError(f"Missing required evidence: {sorted(required - names)}")
    print("Completed all eight required figures and convergence.json; raw artifacts preserved under artifacts/.")


if __name__ == "__main__":
    main()
