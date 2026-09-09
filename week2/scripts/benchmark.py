"""Record reproducible wall-clock benchmarks; never substitute reference timings."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def run_case(label, command, cwd, repetitions=3):
    cwd.mkdir(parents=True,exist_ok=True)
    rows=[]
    for trial in range(1,repetitions+1):
        started=time.perf_counter()
        result=subprocess.run(command,cwd=cwd,capture_output=True,text=True)
        elapsed=time.perf_counter()-started
        row={"trial":trial,"elapsed_seconds":elapsed,"returncode":result.returncode,
             "stdout":result.stdout,"stderr":result.stderr}
        rows.append(row)
        print(f"{label}, run {trial}: {elapsed:.6f} s",flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed: {result.stderr}")
    values=[r["elapsed_seconds"] for r in rows]
    return {"label":label,"command":command,"cwd":str(cwd),"runs":rows,
            "median_seconds":statistics.median(values),
            "min_seconds":min(values),"max_seconds":max(values)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["baseline","scaling"])
    parser.add_argument("--python",default="/tmp/venv/bin/python")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--steps",type=int,default=500,help="scaling production steps")
    parser.add_argument("--eq-steps",type=int,default=100,help="scaling equilibration steps")
    args=parser.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work=ROOT/"benchmark-runs"/f"{args.mode}-{stamp}"
    cases=[]
    if args.mode=="baseline":
        specs=[("NumPy",[args.python,str(ROOT/"week2-sim.py")]),
               ("Rust debug",[str(ROOT/"md/target/debug/md"),"run","--out","artifacts"]),
               ("Rust release naive",[str(ROOT/"md/target/release/md"),"run","--out","artifacts"])]
        for index,(label,command) in enumerate(specs):
            cases.append(run_case(label,command,work/f"case-{index}"))
    else:
        for n in [100,400,1600]:
            for method in ["naive","cells"]:
                command=[str(ROOT/"md/target/release/md"),"run","--force",method,"--n",str(n),
                         "--steps",str(args.steps),"--eq-steps",str(args.eq_steps),"--out","artifacts"]
                case=run_case(f"N={n} {method}",command,work/f"n-{n}-{method}")
                case.update(n=n,force=method,total_steps=args.steps+args.eq_steps)
                cases.append(case)
    result={"mode":args.mode,"recorded_at_utc":stamp,"platform":platform.platform(),
            "machine":platform.machine(),
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "timer":"time.perf_counter around subprocess, wall time including startup and output I/O",
            "repetitions":3,"cases":cases,
            "seed_note":"Unmodified supplied NumPy uses seed 42; Rust default uses seed 2026.",
            "raw_outputs_retained_in":str(work)}
    if args.mode=="baseline":
        result["numpy_version"]=subprocess.check_output([args.python,"-c","import numpy; print(numpy.__version__)"],text=True).strip()
    out=args.out or ROOT/"evidence"/f"{args.mode}-timings.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise RuntimeError(f"refusing to overwrite timing evidence: choose a new --out ({out})")
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(f"saved {out}")


if __name__=="__main__":
    main()
