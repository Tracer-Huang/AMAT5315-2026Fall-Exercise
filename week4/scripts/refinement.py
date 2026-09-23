"""Self-convergence using the actual field/fluid executables and stored omega."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from numerics import frames,grid,relative_error

def solver_commands(bin_dir=None):
    if bin_dir is not None:
        directory=Path(bin_dir).resolve()
        return str(directory/"field"),str(directory/"fluid")
    commands=tuple(shutil.which(name) for name in ("field","fluid"))
    if not all(commands):
        raise RuntimeError("Install field and fluid with cargo install --path . --locked, then add $HOME/.cargo/bin to PATH")
    return commands

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bin-dir",type=Path,help="optional explicit binary directory; default: installed field/fluid on PATH"); p.add_argument("--out-dir",type=Path,default=Path("artifacts/refinement")); p.add_argument("--output",type=Path,default=Path("artifacts/convergence.json")); p.add_argument("--seed",type=int,default=int(os.environ.get("SEED","2026"))); a=p.parse_args()
    field,fluid=solver_commands(a.bin_dir)
    if a.out_dir.exists(): raise SystemExit(f"Refusing existing study directory: {a.out_dir}; choose a fresh --out-dir")
    a.out_dir.mkdir(parents=True); start=time.monotonic()
    def run(n,dt):
        folder=a.out_dir/f"n{n}-dt{dt:g}"
        fcmd=[field,"random","--n",str(n),"--seed",str(a.seed),"--k-min","2","--k-max","6"]
        cmd=[fluid,"--nu","0.004","--dt",str(dt),"--t-end","2","--every","2","--out",str(folder)]
        initial=subprocess.run(fcmd,check=True,capture_output=True).stdout
        result=subprocess.run(cmd,input=initial,check=True,capture_output=True)
        folder.joinpath("stdout.txt").write_bytes(result.stdout)
        print(f"completed N={n}, dt={dt}",flush=True)
        return grid(list(frames(folder/"fields.jsonl"))[-1]["omega"])
    spatial_ref=run(256,.0025); temporal_ref=run(128,.0025)
    n_list=[32,64,128]; n_errors=[]; cache={}
    for n in n_list:
        value=run(n,.01); cache[(n,.01)]=value
        n_errors.append(relative_error(value,spatial_ref[::256//n,::256//n]))
    dt_list=[.02,.0125,.01]; dt_errors=[]
    for dt in dt_list:
        value=cache[(128,dt)] if (128,dt) in cache else run(128,dt)
        dt_errors.append(relative_error(value,temporal_ref))
    slope=float(np.polyfit(np.log(dt_list),np.log(dt_errors),1)[0])
    ratios=[n_errors[0]/n_errors[1],n_errors[1]/n_errors[2]]
    result={"n_list":n_list,"n_errors":n_errors,"dt_list":dt_list,"dt_errors":dt_errors,"dt_slope":slope,"grid_ratios":ratios,"seed":a.seed,"nu":.004,"t_end":2,"spatial_reference":{"n":256,"dt":.0025},"temporal_reference":{"n":128,"dt":.0025},"comparison":"saved six-decimal omega at coincident grid points","wall_seconds":time.monotonic()-start}
    a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if not (ratios[0]>=3 and ratios[1]>=10 and 3.7<=slope<=4.3): raise SystemExit("Convergence acceptance failed")
if __name__=="__main__": main()
