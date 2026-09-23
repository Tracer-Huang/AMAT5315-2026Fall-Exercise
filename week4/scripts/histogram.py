import argparse,json
from pathlib import Path
from numerics import frames,histogram

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=Path("artifacts/random/tracers.jsonl")); p.add_argument("--out",type=Path,default=Path("evidence/tracers.txt")); a=p.parse_args()
    records=list(frames(a.source)); final=records[-1]; counts,chi=histogram(final["x"],final["y"])
    meta=json.loads(a.source.with_name("run.json").read_text())
    text=(f"source = {a.source}\nfluid seed = {meta['seed']}; tracer seed = {meta['tracer_seed']}\n"
          f"frames = {len(records)}; tracers = {len(final['x'])}; t = {final['t']}\n"
          f"bins = 16 x 16; expected count = {len(final['x'])/256}; dof = 255\n"
          f"chi2/dof = {chi:.9f} (required < 2): {'PASS' if chi<2 else 'FAIL'}\n"
          "This tests uniform density, not correctness of all trajectories.\n")
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(text); print(text,end="")
    a.out.with_suffix(".json").write_text(json.dumps({"t":final["t"],"chi2_dof":chi,"counts_yx":counts.tolist()},indent=2)+"\n")
    if chi>=2: raise SystemExit(1)
if __name__=="__main__": main()
