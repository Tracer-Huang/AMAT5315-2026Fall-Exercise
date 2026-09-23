"""Recompute budgets and spectra from the recorded velocities, independently."""
import argparse,json
from pathlib import Path
import numpy as np
from numerics import frames,diagnostics,budget,transfer,waves

def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifacts",type=Path,default=Path("artifacts")); p.add_argument("--evidence",type=Path,default=Path("evidence")); a=p.parse_args(); a.evidence.mkdir(exist_ok=True,parents=True)
    results={}; lines=["Recomputed with NumPy from saved u and v; omega used only for consistency."]
    for case in ["taylor-green","random"]:
        folder=a.artifacts/case; meta=json.loads((folder/"run.json").read_text()); ts=[]; es=[]; zs=[]; ps=[]; divergences=[]; consistencies=[]; first=None
        for frame in frames(folder/"fields.jsonl"):
            d=diagnostics(frame); w=d["omega"]
            if first is None: first=w.copy()
            ts.append(frame["t"]); es.append(d["E"]); zs.append(d["Z"]); divergences.append(d["divergence"]); consistencies.append(d["consistency"])
            kx,ky=waves(meta["n"],True); wh=np.fft.fft2(w)
            wx=np.fft.ifft2(1j*kx*wh).real; wy=np.fft.ifft2(1j*ky*wh).real
            ps.append(float(np.mean(wx*wx+wy*wy)/2))
        prediction,residual=budget(ts,es,zs,meta["nu"])
        z_prediction,z_residual=budget(ts,zs,ps,meta["nu"])
        moved=transfer(first,w,meta["nu"],ts[-1])
        kx,ky=waves(meta["n"]); radius=np.sqrt(kx*kx+ky*ky); shell=np.rint(radius).astype(int)
        count=np.bincount(shell.ravel()); mode_norm=meta["n"]**2
        w0=np.fft.fft2(first); wf=np.fft.fft2(w)
        diffused=w0*np.exp(-meta["nu"]*radius**2*ts[-1])
        average=lambda h: (np.bincount(shell.ravel(),weights=(np.abs(h)/mode_norm).ravel())/np.maximum(count,1)).tolist()
        results[case]={"metadata":meta,"t":ts,"E":es,"Z":zs,"P":ps,"E_predicted":prediction.tolist(),"Z_predicted":z_prediction.tolist(),"budget_residual":residual,"enstrophy_budget_residual":z_residual,"transfer":moved,"max_divergence":max(divergences),"max_consistency":max(consistencies),"spectrum":{"k":list(range(len(count))),"initial":average(w0),"final":average(wf),"pure_diffusion":average(diffused)}}
        lines.extend([f"\n{case}: seed={meta['seed']}, N={meta['n']}, nu={meta['nu']}, dt={meta['dt']}, end={ts[-1]}",f"frames = {len(ts)}; E: {es[0]:.9f} -> {es[-1]:.9f}; Z: {zs[0]:.9f} -> {zs[-1]:.9f}",f"divergence = {max(divergences):.9e} (required < 1e-3)",f"consistency = {max(consistencies):.9e} (required < 1e-3)",f"energy-budget = {residual:.9e}"+(" (required < 1e-3)" if case=="random" else " (diagnostic; TG exact field/energy gates in course-check.txt)"),f"transfer = {moved:.9f}"+(" (required >= 0.2)" if case=="random" else " (expected near zero)"),f"optional enstrophy-budget = {z_residual:.9e} (101/11-frame trapezoid; no course threshold)"])
        if max(divergences)>=1e-3 or max(consistencies)>=1e-3 or (case=="random" and (residual>=1e-3 or moved<.2)): raise RuntimeError(f"physics acceptance failed: {case}")
    lines.append("\nBudget tests conservation; transfer tests departure from pure diffusion. Neither proves every detail of nonlinear dynamics.")
    (a.evidence/"physics.txt").write_text("\n".join(lines)+"\n")
    (a.evidence/"analysis.json").write_text(json.dumps(results,indent=2)+"\n")
    print("\n".join(lines))
if __name__=="__main__": main()
