"""Export standalone scientific figures inside offline HTML, with their source rows."""
import io,json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FixedLocator,FixedFormatter

OUT=Path("evidence")
BLUE="#2463a5"; ORANGE="#c06a20"; INK="#25344a"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":INK,"text.color":INK,"axes.edgecolor":"#bac3cc","grid.color":"#e5e9ef","axes.grid":True,"grid.linewidth":.7,"svg.fonttype":"none"})

def save(fig,name,title,caption,source,command):
    stream=io.StringIO(); fig.savefig(stream,format="svg",bbox_inches="tight")
    svg=stream.getvalue(); svg=svg[svg.index("<svg"):]
    svg="\n".join(line.rstrip() for line in svg.splitlines())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#25344a;background:#f6f8fb;margin:0}}main{{max-width:1100px;margin:auto;padding:32px 24px}}h1{{font-size:28px;line-height:1.25}}figure{{margin:24px 0;background:white;padding:16px;border:1px solid #e4e9ef}}svg{{width:100%;height:auto}}figcaption{{font-size:14px;color:#526176}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}a{{color:#2463a5}}</style>
<main><a href="../README.md">← Week 4 instructions</a><h1>{html.escape(title)}</h1><figure>{svg}<figcaption>{html.escape(caption)}</figcaption></figure><p>Generate: <code>{html.escape(command)}</code></p><details><summary>Source data and method</summary><pre>{html.escape(json.dumps(source,indent=2))}</pre></details></main></html>'''
    (OUT/f"{name}.html").write_text(page); plt.close(fig)

def main():
    data=json.loads((OUT/"analysis.json").read_text()); random=data["random"]; spectrum=random["spectrum"]
    fig,ax=plt.subplots(figsize=(9.5,5.3),layout="constrained")
    for key,label,color,style in [("initial","Initial field (t = 0)",INK,":"),("pure_diffusion","Pure diffusion prediction (t = 10)",ORANGE,"--"),("final","Computed flow (t = 10)",BLUE,"-")]:
        ax.semilogy(spectrum["k"],np.maximum(spectrum[key],1e-16),label=label,color=color,linestyle=style,linewidth=2)
    ax.axvspan(2,6,color=INK,alpha=.06); ax.set(xlabel="Radial wavenumber |k|",ylabel="Shell mean |FFT(ω)| / N²",xlim=(0,60),ylim=(1e-9,1)); ax.legend(frameon=False)
    ax.set_title(f"Random flow · seed {random['metadata']['seed']} · transfer = {random['transfer']:.3f}",loc="left",pad=18)
    save(fig,"spectrum","Vorticity transfer beyond the initial band","Recomputed from full saved velocities on N=128. Shells use nearest integer radial wavenumber. The low-amplitude tail includes six-decimal storage noise; it is not evidence of resolved physical transfer at every mode.",spectrum,"python scripts/analyze.py && python scripts/plots.py")
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for column,case in enumerate(["taylor-green","random"]):
        d=data[case]; t=d["t"]
        axes[0,column].plot(t,d["E"],color=BLUE,label="Recorded energy",linewidth=2)
        axes[0,column].plot(t,d["E_predicted"],color=ORANGE,linestyle="--",label="E(0) − 2ν ∫Z dt",linewidth=2)
        axes[0,column].set(title=case,xlabel="Time",ylabel="Mean kinetic energy E"); axes[0,column].legend(frameon=False,fontsize=9)
        gap=np.array(d["E"])-np.array(d["E_predicted"])
        axes[1,column].plot(t,gap,color=BLUE); axes[1,column].axhline(0,color=INK,linewidth=.8)
        axes[1,column].set(xlabel="Time",ylabel="E − budget prediction",title=f"Relative final residual {d['budget_residual']:.2e}")
        axes[1,column].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    save(fig,"budget","Energy loss matches viscous dissipation","Trapezoidal integration uses the actual stored times (spacing 0.1). The lower panels magnify the gap. Pure diffusion also satisfies this budget, so conservation alone does not verify advection.",{case:{key:data[case][key] for key in ["t","E","Z","E_predicted","budget_residual"]} for case in data},"python scripts/analyze.py && python scripts/plots.py")
    c=json.loads(Path("artifacts/convergence.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),layout="constrained")
    axes[0].loglog(c["n_list"],c["n_errors"],"o-",color=BLUE,linewidth=2)
    axes[0].set(xlabel="Grid size N",ylabel="Relative L2 vorticity error at t = 2",title=f"Grid refinement · ratios {c['grid_ratios'][0]:.2f}×, {c['grid_ratios'][1]:.2f}×")
    axes[0].xaxis.set_major_locator(FixedLocator(c["n_list"])); axes[0].xaxis.set_major_formatter(ScalarFormatter()); axes[0].minorticks_off()
    dt=np.array(c["dt_list"]); err=np.array(c["dt_errors"])
    axes[1].loglog(dt,err,"o-",color=BLUE,linewidth=2,label=f"Measured slope {c['dt_slope']:.3f}")
    axes[1].loglog(dt,err[0]*(dt/dt[0])**4*1.6,"--",color=INK,label="Fourth order guide (offset)")
    axes[1].set(xlabel="Time step Δt",ylabel="Relative L2 vorticity error at t = 2",title="Time refinement · fixed N = 128")
    axes[1].xaxis.set_major_locator(FixedLocator(sorted(dt))); axes[1].xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in sorted(dt)])); axes[1].minorticks_off(); axes[1].legend(frameon=False,fontsize=9)
    save(fig,"convergence","Spatial and temporal self-convergence",f"Grid series: dt=0.01, reference N=256 / dt=0.0025, coincident grid points. Time series: N=128, same-grid reference dt=0.0025. All runs use seed {c['seed']}, band 2–6, ν=0.004, t=2. Acceptance: grid ratios ≥3 and ≥10; time slope 3.7–4.3.",c,"python scripts/refinement.py && python scripts/plots.py")
    print("Wrote spectrum.html, budget.html, convergence.html.")
if __name__=="__main__": main()
