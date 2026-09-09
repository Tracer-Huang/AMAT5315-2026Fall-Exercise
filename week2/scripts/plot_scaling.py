"""Plot the three required measured sizes with median and min-max ranges."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("data",type=Path)
parser.add_argument("--out",default="scaling.png")
args=parser.parse_args()
source=json.loads(args.data.read_text())
cases={(c["n"],c["force"]):c for c in source["cases"]}
ns=[100,400,1600]
fig,ax=plt.subplots(figsize=(7.1,4.9),layout="constrained")
for method,color,marker in [("naive","#b0413e","o"),("cells","#33548c","s")]:
    rows=[cases[n,method] for n in ns]
    median=np.array([c["median_seconds"]/c["total_steps"] for c in rows])
    low=np.array([c["min_seconds"]/c["total_steps"] for c in rows])
    high=np.array([c["max_seconds"]/c["total_steps"] for c in rows])
    ax.errorbar(ns,median,yerr=[median-low,high-median],label=method,color=color,
                marker=marker,markerfacecolor="white",capsize=4,linewidth=1.5)
ax.set(xscale="log",yscale="log",xlabel="Atom count N",ylabel="Wall time / total steps (s)",
       title="Force-search scaling\n100 equilibration + 500 production steps; 3 runs per case")
ax.set_xticks(ns,labels=[str(n) for n in ns]);ax.minorticks_off();ax.grid(alpha=0.18)
ax.legend(frameon=False)
fig.supxlabel("Points: medians. Error bars: min-max. Timings include startup and trajectory output.",fontsize=9)
fig.savefig(args.out,dpi=180)
print("| N | Naive median [min-max] (s) | Cells median [min-max] (s) | Speedup [range] |")
print("| ---: | ---: | ---: | ---: |")
ratios=[]
for n in ns:
    a,b=cases[n,"naive"],cases[n,"cells"]
    ratio=a["median_seconds"]/b["median_seconds"]
    ratios.append(ratio)
    low=a["min_seconds"]/b["max_seconds"];high=a["max_seconds"]/b["min_seconds"]
    print(f'| {n} | {a["median_seconds"]:.6f} [{a["min_seconds"]:.6f}-{a["max_seconds"]:.6f}] | '
          f'{b["median_seconds"]:.6f} [{b["min_seconds"]:.6f}-{b["max_seconds"]:.6f}] | {ratio:.3f}x [{low:.3f}-{high:.3f}] |')
print("\nSpeedup is the ratio of medians; its range is [min(naive)/max(cells), max(naive)/min(cells)].")
assert ratios==sorted(ratios) and ratios[-1]>2,"measured scaling does not satisfy the PDF"
