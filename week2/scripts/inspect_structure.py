"""Compare cold/hot structure and audit the saved heating trajectory."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

parser=argparse.ArgumentParser()
parser.add_argument("--heating",type=Path,required=True)
parser.add_argument("--cold",type=Path,required=True)
parser.add_argument("--hot",type=Path,required=True)
parser.add_argument("--out",type=Path,default=Path("evidence/structure-analysis.json"))
parser.add_argument("--figure",default="structure.png")
args=parser.parse_args()

def read(path):
    return json.loads((path/"run.json").read_text()),[json.loads(line) for line in (path/"traj.jsonl").read_text().splitlines()]

def gr(frame,box,nb):
    p=np.array(frame["pos"]);box=np.array(box);rmax=box.min()/2
    d=p[:,None,:]-p[None,:,:];d-=box*np.round(d/box)
    r=np.sqrt(np.sum(d*d,axis=2))[np.triu_indices(len(p),1)]
    edges=np.linspace(0,rmax,nb+1)
    counts=np.histogram(r[r<rmax],bins=edges)[0]
    return (edges[:-1]+edges[1:])/2,2*counts/(len(p)*(len(p)/np.prod(box))*np.pi*np.diff(edges**2))

def speed_temperature(frame):
    v=np.array(frame["vel"])
    return float(np.mean(np.sum(v*v,axis=1))/2)

def msd(frames,box):
    p=np.array([f["pos"] for f in frames]);delta=np.diff(p,axis=0);box=np.array(box)
    delta-=box*np.round(delta/box)
    displacement=np.sum(delta,axis=0)
    return float(np.mean(np.sum(displacement**2,axis=1)))

run,frames=read(args.heating)
assert run["n"]==400 and len(frames)==200 and run["ramp_to"]==1.2
assert frames[0]["step"]==100 and frames[-1]["step"]==20000
r,initial=gr(frames[0],run["box"],48)
last=np.mean([gr(f,run["box"],48)[1] for f in frames[-10:]],axis=0)
mask=r>=2
initial_contrast=float(np.sqrt(np.mean((initial[mask]-1)**2)))
final_contrast=float(np.sqrt(np.mean((last[mask]-1)**2)))
assert final_contrast<initial_contrast
temperature=[speed_temperature(f) for f in frames]
assert abs(temperature[0]-0.2)<0.02 and abs(temperature[-1]-1.2)<0.02
fig,axes=plt.subplots(1,3,figsize=(11,4),gridspec_kw={"width_ratios":[1.6,1,1]},layout="constrained")
details={}
for (name,path,color),ax in zip([("cold",args.cold,"#33548c"),("hot",args.hot,"#b0413e")],axes[1:]):
    c,f=read(path)
    distance,g=gr(f[0],c["box"],104)
    g=np.mean([gr(frame,c["box"],104)[1] for frame in f],axis=0)
    axes[0].plot(distance,g,label=f'{name}: target T={c["temperature"]}',color=color)
    ax.add_collection(PatchCollection([Circle(p,0.45) for p in f[-1]["pos"]],facecolor=color,edgecolor="none"))
    ax.set(xlim=(0,c["box"][0]),ylim=(0,c["box"][1]),aspect="equal",xlabel="x",ylabel="y",title=f'{name} | T={c["temperature"]}')
    details[name]={"n":c["n"],"frames":len(f),"seed":c["seed"],
                   "mean_speed_temperature":float(np.mean([speed_temperature(frame) for frame in f])),
                   "mean_squared_displacement":msd(f,c["box"]),
                   "time_averaged_gr_contrast_r_gt_2":float(np.sqrt(np.mean((g[distance>=2]-1)**2)))}
axes[0].axhline(1,color="#777777",linestyle="--",linewidth=0.8)
axes[0].set(xlabel="Pair distance r",ylabel="g(r)",title="Cold and hot pair structure",xlim=(0,distance[-1]))
axes[0].legend(frameon=False);axes[0].grid(alpha=0.18)
fig.supxlabel("N=100, density=0.8, seed=2026. g(r) averages all 200 saved frames; snapshots show final positions.",fontsize=9)
fig.savefig(args.figure,dpi=180)
report={"source_directories":{"heating":str(args.heating),"cold":str(args.cold),"hot":str(args.hot)},
        "heating":{"n":run["n"],"frames":len(frames),"seed":run["seed"],
                  "first_saved_T_speed":temperature[0],"last_saved_T_speed":temperature[-1],
                  "viewer_first_frame_contrast":initial_contrast,"viewer_last_frame_contrast":final_contrast,
                  "viewer_definition":"48 radial bins up to half the shorter box, RMS(g-1) for bin centres >=2; current window is 1 frame at start and 10 at end.",
                  "frame_nearest_T_1":int(np.argmin(np.abs(np.array(temperature)-1)))},
        "fixed_temperature_runs":details}
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
