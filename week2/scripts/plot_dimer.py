"""Plot Rust dimer energy errors with the learning sheet's display sampling.

The CSV retains every dt=0.01 integration step. PDF page 8's vector paths
contain 250 left-panel points (steps 2,4,...,500) and 500 right-panel points
(steps 10,20,...,5000). Plotting every step creates a visibly denser waveform;
it does not change the underlying trajectory or conservation diagnostic.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("--out", default="dimer.png")
parser.add_argument("--full-resolution", action="store_true",
                    help="draw every saved step instead of matching PDF display sampling")
args = parser.parse_args()
d = np.genfromtxt(args.csv, delimiter=",", names=True)
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
short = d[d["time"] <= 5]
long = d
if not args.full_resolution:
    short = short[2::2]
    long = long[10::10]
left.plot(short["time"], short["euler"], label="Forward Euler", color="#b0413e")
left.plot(short["time"], short["verlet"], label="Velocity-Verlet", color="#33548c")
left.set(xlabel="Time t", ylabel=r"$(E(t)-E_0)/|E_0|$", title="Same initial state | 500 steps", xlim=(0, 5))
left.legend(frameon=False)
right.plot(long["time"], 1000*long["verlet"], color="#33548c", linewidth=0.8)
right.set(xlabel="Time t", ylabel=r"Relative energy error $\times 1000$",
          title="Velocity-Verlet | 5000 steps", xlim=(0, 50), ylim=(-0.5, 0.5),
          xticks=[0,25,50], yticks=[-0.5,-0.25,0,0.25,0.5])
for ax in (left, right):
    ax.grid(alpha=0.18)
sampling = "Every integration step shown" if args.full_resolution else "Display sampling: every 2 steps (left), every 10 steps (right), matching the PDF"
fig.supxlabel(r"Integration $\Delta t=0.01$. " + sampling + ".", fontsize=9)
fig.savefig(args.out, dpi=170)
