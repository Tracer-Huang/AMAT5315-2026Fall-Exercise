"""Plot energy errors measured by the two Rust integrators."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("--out", default="dimer.png")
args = parser.parse_args()
d = np.genfromtxt(args.csv, delimiter=",", names=True)
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
short = d["time"] <= 5
left.plot(d["time"][short], d["euler"][short], label="Forward Euler", color="#c24135")
left.plot(d["time"][short], d["verlet"][short], label="Velocity-Verlet", color="#215f9a")
left.set(xlabel="Time t", ylabel=r"$(E(t)-E_0)/|E_0|$", title="Same initial state | 500 steps", xlim=(0, 5))
left.legend(frameon=False)
right.plot(d["time"], 1000*d["verlet"], color="#215f9a", linewidth=0.7)
right.set(xlabel="Time t", ylabel=r"Relative energy error $\times 1000$",
          title="Velocity-Verlet | 5000 steps", xlim=(0, 50), ylim=(-0.5, 0.5))
for ax in (left, right):
    ax.grid(alpha=0.18)
fig.savefig(args.out, dpi=170)
