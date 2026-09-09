"""Plot CSV exported by the Rust field example; no duplicate LJ formulas."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("--out", default="field.png")
args = parser.parse_args()
data = np.genfromtxt(args.csv, delimiter=",", names=True)
side = int(np.sqrt(len(data)))
x, y, u, fx, fy = (data[k].reshape(side, side) for k in data.dtype.names)
fig, ax = plt.subplots(figsize=(6.7, 5.7), layout="constrained")
heat = ax.pcolormesh(x, y, np.clip(u, -1, 1), cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading="auto")
sl = (slice(0, None, 5), slice(0, None, 5))
length = np.hypot(fx[sl], fy[sl])
scaled = 0.12 + 0.06 * np.tanh(np.log1p(length) / 5)
factor = np.divide(scaled, length, out=np.zeros_like(length), where=length > 0)
ax.quiver(x[sl], y[sl], fx[sl]*factor, fy[sl]*factor, color="#202124",
          angles="xy", scale_units="xy", scale=1, width=0.0025)
r0 = 2**(1/6)
ax.add_patch(plt.Circle((0, 0), r0, fill=False, linestyle="--", color="black", linewidth=1.2))
ax.scatter([0], [0], s=26, color="black", zorder=5)
ax.set(xlabel=r"$x_j-x_i$", ylabel=r"$y_j-y_i$", aspect="equal",
       title=f"Lennard-Jones pair field | r0 = {r0:.3f}\nArrow lengths compressed; dashed circle: zero radial force")
fig.colorbar(heat, ax=ax, label="Potential energy U (clipped to [-1, 1])")
fig.savefig(args.out, dpi=170)
