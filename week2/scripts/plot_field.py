"""Static field.png: energy heatmap plus normalized force-direction arrows.

Contract: show outward force inside r0, inward force outside, and a negative
energy ring around the repulsive core. All values come from the Rust CSV.
Blue/red encodes signed energy, matching the course reference; arrows encode
direction only, with equal length to keep weak outer attraction visible.
"""
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
fig, ax = plt.subplots(figsize=(7.5, 6.7), layout="constrained")
heat = ax.pcolormesh(x, y, np.clip(u, -1, 1), cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading="auto")
r0 = 2**(1/6)
# A fixed spatial spacing keeps arrow density stable if CSV resolution changes.
stride = max(1, round(0.25 / (x[0, 1] - x[0, 0])))
margin = max(1, stride // 2)
sl = (slice(margin, -margin, stride), slice(margin, -margin, stride))
qx, qy = x[sl], y[sl]
length = np.hypot(fx[sl], fy[sl])
radius = np.hypot(qx, qy)
visible = (length > 1e-12) & (radius > 0.22) & (np.abs(radius-r0) > 0.10)
factor = np.divide(0.17, length, out=np.zeros_like(length), where=visible)
for mask, color in [(visible & (radius < r0), "#ffffff"),
                    (visible & (radius > r0), "#202124")]:
    ax.quiver(qx[mask], qy[mask], (fx[sl]*factor)[mask], (fy[sl]*factor)[mask],
              color=color, angles="xy", scale_units="xy", scale=1,
              pivot="middle", width=0.0036, headwidth=3.7, headlength=4.8)
ax.add_patch(plt.Circle((0, 0), r0, fill=False, linestyle="--", color="#202124", linewidth=1.5))
ax.scatter([0], [0], s=35, color="#202124", zorder=5)
ax.set(xlabel=r"$x_j-x_i$", ylabel=r"$y_j-y_i$", aspect="equal",
       xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
       title="Lennard-Jones energy and force direction\n"
             + rf"Inside $r_0$: outward   |   Outside $r_0$: inward   ($r_0={r0:.3f}$)")
bar = fig.colorbar(heat, ax=ax, shrink=0.94, pad=0.035,
                  label="Potential energy U (reduced units)", ticks=[-1, 0, 1])
bar.ax.set_yticklabels(["−1", "0", "≥ 1"])
fig.supxlabel("Equal-length arrows show direction only. Dashed circle: zero force. Blue ring: U < 0.",
              fontsize=10)
fig.savefig(args.out, dpi=190)
