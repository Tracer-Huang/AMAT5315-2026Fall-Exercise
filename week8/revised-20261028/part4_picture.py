"""Recall a 32x32 character using the four PDF Part 4 memory stores."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from memory import recall

ROOT = Path(__file__).resolve().parent
SEED = 531508
DIM = 64
SIZES = (64, 256, 1024, 4096)
METHODS = ('outer', 'relu', 'relu2', 'softmax')
LABELS = ('Outer product', 'MLP, ReLU', 'MLP, ReLU²', 'Attention, softmax β = 8')


def picture(path):
    lines = path.read_text().splitlines()
    if len(lines) != 32 or any(len(row) != 32 or set(row) - {'.', '#'} for row in lines):
        raise ValueError('the stored picture must contain 32 rows of 32 . / # pixels')
    return np.array([[1. if c == '#' else -1. for c in row] for row in lines]).ravel()


def run():
    (ROOT/'artifacts').mkdir(parents=True,exist_ok=True)
    (ROOT/'evidence').mkdir(parents=True,exist_ok=True)
    asset = ROOT/'assets/suan-32.txt'
    signal = picture(asset)
    rng = np.random.default_rng(SEED)
    images = np.zeros((len(SIZES), len(METHODS), 32, 32))
    ratios = np.zeros((len(SIZES), len(METHODS)))
    for i, count in enumerate(SIZES):
        keys = rng.normal(size=(count, DIM))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = rng.integers(0, 2, size=(count, 1024), dtype=np.int8) * 2 - 1
        values[0] = signal.astype(np.int8)
        for j, method in enumerate(METHODS):
            output, ratios[i, j] = recall(keys, values, 0, method, beta=8)
            images[i, j] = output.reshape(32, 32)
            print(f'M={count:4d} {method:7s} R={ratios[i,j]:.6f}', flush=True)
    art = ROOT/'artifacts'/'part4-picture.npz'
    with art.open('xb') as f:
        np.savez_compressed(f, images=images, ratios=ratios, sizes=SIZES, methods=METHODS,
                            signal=signal, seed=SEED, dimension=DIM, beta=8)
    report = dict(seed=SEED, dimension=DIM, beta=8, sizes=SIZES, methods=METHODS,
                  ratios=ratios.tolist(), asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                  artifact_sha256=hashlib.sha256(art.read_bytes()).hexdigest(),
                  notes='R is the norm of interference divided by the norm of the target signal; one draw per M.')
    (ROOT/'evidence'/'part4-picture.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(images, ratios)


def plot(images, ratios):
    fig, axes = plt.subplots(4, 4, figsize=(9.0, 9.2), constrained_layout=True)
    x, y = np.meshgrid(np.arange(32), np.arange(32))
    for i, count in enumerate(SIZES):
        for j, title in enumerate(LABELS):
            ax = axes[i, j]
            pixels = np.maximum(images[i, j], 0)
            # The scale is common within a panel; area remains proportional to the recalled value.
            area = 16 * pixels / max(1., pixels.max())
            ax.scatter(x.ravel(), y.ravel(), s=area.ravel(),
                       c='#ce4038' if ratios[i, j] < 1 else '#4b5156', linewidths=0)
            ax.set(xlim=(-1, 32), ylim=(32, -1), aspect='equal', xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'M = {count}', fontsize=9)
            ax.set_xlabel(f'R = {ratios[i, j]:.3f}', fontsize=8)
    fig.suptitle('Picture recall by four key–value stores (one draw per row)', fontsize=12)
    fig.savefig(ROOT/'evidence'/'part4-picture-recall.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    run()
