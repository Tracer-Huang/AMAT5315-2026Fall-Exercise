"""Plot the observed MLP training loss and its full held-out comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'evidence'/'part3-mlp.json').read_text())
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
ax.plot([x['step'] for x in r['history']],[x['train_loss'] for x in r['history']],label='training batch mean')
ax.axhline(r['bigram_loss'],color='#bd632e',ls='--',label=f"bigram held-out {r['bigram_loss']:.4f}")
ax.scatter([r['config']['steps']],[r['heldout_loss']],color='#348b64',zorder=3,label=f"MLP held-out {r['heldout_loss']:.4f}")
ax.set(xlabel='optimizer step',ylabel='cross-entropy (nats / token)');ax.legend()
fig.savefig(ROOT/'evidence'/'part3-mlp-loss.png',dpi=180);plt.close(fig)
