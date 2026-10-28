"""Part 1: relationships in the pinned pretrained embedding table."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RELATIONS = {
    'Country → capital': [('iran','tehran'),('turkey','ankara'),('russia','moscow'),('china','beijing'),
                          ('france','paris'),('sweden','stockholm'),('japan','tokyo'),('peru','lima'),('kenya','nairobi')],
    'Male → female': [('king','queen'),('man','woman'),('he','she'),('brother','sister'),('father','mother'),
                      ('son','daughter'),('uncle','aunt'),('boy','girl'),('husband','wife')],
    'Adjective → comparative': [('big','bigger'),('small','smaller'),('fast','faster'),('dark','darker'),
                                ('cold','colder'),('old','older'),('strong','stronger'),('long','longer'),('tall','taller')],
}


def loo_cosines(offsets):
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 2 or len(offsets) < 2:
        raise ValueError('at least two direction vectors are required')
    others = (offsets.sum(axis=0)-offsets)/(len(offsets)-1)
    denominator = np.linalg.norm(offsets, axis=1)*np.linalg.norm(others, axis=1)
    if np.any(denominator == 0):
        raise ValueError('direction or leave-one-out mean has zero length')
    return np.sum(offsets*others, axis=1)/denominator


def relations(table, vocab):
    result = {}
    for name, pairs in RELATIONS.items():
        offsets = []
        for left,right in pairs:
            rows = np.asarray([table[vocab[left]],table[vocab[right]]], dtype=np.float64)
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            offsets.append(rows[1]-rows[0])
        vals = loo_cosines(offsets)
        result[name] = dict(pairs=pairs, cosines=vals.tolist(), mean=float(vals.mean()))
    return result


def main():
    path = ROOT/'data'/'potion-model.safetensors'
    vocab = json.loads((ROOT/'data'/'potion-tokenizer.json').read_text())['model']['vocab']
    with safe_open(path, framework='numpy') as f:
        table = f.get_tensor('embeddings')
    if table.shape != (29528, 256) or len(vocab) != 29528:
        raise ValueError('unexpected pretrained table or tokenizer')
    permutation = np.random.default_rng(1730).permutation(len(table))
    real, shuffled = relations(table,vocab), relations(table[permutation],vocab)
    report = dict(model='minishlab/potion-base-8M', revision='bf8b056', shape=list(table.shape),
                  model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shuffle_seed=1730,
                  real=real, shuffled=shuffled)
    (ROOT/'evidence'/'part1-relations.json').write_text(json.dumps(report,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,(name,_) in zip(axes,RELATIONS.items()):
        for data,color,sign in [(real,'#2a6f9d',1),(shuffled,'#888888',-1)]:
            for cosine in data[name]['cosines']:
                ax.arrow(0,0,cosine,sign*np.sqrt(max(0,1-cosine*cosine)),
                         color=color, alpha=.7, width=.006, head_width=.04, length_includes_head=True)
        ax.axhline(0,color='black',lw=.8); ax.axvline(0,color='#aaa',ls=':',lw=.8)
        ax.set(xlim=(-1.05,1.05),ylim=(-1.1,1.1),aspect='equal',xticks=[-1,0,1],yticks=[])
        ax.set_title(f"{name}\ntable {real[name]['mean']:.3f} · shuffled {shuffled[name]['mean']:.3f}",fontsize=10)
    fig.suptitle('Each arrow compares one offset with the mean of the other eight',fontsize=11)
    fig.savefig(ROOT/'evidence'/'part1-relations.png',dpi=180,bbox_inches='tight');plt.close(fig)
    for name in RELATIONS:
        print(name, 'table',round(real[name]['mean'],4),'shuffled',round(shuffled[name]['mean'],4))


if __name__ == '__main__':
    main()
