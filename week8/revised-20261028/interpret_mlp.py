"""Optional analysis of the positive and negative sides of MLP units."""
import json
from pathlib import Path

import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm

import mlp

ROOT=Path(__file__).resolve().parent


def side_concentration(values, sign, k=5):
    if sign not in (-1,1) or k<1:raise ValueError('sign must be ±1 and k positive')
    side=np.maximum(sign*np.asarray(values,dtype=np.float64),0)
    indices=np.argsort(side)[::-1][:k]
    total=float(np.sum(side**2))
    return dict(indices=indices.tolist(),share=float(np.sum(side[indices]**2)/total) if total else 0.)


def describe(params,sp):
    E,W1,b1,W2=[np.asarray(params[k],dtype=np.float64) for k in ('E','W1','b1','W2')]
    activation=np.tanh(E@W1+b1)
    units=[]
    for unit in range(activation.shape[1]):
        for sign in (1,-1):
            result=side_concentration(activation[:,unit],sign)
            next_ids=np.argsort(sign*W2[unit])[::-1][:5].tolist()
            units.append(dict(unit=unit,side='positive' if sign==1 else 'negative',
                              share=result['share'],previous_ids=result['indices'],next_ids=next_ids,
                              previous=[sp.id_to_piece(i) for i in result['indices']],
                              next=[sp.id_to_piece(i) for i in next_ids]))
    return units


def main():
    cfg=json.loads((ROOT/'artifacts'/'mlp'/'config.json').read_text())
    with np.load(ROOT/'artifacts'/'mlp'/'model.npz',allow_pickle=False) as z:trained={k:z[k] for k in z.files}
    key=jax.random.PRNGKey(cfg['seed']);_,initkey=jax.random.split(key)
    initial=mlp.initialize(cfg,initkey)
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/'data'/'tok512.model'))
    learned=describe(trained,sp);baseline=describe(initial,sp)
    report=dict(trained_median_share=float(np.median([r['share'] for r in learned])),
                initial_median_share=float(np.median([r['share'] for r in baseline])),
                trained_sides=learned,initial_sides=baseline,
                definition='share of squared activation among the five strongest tokens on each positive/negative side')
    (ROOT/'evidence'/'part3-mlp-units.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    selected=sorted((r for r in learned if r['side']=='positive'),key=lambda r:r['share'],reverse=True)[:4]
    fig,ax=plt.subplots(figsize=(11,3.5),constrained_layout=True);ax.axis('off')
    table=[[str(r['unit']),f"{r['share']:.3f}",', '.join(r['previous']).replace('▁','·'),
            ', '.join(r['next']).replace('▁','·')] for r in selected]
    t=ax.table(cellText=table,colLabels=['unit','top-5 share','previous tokens','next tokens'],
               cellLoc='left',loc='center',colWidths=[.08,.12,.4,.4]);t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,2)
    ax.set_title('Four most concentrated positive MLP sides · marks a space',fontsize=11)
    fig.savefig(ROOT/'evidence'/'part3-mlp-units.png',dpi=180);plt.close(fig)
    print('median share trained',report['trained_median_share'],'initial',report['initial_median_share'])


if __name__=='__main__':main()
