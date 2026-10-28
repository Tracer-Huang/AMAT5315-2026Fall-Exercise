"""Optional eight-head attention maps for the first held-out story opening."""
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm

from decoder import forward

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'artifacts'/'transformer';cfg=json.loads((out/'config.json').read_text())
    with np.load(out/'model.npz',allow_pickle=False) as z:params={k:jnp.asarray(z[k]) for k in z.files}
    valid=np.load(ROOT/'artifacts'/'valid-tokens.npy');ids=valid[1:33].astype(np.int32)
    _,attention=forward(params,jnp.asarray(ids[None,:]),cfg,return_attention=True)
    weights=np.stack([np.asarray(a)[0] for a in attention])
    if np.max(np.abs(np.triu(weights,1)))!=0:raise ValueError('future attention weight is nonzero')
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/'data'/'tok512.model'))
    labels=[sp.id_to_piece(int(i)).replace('▁','·') for i in ids]
    with (out/'attention-maps.npz').open('xb') as f:np.savez_compressed(f,weights=weights,ids=ids)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for layer in range(2):
        for head in range(4):
            ax=axes[layer,head];im=ax.imshow(weights[layer,head],vmin=0,vmax=1,cmap='viridis')
            ax.set_title(f'Layer {layer+1} · head {head+1}',fontsize=9)
            ax.set_xticks(range(len(ids)),labels,rotation=90,fontsize=4)
            ax.set_yticks(range(len(ids)),labels,fontsize=4)
            if head==0:ax.set_ylabel('query position')
            if layer==1:ax.set_xlabel('key position')
    fig.colorbar(im,ax=axes,shrink=.7,label='attention weight')
    fig.suptitle('Eight causal heads · first 32 tokens of a held-out story',fontsize=11)
    fig.savefig(ROOT/'evidence'/'part5-attention-heads.png',dpi=180);plt.close(fig)
    print('attention maps',weights.shape,'upper triangle',float(np.max(np.abs(np.triu(weights,1)))),
          'row error',float(np.max(np.abs(weights.sum(-1)-1))))


if __name__=='__main__':main()
