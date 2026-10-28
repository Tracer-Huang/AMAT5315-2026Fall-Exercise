"""Measure key-value memory interference and capacity."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('outer','least_squares','relu','relu2','relu3')
DIMENSIONS=(16,32,64)
SIZES=tuple(2**n for n in range(3,17))
MAX_QUERIES=256


def measure(keys, values, indices):
    keys=np.asarray(keys,dtype=np.float64);values=np.asarray(values,dtype=np.float64)
    indices=np.asarray(indices,dtype=int);queries=keys[indices];targets=values[indices]
    if len(keys)!=len(values) or len(keys)==0:raise ValueError('keys and values must have matching rows')
    own=np.einsum('ij,ij->i',queries,queries);value_norm=np.linalg.norm(targets,axis=1)
    if np.any(own<=0) or np.any(value_norm==0):raise ValueError('zero key or value')
    outer=queries@(keys.T@values)-own[:,None]*targets
    result={'outer':np.linalg.norm(outer,axis=1)/(own*value_norm)}
    if len(keys)<=2*keys.shape[1]:
        inverse=np.linalg.pinv(keys)
        mapping=inverse@values
        self_weight=np.einsum('ij,ij->i',queries,inverse.T[indices])
    else:
        gram_inv=np.linalg.inv(keys.T@keys)
        mapping=gram_inv@(keys.T@values)
        self_weight=np.einsum('ij,ij->i',queries@gram_inv,queries)
    noise=queries@mapping-self_weight[:,None]*targets
    result['least_squares']=np.linalg.norm(noise,axis=1)/(self_weight*value_norm)
    accum={name:[] for name in ('relu','relu2','relu3')}
    for start in range(0,len(indices),32):
        index=indices[start:start+32];s=keys[index]@keys.T
        s[np.arange(len(index)),index]=0
        positive=np.maximum(s,0)
        for power,name in [(1,'relu'),(2,'relu2'),(3,'relu3')]:
            interference=(positive**power)@values
            signal=(own[start:start+len(index)]**power)*value_norm[start:start+len(index)]
            accum[name].extend((np.linalg.norm(interference,axis=1)/signal).tolist())
    result.update({k:np.asarray(v) for k,v in accum.items()})
    return result


def crossing(sizes, values):
    for i in range(1,len(sizes)):
        if values[i]>=1 and values[i-1]<1:
            x0,x1=np.log(sizes[i-1]),np.log(sizes[i]); y0,y1=np.log(max(values[i-1],1e-15)),np.log(values[i])
            return float(np.exp(x0+(x1-x0)*(-y0)/(y1-y0)))
    return None


def main():
    folder=ROOT/'artifacts'/'capacity';folder.mkdir(parents=True,exist_ok=True)
    ratios={str(d):{m:[] for m in METHODS} for d in DIMENSIONS}
    for d in DIMENSIONS:
        for M in SIZES:
            per_method={m:[] for m in METHODS}
            for seed in range(3):
                path=folder/f'd{d}-m{M}-seed{seed}.npz'
                if not path.exists():
                    rng=np.random.default_rng(np.random.SeedSequence([531508,d,M,seed]))
                    k=rng.normal(size=(M,d));k/=np.linalg.norm(k,axis=1,keepdims=True)
                    v=rng.normal(size=(M,d));v/=np.linalg.norm(v,axis=1,keepdims=True)
                    indexes=np.arange(min(M,MAX_QUERIES));r=measure(k,v,indexes)
                    with path.open('xb') as f:np.savez_compressed(f,**r,dimension=d,M=M,seed=seed,queries=len(indexes))
                with np.load(path,allow_pickle=False) as z:
                    if int(z['queries'])!=min(M,MAX_QUERIES):raise ValueError('cached query policy differs')
                    for name in METHODS:per_method[name].append(float(np.median(z[name])))
            for name in METHODS:ratios[str(d)][name].append(float(np.median(per_method[name])))
            print('d',d,'M',M,'queries',min(M,MAX_QUERIES),'median', {k:round(v[-1],4) for k,v in ratios[str(d)].items()},flush=True)
    caps={name:[crossing(SIZES,ratios[str(d)][name]) for d in DIMENSIONS] for name in METHODS}
    slopes={name:float(np.polyfit(np.log(DIMENSIONS),np.log(v),1)[0]) if all(x is not None for x in v) else None
            for name,v in caps.items()}
    report=dict(dimensions=DIMENSIONS,sizes=SIZES,draws=3,query_policy='first min(M, 256) stored keys per draw',
                max_queries=MAX_QUERIES,seed_scheme='SeedSequence([531508,d,M,draw])',ratios=ratios,
                capacities=caps,capacity_slopes=slopes)
    (ROOT/'evidence'/'part3-capacity.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for name in METHODS:ax.loglog(SIZES,np.maximum(ratios['64'][name],1e-15),'o-',markersize=3,label=name)
    m=np.asarray(SIZES,dtype=float)
    for value,label in [(np.sqrt(m/64),'outer theory'),(np.sqrt(m/128),'ReLU theory'),
                        (np.sqrt(3*m/(2*64**2)),'ReLU² theory'),(np.sqrt(15*m/(2*64**3)),'ReLU³ theory')]:
        ax.loglog(m,value,'--',alpha=.35,label=label)
    ax.axhline(1,color='black',lw=.7);ax.set(xlabel='number of memories M',ylabel='interference / signal R');ax.legend(fontsize=7,ncol=2)
    fig.savefig(ROOT/'evidence'/'part3-interference.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for name in METHODS:ax.loglog(DIMENSIONS,caps[name],'o-',label=f"{name} slope={slopes[name]:.2f}" if slopes[name] is not None else name)
    ax.set(xlabel='key dimension d',ylabel='capacity at R = 1');ax.legend(fontsize=8)
    fig.savefig(ROOT/'evidence'/'part3-capacity.png',dpi=180);plt.close(fig)
    print('capacity slopes',slopes,flush=True)


if __name__=='__main__':main()
