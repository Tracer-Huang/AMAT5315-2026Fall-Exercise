"""Exact temperature and truncation distributions."""
import numpy as np


def distribution(logits, temperature=1., mode='temperature', value=None):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim != 1 or temperature <= 0 or not np.all(np.isfinite(logits)):
        raise ValueError('finite one-dimensional logits and a positive temperature are required')
    z=logits/temperature; p=np.exp(z-z.max());p/=p.sum()
    if mode=='temperature':
        keep=np.ones(len(p),dtype=bool)
    elif mode=='top_k':
        if not isinstance(value,int) or not 1 <= value <= len(p):raise ValueError('invalid top-k')
        keep=np.zeros(len(p),dtype=bool);keep[np.argsort(p)[-value:]]=True
    elif mode=='top_p':
        if not 0 < value <= 1:raise ValueError('invalid top-p')
        order=np.argsort(p)[::-1]; previous=np.cumsum(p[order])-p[order]
        keep=np.zeros(len(p),dtype=bool);keep[order[previous < value]]=True
    elif mode=='min_p':
        if not 0 <= value <= 1:raise ValueError('invalid min-p')
        keep=p >= p.max()*value
    else: raise ValueError('unknown sampling mode')
    p=np.where(keep,p,0);return p/p.sum()


def broadest_context(logits):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or not np.all(np.isfinite(logits)):raise ValueError('expected finite rows of logits')
    z=logits-logits.max(axis=1,keepdims=True);p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    entropy=-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=1)
    index=int(np.argmax(entropy))
    return index,float(entropy[index])
