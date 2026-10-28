"""Part 2: sparse n-gram models over the pinned BPE token stream."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
KS = (0.0001, .001, .01, .1, 1.)


def pack_tokens(tokens, n, eos=2):
    if not 1 <= n <= 6:
        raise ValueError('n must be between 1 and 6')
    tokens = np.asarray(tokens, dtype=np.uint16)
    if len(tokens) < 2:
        raise ValueError('at least one context and one target are required')
    if n == 1:
        return tokens[1:].astype(np.uint64)
    padded = np.concatenate((np.full(n-2, eos, dtype=np.uint16), tokens))
    count = len(tokens)-1
    packed = np.zeros(count, dtype=np.uint64)
    for j in range(n):
        packed = (packed << np.uint64(9)) | padded[j:j+count].astype(np.uint64)
    return packed


def fit_counts(tokens, n, vocab=512, k=.1, eos=2):
    if not 1 <= vocab <= 512 or k <= 0:
        raise ValueError('vocabulary must fit in nine bits and k must be positive')
    keys, counts = np.unique(pack_tokens(tokens,n,eos),return_counts=True)
    contexts = np.zeros_like(keys) if n == 1 else keys >> np.uint64(9)
    unique_contexts, first = np.unique(contexts,return_index=True)
    totals = np.add.reduceat(counts,first)
    return dict(n=n,vocab=vocab,k=k,eos=eos,keys=keys,counts=counts,contexts=unique_contexts,totals=totals)


def score_tokens(model, tokens):
    keys=pack_tokens(tokens,model['n'],model['eos'])
    contexts=np.zeros_like(keys) if model['n']==1 else keys >> np.uint64(9)
    def lookup(sorted_keys, values, query):
        indexes=np.searchsorted(sorted_keys,query)
        valid=indexes<len(sorted_keys)
        safe=np.minimum(indexes,len(sorted_keys)-1)
        matched=valid & (sorted_keys[safe]==query)
        return np.where(matched,values[safe],0),matched
    count, seen=lookup(model['keys'],model['counts'],keys)
    total,_=lookup(model['contexts'],model['totals'],contexts)
    return (count+model['k'])/(total+model['k']*model['vocab']),seen


def next_distribution(model, history):
    n=model['n'];v=model['vocab']; k=model['k']
    needed=n-1
    past=([model['eos']]*needed+list(history))[-needed:] if needed else []
    context=0
    for token in past: context=(context<<9)|int(token)
    candidates=(np.uint64(context)<<np.uint64(9)) | np.arange(v,dtype=np.uint64) if n>1 else np.arange(v,dtype=np.uint64)
    i=np.searchsorted(model['keys'],candidates); safe=np.minimum(i,len(model['keys'])-1)
    counts=np.where((i<len(model['keys'])) & (model['keys'][safe]==candidates),model['counts'][safe],0)
    return (counts+k)/(counts.sum()+v*k)


def next_with_short_context(models, n, history):
    return next_distribution(models[min(n, len(history)+1)], history)


def save_model(model, path):
    with path.open('xb') as f:
        np.savez_compressed(f, **model)


def load_model(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:(z[k].item() if z[k].ndim==0 else z[k]) for k in z.files}


def sample_check(p, context_counts, seed=531508, length=250000, output_path=None):
    rng=np.random.default_rng(seed); cdf=np.cumsum(p,axis=1); cdf[:,-1]=1.
    seq=np.empty(length+1,dtype=np.int32);seq[0]=2
    draws=rng.random(length)
    for i,u in enumerate(draws):
        seq[i+1]=np.searchsorted(cdf[seq[i]],u)
    obs=np.bincount(seq[:-1]*512+seq[1:],minlength=512**2).reshape(512,512)
    if output_path is not None:
        with Path(output_path).open('xb') as f:
            np.savez_compressed(f,sequence=seq,observed=obs,probabilities=p,seed=seed)
    top=np.argsort(context_counts)[-20:]
    statistic=0.;dof=0; group_details=[]
    for row in top:
        expected=obs[row].sum()*p[row]
        common=expected>=5
        o=np.r_[obs[row,common],obs[row,~common].sum()]
        e=np.r_[expected[common],expected[~common].sum()]
        use=e>0; o=o[use];e=e[use]
        statistic+=float(np.sum((o-e)**2/e));dof+=len(e)-1
        group_details.append(dict(context=int(row),draws=int(obs[row].sum()),bins=len(e)))
    return dict(seed=seed,generated=length,chi_square=statistic,dof=dof,p_value=float(chi2.sf(statistic,dof)),
                groups=group_details)


def main():
    train=np.load(ROOT/'artifacts'/'train-tokens.npy',mmap_mode='r')
    valid=np.load(ROOT/'artifacts'/'valid-tokens.npy',mmap_mode='r')
    meta=json.loads((ROOT/'artifacts'/'data-manifest.json').read_text())
    model_dir=ROOT/'artifacts'/'ngrams';model_dir.mkdir(exist_ok=True)
    results={}; bigram=None
    split=int(.9*len(train))
    for n in range(1,7):
        print(f'n={n}: tuning on last 10% of training tokens',flush=True)
        pre=fit_counts(train[:split],n,k=1.)
        prefix=max(1,n-1); dev=train[split-prefix:]; trim=prefix-1
        tuning=[]
        for k in KS:
            pre['k']=k; probabilities,_=score_tokens(pre,dev)
            tuning.append(float(-np.log(probabilities[trim:]).mean()))
        best=KS[int(np.argmin(tuning))]
        del pre
        model=fit_counts(train,n,k=best)
        probabilities,seen=score_tokens(model,valid)
        loss=float(-np.log(probabilities).mean());nll=loss*len(probabilities)
        results[str(n)]=dict(k=best,dev_losses=tuning,loss=loss,ppl=float(np.exp(loss)),
                             bpc=float(nll/(np.log(2)*meta['valid_characters'])),
                             targets=len(probabilities),unseen_fraction=float(np.mean(~seen)),
                             distinct_ngrams=len(model['keys']))
        save_model(model,model_dir/f'n{n}.npz')
        if n==2: bigram=model
        print(' ',results[str(n)],flush=True)
    uniform_loss=float(np.log(512));uniform=dict(loss=uniform_loss,ppl=512.,
            bpc=float(uniform_loss*(len(valid)-1)/(np.log(2)*meta['valid_characters'])),targets=len(valid)-1)
    counts=np.bincount(train[:-1].astype(np.int64)*512+train[1:],minlength=512**2).reshape(512,512)
    p=(counts+bigram['k'])/(counts.sum(axis=1,keepdims=True)+512*bigram['k'])
    first=p[meta['eos']]
    sums=[float(first.sum()),float((first[:,None]*p).sum()),
          float(sum(np.sum(first[i]*p[i,:,None]*p) for i in range(512)))]
    sample=sample_check(p,counts.sum(axis=1))
    result=dict(uniform=uniform,models=results,normalization_sums=sums,bigram_sample=sample,
                selection='last 10% of training tokens; counts fit on first 90%, then refit on all training',
                validation_characters=meta['valid_characters'],validation_targets=len(valid)-1)
    (ROOT/'evidence'/'part2-ngrams.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    ns=np.arange(1,7);axes[0].plot(ns,[results[str(n)]['bpc'] for n in ns],'o-')
    axes[0].set(xlabel='n in n-gram',ylabel='held-out bits / character',xticks=ns)
    axes[1].plot(ns,[100*results[str(n)]['unseen_fraction'] for n in ns],'o-',color='#c96a38')
    axes[1].set(xlabel='n in n-gram',ylabel='unseen held-out n-grams (%)',xticks=ns)
    fig.savefig(ROOT/'evidence'/'part2-ngram-ladder.png',dpi=180);plt.close(fig)
    print('uniform',uniform,'normalization',sums,'sampling p',sample['p_value'],flush=True)


if __name__ == '__main__':
    main()
