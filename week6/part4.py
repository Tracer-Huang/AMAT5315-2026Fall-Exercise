import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,time
import numpy as np
import jax
import jax.numpy as jnp
from data import mnist,ROOT
from models import recall,modern,attention
from plotting import plt,save,rows_gallery
RAW=ROOT/'artifacts/modern'

def main():
    RAW.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    images,labels=mnist('train');spins=jnp.asarray(2*(images[:1000]>=128).reshape(1000,784).astype(np.int32)-1)
    Ps=[3,10,30,100,300,1000];betas=[.001,.003,.01,.03,.1,.3,1.];records=[];maxdiff=0;example=None
    for pi,P in enumerate(Ps):
        X=spins[:P];target_indices=jnp.arange(100)%P;targets=X[target_indices]
        seed=6640+pi;random=2*jax.random.bernoulli(jax.random.key(seed),.5,targets.shape).astype(jnp.int32)-1
        queries=jnp.where(jnp.arange(784)<392,targets,random)
        C=X.T@X;C=C.at[jnp.diag_indices(784)].set(0)
        classical,sweeps,converged=recall(C,queries,jax.random.key(seed+1000));assert bool(jnp.all(converged))
        assert bool(jnp.all(classical*(classical@C)>=0))
        def success(results):
            signs=jnp.where(results>=0,1,-1)
            overlaps=jnp.mean(signs*targets,axis=-1)
            nearest=jnp.argmax(signs@X.T,axis=-1)
            return np.array((overlaps>=.95)&(nearest==target_indices)),np.array(overlaps),np.array(nearest)
        classic_success,classic_overlap,classic_nearest=success(classical);results=[];modern_successes=[];differences=[];weights=[]
        for beta in betas:
            out=modern(X.astype(jnp.float32),queries.astype(jnp.float32),beta)
            att=attention(X.astype(jnp.float32),queries.astype(jnp.float32),beta)
            diff=float(jnp.max(jnp.abs(out-att)));maxdiff=max(maxdiff,diff)
            succ,overlap,nearest=success(out);modern_successes.append(succ);results.append(np.array(out));differences.append(diff)
            weights.append(np.array(jax.nn.softmax(beta*queries[:1].astype(jnp.float32)@X.T,axis=-1))[0])
            records.append(dict(P=P,beta=beta,modern_success=float(succ.mean()),classical_success=float(classic_success.mean()),attention_max_abs_difference=diff,seed=seed))
        np.savez(RAW/f'p{P}.npz',patterns=np.array(X),targets=np.array(targets),target_indices=np.array(target_indices),queries=np.array(queries),classical=np.array(classical),sweeps=np.array(sweeps),modern=np.stack(results),modern_success=np.stack(modern_successes),classical_success=classic_success,betas=betas,seed=seed)
        if P==100:example=dict(X=X,queries=queries,targets=targets,classical=classical,results=results,weights=weights)
        print(f'P={P}: classical={classic_success.mean():.2f}; modern beta .3={np.mean(modern_successes[5]):.2f}; attention max={max(differences):.3g}',flush=True)
    ex=example;fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for ax,b in zip(axes,[.003,.03,.3]):
        weights=ex['weights'][betas.index(b)];ax.bar(np.arange(100),weights);ax.set_ylabel('Softmax weight');ax.set_title(f'beta={b}; target index 0 weight={weights[0]:.4f}')
    axes[-1].set_xlabel('Stored pattern index');fig.tight_layout();save(fig,'attention-weights.png')
    rows_gallery([np.array(ex['targets'][:6]),np.array(ex['queries'][:6])]+[r[:6] for r in ex['results']]+[np.array(ex['classical'][:6])],['target','half-erased query']+[f'modern beta={b}' for b in betas]+['classical'], 'modern-beta-gallery.png','P=100: one attention step vs asynchronous classical recall',vmin=-1,vmax=1)
    rows_gallery([np.array(ex['targets'][:6]),np.array(ex['queries'][:6]),np.array(ex['classical'][:6]),ex['results'][5][:6]],['target','query','classical','modern beta=.3'],'modern-retrieval-gallery.png','Stored-digit retrieval at P=100',vmin=-1,vmax=1)
    fig,ax=plt.subplots(figsize=(9,5))
    for b in betas:
        rr=[r for r in records if r['beta']==b];ax.plot(Ps,[r['modern_success'] for r in rr],marker='o',label=f'modern beta={b}')
    ax.plot(Ps,[r['classical_success'] for r in records if r['beta']==.3],marker='s',c='black',ls='--',label='classical')
    ax.set_xscale('log');ax.set_xlabel('Number P of stored digits');ax.set_ylabel('Success (overlap >= .95 and target nearest)');ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'modern-success.png')
    print(f'Largest update-attention difference over all queries/P/beta: {maxdiff:.9g}',flush=True)
    metrics=dict(Ps=Ps,betas=betas,queries=100,seed_rule='6640+P_index, recall adds 1000',nearest_tie_rule='argmax first stored index; a duplicate at an earlier index counts as failure',attention_max_abs_difference=maxdiff,runtime_seconds=time.perf_counter()-start,records=records)
    (RAW/'settings-and-metrics.json').write_text(json.dumps(metrics,indent=2))
    (ROOT/'evidence/modern-metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=str,default=str(RAW),help='New empty raw-output directory; existing directory is refused')
    args=parser.parse_args()
    from pathlib import Path
    RAW=Path(args.raw)
    main()
