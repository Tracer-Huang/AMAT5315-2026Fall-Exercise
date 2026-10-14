"""Full 4 sizes x 21 loads x 20 independent pattern sets, exact integer recall fields."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,time,math
import jax
import jax.numpy as jnp
import numpy as np
from models import recall
from data import ROOT
from plotting import plt,save
RAW=ROOT/'artifacts/hopfield'

def main():
    RAW.mkdir(parents=True,exist_ok=False)
    sizes=[100,200,400,800];loads=np.round(np.arange(.05,.251,.01),2);records=[];start=time.perf_counter()
    for n in sizes:
        t=time.perf_counter()
        for li,alpha in enumerate(loads):
            p=int(round(n*alpha));unstable=[];overlaps=[];success=[];sweepcounts=[]
            for rep in range(20):
                seed=6630+n*10000+li*100+rep
                patterns=(2*jax.random.bernoulli(jax.random.key(seed),.5,(p,n)).astype(jnp.int32)-1)
                C=patterns.T@patterns;C=C.at[jnp.diag_indices(n)].set(0)
                # C=N*J has the same signs/zero fields as J; integers avoid tie roundoff.
                unstable.append(float(jnp.mean(patterns*(patterns@C)<0)))
                states,sweeps,converged=recall(C,patterns[:min(10,p)],jax.random.key(seed+1_000_000))
                assert bool(jnp.all(converged)),f'Unconverged recall N={n} P={p} seed={seed}'
                overlap=np.array(jnp.mean(states*patterns[:min(10,p)],axis=-1))
                # Independent final-field/energy checks on every recall.
                assert bool(jnp.all(states*(states@C)>=0))
                e0=-np.einsum('bi,ij,bj->b',np.array(patterns[:len(states)],dtype=np.int64),np.array(C,dtype=np.int64),np.array(patterns[:len(states)],dtype=np.int64))/2
                e1=-np.einsum('bi,ij,bj->b',np.array(states,dtype=np.int64),np.array(C,dtype=np.int64),np.array(states,dtype=np.int64))/2
                assert np.all(e1<=e0)
                overlaps.append(overlap);success.append(float(np.mean(overlap>=.9)));sweepcounts.append(np.array(sweeps))
                np.savez(RAW/f'n{n}-p{p}-r{rep:02d}.npz',patterns=np.array(patterns),final=np.array(states),overlap=overlap,sweeps=np.array(sweeps),seed=seed)
            values=np.array([np.mean(o) for o in overlaps]);u=np.array(unstable);success=np.array(success)
            rec=dict(N=n,P=p,alpha=float(alpha),unstable_mean=float(u.mean()),unstable_se=float(u.std(ddof=1)/np.sqrt(20)),overlap_mean=float(values.mean()),overlap_se=float(values.std(ddof=1)/np.sqrt(20)),success_mean=float(success.mean()),success_se=float(success.std(ddof=1)/np.sqrt(20)),max_sweeps=int(np.max(sweepcounts)),se_unit='20 independent sets')
            records.append(rec);print(f'N={n} P={p} unstable={rec["unstable_mean"]:.5f} overlap={rec["overlap_mean"]:.4f}',flush=True)
            (RAW/'progress.json').write_text(json.dumps(records,indent=2))
        print(f'N={n} seconds={time.perf_counter()-t:.1f}',flush=True)
    fig,ax=plt.subplots(figsize=(9,5));grid=np.linspace(.05,.25,250)
    ax.plot(grid,[.5*math.erfc(1/math.sqrt(2*a)) for a in grid],c='black',label='Large N: Phi(-1/sqrt(alpha))')
    crossings={};fig2,axes=plt.subplots(1,2,figsize=(12,4.5))
    for n in sizes:
        rr=[r for r in records if r['N']==n];x=np.array([r['alpha'] for r in rr]);m=np.array([r['overlap_mean'] for r in rr])
        line=ax.errorbar(x,[r['unstable_mean'] for r in rr],yerr=[r['unstable_se'] for r in rr],marker='o',ms=3,label=f'N={n}')
        ax.plot(x,[.5*math.erfc(math.sqrt((n-1)/(r['P']-1))/math.sqrt(2)) for r in rr],ls=':',c=line.lines[0].get_color(),alpha=.6)
        for a,metric,se in [(axes[0],'overlap_mean','overlap_se'),(axes[1],'success_mean','success_se')]:a.errorbar(x,[r[metric] for r in rr],yerr=[r[se] for r in rr],label=f'N={n}',marker='o',ms=3)
        crossing=None
        for i in range(1,len(x)):
            if m[i-1]>=.9 and m[i]<.9:
                crossing=float(x[i-1]+(.9-m[i-1])*(x[i]-x[i-1])/(m[i]-m[i-1]));break
        crossings[str(n)]=crossing
    ax.set_xlabel('Load alpha=P/N');ax.set_ylabel('Fraction of unstable stored bits');ax.legend();ax.grid(alpha=.2);save(fig,'hopfield-unstable.png')
    for ax in axes:ax.axvline(.138,ls='--',c='black',label='Thermodynamic capacity .138');ax.set_xlabel('Load alpha=P/N');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean final target overlap');axes[0].axhline(.9,c='gray',ls=':');axes[1].set_ylabel('Fraction with final overlap >= .9');save(fig2,'hopfield-capacity.png')
    summary=dict(sizes=sizes,loads=loads.tolist(),sets_per_point=20,recalls_per_set='min(10,P)',seed_rule='6630+N*10000+load_index*100+repeat; recall adds 1000000',random_sweep_order='independent per recall and sweep',field_representation='int32 C=N*J, zero diagonal; exact signs and ties',runtime_seconds=time.perf_counter()-start,crossings_at_point9=crossings,records=records)
    (RAW/'settings-and-metrics.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'evidence/hopfield-metrics.json').write_text(json.dumps(summary,indent=2));print('Crossings',crossings,flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=str,default=str(RAW),help='New empty raw-output directory; existing directory is refused')
    args=parser.parse_args()
    from pathlib import Path
    RAW=Path(args.raw)
    main()
