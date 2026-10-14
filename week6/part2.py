"""Full-budget CD training, matched reconstructions/generation and controlled completion."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,time,tomllib,platform
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from data import mnist,ROOT
from models import train_batch,gibbs,ink_overlap
from plotting import plt,save,rows_gallery,gallery
RAW=ROOT/'artifacts/rbm-shared-seed'

@partial(jax.jit,static_argnames=('steps',))
def epoch(p,batches,key,steps,lr):
    def batch(carry,v):
        p,key=carry;p,key,loss=train_batch(p,v,key,steps,lr)
        return (p,key),loss
    (p,key),losses=jax.lax.scan(batch,(p,key),batches)
    return p,key,losses.mean()

def snapshots(p,start,key,steps,clamp=None,mask=None):
    states=[np.array(start)];v=start;previous=0
    for step in steps:
        v,key=gibbs(p,v,key,step-previous,clamp,mask);states.append(np.array(v));previous=step
    return np.stack(states)

def main():
    RAW.mkdir(parents=True,exist_ok=False)
    cfg=tomllib.loads((ROOT/'rbm.toml').read_text());(RAW/'rbm.toml').write_text((ROOT/'rbm.toml').read_text());t=cfg['training'];init=cfg['initialization'];ev=cfg['evaluation'];comp=cfg['completion']
    train,trainlabels=mnist('train');test,labels=mnist('test')
    data=jnp.asarray((train>=128).reshape(-1,784),jnp.float32)
    test=jnp.asarray((test[:ev['test_images']]>=128).reshape(-1,784),jnp.float32)
    freq=jnp.clip(data.mean(0),*init['pixel_frequency_clip'])
    p0=(jax.random.normal(jax.random.key(init['seed']),(784,128),dtype=jnp.float32)*init['weight_std'],jnp.log(freq/(1-freq)),jnp.full(128,init['hidden_bias'],dtype=jnp.float32))
    np.savez(RAW/'initialization.npz',W=np.array(p0[0]),a=np.array(p0[1]),b=np.array(p0[2]))
    def error(p):
        recon,_=gibbs(p,test,jax.random.key(ev['seed']),1)
        return float(jnp.mean(recon!=test))
    initial_error=error(p0);steps=comp['steps'];all_metrics={};models={}
    shuffle_key,gibbs_key=jax.random.split(jax.random.key(t['seed']))
    for k in [1,10]:
        p=p0;history=[dict(epoch=0,surrogate=None,reconstruction_error=initial_error)];start=time.perf_counter()
        for e in range(t['epochs']):
            order=jax.random.permutation(jax.random.fold_in(shuffle_key,e),t['images'])
            batches=data[order].reshape(-1,t['batch_size'],784)
            p,_,loss=epoch(p,batches,jax.random.fold_in(gibbs_key,e),k,t['learning_rate'])
            history.append(dict(epoch=e+1,surrogate=float(loss),reconstruction_error=error(p)))
            np.savez(RAW/f'cd{k}-epoch{e+1:02d}.npz',W=np.array(p[0]),a=np.array(p[1]),b=np.array(p[2]))
            print(f'CD-{k} epoch {e+1}: surrogate {float(loss):.5f}; held-out Hamming {history[-1]["reconstruction_error"]:.5f}',flush=True)
        models[k]=p;all_metrics[str(k)]=dict(history=history,training_seconds=time.perf_counter()-start)
        (RAW/f'cd{k}-history.json').write_text(json.dumps(all_metrics[str(k)],indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in [1,10]:
        h=all_metrics[str(k)]['history'];axes[0].plot(range(1,21),[v['surrogate'] for v in h[1:]],label=f'CD-{k}')
        axes[1].plot(range(21),[v['reconstruction_error'] for v in h],label=f'CD-{k}')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean pre-update CD surrogate (not NLL)');axes[1].set_ylabel('Held-out sampled Hamming fraction');save(fig,'rbm-training.png')
    gallery(np.array(data[:16]).reshape(-1,28,28),trainlabels[:16],title='784 visible bits: ink +1, background -1',name='binarized-spins.png')
    fig,ax=plt.subplots(figsize=(9,4))
    visible=np.arange(8);hidden=np.arange(5)+1.5
    for i in visible:
        for h in hidden:ax.plot([i,h],[0,1],c='lightgray',lw=.7)
    ax.scatter(visible,np.zeros(8),s=180,label='784 visible bits v (8 drawn)');ax.scatter(hidden,np.ones(5),s=180,label='128 hidden bits h (5 drawn)')
    ax.text(3.5,.45,'All visible-hidden couplings W; no within-layer edges',ha='center',bbox=dict(facecolor='white',alpha=.85));ax.set_ylim(-.4,1.4);ax.legend(loc='upper center');ax.axis('off');save(fig,'rbm-architecture.png')
    start_random=jax.random.bernoulli(jax.random.key(6608),.5,(64,784)).astype(jnp.float32)
    for k,p in models.items():
        gen=snapshots(p,start_random,jax.random.key(6609),steps)
        recon=snapshots(p,test,jax.random.key(ev['seed']),steps)
        np.savez(RAW/f'cd{k}-sampling.npz',generation=gen,reconstruction=recon,steps=[0]+steps)
        rows_gallery(list(recon[:,:6])+list(gen[:,:6]),[f'recon t={s}' for s in [0]+steps]+[f'fresh t={s}' for s in [0]+steps],f'rbm-cd{k}-sampling.png',f'CD-{k}: held-out reconstruction vs fresh generation')
    labels=labels[:64];wrong_indices=[]
    for i in range(64):
        wrong_indices.append(next((i+offset)%64 for offset in range(1,64) if labels[(i+offset)%64]!=labels[i]))
    obs_true=jnp.repeat(test,comp['chains'],axis=0);obs_wrong=jnp.repeat(test[jnp.array(wrong_indices)],comp['chains'],axis=0)
    observed=jnp.concatenate([obs_true,obs_wrong]);mask=jnp.arange(784)<392
    key=jax.random.key(comp['seed']);key,kstart=jax.random.split(key)
    random=jax.random.bernoulli(kstart,.5,obs_true.shape).astype(jnp.float32)
    start=jnp.concatenate([random,random]);start=jnp.where(mask,observed,start)
    completion={};start_time=time.perf_counter()
    for k,p in models.items():
        states=snapshots(p,start,key,steps,observed,mask)
        assert np.array_equal(states[:,:,:392],np.broadcast_to(np.array(observed[:,:392]),states[:,:,:392].shape))
        dice=np.array(ink_overlap(jnp.asarray(states[1:,:,392:]),jnp.tile(obs_true[:,392:],(2,1)))).reshape(4,2,64,comp['chains'])
        per_digit=dice.mean(-1);means=per_digit.mean(-1);se=per_digit.std(-1,ddof=1)/8
        completion[str(k)]=dict(mean=means.tolist(),se=se.tolist(),advantage_at_1000=float(means[-1,0]-means[-1,1]))
        np.savez(RAW/f'cd{k}-completion.npz',states=states,per_digit_dice=per_digit,wrong_indices=wrong_indices,steps=[0]+steps)
        for kind,offset in [('true',0),('wrong',512)]:
            # Show two chains for each of three held-out digits (columns paired).
            ix=np.array([0,1,8,9,16,17])+offset
            rows_gallery([np.array(test)[[0,0,1,1,2,2]]]+[s[ix] for s in states],['true target']+[f'{kind} t={s}' for s in [0]+steps],f'rbm-cd{k}-completion-{kind}.png',f'CD-{k}, {kind} upper-half clamp: two chains per digit')
    fig,ax=plt.subplots(figsize=(8,4))
    for k in [1,10]:
        m=np.array(completion[str(k)]['mean']);se=np.array(completion[str(k)]['se'])
        for c,label in [(0,'true upper half'),(1,'wrong upper half')]:ax.errorbar(steps,m[:,c],yerr=se[:,c],marker='o',ls='-' if c==0 else '--',label=f'CD-{k}, {label}')
    ax.set_xscale('log');ax.set_xlabel('Gibbs steps');ax.set_ylabel('Lower-half ink Dice overlap');ax.legend();ax.grid(alpha=.2);save(fig,'rbm-completion-overlap.png')
    summary=dict(settings=cfg,random_streams='split training seed 6602 into shuffle and Gibbs roots; fold in epoch; reset roots per model',hidden_units=128,training=all_metrics,completion=completion,completion_seconds=time.perf_counter()-start_time,se_unit='64 digits; average eight chains within each digit first',environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,devices=str(jax.devices())))
    (RAW/'settings-and-metrics.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'evidence/rbm-metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(completion,indent=2),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=str,default=str(RAW),help='New empty raw-output directory; existing directory is refused')
    args=parser.parse_args()
    from pathlib import Path
    RAW=Path(args.raw)
    main()
