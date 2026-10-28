"""Full held-out scoring, sampler checks and generation for the trained decoder."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
from scipy.stats import chi2

from decoder import forward
from ngrams import load_model, next_with_short_context
from sampling import distribution, broadest_context

ROOT=Path(__file__).resolve().parent


def calibration(p,rng,draws=100000):
    samples=rng.choice(len(p),size=draws,p=p)
    observed=np.bincount(samples,minlength=len(p));expected=draws*p
    common=expected>=5
    o=np.r_[observed[common],observed[~common].sum()]
    e=np.r_[expected[common],expected[~common].sum()]
    use=e>0;o=o[use];e=e[use]
    statistic=float(np.sum((o-e)**2/e));dof=len(e)-1
    if dof < 1:raise ValueError('context is too concentrated for a chi-squared goodness-of-fit test')
    return dict(draws=draws,chi_square=statistic,dof=dof,p_value=float(chi2.sf(statistic,dof))),observed


def evaluate(params,cfg,valid,characters):
    count=len(valid)-1;context=cfg['context'];blocks=(count+context-1)//context
    x=np.zeros(blocks*context,dtype=np.int32);y=np.zeros_like(x);mask=np.zeros_like(x,dtype=bool)
    x[:count]=valid[:-1];y[:count]=valid[1:];mask[:count]=True
    x=x.reshape(blocks,context);y=y.reshape(blocks,context);mask=mask.reshape(blocks,context)
    @jax.jit
    def nll(inputs,targets):
        lp=jax.nn.log_softmax(forward(params,inputs,cfg),axis=-1)
        return -jnp.take_along_axis(lp,targets[...,None],axis=-1)[...,0]
    batch=128;total=0.;positions=np.zeros(context);squares=np.zeros(context);counts=np.zeros(context,dtype=int)
    for start in range(0,blocks,batch):
        used=min(batch,blocks-start)
        xx=np.zeros((batch,context),dtype=np.int32);yy=np.zeros_like(xx)
        xx[:used]=x[start:start+used];yy[:used]=y[start:start+used]
        v=np.asarray(nll(xx,yy),dtype=np.float64)[:used]; m=mask[start:start+used]
        total+=float((v*m).sum());positions+=(v*m).sum(axis=0);squares+=(v*v*m).sum(axis=0);counts+=m.sum(axis=0)
        if start and start%(batch*40)==0:print('scored',min(start+batch,blocks),'/',blocks,'blocks',flush=True)
    means=positions/counts
    se=np.sqrt(np.maximum(squares/counts-means**2,0)/(counts-1))
    return dict(summed_nll=total,targets=count,loss=total/count,ppl=float(np.exp(total/count)),
                bpc=float(total/(np.log(2)*characters)),blocks=blocks,last_block=int(mask[-1].sum()),
                position_loss=means.tolist(),position_se=se.tolist(),position_counts=counts.tolist())


def entropy_check(params,cfg,temperature,seed):
    batch=128;length=128; context=cfg['context']
    @jax.jit
    def generate(key,temp):
        initial=jnp.zeros((batch,length+1),dtype=jnp.int32).at[:,0].set(2)
        def step(carry,t):
            ids,key=carry;key,sub=jax.random.split(key)
            start=jnp.maximum(t-(context-1),0)
            window=jax.lax.dynamic_slice(ids,(0,start),(batch,context))
            logits=forward(params,window,cfg)[:,jnp.minimum(t,context-1),:]
            logp=jax.nn.log_softmax(logits,axis=-1);p=jnp.exp(logp)
            entropy=-jnp.sum(p*logp,axis=-1)
            chosen=jax.random.categorical(sub,logits/temp,axis=-1)
            surprise=-jnp.take_along_axis(logp,chosen[:,None],axis=-1)[:,0]
            ids=ids.at[:,t+1].set(chosen)
            return (ids,key),(surprise,entropy)
        return jax.lax.scan(step,(initial,key),jnp.arange(length))
    (_, _),(surprise,entropy)=generate(jax.random.PRNGKey(seed),jnp.asarray(temperature))
    surprise=np.asarray(surprise,dtype=np.float64);entropy=np.asarray(entropy,dtype=np.float64)
    per_sequence=(surprise-entropy).mean(axis=0)
    z=float(per_sequence.mean()/(per_sequence.std(ddof=1)/np.sqrt(batch)))
    return dict(temperature=temperature,seed=seed,sequences=batch,tokens_per_sequence=length,
                mean_surprise=float(surprise.mean()),mean_entropy=float(entropy.mean()),z=z),surprise,entropy


def main(version):
    out=ROOT/'artifacts'/'transformer';cfg=json.loads((out/'config.json').read_text())
    training=json.loads((out/'training.json').read_text());model=out/'model.npz'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=training['model_sha256']:raise ValueError('model checksum mismatch')
    with np.load(model,allow_pickle=False) as z:params={k:jnp.asarray(z[k]) for k in z.files}
    meta=json.loads((ROOT/'artifacts'/'data-manifest.json').read_text()); valid=np.load(ROOT/'artifacts'/'valid-tokens.npy')
    measurements=evaluate(params,cfg,valid,meta['valid_characters'])
    print('held-out loss',measurements['loss'],'BPC',measurements['bpc'],flush=True)
    p2=json.loads((ROOT/'evidence'/'part2-ngrams.json').read_text())
    starts=np.linspace(0,len(valid)-cfg['context'],64,dtype=np.int64)
    candidates=np.stack([valid[start:start+cfg['context']] for start in starts]).astype(np.int32)
    candidate_logits=np.asarray(forward(params,jnp.asarray(candidates),cfg)[:,-1,:],dtype=np.float64)
    chosen,entropy=broadest_context(candidate_logits)
    context_ids=candidates[chosen];broad=candidate_logits[chosen]
    @jax.jit
    def predict(window,index):return forward(params,window,cfg)[0,index]
    def next_logits(ids):
        ids=ids[-cfg['context']:];x=np.zeros((1,cfg['context']),dtype=np.int32);x[0,:len(ids)]=ids
        return np.asarray(predict(x,len(ids)-1),dtype=np.float64)
    settings=[('temperature_1',1.,'temperature',None),('temperature_0.8',.8,'temperature',None),
              ('top_k_20',.8,'top_k',20),('top_p_0.9',.8,'top_p',.9),('min_p_0.1',.7,'min_p',.1)]
    rng=np.random.default_rng(531508);checks={};arrays={}
    for name,temp,mode,value in settings:
        p=distribution(broad,temp,mode,value);checks[name],arrays[name]=calibration(p,rng)
        arrays[name+'_probabilities']=p
    greedy=distribution(broad,.0001);checks['greedy']=dict(mode=int(np.argmax(broad)),
        all_greedy=bool(np.all(rng.choice(len(greedy),size=100000,p=greedy)==np.argmax(broad))))
    with (out/f'sampler-counts-v{version}.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/'data'/'tok512.model'))
    def story(decoder,temp=1.,mode='temperature',value=None,seed=1848):
        rng=np.random.default_rng(seed);tokens=[meta['eos']]
        for _ in range(60):
            probs=decoder(tokens,temp,mode,value);tokens.append(int(rng.choice(len(probs),p=probs)))
        return dict(seed=seed,tokens=tokens[1:],text=sp.decode([x for x in tokens[1:] if x!=meta['eos']]))
    def trans_decoder(tokens,temp,mode,value):return distribution(next_logits(tokens),temp,mode,value)
    snippets={}
    snippets['uniform']=story(lambda t,temp,mode,value:np.full(cfg['vocab'],1/cfg['vocab']))
    models={n:load_model(ROOT/'artifacts'/'ngrams'/f'n{n}.npz') for n in (1,2,3)}
    for n in (1,2,3):
        snippets[f'ngram_{n}']=story(lambda t,temp,mode,value,order=n:next_with_short_context(models,order,t))
    snippets['transformer']=story(trans_decoder)
    generated={name:story(trans_decoder,temp,mode,value,seed=1848) for name,temp,mode,value in settings if name!='temperature_1'}
    e1,s1,h1=entropy_check(params,cfg,1.,531508)
    e07,s07,h07=entropy_check(params,cfg,.7,531509)
    with (out/f'entropy-samples-v{version}.npz').open('xb') as f:np.savez_compressed(f,surprise1=s1,entropy1=h1,surprise07=s07,entropy07=h07)
    report=dict(model_sha256=training['model_sha256'],parameters=training['parameters'],heldout=measurements,
                causality='causal model; exact probe separately tested',sampler_checks=checks,
                sampler_context=dict(selection='maximum entropy of 64 evenly spaced validation windows, before sampling',
                                     start=int(starts[chosen]),tokens=context_ids.tolist(),entropy=entropy),
                evaluation_version=version, ngram_startup='use the longest available n-gram until enough context exists',
                entropy_checks=[e1,e07],comparison_samples=snippets,generated_samples=generated,
                position_context_policy='consecutive 64-token blocks; input at position t predicts shifted target')
    (ROOT/'evidence'/'part5-transformer.json').write_text(json.dumps(report,indent=2)+'\n')
    names=['uniform','unigram','bigram','trigram','transformer']
    values=[p2['uniform']['bpc']]+[p2['models'][str(n)]['bpc'] for n in (1,2,3)]+[measurements['bpc']]
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);bars=ax.bar(names,values,color=['#999','#567fa9','#567fa9','#567fa9','#389466'])
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.06,f'{value:.3f}',ha='center')
    ax.set(ylabel='held-out bits / character',ylim=(0,max(values)*1.16))
    fig.savefig(ROOT/'evidence'/'part5-model-comparison.png',dpi=180);plt.close(fig)
    means=np.array(measurements['position_loss']);se=np.array(measurements['position_se']);pos=np.arange(1,len(means)+1)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);ax.plot(pos,means,label='transformer')
    ax.fill_between(pos,means-2*se,means+2*se,alpha=.2)
    for n in (2,3):ax.axhline(p2['models'][str(n)]['loss'],ls='--',label=f'{n}-gram')
    ax.set(xlabel='tokens read',ylabel='held-out nats / token');ax.legend()
    fig.savefig(ROOT/'evidence'/'part5-context-loss.png',dpi=180);plt.close(fig)
    print('sampler p-values',{k:v.get('p_value') for k,v in checks.items()},'entropy z',[e1['z'],e07['z']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=1)
    main(parser.parse_args().version)
