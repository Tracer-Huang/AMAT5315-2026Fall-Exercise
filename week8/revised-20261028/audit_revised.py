"""Audit the 2026-10-28 Week 8 learning-sheet outputs and raw records."""
import datetime
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.special import erf, logsumexp
from scipy.stats import chi2
from safetensors import safe_open

import decoder

ROOT=Path(__file__).resolve().parent
CHECKS=[]


def check(name, condition, detail=None):
    CHECKS.append(dict(name=name,passed=bool(condition),detail=detail))


def read(path):
    def reject(value):raise ValueError(f'non-finite JSON value {value}')
    return json.loads(path.read_text(),parse_constant=reject)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_forward(p,ids,cfg):
    """Float64 NumPy decoder; independently implements layer norm and exact GELU."""
    p={k:np.asarray(v,dtype=np.float64) for k,v in p.items()};b,t=ids.shape;d=cfg['width'];h=cfg['heads']
    def dense(x,name):return x@p[name+'.weight']+p[name+'.bias']
    def norm(x,name):
        mean=x.mean(axis=-1,keepdims=True);var=((x-mean)**2).mean(axis=-1,keepdims=True)
        return (x-mean)/np.sqrt(var+1e-5)*p[name+'.scale']+p[name+'.bias']
    x=p['token_embedding'][ids]+p['position_embedding'][:t]
    for i in range(cfg['layers']):
        prefix=f'blocks.{i}.';qkv=dense(norm(x,prefix+'norm1'),prefix+'attention.qkv')
        q,k,v=[a.reshape(b,t,h,d//h).transpose(0,2,1,3) for a in np.split(qkv,3,axis=-1)]
        s=q@k.swapaxes(-1,-2)/np.sqrt(d//h);s=np.where(np.arange(t)[:,None]>=np.arange(t)[None,:],s,-np.inf)
        weights=np.exp(s-logsumexp(s,axis=-1,keepdims=True))
        attended=(weights@v).transpose(0,2,1,3).reshape(b,t,d)
        x=x+dense(attended,prefix+'attention.output')
        u=dense(norm(x,prefix+'norm2'),prefix+'mlp.up')
        gelu=.5*u*(1+erf(u/np.sqrt(2)))
        x=x+dense(gelu,prefix+'mlp.down')
    return dense(norm(x,'final_norm'),'output')


def main():
    data=read(ROOT/'artifacts'/'data-manifest.json'); p1=read(ROOT/'evidence'/'part1-relations.json')
    p2=read(ROOT/'evidence'/'part2-ngrams.json');cap=read(ROOT/'evidence'/'part3-capacity.json')
    mlp=read(ROOT/'evidence'/'part3-mlp.json');pic=read(ROOT/'evidence'/'part4-picture.json')
    p5=read(ROOT/'evidence'/'part5-transformer.json');p6=read(ROOT/'evidence'/'part6-chinchilla.json')
    for name,digest in data['source_sha256'].items():check('input SHA256: '+name,sha(ROOT/'data'/name)==digest,digest)
    check('story split',(data['train_stories'],data['valid_stories'])==(24867,2763))
    ratio=data['valid_characters']/data['valid_tokens']
    check('tokenizer vocabulary and characters per token',data['vocab']==512 and abs(ratio-2.05)<.03,ratio)
    check('validation decode mismatches',len(data['valid_decode_mismatches'])==3,data['valid_decode_mismatches'])
    for name in p1['real']:
        real=p1['real'][name]['mean'];control=p1['shuffled'][name]['mean']
        check('embedding relation '+name,real>=.3 and abs(control)<=.15,[real,control])
    vocab=json.loads((ROOT/'data'/'potion-tokenizer.json').read_text())['model']['vocab']
    with safe_open(ROOT/'data'/'potion-model.safetensors',framework='numpy') as f: table=f.get_tensor('embeddings')
    permutation=np.random.default_rng(1730).permutation(len(table))
    for name in p1['real']:
        pairs=p1['real'][name]['pairs']
        for label,indices in [('real',np.arange(len(table))),('shuffled',permutation)]:
            vectors=[]
            for left,right in pairs:
                a,b=table[indices[vocab[left]]].astype(float),table[indices[vocab[right]]].astype(float)
                vectors.append(b/np.linalg.norm(b)-a/np.linalg.norm(a))
            vectors=np.asarray(vectors);others=(vectors.sum(axis=0)-vectors)/8
            cosine=(vectors*others).sum(axis=1)/(np.linalg.norm(vectors,axis=1)*np.linalg.norm(others,axis=1))
            check('embedding cosine replay '+name+' '+label,
                  np.max(abs(cosine-np.asarray(p1[label][name]['cosines'])))<1e-12)
    check('bigram normalization through explicit length 1,2,3 enumeration',
          all(abs(x-1)<1e-12 for x in p2['normalization_sums']),p2['normalization_sums'])
    check('bigram sample goodness of fit',p2['bigram_sample']['generated']==250000 and p2['bigram_sample']['p_value']>.001,
          p2['bigram_sample']['p_value'])
    replay=read(ROOT/'evidence'/'part2-bigram-sampler-audit.json')
    sample_path=ROOT/'artifacts'/'bigram-sample.npz'
    with np.load(sample_path,allow_pickle=False) as z:
        seq=z['sequence'];observed=z['observed'];prob=z['probabilities']
        reconstructed=np.bincount(seq[:-1]*512+seq[1:],minlength=512**2).reshape(512,512)
        check('bigram sample raw transitions',len(seq)==250001 and np.array_equal(observed,reconstructed) and
              np.max(abs(prob.sum(axis=1)-1))<1e-12 and sha(sample_path)==replay['artifact_sha256'])
        statistic=0.;dof=0
        for group in replay['replay']['groups']:
            row=group['context'];expected=observed[row].sum()*prob[row];common=expected>=5
            o=np.r_[observed[row,common],observed[row,~common].sum()]
            e=np.r_[expected[common],expected[~common].sum()];use=e>0;o=o[use];e=e[use]
            statistic+=float(np.sum((o-e)**2/e));dof+=len(e)-1
        sample_p=float(chi2.sf(statistic,dof))
        check('bigram sample independently recalculated p-value',abs(sample_p-p2['bigram_sample']['p_value'])<1e-12,sample_p)
    refs=[(1,4.839,126.4,3.403),(2,3.038,20.9,2.136),(3,2.094,8.12,1.473)]
    check('uniform probability baseline',abs(p2['uniform']['loss']-np.log(512))<1e-12 and p2['uniform']['ppl']==512)
    for n,loss,ppl,bpc in refs:
        r=p2['models'][str(n)]
        check(f'{n}-gram reference tolerance',abs(r['loss']-loss)<.03 and abs(r['ppl']/ppl-1)<.03 and abs(r['bpc']-bpc)<.03,
              [r['loss'],r['ppl'],r['bpc']])
    models=p2['models'];check('n-gram minimum and 5/6-gram scores',
        min(range(1,7),key=lambda n:models[str(n)]['bpc'])==4 and abs(models['4']['bpc']-1.310)<.03 and
        abs(models['5']['bpc']-1.46)<.03 and abs(models['6']['bpc']-1.79)<.03)
    for n in range(1,7):
        r=models[str(n)];options=[.0001,.001,.01,.1,1.]
        check(f'{n}-gram smoothing selected on training holdout',r['k']==options[int(np.argmin(r['dev_losses']))] and
              r['targets']==data['valid_tokens']-1)
    for n,reference in [(3,.006),(4,.042),(5,.122),(6,.235)]:
        check(f'{n}-gram unseen fraction',abs(models[str(n)]['unseen_fraction']-reference)<.02,models[str(n)]['unseen_fraction'])
    c=cap['capacities'];s=cap['capacity_slopes']
    check('outer and least-squares capacities at d=64',50<=c['outer'][2]<=85 and 100<=c['least_squares'][2]<=170,
          [c['outer'][2],c['least_squares'][2]])
    check('least-squares exact interpolation through d=64',max(cap['ratios']['64']['least_squares'][:4])<1e-10)
    check('capacity scaling slopes',all(abs(s[k]-1)<.1 for k in ['outer','least_squares','relu']) and
          abs(s['relu2']-2)<.15 and s['relu3']>2.5,s)
    for dimension in (16,32,64):
        for method in ('outer','least_squares','relu','relu2','relu3'):
            medians=[]
            for M in cap['sizes']:
                draw=[]
                for seed in range(3):
                    path=ROOT/'artifacts'/'capacity'/f'd{dimension}-m{M}-seed{seed}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        draw.append(float(np.median(z[method])))
                medians.append(float(np.median(draw)))
            check(f'capacity raw curve d={dimension} {method}',np.allclose(medians,cap['ratios'][str(dimension)][method],rtol=1e-12))
    cfg=mlp['config'];check('MLP architecture and training budget',cfg['vocab']==512 and cfg['width']==64 and
        cfg['hidden']==512 and cfg['batch']==16384 and cfg['steps']==4000 and cfg['learning_rate']==.006)
    check('MLP matches held-out bigram',abs(mlp['difference'])<.01,mlp['difference'])
    check('MLP saved weights',sha(ROOT/'artifacts'/'mlp'/'model.npz')==mlp['model_sha256'])
    unit_report=read(ROOT/'evidence'/'part3-mlp-units.json')
    with np.load(ROOT/'artifacts'/'mlp'/'model.npz',allow_pickle=False) as z:
        activation=np.tanh(np.asarray(z['E'],dtype=np.float64)@z['W1']+z['b1'])
        logits=activation@np.asarray(z['W2'],dtype=np.float64)+z['b2']
        logp=logits-logsumexp(logits,axis=1,keepdims=True)
        val_mlp=np.load(ROOT/'artifacts'/'valid-tokens.npy')
        independent_mlp=float(-np.mean(logp[val_mlp[:-1],val_mlp[1:]]))
    check('MLP independent NumPy held-out loss',abs(independent_mlp-mlp['heldout_loss'])<1e-5,independent_mlp)
    shares=[]
    for column in range(activation.shape[1]):
        for sign in (1,-1):
            side=np.maximum(sign*activation[:,column],0);power=side**2
            shares.append(np.sort(power)[-5:].sum()/power.sum() if power.sum() else 0.)
    check('MLP extension all 1024 sides and concentration',len(unit_report['trained_sides'])==1024 and
          abs(np.median(shares)-unit_report['trained_median_share'])<1e-12,unit_report['trained_median_share'])
    ratios=np.array(pic['ratios']);check('picture experiment thresholds',
        abs(ratios[2,0]/4-1)<.15 and abs(ratios[2,1]/np.sqrt(8)-1)<.15 and ratios[3,2]>1 and np.all(ratios[:,3]<.1),
        ratios.tolist())
    check('picture asset and artifact checksums',sha(ROOT/'assets'/'suan-32.txt')==pic['asset_sha256'] and
          sha(ROOT/'artifacts'/'part4-picture.npz')==pic['artifact_sha256'])
    with np.load(ROOT/'artifacts'/'part4-picture.npz',allow_pickle=False) as z:
        images=z['images'];signal=z['signal'];rng=np.random.default_rng(pic['seed'])
        for i,M in enumerate(pic['sizes']):
            k=rng.normal(size=(M,64));k/=np.linalg.norm(k,axis=1,keepdims=True)
            v=rng.integers(0,2,size=(M,1024),dtype=np.int8)*2-1;v[0]=signal.astype(np.int8)
            score=k@k[0]
            for j,method in enumerate(pic['methods']):
                if method=='outer':w=score
                elif method=='relu':w=np.maximum(score,0)
                elif method=='relu2':w=np.maximum(score,0)**2
                else:
                    w=np.exp(8*(score-score.max()));w/=w.sum()
                noise=w[1:]@v[1:];R=np.linalg.norm(noise)/(abs(w[0])*np.linalg.norm(v[0]))
                reconstructed=v[0]+noise/w[0]
                check(f'picture M={M} {method} independently recomputed',abs(R-ratios[i,j])<1e-11 and
                      np.max(abs(reconstructed.reshape(32,32)-images[i,j]))<1e-10,float(R))
    train=read(ROOT/'artifacts'/'transformer'/'training.json');t=train['config'];model=ROOT/'artifacts'/'transformer'/'model.npz'
    check('transformer architecture and training budget',t['layers']==2 and t['width']==96 and t['heads']==4 and
          t['context']==64 and t['steps']==6000 and t['batch']==32 and t['warmup']==300 and
          t['learning_rate']==.003 and t['min_learning_rate']==.0003)
    check('transformer saved weights',sha(model)==train['model_sha256']==p5['model_sha256'])
    check('transformer training history',train['history'][-1]['step']==6000 and train['history'][-1]['tokens']==6000*32*64)
    h=p5['heldout'];check('transformer better than 4-gram in bits/character',h['bpc']<models['4']['bpc'],h['bpc'])
    check('transformer uses long context',max(h['position_loss'][31:])<models['3']['loss'],max(h['position_loss'][31:]))
    with np.load(model,allow_pickle=False) as z:p={k:z[k] for k in z.files}
    val=np.load(ROOT/'artifacts'/'valid-tokens.npy');batch=np.stack([val[200:264],val[1000:1064]]).astype(np.int32)
    reference=independent_forward(p,batch,t);jax_out=np.asarray(decoder.forward({k:jnp.asarray(v) for k,v in p.items()},jnp.asarray(batch),t))
    error=float(np.max(abs(reference-jax_out)))
    check('transformer independent NumPy / JAX forward',error<2e-4,error)
    altered=batch.copy();altered[:,32:]=(altered[:,32:]+1)%512
    changed=independent_forward(p,altered,t)
    check('transformer causal prefix',np.max(abs(reference[:,:32]-changed[:,:32]))==0.)
    with np.load(ROOT/'artifacts'/'transformer'/f"sampler-counts-v{p5['evaluation_version']}.npz",allow_pickle=False) as z:
        for name,entry in p5['sampler_checks'].items():
            if name=='greedy':continue
            observed=z[name];p=z[name+'_probabilities'];expected=100000*p;common=expected>=5
            o=np.r_[observed[common],observed[~common].sum()];e=np.r_[expected[common],expected[~common].sum()];use=e>0;o=o[use];e=e[use]
            stat=float(np.sum((o-e)**2/e));prob=float(chi2.sf(stat,len(e)-1))
            check('sampler '+name,observed.sum()==100000 and abs(prob-entry['p_value'])<1e-12 and prob>.001,prob)
    check('near-zero temperature is greedy',p5['sampler_checks']['greedy']['all_greedy'])
    e1,e07=p5['entropy_checks'];check('sample surprise / entropy z-scores',abs(e1['z'])<=4 and abs(e07['z'])>40,[e1['z'],e07['z']])
    with np.load(ROOT/'artifacts'/'transformer'/f"entropy-samples-v{p5['evaluation_version']}.npz",allow_pickle=False) as z:
        for suffix,report in [('1',e1),('07',e07)]:
            diff=(z['surprise'+suffix]-z['entropy'+suffix]).mean(axis=0)
            measured=float(diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))))
            check('entropy z-score replay '+suffix,abs(measured-report['z'])<1e-9,measured)
    check('60-token comparison samples',all(len(x['tokens'])==60 for x in p5['comparison_samples'].values()))
    check('four requested generation settings',set(p5['generated_samples'])==
          {'temperature_0.8','top_k_20','top_p_0.9','min_p_0.1'} and
          all(len(x['tokens'])==60 for x in p5['generated_samples'].values()))
    with np.load(ROOT/'artifacts'/'transformer'/'attention-maps.npz',allow_pickle=False) as z:
        w=z['weights']
        check('all eight attention-head extension maps',w.shape==(2,4,32,32) and np.max(abs(np.triu(w,1)))==0 and
              np.max(abs(w.sum(axis=-1)-1))<1e-6)
    pa=p6['parameters'];check('Chinchilla exponents and floor',.32<=pa['alpha']<=.38 and .32<=pa['beta']<=.42 and 1.75<=pa['E']<=1.90,pa)
    files=sorted((ROOT/'artifacts'/'chinchilla').glob('bootstrap-*.json'))
    check('Chinchilla bootstrap 400 draws',len(files)==400 and p6['bootstrap']==400)
    draws=[]
    for file in files:
        r=read(file);q=np.array(r['theta']);e,a,b=np.exp(q[:3]);alpha,beta=q[3:]
        draws.append([e,a*1e9**alpha,b*1e10**beta,alpha,beta])
    draws=np.asarray(draws);errors=draws.std(axis=0,ddof=1)
    check('Chinchilla bootstrap standard errors',np.allclose(errors,list(p6['standard_errors'].values()),rtol=1e-10))
    A=draws[:,1];B=draws[:,2];alpha=draws[:,3];beta=draws[:,4]
    G=(alpha*A/(beta*B))**(1/(alpha+beta));N=G*(p6['budget']/6)**(beta/(alpha+beta));D=p6['budget']/(6*N)
    check('Chinchilla bootstrap optimal-ratio interval',np.allclose(np.percentile(D/N,[2.5,97.5]),p6['tokens_per_parameter_95'],rtol=1e-10))
    check('Chinchilla residual comparison',abs(p6['mean_log_residual'])<=.005 and p6['printed_mean_log_residual']>.01)
    check('compute-optimal tokens per parameter',10<=p6['tokens_per_parameter']<=30,p6['tokens_per_parameter'])
    figures=['part1-relations.png','part2-ngram-ladder.png','part3-interference.png','part3-capacity.png',
             'part3-mlp-loss.png','part3-mlp-units.png','part4-picture-recall.png','part5-model-comparison.png','part5-context-loss.png',
             'part5-attention-heads.png',
             'part6-chinchilla-fit.png','part6-residuals.png','part6-optimal.png']
    for name in figures:
        with Image.open(ROOT/'evidence'/name) as im:size=im.size;im.verify()
        check('figure '+name,min(size)>=400,list(size))
    failed=[r['name'] for r in CHECKS if not r['passed']]
    report=dict(pdf_sha256='4e33bea929459c01511efc0298308a93d08d99ee7df38372bbf52e89be69cc35',
                checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                passed=len(CHECKS)-len(failed),total=len(CHECKS),failed=failed,checks=CHECKS,
                limitations=['Capacity uses all stored keys through M=256, and a fixed 256 stored-key sample per draw beyond that.',
                             'NumPy decoder forward is independently compared on two held-out windows, not the entire validation set.',
                             'Sampler v1 was too concentrated for two chi-squared tests and is preserved; v3 uses a context selected by entropy before sampling.'])
    (ROOT/'evidence'/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{report['passed']}/{report['total']} checks passed; failed={failed}")
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
