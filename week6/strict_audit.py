"""Recompute course results from retained raw arrays, not from the previous validator."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,time,tomllib,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from data import ROOT,mnist,cifar
from models import gibbs,attention
checks=[]
def check(name,condition,detail=''):
    checks.append(dict(check=name,passed=bool(condition),detail=detail))
    print(('PASS ' if condition else 'FAIL ')+name+(' — '+str(detail) if detail else ''),flush=True)
    if not condition:raise AssertionError(name)

def close(a,b,tol=1e-6):return np.allclose(a,b,rtol=0,atol=tol)

def main():
    start=time.perf_counter();cfg=tomllib.loads((ROOT/'rbm.toml').read_text())
    expected=dict(images=60000,epochs=20,batch_size=100,shuffle_each_epoch=True,optimizer='SGD',learning_rate=.1,momentum=0.,weight_decay=0.,precision='float32',seed=6602)
    check('PDF p4-5: every prescribed training setting',cfg['training']==expected)
    check('PDF p5: initialization/evaluation/completion settings',cfg['initialization']==dict(weight_std=.01,hidden_bias=0.,visible_bias='logit(clipped training pixel frequency)',pixel_frequency_clip=[.01,.99],seed=6601) and cfg['evaluation']==dict(test_images=64,seed=6606,reconstruction_at='initialization and each epoch',surrogate='epoch mean of batch surrogates before updates') and cfg['completion']==dict(test_images=64,chains=8,steps=[1,10,100,1000],seed=6607,wrong_upper='next evaluation image with a different label, wrapping within the 64 images'))
    check('PDF p2: provenance URLs and original byte identities',all((ROOT/'inputs'/r['file']).stat().st_size==r['bytes'] and hashlib.file_digest(open(ROOT/'inputs'/r['file'],'rb'),'sha256').hexdigest()==r['sha256'] for r in json.loads((ROOT/'evidence/provenance.json').read_text())))
    ds=json.loads((ROOT/'evidence/dataset-stats.json').read_text())
    for name,reader in [('MNIST',mnist),('CIFAR-10',cifar)]:
        for split in ['train','test']:
            images,labels=reader(split)
            check(f'PDF p3-4: {name} {split} full labels and native shape',list(images.shape)==ds[name][split+'_shape'] and np.bincount(labels,minlength=10).tolist()==ds[name][split+'_counts'] and images.dtype==np.uint8)
    train,train_labels=mnist('train');test,testlabels=mnist('test');binary=(train>=128).reshape(-1,784).astype(np.float32);targets=(test[:64]>=128).reshape(64,784).astype(np.float32)
    raw=ROOT/'artifacts/rbm-shared-seed';r=json.loads((ROOT/'evidence/rbm-metrics.json').read_text());initial=np.load(raw/'initialization.npz')
    expectedW=np.array(jax.random.normal(jax.random.key(6601),(784,128),dtype=jnp.float32)*.01);freq=np.clip(binary.mean(0),.01,.99)
    check('PDF p5: training-only bias initialization and exact weight seed',np.array_equal(initial['W'],expectedW) and close(initial['a'],np.log(freq/(1-freq)),1e-5) and np.all(initial['b']==0))
    generation=[];completion=[]
    for k in [1,10]:
        hist=r['training'][str(k)]['history'];check(f'PDF p5: CD-{k} all 20 checkpoints',len(hist)==21 and all((raw/f'cd{k}-epoch{e:02d}.npz').exists() for e in range(1,21)))
        maxerr=0
        for e in range(21):
            weights=initial if e==0 else np.load(raw/f'cd{k}-epoch{e:02d}.npz')
            params=tuple(jnp.asarray(weights[n]) for n in ['W','a','b'])
            checkshape=all(x.dtype==jnp.float32 for x in params) and [x.shape for x in params]==[(784,128),(784,),(128,)]
            assert checkshape
            sampled,_=gibbs(params,jnp.asarray(targets),jax.random.key(6606),1)
            err=np.mean(np.array(sampled)!=targets);maxerr=max(maxerr,abs(err-hist[e]['reconstruction_error']))
        check(f'PDF p8-9: CD-{k} fixed-seed reconstruction at all states',maxerr<1e-7,f'max independent Hamming discrepancy={maxerr:.3g}')
        params=tuple(jnp.asarray(weights[n]) for n in ['W','a','b']);sample=np.load(raw/f'cd{k}-sampling.npz');gen=sample['generation'];recon=sample['reconstruction'];generation.append(gen)
        check(f'PDF p9: CD-{k} stored steps and held-out reconstruction starts',sample['steps'].tolist()==[0,1,10,100,1000] and gen.shape==recon.shape==(5,64,784) and np.array_equal(recon[0],targets))
        # Replay trajectories, preserving the key sequence between requested frames.
        for label,states,key in [('generation',gen,jax.random.key(6609)),('reconstruction',recon,jax.random.key(6606))]:
            state=jnp.asarray(states[0]);previous=0
            for pos,step in enumerate([1,10,100,1000],1):
                state,key=gibbs(params,state,key,step-previous);previous=step
                assert np.array_equal(state,states[pos])
            check(f'PDF p9: CD-{k} exact replay of {label}',True)
        data=np.load(raw/f'cd{k}-completion.npz');states=data['states'];completion.append(states)
        wrong=np.array([next((i+d)%64 for d in range(1,64) if testlabels[(i+d)%64]!=testlabels[i]) for i in range(64)])
        check(f'PDF p5/10: CD-{k} wrong-upper next-different-label rule',np.array_equal(data['wrong_indices'],wrong))
        truth=np.repeat(targets[:,392:],8,axis=0);truth=np.concatenate([truth,truth]);a=states[1:,:,392:];den=(a+truth).sum(-1)
        dice=np.divide(2*(a*truth).sum(-1),den,out=np.zeros_like(den),where=den>0).reshape(4,2,64,8).mean(-1)
        means=dice.mean(-1);se=dice.std(-1,ddof=1)/8
        check(f'PDF p10: CD-{k} independently recomputed missing-half Dice/SE',close(dice,data['per_digit_dice']) and close(means,r['completion'][str(k)]['mean']) and close(se,r['completion'][str(k)]['se']),f'1000-step true/wrong={means[-1].tolist()}')
        obs=np.concatenate([np.repeat(targets,8,axis=0),np.repeat(targets[wrong],8,axis=0)])
        check(f'PDF p10: CD-{k} 1024 chains and all stored observed halves exact',states.shape==(5,1024,784) and np.array_equal(states[:,:,:392],np.broadcast_to(obs[:,:392],states[:,:,:392].shape)))
        key,kstart=jax.random.split(jax.random.key(6607));random=np.array(jax.random.bernoulli(kstart,.5,(512,784)).astype(jnp.float32));initialstate=np.concatenate([random,random]);initialstate[:,:392]=obs[:,:392]
        state=jnp.asarray(initialstate);previous=0
        for pos,step in enumerate([1,10,100,1000],1):
            state,key=gibbs(params,state,key,step-previous,jnp.asarray(obs),jnp.arange(784)<392);previous=step
            assert np.array_equal(state,states[pos])
        check(f'PDF p10: CD-{k} exact clamped trajectory replay',True)
    check('PDF p9: matched fair generation starts across CD models',np.array_equal(generation[0][0],generation[1][0]) and np.array_equal(generation[0][0],jax.random.bernoulli(jax.random.key(6608),.5,(64,784))))
    check('PDF p5: completion start/randomness matched across CD models',np.array_equal(completion[0][0],completion[1][0]))
    hop=json.loads((ROOT/'evidence/hopfield-metrics.json').read_text());seeds=[];maxdiscrepancy=0;maxsweeps=0
    for rec in hop['records']:
        n,p=rec['N'],rec['P'];u=[];overlap=[];success=[]
        for rep in range(20):
            data=np.load(ROOT/f'artifacts/hopfield/n{n}-p{p}-r{rep:02d}.npz');xi=data['patterns'].astype(np.float64);state=data['final'].astype(np.float64)
            assert xi.shape==(p,n) and state.shape==(min(10,p),n) and np.all(np.isin(xi,[-1,1])) and np.all(np.isin(state,[-1,1]))
            C=xi.T@xi;np.fill_diagonal(C,0);fields=state@C
            assert np.all(state*fields>=0)
            e0=-np.sum((xi[:len(state)]@C)*xi[:len(state)],axis=1)/2;e1=-np.sum(fields*state,axis=1)/2;assert np.all(e1<=e0)
            observed=(state*xi[:len(state)]).mean(-1);assert close(observed,data['overlap'])
            u.append(np.mean(xi*(xi@C)<0));overlap.append(observed.mean());success.append(np.mean(observed>=.9));seeds.append(int(data['seed']));maxsweeps=max(maxsweeps,int(data['sweeps'].max()))
        for a,mean,se in [(u,'unstable_mean','unstable_se'),(overlap,'overlap_mean','overlap_se'),(success,'success_mean','success_se')]:
            maxdiscrepancy=max(maxdiscrepancy,abs(np.mean(a)-rec[mean]),abs(np.std(a,ddof=1)/np.sqrt(20)-rec[se]))
        print(f'Recomputed N={n} P={p}',flush=True)
    check('PDF p11-13: all 1680 sets recomputed, stable and nonincreasing energy',len(seeds)==1680 and len(set(seeds))==1680 and maxdiscrepancy<1e-6 and maxsweeps<1000,f'max summary discrepancy={maxdiscrepancy:.3g}; max sweeps={maxsweeps}')
    modern=json.loads((ROOT/'evidence/modern-metrics.json').read_text());attention_diff=0;formula_diff=0
    for P in modern['Ps']:
        data=np.load(ROOT/f'artifacts/modern/p{P}.npz');X=data['patterns'];query=data['queries'];idx=data['target_indices'];target=data['targets'];classical=data['classical']
        check(f'PDF p13-14: P={P} original-order patterns and paired half-erased queries',np.array_equal(X,2*binary[:P]-1) and np.array_equal(idx,np.arange(100)%P) and np.array_equal(target,X[idx]) and np.array_equal(query[:,:392],target[:,:392]))
        for bi,beta in enumerate(data['betas']):
            out=data['modern'][bi];logits=beta*query.astype(np.float64)@X.astype(np.float64).T;logits-=logits.max(-1,keepdims=True);probs=np.exp(logits);probs/=probs.sum(-1,keepdims=True);independent=probs@X
            formula_diff=max(formula_diff,float(np.max(np.abs(out-independent))))
            att=np.array(attention(jnp.asarray(X,jnp.float32),jnp.asarray(query,jnp.float32),float(beta)));attention_diff=max(attention_diff,float(np.max(np.abs(out-att))))
            signs=np.where(out>=0,1,-1);succ=((signs*target).mean(-1)>=.95)&(np.argmax(signs@X.T,axis=-1)==idx)
            assert np.array_equal(succ,data['modern_success'][bi])
            row=next(r for r in modern['records'] if r['P']==P and r['beta']==beta);assert abs(succ.mean()-row['modern_success'])<1e-7
        signs=classical;succ=((signs*target).mean(-1)>=.95)&(np.argmax(signs@X.T,axis=-1)==idx)
        C=X.astype(np.float64).T@X;np.fill_diagonal(C,0)
        assert np.array_equal(succ,data['classical_success']) and np.all(signs*(signs@C)>=0)
        assert np.all(-np.sum((signs@C)*signs,axis=-1)<=-np.sum((query@C)*query,axis=-1))
    check('PDF p15: independent NumPy softmax and actual JAX attention',formula_diff<1e-5 and attention_diff<1e-5 and abs(attention_diff-modern['attention_max_abs_difference'])<1e-7,f'NumPy discrepancy={formula_diff:.9g}; attention max={attention_diff:.9g}')
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,runtime_seconds=time.perf_counter()-start,scope='PDF exact settings, original inputs, every saved epoch evaluation, all RBM sampling/completion frames, all 1680 Hopfield sets, all modern outputs; independent NumPy statistics and formulas',optional_AIS='Explicit optional extension, excluded from required tasks',remaining_personal_activity='Reading, oral explanation and Week7 presentation by the student are not claimed as completed')
    (ROOT/'evidence/strict-audit.json').write_text(json.dumps(report,indent=2));print('ALL STRICT RAW-ARRAY CHECKS PASS',flush=True)
if __name__=='__main__':main()
