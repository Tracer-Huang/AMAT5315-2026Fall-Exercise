"""One-token MLP language model for revised Week 8 Part 3."""
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train_transformer import save_checkpoint

ROOT=Path(__file__).resolve().parent
CONFIG=dict(vocab=512,width=64,hidden=512,batch=16384,steps=4000,learning_rate=.006,
            min_learning_rate=.0006,warmup=200,weight_decay=.01,beta1=.9,beta2=.99,
            seed=531508,chunk=25)


def initialize(cfg, key):
    a,b,c=jax.random.split(key,3);v,d,h=cfg['vocab'],cfg['width'],cfg['hidden']
    return dict(E=.02*jax.random.normal(a,(v,d)),W1=jax.random.normal(b,(d,h))/np.sqrt(d),
                b1=jnp.zeros(h),W2=jax.random.normal(c,(h,v))/np.sqrt(h),b2=jnp.zeros(v))


def logits(params, ids):
    hidden=jnp.tanh(params['E'][ids]@params['W1']+params['b1'])
    return hidden@params['W2']+params['b2']


def grouped_loss(params, x, y, cfg):
    v=cfg['vocab']
    counts=jnp.bincount(x*v+y,length=v*v).reshape(v,v)
    logp=jax.nn.log_softmax(logits(params,jnp.arange(v)),axis=-1)
    return -jnp.sum(counts*logp)/len(x)


def direct_loss(params, x, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits(params,x),y).mean()


def main():
    cfg=CONFIG.copy();out=ROOT/'artifacts'/'mlp';out.mkdir(exist_ok=True)
    with (out/'config.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    train=jnp.asarray(np.load(ROOT/'artifacts'/'train-tokens.npy'),dtype=jnp.int32)
    schedule=optax.warmup_cosine_decay_schedule(0.,cfg['learning_rate'],cfg['warmup'],cfg['steps'],cfg['min_learning_rate'])
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,b1=cfg['beta1'],b2=cfg['beta2'],weight_decay=cfg['weight_decay']))
    key=jax.random.PRNGKey(cfg['seed']);key,initkey=jax.random.split(key)
    p=initialize(cfg,initkey);state=tx.init(p);start=time.time();history=[];window=[]
    def one(carry,_):
        p,state,key=carry;key,sub=jax.random.split(key)
        indices=jax.random.randint(sub,(cfg['batch'],),0,len(train)-1)
        x,y=train[indices],train[indices+1]
        loss,grad=jax.value_and_grad(lambda prm:grouped_loss(prm,x,y,cfg))(p)
        upd,state=tx.update(grad,state,p)
        return (optax.apply_updates(p,upd),state,key),loss
    update=jax.jit(lambda carry:jax.lax.scan(one,carry,None,length=cfg['chunk']))
    carry=(p,state,key)
    for step in range(cfg['chunk'],cfg['steps']+1,cfg['chunk']):
        carry,loss=update(carry);window.extend(np.asarray(loss).tolist())
        if step%200==0:
            row=dict(step=step,train_loss=float(np.mean(window)),lr=float(schedule(step-1)),
                     elapsed_seconds=time.time()-start);history.append(row);window=[]
            print(f"step {step:4d} loss {row['train_loss']:.4f} elapsed {row['elapsed_seconds']:.1f}s",flush=True)
        if step%1000==0:save_checkpoint(out/f'checkpoint-{step:04d}.npz',*carry,step)
    p,state,key=carry;model=out/'model.npz'
    with model.open('xb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in p.items()})
    val=np.load(ROOT/'artifacts'/'valid-tokens.npy');logp=np.asarray(jax.nn.log_softmax(logits(p,jnp.arange(cfg['vocab'])),axis=-1))
    loss=float(-np.mean(logp[val[:-1],val[1:]]))
    ref=json.loads((ROOT/'evidence'/'part2-ngrams.json').read_text())['models']['2']['loss']
    report=dict(config=cfg,history=history,device=str(jax.devices()[0]),jax_version=jax.__version__,
                heldout_loss=loss,bigram_loss=ref,difference=loss-ref,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                implementation='batch counts grouped by (input,target), exactly the same empirical loss and gradient',
                parameters=int(sum(v.size for v in p.values())),seconds=time.time()-start)
    (ROOT/'evidence'/'part3-mlp.json').write_text(json.dumps(report,indent=2)+'\n')
    print('held-out MLP',loss,'bigram',ref,'difference',loss-ref,flush=True)


if __name__=='__main__':
    main()
