"""Train the revised Week 8 causal transformer on BPE training windows."""
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from decoder import initialize, forward

ROOT=Path(__file__).resolve().parent
CONFIG=dict(vocab=512,context=64,width=96,heads=4,layers=2,batch=32,steps=6000,
            learning_rate=.003,min_learning_rate=.0003,warmup=300,weight_decay=.1,
            beta1=.9,beta2=.99,seed=531508,chunk=25)


def lm_loss(params, inputs, targets, cfg):
    logits=forward(params,inputs,cfg)
    return optax.softmax_cross_entropy_with_integer_labels(logits,targets).mean()


def save_checkpoint(path, params, state, key, step):
    leaves,_=jax.tree_util.tree_flatten(state)
    data={f'p::{k}':np.asarray(v) for k,v in params.items()}
    data.update({f's::{i}':np.asarray(v) for i,v in enumerate(leaves)})
    data.update(key=np.asarray(key),step=np.asarray(step,dtype=np.int64))
    with path.open('xb') as f: np.savez_compressed(f,**data)


def main():
    cfg=CONFIG.copy(); out=ROOT/'artifacts'/'transformer';out.mkdir(exist_ok=True)
    config_file=out/'config.json'
    with config_file.open('x') as f: json.dump(cfg,f,indent=2);f.write('\n')
    training=jnp.asarray(np.load(ROOT/'artifacts'/'train-tokens.npy'),dtype=jnp.int32)
    schedule=optax.warmup_cosine_decay_schedule(0.,cfg['learning_rate'],cfg['warmup'],cfg['steps'],cfg['min_learning_rate'])
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,b1=cfg['beta1'],b2=cfg['beta2'],weight_decay=cfg['weight_decay']))
    key=jax.random.PRNGKey(cfg['seed']);key,init_key=jax.random.split(key)
    params=initialize(cfg,init_key);state=tx.init(params)
    def one_step(carry,_):
        params,state,key=carry;key,sub=jax.random.split(key)
        starts=jax.random.randint(sub,(cfg['batch'],),0,len(training)-cfg['context'])
        indexes=starts[:,None]+jnp.arange(cfg['context'])
        inputs=training[indexes];targets=training[indexes+1]
        loss,grad=jax.value_and_grad(lambda p:lm_loss(p,inputs,targets,cfg))(params)
        updates,state=tx.update(grad,state,params)
        return (optax.apply_updates(params,updates),state,key),loss
    update=jax.jit(lambda carry:jax.lax.scan(one_step,carry,None,length=cfg['chunk']))
    carry=(params,state,key);history=[];window=[];started=time.time()
    for step in range(cfg['chunk'],cfg['steps']+1,cfg['chunk']):
        carry,losses=update(carry);window.extend(np.asarray(losses).tolist())
        if step%200==0:
            row=dict(step=step,train_loss=float(np.mean(window)),lr=float(schedule(step-1)),
                     tokens=step*cfg['batch']*cfg['context'],wall_seconds=time.time()-started)
            history.append(row);window=[]
            print(f"step {step:5d} loss {row['train_loss']:.4f} elapsed {row['wall_seconds']:.1f}s",flush=True)
        if step%1000==0:
            save_checkpoint(out/f'checkpoint-{step:04d}.npz',*carry,step)
    params,state,key=carry
    model=out/'model.npz'
    with model.open('xb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in params.items()})
    record=dict(config=cfg,history=history,device=str(jax.devices()[0]),jax_version=jax.__version__,
                parameters=int(sum(v.size for v in params.values())),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                seconds=time.time()-started)
    (out/'training.json').write_text(json.dumps(record,indent=2)+'\n')
    print('saved transformer',record['parameters'],'parameters; SHA256',record['model_sha256'],flush=True)


if __name__=='__main__':
    main()
