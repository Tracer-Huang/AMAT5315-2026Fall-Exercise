"""Pre-norm causal transformer for the revised Week 8 task."""
import jax
import jax.numpy as jnp
import numpy as np


def initialize(cfg, key):
    d, layers, vocab, context = cfg['width'], cfg['layers'], cfg['vocab'], cfg['context']
    if d % cfg['heads']:
        raise ValueError('width must be divisible by the head count')
    shapes = {'token_embedding': (vocab,d), 'position_embedding': (context,d),
              'final_norm.scale': (d,), 'final_norm.bias': (d,),
              'output.weight': (d,vocab), 'output.bias': (vocab,)}
    for i in range(layers):
        prefix=f'blocks.{i}.'
        for norm in ('norm1','norm2'):
            shapes[prefix+norm+'.scale']=(d,);shapes[prefix+norm+'.bias']=(d,)
        for name,shape in [('attention.qkv',(d,3*d)),('attention.output',(d,d)),
                           ('mlp.up',(d,4*d)),('mlp.down',(4*d,d))]:
            shapes[prefix+name+'.weight']=shape;shapes[prefix+name+'.bias']=(shape[1],)
    keys=jax.random.split(key,len(shapes))
    return {name:(jnp.ones(shape) if name.endswith('.scale') else
                  jnp.zeros(shape) if name.endswith('.bias') else .02*jax.random.normal(k,shape))
            for (name,shape),k in zip(shapes.items(),keys)}


def forward(params, ids, cfg, return_attention=False):
    batch,t=ids.shape; d=cfg['width']; heads=cfg['heads']
    def dense(x,name):
        return jnp.matmul(x,params[name+'.weight'],precision=jax.lax.Precision.HIGHEST)+params[name+'.bias']
    def norm(x,name):
        mean=jnp.mean(x,axis=-1,keepdims=True);var=jnp.mean((x-mean)**2,axis=-1,keepdims=True)
        return (x-mean)*jax.lax.rsqrt(var+1e-5)*params[name+'.scale']+params[name+'.bias']
    x=params['token_embedding'][ids]+params['position_embedding'][:t]
    attentions=[]
    for i in range(cfg['layers']):
        prefix=f'blocks.{i}.'
        qkv=dense(norm(x,prefix+'norm1'),prefix+'attention.qkv')
        q,k,v=[z.reshape(batch,t,heads,d//heads).transpose(0,2,1,3) for z in jnp.split(qkv,3,axis=-1)]
        scores=jnp.matmul(q,k.swapaxes(-1,-2),precision=jax.lax.Precision.HIGHEST)/np.sqrt(d//heads)
        scores=jnp.where(jnp.arange(t)[:,None]>=jnp.arange(t)[None,:],scores,-jnp.inf)
        weights=jax.nn.softmax(scores,axis=-1)
        attended=jnp.matmul(weights,v,precision=jax.lax.Precision.HIGHEST).transpose(0,2,1,3).reshape(batch,t,d)
        x=x+dense(attended,prefix+'attention.output')
        x=x+dense(jax.nn.gelu(dense(norm(x,prefix+'norm2'),prefix+'mlp.up'),approximate=False),prefix+'mlp.down')
        if return_attention: attentions.append(weights)
    logits=dense(norm(x,'final_norm'),'output')
    return (logits,attentions) if return_attention else logits
