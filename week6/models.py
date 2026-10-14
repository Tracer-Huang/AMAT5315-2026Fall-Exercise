"""Course equations; patterns are stored as rows (transpose of sheet's X)."""
from functools import partial
import jax
import jax.numpy as jnp

def energy(p,v,h):
    w,a,b=p
    return -v@a-h@b-jnp.sum((v@w)*h,axis=-1)

def free_energy(p,v):
    w,a,b=p
    return -v@a-jnp.sum(jax.nn.softplus(v@w+b),axis=-1)

def cd_surrogate(p,v,end):
    return jnp.mean(free_energy(p,v)-free_energy(p,jax.lax.stop_gradient(end)))

@partial(jax.jit,static_argnames=('steps',))
def gibbs(p,v,key,steps,clamp=None,mask=None):
    w,a,b=p
    def step(_,carry):
        v,key=carry
        key,kh,kv=jax.random.split(key,3)
        h=jax.random.bernoulli(kh,jax.nn.sigmoid(v@w+b)).astype(jnp.float32)
        v=jax.random.bernoulli(kv,jax.nn.sigmoid(h@w.T+a)).astype(jnp.float32)
        if clamp is not None: v=jnp.where(mask,clamp,v)
        return v,key
    return jax.lax.fori_loop(0,steps,step,(v,key))

@partial(jax.jit,static_argnames=('steps',))
def train_batch(p,v,key,steps,lr):
    end,key=gibbs(p,v,key,steps)
    loss,grad=jax.value_and_grad(cd_surrogate)(p,v,end)
    return jax.tree.map(lambda x,g:x-lr*g,p,grad),key,loss

def hebbian(patterns):
    J=patterns.T@patterns/patterns.shape[1]
    return J.at[jnp.diag_indices(J.shape[0])].set(0)

def _recall_one(J,start,key):
    n=len(start)
    def sweep(carry):
        s,h,key,count,_=carry
        key,korder=jax.random.split(key)
        order=jax.random.permutation(korder,n)
        def update(t,c):
            s,h,flips=c; i=order[t]
            new=jnp.where(h[i]>0,1,jnp.where(h[i]<0,-1,s[i])).astype(s.dtype)
            delta=new-s[i]
            return s.at[i].set(new),h+delta*J[:,i],flips+(delta!=0)
        s,h,flips=jax.lax.fori_loop(0,n,update,(s,h,jnp.int32(0)))
        return s,h,key,count+1,flips
    s,h,key,count,flips=jax.lax.while_loop(lambda c:(c[4]>0)&(c[3]<1000),sweep,(start,J@start,key,jnp.int32(0),jnp.int32(1)))
    return s,count,flips==0

@jax.jit
def recall(J,starts,key):
    return jax.vmap(_recall_one,in_axes=(None,0,0))(J,starts,jax.random.split(key,len(starts)))

@jax.jit
def modern(patterns,queries,beta):
    return jax.nn.softmax(beta*(queries@patterns.T),axis=-1)@patterns

@jax.jit
def attention(patterns,queries,beta):
    # q: batch=1, query sequence, heads=1, channels=784; keys/values same.
    q=queries[None,:,None,:]; kv=patterns[None,:,None,:]
    return jax.nn.dot_product_attention(q,kv,kv,scale=beta,implementation='xla')[0,:,0,:]

def ink_overlap(a,b):
    both=jnp.sum(a*b,axis=-1); denom=jnp.sum(a+b,axis=-1)
    return jnp.where(denom>0,2*both/jnp.maximum(denom,1),0)
