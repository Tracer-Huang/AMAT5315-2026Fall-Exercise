"""Week 7 equations, 1-based mathematical diffusion levels, CPU JAX."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import jax.numpy as j
import numpy as np
from scipy.spatial.distance import cdist, pdist
CENTRES=j.array([[3.,0.],[0.,3.],[-3.,0.],[0.,-3.]],dtype=j.float32)
B=j.concatenate([j.zeros(1),j.linspace(.0001,.02,1000)])
A=j.concatenate([j.ones(1),j.cumprod(1-B[1:])])

def draw(key,n):
    a,b=jax.random.split(key)
    return CENTRES[jax.random.randint(a,(n,),0,4)]+.3*jax.random.normal(b,(n,2))

def exact_noise(x,t):
    a=A[t]; v=.09*a+1-a
    delta=x[:,None,:]-j.sqrt(a)*CENTRES
    w=jax.nn.softmax(-j.sum(delta**2,axis=-1)/(2*v),axis=1)
    return j.sqrt(1-a)/v*j.sum(w[:,:,None]*delta,axis=1)

def exact_flow(x,s):
    v=.09*(1-s)**2+s*s
    delta=x[:,None,:]-(1-s)*CENTRES
    w=jax.nn.softmax(-j.sum(delta**2,axis=-1)/(2*v),axis=1)
    component=-CENTRES+(s-.09*(1-s))/v*delta
    return j.sum(w[:,:,None]*component,axis=1)

def init(key,width=128,architecture="dense"):
    dims=([16,width,width,width,3] if architecture=="scalar_basis" else [17,width,width,width,1] if architecture=="factorized" else [18,width,width,width,2]);keys=jax.random.split(key,4)
    return [(jax.random.normal(k,(a,b))*j.sqrt(2/a),j.zeros(b)) for k,a,b in zip(keys,dims[:-1],dims[1:])]

def net(p,x,s):
    if p[0][0].shape[0]==16:
        # A learned odd scalar family in the target's independent orthogonal coordinates.
        # All three time-dependent coefficients are trained by original regression targets.
        # The exact noise/velocity functions below are never called by this predictor.
        rotation=j.array([[1.,1.],[1.,-1.]],dtype=x.dtype)/j.sqrt(2.)
        q=x@rotation;times=j.broadcast_to(j.asarray(s),x.shape[:-1])
        phase=times[...,None]*j.arange(1,9)*j.pi
        h=j.concatenate([j.sin(phase),j.cos(phase)],axis=-1)
        for w,b in p[:-1]:h=jax.nn.silu(h@w+b)
        w,b=p[-1];coeff=h@w+b
        a=coeff[...,0,None];b=coeff[...,1,None]
        sharpness=jax.nn.softplus(coeff[...,2,None])/(times[...,None]+.05)
        return (a*q+b*j.tanh(sharpness*q))@rotation.T
    if p[0][0].shape[0]==17:
        # Rotation maps the four Gaussian centres to all sign pairs (±3/√2, ±3/√2).
        # Their scalar marginals are independent and share one odd learned regressor.
        # No exact conditional noise/velocity or analytic oracle is used here.
        rotation=j.array([[1.,1.],[1.,-1.]],dtype=x.dtype)/j.sqrt(2.)
        q=x@rotation
        times=j.broadcast_to(j.asarray(s),x.shape[:-1])
        phase=times[...,None]*j.arange(1,9)*j.pi
        features=j.concatenate([j.sin(phase),j.cos(phase)],axis=-1)
        def scalar(values):
            h=j.concatenate([values[...,None],j.broadcast_to(features[...,None,:],values.shape+(16,))],axis=-1)
            for w,b in p[:-1]:h=jax.nn.silu(h@w+b)
            w,b=p[-1];return (h@w+b)[...,0]
        return (.5*(scalar(q)-scalar(-q)))@rotation.T
    s=j.broadcast_to(j.asarray(s),x.shape[:-1])
    f=s[...,None]*j.array([1,2,4,8,16,32,64,128])*j.pi
    h=j.concatenate([x,j.sin(f),j.cos(f)],axis=-1)
    for w,b in p[:-1]:h=jax.nn.silu(h@w+b)
    w,b=p[-1];return h@w+b

def train(kind,updates=25000,batch=512,seed=7101,lr=.001,architecture="dense",out_dir="artifacts",width=128):
    from pathlib import Path
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    key=jax.random.key(seed); key,k=jax.random.split(key); p=init(k,width=width,architecture=architecture)
    m=jax.tree.map(j.zeros_like,p);v=jax.tree.map(j.zeros_like,p)
    stages=np.unique(np.rint(np.geomspace(1,updates,9)).astype(int));stages=np.r_[0,stages]
    saved=[p];losses=[];done=0
    @jax.jit
    def step(state,unused):
        p,m,v,key,i=state;key,a,b,c=jax.random.split(key,4)
        x0=draw(a,batch);z=jax.random.normal(b,x0.shape)
        if kind=='noise':
            t=jax.random.randint(c,(batch,),1,1001); aa=A[t,None]
            x=j.sqrt(aa)*x0+j.sqrt(1-aa)*z;s=t/1000;target=z
        else:
            s=jax.random.uniform(c,(batch,));x=(1-s[:,None])*x0+s[:,None]*z;target=z-x0
        loss,g=jax.value_and_grad(lambda pp:j.mean((net(pp,x,s)-target)**2))(p)
        i=i+1;rate=lr*.5*(1+j.cos(j.pi*(i-1)/(updates-1)))
        m=jax.tree.map(lambda mm,gg:.9*mm+.1*gg,m,g)
        v=jax.tree.map(lambda vv,gg:.999*vv+.001*gg*gg,v,g)
        p=jax.tree.map(lambda pp,mm,vv:pp-rate*(mm/(1-.9**i))/(j.sqrt(vv/(1-.999**i))+1e-8),p,m,v)
        return (p,m,v,key,i),loss
    state=(p,m,v,key,j.array(0))
    for end in stages[1:]:
        state,ls=jax.jit(lambda st:jax.lax.scan(step,st,None,length=int(end-done)))(state)
        losses.extend(np.asarray(ls).tolist());saved.append(state[0]);done=int(end)
        print(kind,'stage',done,'loss',float(np.mean(losses[-100:])),flush=True)
    data={f'p_{si}_{li}_{k}':np.asarray(arr) for si,pp in enumerate(saved) for li,pair in enumerate(pp) for k,arr in enumerate(pair)}
    np.savez(out/f'{kind}-training.npz',**data,stages=stages,loss=np.array(losses),settings=np.array(__import__('json').dumps(dict(kind=kind,updates=updates,batch=batch,seed=seed,lr=lr,architecture=architecture,width=width))))
    return saved,stages,np.array(losses)

def load(kind,root="artifacts"):
    z=np.load(f'{root}/{kind}-training.npz');return [[(j.array(z[f'p_{s}_{l}_0']),j.array(z[f'p_{s}_{l}_1'])) for l in range(4)] for s in range(10)],z['stages'],z['loss']

def sample(predict,x,steps,method,seed=1):
    levels=j.array(np.rint(np.linspace(1000,0,steps+1)).astype(int))
    key=jax.random.key(seed)
    def body(state,pair):
        x,key=state;t,u=pair;key,k=jax.random.split(key);e=predict(x,t)
        if method=='ddim':
            clean=(x-j.sqrt(1-A[t])*e)/j.sqrt(A[t])
            y=j.sqrt(A[u])*clean+j.sqrt(1-A[u])*e
        else:
            a=A[t]/A[u];mu=(x-(1-a)/j.sqrt(1-A[t])*e)/j.sqrt(a)
            y=mu+j.where(u>0,j.sqrt(1-a),0)*jax.random.normal(k,x.shape)
        return (y,key),y
    (y,_),path=jax.jit(lambda xx,kk:jax.lax.scan(body,(xx,kk),(levels[:-1],levels[1:])))(x,key)
    return y,j.concatenate([x[None],path])

def flow_sample(predict,x,steps):
    def body(x,s):
        y=x-predict(x,s)/steps;return y,y
    y,path=jax.jit(lambda xx:jax.lax.scan(body,xx,j.linspace(1,1/steps,steps)))(x)
    return y,j.concatenate([x[None],path])

def repaint(predict,y,mask,seed):
    # This interface intentionally has no hidden truth argument.
    key=jax.random.key(seed);key,k=jax.random.split(key);x=jax.random.normal(k,y.shape)
    def jump(state,t):
        x,key=state;u=t-20;a=A[t]/A[u]
        def attempt(i,state):
            x,key=state;key,f,n,o=jax.random.split(key,4)
            x=jax.lax.cond(i>0,lambda xx:j.sqrt(a)*xx+j.sqrt(1-a)*jax.random.normal(f,xx.shape),lambda xx:xx,x)
            e=predict(x,t);clean=j.clip((x-j.sqrt(1-A[t])*e)/j.sqrt(A[t]),-1,1)
            mu=j.sqrt(A[u])*(1-a)/(1-A[t])*clean+j.sqrt(a)*(1-A[u])/(1-A[t])*x
            var=(1-A[u])*(1-a)/(1-A[t])
            z=mu+j.sqrt(var)*jax.random.normal(n,x.shape)
            obs=j.sqrt(A[u])*y+j.sqrt(1-A[u])*jax.random.normal(o,y.shape)
            return mask*obs+(1-mask)*z,key
        x,key=jax.lax.fori_loop(0,j.where(u>0,2,1),attempt,(x,key))
        return (x,key),None
    (x,_),_=jax.jit(lambda xx,kk:jax.lax.scan(jump,(xx,kk),j.arange(1000,0,-20)))(x,key)
    return x,99

def energy(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    # U statistic: exclude all same-set diagonal pairs.
    return float(2*cdist(x,y).mean()-pdist(x).mean()-pdist(y).mean())

def metrics(x):
    x=np.asarray(x);mu=np.asarray(CENTRES);idx=np.argmin(((x[:,None]-mu)**2).sum(-1),axis=1)
    frac=[];means=[];std=[]
    for k in range(4):
        z=x[idx==k];frac.append(len(z)/len(x));means.append(z.mean(0).tolist() if len(z) else [float('nan')]*2);std.append(z.std(0).tolist() if len(z) else [float('nan')]*2)
    return dict(fractions=frac,centres=means,widths=std,fraction_error=float(np.max(np.abs(np.array(frac)-.25))),centre_error=float(np.max(np.linalg.norm(np.array(means)-mu,axis=1))),width_error=float(np.max(np.abs(np.array(std)-.3))))
