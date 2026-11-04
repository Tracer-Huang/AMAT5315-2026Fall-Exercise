"""Independent NumPy baseline for UAI factor models and sum-product BP."""
import itertools
import numpy as np

def read_uai(path):
    from pathlib import Path
    t = iter(Path(path).read_text().split())
    if next(t) != 'MARKOV': raise ValueError('Expected MARKOV model')
    n = int(next(t)); dims = [int(next(t)) for _ in range(n)]
    nf = int(next(t)); scopes = [tuple(int(next(t)) for _ in range(int(next(t)))) for _ in range(nf)]
    factors=[]
    for scope in scopes:
        count=int(next(t)); data=np.array([float(next(t)) for _ in range(count)])
        factors.append((scope, data.reshape(tuple(dims[i] for i in scope))))
    return dims, factors

def enumerate_asia(model, evidence):
    dims, factors = model
    joint=np.zeros(dims)
    for state in itertools.product(*(range(d) for d in dims)):
        if any(state[v]!=x for v,x in evidence.items()): continue
        joint[state]=np.prod([a[tuple(state[v] for v in scope)] for scope,a in factors])
    z=joint.sum(); joint/=z
    marg=[joint.sum(axis=tuple(j for j in range(len(dims)) if j!=i)) for i in range(len(dims))]
    return z,marg,joint.sum(axis=tuple(j for j in range(len(dims)) if j not in (1,3)))

def bp(model, evidence, max_sweeps=1000, tolerance=1e-10, damping=.5):
    dims, factors=model; n=len(dims)
    neighbors=[[f for f,(scope,_) in enumerate(factors) if v in scope] for v in range(n)]
    factors2=[]
    for scope,a in factors:
        a=a.copy()
        for axis,v in enumerate(scope):
            if v in evidence:
                mask=np.zeros(dims[v]); mask[evidence[v]]=1
                shape=[1]*a.ndim; shape[axis]=dims[v]; a=a*mask.reshape(shape)
        factors2.append((scope,a))
    vf={(v,f):np.ones(dims[v])/dims[v] for v in range(n) for f in neighbors[v]}
    fv={(f,v):np.ones(dims[v])/dims[v] for v in range(n) for f in neighbors[v]}
    def norm(x):
        s=x.sum()
        if s<=0: raise ValueError('zero message')
        return x/s
    def update(vf,fv):
        nv={}; nf={}
        for v in range(n):
            for f in neighbors[v]:
                x=np.ones(dims[v])
                for g in neighbors[v]:
                    if g!=f: x*=fv[g,v]
                nv[v,f]=norm(x)
        for f,(scope,a) in enumerate(factors2):
            for axis,v in enumerate(scope):
                x=a.copy()
                for ax,u in enumerate(scope):
                    if u!=v:
                        shape=[1]*a.ndim; shape[ax]=dims[u]; x=x*vf[u,f].reshape(shape)
                axes=tuple(ax for ax in range(a.ndim) if ax!=axis)
                nf[f,v]=norm(x.sum(axis=axes) if axes else x)
        return nv,nf
    def beliefs(fv):
        out=[]
        for v in range(n):
            x=np.ones(dims[v])
            for f in neighbors[v]: x*=fv[f,v]
            out.append(norm(x))
        return out
    history=[]
    for sweep in range(max_sweeps+1):
        nv,nf=update(vf,fv)
        res=max([np.max(np.abs(nv[k]-vf[k])) for k in vf]+[np.max(np.abs(nf[k]-fv[k])) for k in fv])
        b=beliefs(fv)
        history.append({'sweep':sweep,'residual':float(res),'beliefs':[x.tolist() for x in b]})
        if res<tolerance or sweep==max_sweeps: break
        vf={k:(1-damping)*vf[k]+damping*nv[k] for k in vf}
        fv={k:(1-damping)*fv[k]+damping*nf[k] for k in fv}
    return b,history

def contract_asia(model,evidence):
    dims,factors=model
    operands=[]
    for scope,arr in factors: operands += [arr,list(scope)]
    for v,x in evidence.items():
        mask=np.zeros(dims[v]); mask[x]=1; operands += [mask,[v]]
    def query(out): return np.einsum(*operands,list(out),optimize='greedy')
    z=float(query([])); marg=[query([i])/z for i in range(len(dims))]
    return z,marg,query([1,3])/z

def initial_factors(model,evidence):
    dims,factors=model; result={f'f{i}':(tuple(s),a) for i,(s,a) in enumerate(factors)}
    for v,x in sorted(evidence.items()):
        mask=np.zeros(dims[v]); mask[x]=1; result[f'e{v}']=((v,),mask)
    return result

def _pair(scopes,a,b):
    union=set(scopes[a])|set(scopes[b]); outside=set().union(*(set(s) for k,s in scopes.items() if k not in (a,b)))
    return tuple(sorted(union & outside)), tuple(sorted(union))

def make_plan(model,evidence,method='greedy',seed=42):
    import random
    dims,_=model; scopes={k:s for k,(s,_) in initial_factors(model,evidence).items()}; rng=random.Random(seed); steps=[]
    while len(scopes)>1:
        keys=list(scopes); pairs=list(itertools.combinations(keys,2))
        if method=='random': a,b=rng.choice(pairs)
        elif method=='greedy': a,b=min(pairs,key=lambda pair:(np.prod([dims[i] for i in _pair(scopes,*pair)[0]],dtype=int),np.prod([dims[i] for i in _pair(scopes,*pair)[1]],dtype=int),pair))
        else: raise ValueError(method)
        keep,_=_pair(scopes,a,b); key=f'i{len(steps)}'; steps.append([a,b,key]); del scopes[a],scopes[b]; scopes[key]=keep
    return {'method':method,'seed':seed if method=='random' else None,'steps':steps}

def execute_plan(model,evidence,plan):
    dims,_=model; factors=initial_factors(model,evidence); largest=max(a.size for _,a in factors.values()); work=0; steps=[]
    for a,b,key in plan['steps']:
        scopes={k:s for k,(s,_) in factors.items()}; keep,union=_pair(scopes,a,b); cost=int(np.prod([dims[i] for i in union],dtype=int)); work+=cost
        sa,aa=factors.pop(a); sb,ab=factors.pop(b)
        result=np.einsum(aa,list(sa),ab,list(sb),list(keep)); size=int(result.size); largest=max(largest,size)
        factors[key]=(keep,result); steps.append({'inputs':[a,b],'retained':keep,'union':union,'work':cost,'entries':size})
    if len(factors)!=1: raise ValueError('incomplete plan')
    scope,result=next(iter(factors.values()))
    if scope: raise ValueError('non-scalar plan')
    return float(result),{'work':work,'largest_tensor':largest,'steps':steps}
