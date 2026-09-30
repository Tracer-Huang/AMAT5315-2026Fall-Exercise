"""Manual node-wise JAX passes; no whole-function autodiff in manual passes."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

def energy(r):
    a = r ** -6
    b = a ** 2
    c = b - a
    return 4 * c

def manual(r):
    r = jnp.asarray(r, dtype=jnp.float64)
    a = r ** -6
    b = a ** 2
    c = b - a
    u = 4 * c
    dr = jnp.ones_like(r)
    da = -6 * r ** -7 * dr
    db = 2 * a * da
    dc = db - da
    du = 4 * dc
    au = jnp.ones_like(r)
    ac = 4 * au
    ab = ac
    aa = -ac + 2 * a * ab
    ar = -6 * r ** -7 * aa
    return {'r': r, 'energy': u,
            'tangents': dict(zip(['r','a','b','c','U'],[dr,da,db,dc,du])),
            'adjoints': dict(zip(['r','a','b','c','U'],[ar,aa,ab,ac,au]))}

def draw_graph(closed, path):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    graph = closed.jaxpr
    nodes = {}; producer = {}; depth = {}; edges = []
    for i, var in enumerate(graph.invars):
        key = 'input'+str(i); nodes[key]='r'; depth[key]=0; producer[str(var)]=key
    for i, eq in enumerate(graph.eqns):
        key='op'+str(i); parents=[]; constants=[]
        for v in eq.invars:
            if str(v) in producer:
                parent=producer[str(v)]; parents.append(parent); edges.append((parent,key))
            else: constants.append(str(getattr(v,'val',v)))
        depth[key]=1+max((depth[p] for p in parents),default=0)
        label=eq.primitive.name
        if 'y' in eq.params: label+='[y='+str(eq.params['y'])+']'
        if constants: label+='\nconstant: '+', '.join(constants)
        nodes[key]=label
        for v in eq.outvars: producer[str(v)]=key
    for i,v in enumerate(graph.outvars):
        key='output'+str(i); parent=producer[str(v)]; nodes[key]='output';depth[key]=depth[parent]+1;edges.append((parent,key))
    layers=defaultdict(list)
    for key,d in depth.items():layers[d].append(key)
    pos={key:((i-(len(keys)-1)/2)*3.0,-d) for d,keys in layers.items() for i,key in enumerate(keys)}
    fig,ax=plt.subplots(figsize=(11,max(5,len(layers)*.8)))
    for start,end in edges:
        ax.annotate('',xy=pos[end],xytext=pos[start],arrowprops=dict(arrowstyle='->',color='#64748b',lw=1.25,shrinkA=23,shrinkB=23,connectionstyle='arc3,rad=.06'),zorder=1)
    for key,label in nodes.items():
        ax.text(*pos[key],label,ha='center',va='center',fontsize=10,zorder=3,bbox=dict(boxstyle='round,pad=.45',fc='#fef3c7' if 'add_any' in label else '#e5f1fb',ec='#477fa5'))
    xs=[p[0] for p in pos.values()];ax.set(xlim=(min(xs)-1.6,max(xs)+1.6),ylim=(-max(depth.values())-.7,.7));ax.axis('off')
    ax.set_title('JAX recorded '+('gradient graph' if 'grad-' in path.name else 'energy graph'))
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def main():
    import json
    import numpy as np
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    out=Path(os.environ.get('WEEK5_ARTIFACTS',Path(__file__).resolve().parents[1]/'artifacts'))/'ad';out.mkdir(parents=True,exist_ok=True)
    value=manual(1.3)
    value['jax_grad']=jax.grad(energy)(1.3)
    (out/'derivatives.json').write_text(json.dumps(value,default=lambda x:float(x),indent=2)+'\n')
    rs=jnp.linspace(.95,2.5,601);ex=24*(rs**-7-2*rs**-13)
    m=manual(rs);fd=(energy(rs+1e-6)-energy(rs-1e-6))/2e-6
    fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    errors={}
    for name,values in [('Forward AD',m['tangents']['U']),('Reverse AD',m['adjoints']['r']),('Finite difference',fd)]:
        err=np.abs(values-ex);errors[name]=float(np.max(err))
        axs[0].plot(rs,values,label=name);axs[1].semilogy(rs,np.maximum(err,1e-17),label=name)
    axs[0].plot(rs,ex,'k:',label='Analytic');axs[0].axhline(0,c='gray',lw=.5)
    axs[0].set(ylabel='dU/dr');axs[1].set(ylabel='Absolute error')
    for ax in axs:ax.set_xlabel('Separation r (reduced units)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'modes.png',dpi=180);plt.close(fig)
    for name,fn in [('graph',energy),('grad-graph',jax.grad(energy))]:
        g=jax.make_jaxpr(fn)(1.3);(out/(name+'.txt')).write_text(str(g)+'\n');draw_graph(g,out/(name+'.png'))
    (out/'errors.json').write_text(json.dumps(errors,indent=2)+'\n');print(errors)
    assert errors['Forward AD']<1e-12 and errors['Reverse AD']<1e-12
    assert max(errors['Forward AD'],errors['Reverse AD'])<errors['Finite difference']

if __name__=='__main__':main()
