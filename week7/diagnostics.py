"""Additional explicit-domain diagnostics, keeping the original grid failure."""
import json
from pathlib import Path
import numpy as np
import jax.numpy as j
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as c
p,_,_=c.load('flow')
x=np.stack(np.meshgrid(np.linspace(-4,4,41),np.linspace(-4,4,41)),axis=-1).reshape(-1,2).astype(np.float32)
fig,axs=plt.subplots(1,3,figsize=(15,5));rows=[]
for ax,s in zip(axs,[.1,.5,.9]):
 v=.09*(1-s)**2+s*s;dens=np.exp(-((x[:,None]-(1-s)*np.array(c.CENTRES))**2).sum(-1)/(2*v)).sum(1)
 supported=dens>.01*dens.max();a=np.array(c.exact_flow(j.array(x),s));b=np.array(c.net(p[-1],j.array(x),s))
 rms=lambda w:float(np.sqrt(np.sum(w[:,None]*(a-b)**2)/np.sum(w[:,None]*a*a)))
 row=dict(s=s,grid='41x41 on [-4,4]^2',support_definition='p_s(x)>0.01 max_grid p_s',support_points=int(supported.sum()),full_grid_relative_rms=rms(np.ones(len(x))),density_weighted_relative_rms=rms(dens),supported_grid_relative_rms=rms(supported));rows.append(row)
 for vv,color,label in [(a,'black','Exact'),(b,'orange','Learned')]:
  z=vv[supported];z=z/np.maximum(1,np.linalg.norm(z,axis=1,keepdims=True)/4);ax.quiver(*x[supported].T,*z.T,color=color,alpha=.6,label=label)
 ax.set(title=f's={s}; support RMS={row["supported_grid_relative_rms"]:.4f}',xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.legend()
fig.tight_layout();fig.savefig('evidence/velocity-supported.png',dpi=160);plt.close(fig)
Path('evidence/velocity-domain-diagnostics.json').write_text(json.dumps(rows,indent=2))
r=json.load(open('evidence/toy-results.json'));fig,axs=plt.subplots(1,3,figsize=(15,4))
for method in ['ddpm','ddim','flow']:
 for kind,ls in [('comparison','-'),('exact_comparison','--')]:
  rr=[v for v in r[kind] if v['method']==method]
  for ax,field in zip(axs,['energy','fraction_error','width_error']):ax.plot([v['steps'] for v in rr],[max(v[field],1e-4) if field=='energy' else v[field] for v in rr],ls,marker='o',label=method+(' learned' if kind=='comparison' else ' exact'))
for ax,field in zip(axs,['energy','fraction_error','width_error']):ax.set(xscale='log',xlabel='Steps',ylabel=field);ax.legend(fontsize=7)
axs[0].set_yscale('log');axs[0].axhspan(1e-4,r['energy_null']['q95'],color='gray',alpha=.2)
fig.tight_layout();fig.savefig('evidence/exact-vs-learned.png',dpi=160);plt.close(fig)
print(json.dumps(rows,indent=2))
