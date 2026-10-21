"""Reproducible Parts 1-3. All comparisons use prescribed counts."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import core as c
from plotting import draw_training_stage,draw_paired_paths
P=Path('evidence');P.mkdir(exist_ok=True);Path('artifacts').mkdir(exist_ok=True)
results={}
def save(name,fig):
    fig.tight_layout();fig.savefig(P/f'{name}.png',dpi=150);plt.close(fig)
def scat(ax,x,title):
    x=np.asarray(x);ax.scatter(*x.T,s=2,alpha=.35);ax.scatter(*np.array(c.CENTRES).T,marker='+',c='red');ax.set(xlim=(-5,5),ylim=(-5,5),title=title,aspect='equal')
def movie(name,paths,levels):
    fig,ax=plt.subplots(figsize=(5,5));scat(ax,paths[0],str(levels[0]));dots=ax.collections[0]
    def frame(i):dots.set_offsets(paths[i]);ax.set_title(f'{name}: level {levels[i]}');return dots,
    FuncAnimation(fig,frame,frames=len(paths),interval=120).save(P/f'{name}.gif',writer=PillowWriter(fps=8));plt.close(fig)
def exact(n,seed):return np.asarray(c.draw(jax.random.key(seed),n))
t0=time.time()
# Train independently, checkpoint all ten stages.
for kind in ['noise','flow']:
    if not Path(f'artifacts/{kind}-training.npz').exists(): c.train(kind,updates=100000,architecture="scalar_basis",width=64)
noise,stages,loss=c.load('noise');flow,fs,floss=c.load('flow')
predict=lambda x,t:c.net(noise[-1],x,t/1000)
velocity=lambda x,s:c.net(flow[-1],x,s)
# True forward Markov trajectories, not independently re-noised snapshots.
x=j.array(exact(2000,7001)); key=jax.random.key(7002)
def forward(st,t):
    x,key=st;key,k=jax.random.split(key);y=j.sqrt(1-c.B[t])*x+j.sqrt(c.B[t])*jax.random.normal(k,x.shape);return (y,key),y
(_, _),fw=jax.jit(lambda x,k:jax.lax.scan(forward,(x,k),j.arange(1,1001)))(x,key)
fw=np.concatenate([np.array(x)[None],np.asarray(fw)])
results['forward']={'mean':fw[-1].mean(0).tolist(),'variance':fw[-1].var(0).tolist()}
np.savez('artifacts/forward.npz',frames=fw[::25],levels=np.arange(0,1001,25),final=fw[-1])
levels=np.unique(np.r_[np.arange(0,1001,25),1000]);movie('forward',fw[levels],levels)
fig,axes=plt.subplots(1,6,figsize=(15,3))
for ax,t in zip(axes,[0,50,200,400,700,1000]):scat(ax,fw[t],f't={t}')
save('forward',fig)
starts=j.array(np.random.default_rng(7003).normal(size=(2000,2)),dtype=j.float32)
y,rev=c.sample(predict,starts,1000,'ddpm',7004);y=np.asarray(y);np.savez('artifacts/ddpm.npz',samples=y,paths=np.asarray(rev)[:, :10],frames=np.asarray(rev)[::25],levels=np.arange(1000,-1,-25))
results['ddpm']=c.metrics(y);movie('reverse',np.asarray(rev)[::25],np.arange(1000,-1,-25))
fig,axs=plt.subplots(1,2,figsize=(10,5));scat(axs[0],exact(2000,7005),'Exact target');scat(axs[1],y,'Learned DDPM');save('ddpm',fig)
# Same starts and all reverse noise draws at every training stage.
paths=[]
for p in noise:
    _,path=c.sample(lambda x,t:c.net(p,x,t/1000),starts[:10],1000,'ddpm',7010);paths.append(np.asarray(path))
np.savez('artifacts/training-paths.npz',paths=np.array(paths),stages=stages)
fig,axs=plt.subplots(1,2,figsize=(10,4))
def stage_frame(i):
    draw_training_stage(axs,paths,stages,loss,i,np.array(c.CENTRES))
FuncAnimation(fig,stage_frame,frames=10,interval=700).save(P/'training.gif',writer=PillowWriter(fps=1));plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for ls,label in [(loss,'Noise'),(floss,'Velocity')]:ax.plot(np.arange(100,len(ls)+1,100),ls[:len(ls)//100*100].reshape(-1,100).mean(1),label=label)
ax.set(xlabel='Updates',ylabel='Batch MSE (100-update mean)');ax.legend();save('loss',fig)
# Determinism and stochasticity.
d1,dp=c.sample(predict,starts,1000,'ddim',11);d2,dp2=c.sample(predict,starts,1000,'ddim',12)
r1,rp=c.sample(predict,starts,1000,'ddpm',11);r2,rp2=c.sample(predict,starts,1000,'ddpm',12)
assign=lambda z:np.argmin(((np.asarray(z)[:,None]-np.asarray(c.CENTRES))**2).sum(-1),axis=1)
results['determinism']={'ddim_max_difference':float(np.max(np.abs(d1-d2))),'ddpm_same_cluster_fraction':float(np.mean(assign(r1)==assign(r2))),'ddim':c.metrics(d1),'ddpm':c.metrics(r1)}
fig,axs=plt.subplots(1,5,figsize=(20,4))
for ax,z,title in zip(axs,[exact(2000,20),d1,d2,r1,r2],['Exact','DDIM run 1','DDIM run 2','DDPM run 1','DDPM run 2']):scat(ax,z,title)
save('determinism',fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
draw_paired_paths(axs,dp,dp2,rp,rp2)
np.savez('artifacts/paired-runs.npz',starts=np.asarray(starts),ddim1=np.asarray(d1),ddim2=np.asarray(d2),ddpm1=np.asarray(r1),ddpm2=np.asarray(r2),ddim_paths1=np.asarray(dp)[:,:10],ddim_paths2=np.asarray(dp2)[:,:10],ddpm_paths1=np.asarray(rp)[:,:10],ddpm_paths2=np.asarray(rp2)[:,:10])
save('trajectories',fig)
Path('evidence/plotting-run.json').write_text(json.dumps(dict(
    source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['run_toy.py','plotting.py']},
    stages=stages.tolist(),loss_points_per_stage=[len(__import__('plotting').loss_history(loss,int(s))[0]) for s in stages],
    paired_data='artifacts/paired-runs.npz',paired_paths_per_panel=20),indent=2))
# Velocity grid: same fixed domain at all times, no density filtering.
xy=np.stack(np.meshgrid(np.linspace(-4,4,21),np.linspace(-4,4,21)),axis=-1).reshape(-1,2).astype(np.float32)
fig,axs=plt.subplots(1,3,figsize=(15,5));errs=[]
for ax,s in zip(axs,[.1,.5,.9]):
    ev=np.array(c.exact_flow(j.array(xy),s));pv=np.array(velocity(j.array(xy),s));err=float(np.sqrt(np.sum((ev-pv)**2)/np.sum(ev**2)));errs.append(err)
    for v,color,label in [(ev,'black','Exact'),(pv,'orange','Learned')]:
        vv=v/np.maximum(1,np.linalg.norm(v,axis=1,keepdims=True)/4);ax.quiver(*xy.T,*vv.T,color=color,alpha=.6,label=label)
    ax.set(title=f's={s}, relative RMS={err:.4f}',aspect='equal');ax.legend()
results['velocity_relative_rms']=errs;save('velocity',fig)
# Null U-statistic scale: 20 independent pairs, each set N=4000.
null=[]
for i in range(20):null.append(c.energy(exact(4000,8000+2*i),exact(4000,8001+2*i)))
q=float(np.quantile(null,.95));results['energy_null']={'values':null,'q95':q,'n':4000,'pairs':20}
reference=exact(4000,8100);start=j.array(np.random.default_rng(8101).normal(size=(4000,2)),dtype=j.float32)
rows=[]
for method in ['ddpm','ddim','flow']:
    for n in ([1,2,5,10,20,50,100]+([] if method=='flow' else [1000])):
        z,_=c.flow_sample(velocity,start,n) if method=='flow' else c.sample(predict,start,n,method,8102)
        row=dict(method=method,steps=n,**c.metrics(z),energy=c.energy(z,reference));rows.append(row)
        print('comparison',method,n,row['energy'],row['width_error'],flush=True)
        np.save(f'artifacts/{method}-{n}.npy',np.asarray(z))
# Exact predictors on the same step grids isolate discretization error.
exact_rows=[]
for method in ['ddpm','ddim','flow']:
    for n in [1,2,5,10,20,50,100]:
        z,_=c.flow_sample(c.exact_flow,start,n) if method=='flow' else c.sample(c.exact_noise,start,n,method,8102)
        exact_rows.append(dict(method=method,steps=n,**c.metrics(z),energy=c.energy(z,reference)))
results['comparison']=rows;results['exact_comparison']=exact_rows
fig,axs=plt.subplots(1,3,figsize=(15,4))
for method in ['ddpm','ddim','flow']:
    rr=[r for r in rows if r['method']==method]
    for ax,field in zip(axs,['energy','fraction_error','width_error']):ax.plot([r['steps'] for r in rr],[max(r[field],1e-4) if field=='energy' else r[field] for r in rr],'-o',label=method)
for ax,field in zip(axs,['energy','fraction_error','width_error']):ax.set(xscale='log',xlabel='Network evaluations',ylabel=field);ax.axvline(20,ls='--',color='gray');ax.legend()
axs[0].axhspan(1e-4,max(q,1e-4),alpha=.2,color='gray',label='Exact null q95');axs[0].set_yscale('log')
axs[1].axhspan(0,2*np.sqrt(.25*.75/4000),alpha=.2,color='gray');axs[2].axhspan(0,.03,alpha=.1,color='gray');save('comparison',fig)
results['settings']=dict(backend=jax.default_backend(),jax=jax.__version__,noise_updates=len(loss),flow_updates=len(floss),batch=512,width=int(noise[-1][0][0].shape[1]),seed=7101,learning_rate=.001,architecture='shared odd linear+tanh scalar family; coefficients learned by a time MLP' if noise[-1][0][0].shape[0]==16 else 'factorized odd scalar MLP' if noise[-1][0][0].shape[0]==17 else 'dense MLP',schedule='cosine to zero',seconds=time.time()-t0)
Path('artifacts/toy-results.json').write_text(json.dumps(results,indent=2));Path('evidence/toy-results.json').write_text(json.dumps(results,indent=2))
print('DONE',results['ddpm'],errs,flush=True)
