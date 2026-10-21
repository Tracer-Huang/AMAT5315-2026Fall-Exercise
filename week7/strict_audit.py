"""Recalculate artifact-based Week 7 checks; fail closed for PDF omissions.
This audits the current run. It does not change models or mask the velocity grid.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from scipy.spatial.distance import cdist,pdist
from PIL import Image
import core as c
checks=[]
def check(name,passed,detail,pages):
    checks.append(dict(requirement=name,passed=bool(passed),detail=detail,pdf_pages=pages))
r=json.loads(Path('evidence/toy-results.json').read_text());im=json.loads(Path('evidence/image-results.json').read_text())
# Independently recalculate cluster moments from saved arrays, not summary flags.
mu=np.array([[3.,0.],[0.,3.],[-3.,0.],[0.,-3.]])
def raw_metrics(z):
    labels=cdist(z,mu).argmin(1);groups=[z[labels==i] for i in range(4)]
    return dict(fractions=np.array([len(g)/len(z) for g in groups]),centres=np.array([g.mean(0) for g in groups]),widths=np.array([g.std(0) for g in groups]))
def matches(a,b):return all(np.allclose(a[k],b[k],atol=2e-6,rtol=2e-6) for k in ['fractions','centres','widths'])
z=np.load('artifacts/ddpm.npz')['samples'];m=raw_metrics(z)
check('P1 final sample counts and moments',z.shape==(2000,2) and matches(m,r['ddpm']) and abs(m['fractions']-.25).max()<.04 and abs(m['widths']-.3).max()<.03 and np.linalg.norm(m['centres']-mu,axis=1).max()<.05,r['ddpm'],[6,7])
for kind in ['noise','flow']:
    z=np.load(f'artifacts/{kind}-training.npz')
    check(f'{kind} ten actual checkpoints and loss history',len(z['stages'])==10 and z['stages'][0]==0 and z['stages'][-1]==len(z['loss']) and np.isfinite(z['loss']).all(),dict(stages=z['stages'].tolist(),loss_updates=len(z['loss'])),[2,4,6,8])
training_settings={kind:json.loads(str(np.load(f'artifacts/{kind}-training.npz')['settings'])) for kind in ['noise','flow']}
noise_params,_,_=c.load('noise');flow_params,_,_=c.load('flow')
same_shapes=[a.shape for a in jax.tree.leaves(noise_params[0])]==[a.shape for a in jax.tree.leaves(flow_params[0])]
check('P1_3 same architecture, separate regression training',same_shapes and all(training_settings[k]['kind']==k and training_settings[k]['architecture']=='scalar_basis' for k in ['noise','flow']),training_settings,[2,8])
for file,frames in [('forward.gif',41),('reverse.gif',41),('training.gif',10)]:
    image=Image.open('evidence/'+file);image.seek(image.n_frames-1);image.load()
    check(file+' decodes',image.n_frames==frames,dict(frames=image.n_frames),[6])
# Inspect the actual frame-rendering artists and bind them to the generated artifacts.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotting import draw_training_stage,draw_paired_paths
plot_run=json.loads(Path('evidence/plotting-run.json').read_text())
plot_binding=all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in plot_run['source_sha256'].items())
training=np.load('artifacts/training-paths.npz');training_loss=np.load('artifacts/noise-training.npz')['loss'];fig,axes=plt.subplots(1,2);coverage=[]
for i,stage in enumerate(training['stages']):
    draw_training_stage(axes,training['paths'],training['stages'],training_loss,i,mu)
    coverage.append(bool(len(axes[0].lines)==10 and (stage==0 or (len(axes[1].lines)==1 and len(axes[1].lines[0].get_xdata())>0 and axes[1].lines[0].get_xdata()[-1]==stage))))
plt.close(fig)
check('P1 training animation includes loss history at every trained stage',all(coverage) and plot_binding,dict(stages=training['stages'].tolist(),coverage=coverage,producer_binding=plot_binding),[6])
starts=j.array(np.random.default_rng(7003).normal(size=(2000,2)),dtype=j.float32)
noise,_,_=c.load('noise');predict=lambda x,t:c.net(noise[-1],x,t/1000)
a,_=c.sample(predict,starts,1000,'ddim',11);b,_=c.sample(predict,starts,1000,'ddim',12)
x,_=c.sample(predict,starts,1000,'ddpm',11);y,_=c.sample(predict,starts,1000,'ddpm',12)
d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));same=float(np.mean(cdist(x,mu).argmin(1)==cdist(y,mu).argmin(1)))
check('P2 independent replay: DDIM equality and two DDPM runs',d==0 and abs(same-r['determinism']['ddpm_same_cluster_fraction'])<1e-9,dict(ddim_max_difference=d,ddpm_same_cluster_fraction=same,ddim_metrics_match=matches(raw_metrics(np.asarray(a)),r['determinism']['ddim'])),[8])
paired=np.load('artifacts/paired-runs.npz');fig,axes=plt.subplots(1,2)
draw_paired_paths(axes,paired['ddim_paths1'],paired['ddim_paths2'],paired['ddpm_paths1'],paired['ddpm_paths2'])
line_counts=[len(ax.lines) for ax in axes]
paired_match=all(np.allclose(np.c_[axes[1].lines[2*k+1].get_xdata(),axes[1].lines[2*k+1].get_ydata()],paired['ddpm_paths2'][:,k]) for k in range(10))
plt.close(fig)
check('P2 trajectories plot includes both stochastic repeats',line_counts==[20,20] and paired_match and plot_binding,dict(lines_per_panel=line_counts,second_ddpm_paths_match=paired_match,producer_binding=plot_binding),[8])
# Exactly the original unweighted grid: no support mask, no clipping of numeric values.
p,_,_=c.load('flow');xy=j.array(np.stack(np.meshgrid(np.linspace(-4,4,21),np.linspace(-4,4,21)),axis=-1).reshape(-1,2),dtype=j.float32)
errs=[]
for s in [.1,.5,.9]:
    v=np.array(c.exact_flow(xy,s));pred=np.array(c.net(p[-1],xy,s));errs.append(float(np.linalg.norm(pred-v)/np.linalg.norm(v)))
check('P3 velocity RMS <0.1 at all three s',all(e<.1 for e in errs),dict(domain='original full 21x21 [-4,4]^2 grid',s=[.1,.5,.9],relative_rms=errs,stored_values_match=bool(np.allclose(errs,r['velocity_relative_rms'],atol=1e-6)),density_mask_applied=False),[11])
expected={(method,n) for method in ['ddpm','ddim','flow'] for n in ([1,2,5,10,20,50,100]+([] if method=='flow' else [1000]))}
seen={(row['method'],row['steps']) for row in r['comparison']}
reference=np.asarray(c.draw(jax.random.key(8100),4000));discrepancies=[]
for row in r['comparison']:
    z=np.load(f'artifacts/{row["method"]}-{row["steps"]}.npy');mat=raw_metrics(z)
    en=2*cdist(z.astype(np.float64),reference.astype(np.float64)).mean()-pdist(z.astype(np.float64)).mean()-pdist(reference.astype(np.float64)).mean()
    if z.shape!=(4000,2) or not matches(mat,row) or not np.isclose(en,row['energy'],atol=1e-9):discrepancies.append([row['method'],row['steps']])
check('P3 every prescribed step, raw N=4000, moments and energy',seen==expected and len(r['comparison'])==len(expected) and not discrepancies,dict(runs=len(seen),mismatches=discrepancies),[8,11])
null=r['energy_null'];check('P3 null scale and 20-step flow targets',len(null['values'])==20 and null['n']==4000 and np.isclose(np.quantile(null['values'],.95),null['q95']) and next(v for v in r['comparison'] if v['method']=='flow' and v['steps']==20)['energy']<.005 and next(v for v in r['comparison'] if v['method']=='flow' and v['steps']==20)['width_error']<.03,dict(null_pairs=20,null_q95=null['q95'],flow20=next(v for v in r['comparison'] if v['method']=='flow' and v['steps']==20)),[8,11])
photos=np.load('artifacts/photos.npz')
check('P4 four photographs and six reverse stages',photos['samples'].shape==(4,3,32,32) and photos['samples'].dtype==np.float32 and photos['stages'].shape==(4,6,3,32,32) and np.isfinite(photos['samples']).all(),dict(visual_evidence=['photographs.png','real-vs-generated.png'],stage_levels=[1000,500,200,100,50,0],source_review='JAX float32; t-1 indexing; no sampling clipping; frozen weights'),[12,13])
truth=np.load('inputs/images.npz')['test'];mask=np.ones_like(truth);mask[:,:,6:26,6:26]=0;allz=[];recomputed=[]
for seed in [7201,7202,7203,7204]:
    z=np.load(f'artifacts/inpaint-{seed}.npy');assert z.shape==truth.shape;allz.append(z)
    for i in range(4):
        mse=float(((z[i,:,6:26,6:26]-truth[i,:,6:26,6:26])**2).mean());delta=float(np.abs((z[i]-truth[i])*mask[i]).max())
        old=next(v for v in im['completions'] if v['image']==i and v['seed']==seed)
        recomputed.append(dict(image=i,seed=seed,mse=mse,observed_max_change=delta,matches_saved=bool(np.isclose(mse,old['hidden_mse'],atol=1e-8))))
var=np.var(allz,axis=0);outside=float((var*mask).max());inside=float(var[:,:,6:26,6:26].mean())
check('P5 all 16 MSEs, exact observed pixels, inside-only variation',all(v['matches_saved'] and v['observed_max_change']<1e-5 for v in recomputed) and outside==0 and inside>0,dict(completions=recomputed,outside_variance=outside,mean_inside_variance=inside),[14,15,16])
# Runtime instrumentation counts actual executions, rather than trusting return n=99.
levels=[]
def counter(x,t):
    jax.debug.callback(lambda tt:levels.append(int(tt)),t,ordered=True)
    return j.zeros_like(x)
tiny=j.zeros((1,3,4,4),dtype=j.float32);result,_=c.repaint(counter,tiny,j.ones_like(tiny),901);result.block_until_ready();jax.effects_barrier()
expected_levels=[t for t in range(1000,0,-20) for _ in range(1 if t==20 else 2)]
check('P5 actual 99 runtime calls and requested jump sequence',levels==expected_levels,dict(calls=len(levels),levels=levels),[15])
sha=hashlib.sha256(Path('inputs/cifar-model/diffusion_pytorch_model.safetensors').read_bytes()).hexdigest()
check('P4_5 pinned model identity',sha=='1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f',dict(sha256=sha,revision='267b167dc01f0e4e61923ea244e8b988f84deb80'),[12])
failed=[v['requirement'] for v in checks if not v['passed']]
summary=dict(checked_against='local original Week 7 16-page PDF, DO / VERIFY / final instructions',submission_ready=not failed,failed_requirements=failed,checks=checks,authorization='User authorizes completion, strict recheck, and course-required push once all checks pass',submission_semantics='PDF says push source, environment, evidence; Nothing is submitted; no LMS submission prescribed',publication_scope='This audit performs no git operations; delivery is recorded separately',manual_plot_review_source=dict(file='run_toy.py',sha256=hashlib.sha256(Path('run_toy.py').read_bytes()).hexdigest(),scope='Actual plotting artists and artifact producer hashes checked for current version'))
Path('evidence/strict-audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
for v in checks:print(('PASS' if v['passed'] else 'FAIL'),v['requirement'])
print('SUBMISSION_READY',summary['submission_ready'])
raise SystemExit(0 if summary['submission_ready'] else 1)
