"""Recompute every plotted quantity from the student Rust outputs, never reference images."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
import os
A=Path(os.environ.get('WEEK5_ARTIFACTS',ROOT/'artifacts'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def load(p):return json.loads(Path(p).read_text())
def save(fig,path):
 fig.tight_layout();fig.savefig(A/path,dpi=170);plt.close(fig)
def panel(ax,data,extent,title,label,cmap='RdBu_r',limit=None):
 if limit is None:limit=max(abs(data.min()),abs(data.max()))
 im=ax.imshow(data,extent=extent,origin='upper',aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
 ax.set(title=title,xlabel='Horizontal position (km)',ylabel='Depth (km)')
 plt.colorbar(im,ax=ax,label=label,shrink=.85);return im

def small():
 e=load(ROOT/'inputs/reflector.json');dx=e['dx']*e['length_unit_m']/1000;dt=e['dt']*e['time_unit_s'];extent=[0,(e['nx']-1)*dx,(e['nz']-1)*dx,0]
 c=np.array(e['background']);m=np.array(e['perturbation']);shots=np.array(e['shots'])*dx;recv=np.array(e['receivers'])*dx
 fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 im=axs[0].imshow(c,extent=extent,origin='upper',vmin=1.5,vmax=2.1);plt.colorbar(im,ax=axs[0],label='Speed (km/s)')
 axs[0].scatter(*shots.T,marker='*',s=100,c='orange',edgecolors='k',label='Shots');axs[0].scatter(*recv.T,marker='v',s=20,c='white',edgecolors='k',label='Receivers');axs[0].axhline(2.1,c='red',ls='--',label='Reflector');axs[0].legend(loc='lower right');axs[0].set(title='Acquisition geometry',xlabel='Horizontal position (km)',ylabel='Depth (km)')
 panel(axs[1],m,extent,'Thin reflector','Velocity perturbation (km/s)');save(fig,'inputs.png')
 traces=np.load(A/'forward/traces.npy');fig,axs=plt.subplots(1,3,figsize=(13,4.5));limit=abs(traces).max()
 peaks=[]
 for i,ax in enumerate(axs):
  im=ax.imshow(traces[i],extent=[recv[0,0],recv[-1,0],e['steps']*dt,dt],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
  ax.set(title=f'Shot {i}; source x={shots[i,0]:.1f} km',xlabel='Receiver position (km)',ylabel='Time (s)');plt.colorbar(im,ax=ax,label='Pressure (a.u.)')
  n,k=np.unravel_index(abs(traces[i]).argmax(),traces[i].shape);peaks.append(dict(shot=i,trace_index=int(n),receiver=int(k),absolute_pressure=float(abs(traces[i,n,k]))))
 save(fig,'forward/gathers.png')
 wave=np.load(A/'forward/wavefield.npy');run=load(A/'forward/run.json');steps=run['recording']['steps']
 recorded=np.asarray(run['recording']['times'],dtype=float)
 if recorded.shape!=(len(steps),) or not np.all(np.isfinite(recorded)) or not np.allclose(recorded,np.asarray(steps)*run['experiment']['dt'],rtol=1e-12,atol=1e-12):
  raise ValueError('recording times do not agree with recorded steps and dt')
 seconds=recorded*run['experiment']['time_unit_s']
 selected_times=[float(seconds[steps.index(step)]) for step in [108,144]]
 elapsed=selected_times[1]-selected_times[0]
 assert abs(elapsed-.72)<1e-12
 expected_displacement=float(c[8,10])*elapsed
 fig,ax=plt.subplots(figsize=(8,4));dist=np.arange(24)*dx*np.sqrt(2);positions=[]
 for step in [108,144]:
  values=wave[steps.index(step),8+np.arange(24),10+np.arange(24)];k=values.argmax();positions.append(float(dist[k]));ax.plot(dist,values,label=f'{seconds[steps.index(step)]:.2f} s');ax.scatter(dist[k],values[k]);ax.annotate(f'{dist[k]:.3f} km',(dist[k],values[k]),xytext=(8,8),textcoords='offset points')
 shift=positions[1]-positions[0];ax.set(xlabel='Distance from shot (km)',ylabel='Pressure (a.u.)',title=f'Wave displacement: {shift:.3f} km; expected {expected_displacement:.3f} km');ax.legend();save(fig,'forward/wave-speed.png')
 assert abs(shift-expected_displacement)<=dx*np.sqrt(2)
 image=np.load(A/'adjoint/image.npy');window=image[10:34,7:34];profile=np.linalg.norm(window,axis=1);peak=10+int(profile.argmax());ext=[.7,3.3,3.3,1.]
 fig,axs=plt.subplots(1,3,figsize=(13,4.8));panel(axs[0],m[10:34,7:34],ext,'Known reflector','Velocity perturbation (km/s)');panel(axs[1],window,ext,'Raw signed RTM image','Image (a.u.)');axs[2].plot(profile,np.arange(10,34)*dx);axs[2].invert_yaxis();axs[2].set(xlabel='Row L2 norm (a.u.)',ylabel='Depth (km)',title=f'Image peak at {peak*dx:.1f} km')
 for ax in axs:ax.axhline(2.1,color='gray',ls='--',lw=1)
 axs[2].scatter(profile.max(),peak*dx,c='red');save(fig,'adjoint/image.png')
 born=np.load(A/'born/born_data.npy');left=float((born*born).sum());right=float((m*image).sum())
 true_depth=float(np.argmax(np.linalg.norm(m[10:34,7:34],axis=1))+10)*dx
 assert abs(peak*dx-true_depth)<=dx+1e-12
 metrics={'trace_norm':float(np.linalg.norm(traces)),'trace_peaks':peaks,'wave_peak_distances_km':positions,'frame_times_s':selected_times,'elapsed_s':elapsed,'expected_displacement_km':expected_displacement,'displacement_km':shift,'born_norm':float(np.linalg.norm(born)),'dot_left':left,'dot_right':right,'dot_relative_error':abs(left-right)/max(abs(left),abs(right)),'image_peak_depth_km':peak*dx,'true_reflector_depth_km':true_depth,'depth_difference_km':abs(peak*dx-true_depth)}
 (A/'reflector-checks.json').write_text(json.dumps(metrics,indent=2)+'\n');print(metrics)

def audit(actions,n,budget):
 counts=dict(missing_reverse_steps=0,duplicated_reverse_steps=0,reverse_steps_out_of_order=0,invalid_restores=0,budget_overruns=0)
 saved={0};work=None;grads=[];calls=0;peak=1
 for event in actions:
  a,s=event['action'],event['step']
  if a=='restore':
   counts['invalid_restores']+=int(s not in saved);work=s
  elif a=='call':assert work==s;work=s+1;calls+=1
  elif a=='store':assert work==s and s not in saved;saved.add(s);peak=max(peak,len(saved))
  elif a=='grad':assert s in saved;grads.append(s)
  elif a=='fetch':assert s!=0 and s in saved;saved.remove(s)
  else:raise AssertionError(a)
  counts['budget_overruns']+=int(len(saved)>budget+1)
  assert event['saved_states']==len(saved)
 counts['missing_reverse_steps']=len(set(range(n))-set(grads));counts['duplicated_reverse_steps']=len(grads)-len(set(grads));counts['reverse_steps_out_of_order']=sum(a!=b for a,b in zip(grads,range(n-1,-1,-1)))
 assert saved=={0} and all(v==0 for v in counts.values())
 return {**counts,'forward_calls':calls,'peak_saved_states':peak}

def checkpoints():
 full=np.load(A/'adjoint/image.npy');full_result=load(A/'adjoint/result.json');full_stats=full_result['statistics']
 full_work=full_stats['scheduler_forward_calls']/len(full_result['shots']);full_bytes=full_stats['peak_saved_bytes']
 reports={};budgets=[1,3,5,10];works=[];storage=[]
 for b in budgets:
  folder=A/f'checkpoint-{b}';result=load(folder/'result.json');audits=[audit(load(folder/p['actions_file']),result['steps'],b) for p in result['statistics']['per_shot']]
  error=float(np.linalg.norm(np.load(folder/'image.npy')-full)/np.linalg.norm(full));assert error<1e-9
  reports[str(b)]={'relative_l2_error':error,'peak_saved_states':result['statistics']['peak_saved_states'],'shots':audits}
  print(json.dumps({'checkpoint_budget':b,**reports[str(b)]},sort_keys=True))
  works.append(result['statistics']['scheduler_forward_calls']/len(result['shots']));storage.append(result['statistics']['peak_saved_bytes'])
 fig,axs=plt.subplots(1,2,figsize=(11,4.5));axs[0].semilogy(budgets,works,'o-',label='Treeverse');axs[0].axhline(full_work,c='gray',ls='--',label='Full history');axs[0].set(ylabel='Forward steps per shot',title='Recomputation cost');axs[0].legend()
 axs[1].plot(budgets,storage,'o-');axs[1].set(ylabel='Peak saved-state bytes',title='State storage (excluding work buffers)');axs[1].text(.03,.93,f'Full history: {full_bytes:,} bytes',transform=axs[1].transAxes,fontsize=9)
 for ax in axs:ax.set_xlabel('Additional checkpoint slots');ax.grid(alpha=.2)
 save(fig,'checkpoint-work.png')
 actions=load(A/'checkpoint-5/actions-0.json');fig,ax=plt.subplots(figsize=(12,4.5))
 for kind in ['store','restore','call','grad','fetch']:
  pairs=[(i,x['step']) for i,x in enumerate(actions) if x['action']==kind];ax.scatter(*np.array(pairs).T,s=6,label=kind)
 ax.set(xlabel='Operation index',ylabel='Time step',title='Treeverse schedule: first shot, five additional slots');ax.legend(ncol=5);save(fig,'checkpoint-actions.png')
 print(json.dumps({'full_history_forward_steps_per_shot':full_work,'full_history_peak_saved_bytes':full_bytes}))
 (A/'checkpoint-audit.json').write_text(json.dumps(reports,indent=2)+'\n');print('Checkpoint audit passed for every shot and budget')

def marmousi():
 e=load(ROOT/'inputs/marmousi.json');m=np.array(e['perturbation']);im=np.load(A/'marmousi-image/image.npy');g=np.load(A/'marmousi-born/born_data.npy')[4];extent=[0,20.1,6.7,0]
 fig,axs=plt.subplots(3,1,figsize=(13,12));panel(axs[0],m,extent,'Marmousi short-wavelength perturbation','Velocity perturbation (km/s)')
 lim=abs(g).max();p=axs[1].imshow(g,extent=[1,19,3.6,.003],aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);axs[1].set(title='Born shot gather: source x=10 km',xlabel='Receiver position (km)',ylabel='Time (s)');plt.colorbar(p,ax=axs[1],label='Scattered pressure (a.u.)',shrink=.85)
 panel(axs[2],im,extent,'Checkpointed RTM: raw image, no depth gain','Image (a.u.)')
 from matplotlib.patches import Rectangle
 for ax in [axs[0],axs[2]]:ax.add_patch(Rectangle((8,0),6,3,fill=False,ec='black',ls='--',lw=1))
 save(fig,'marmousi.png')
 result=load(A/'marmousi-image/result.json');audits=[audit(load(A/'marmousi-image'/p['actions_file']),1200,5) for p in result['statistics']['per_shot']]
 val=float(np.linalg.norm(im));relative=abs(val/6.7037741e-4-1);assert relative<1e-4
 report={'image_l2_norm':val,'relative_norm_error':relative,'peak_saved_states':result['statistics']['peak_saved_states'],'peak_saved_bytes':result['statistics']['peak_saved_bytes'],'full_history_bytes_per_shot':1201*805*269*16,'shots':audits}
 (A/'marmousi-checks.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='shots'})

if __name__=='__main__':
 small();checkpoints();marmousi()
