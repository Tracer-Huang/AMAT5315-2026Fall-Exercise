import os,subprocess,json,csv,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('runs/study-v2'); root.mkdir(parents=True,exist_ok=True)
rows=[]
executor=ThreadPoolExecutor(max_workers=8)
def sweep(method,l,ks,chi):
 return list(executor.map(lambda k: run(method,l,float(k),chi),ks))
def run(method,l,k,chi):
 key=f'{method}-L{l}-chi{chi}-K{round(k*1000000):06d}'; out=root/key; f=out/'evidence.json'
 if f.exists():
  data=json.loads(f.read_text());g=data['g'];sec=data['seconds']
 else:
  proc=subprocess.run(['target/release/thermo','--method',method,'--l',str(l),'--k',f'{k:.8f}','--chi',str(chi),'--out',str(out)],capture_output=True,text=True,check=True)
  g,sec=map(float,proc.stdout.split())
 rows.append(dict(method=method,L=l,K=k,chi=chi,g=g,seconds=sec));return g

def curvature(ks,gs):
 ks=np.array(ks);gs=np.array(gs); h=ks[1]-ks[0];return ks[1:-1],ks[1:-1]**2*(gs[2:]-2*gs[1:-1]+gs[:-2])/h**2

def peak(ks,gs):
 x,y=curvature(ks,gs);j=int(np.argmax(y));assert 0<j<len(x)-1, (x,y);a,b,c=np.polyfit(x[j-1:j+2],y[j-1:j+2],2);assert a<0;return float(-b/(2*a)),(x,y)
results={}
for method,sizes in [('boundary',[16,32,64]),('trg',[64,256,1024])]:
 results[method]={};fig,ax=plt.subplots(1,2,figsize=(10,4))
 for l in sizes:
  print(method,l,'coarse',flush=True)
  ks=np.linspace(.4,.5,41);gs=sweep(method,l,ks,16);kp,(x,y)=peak(ks,gs);ax[0].plot(1/x,y,label=f'L={l} coarse χ=16')
  center=round(kp/.0025)*.0025;fine=center+np.arange(-4,5)*.00125;out={}
  for chi in [16,32]:
   print(method,l,'refined',chi,flush=True)
   vals=sweep(method,l,fine,chi);kpf,(xf,yf)=peak(fine,vals);out[str(chi)]={'k_peak':kpf,'T_peak':1/kpf};
   if chi==32:
    kpwide,_=peak(fine[::2],vals[::2]);out['wide']={'k_peak':kpwide,'T_peak':1/kpwide};ax[0].scatter(1/xf,yf,s=9)
  results[method][str(l)]=out
 t=np.array([results[method][str(l)]['32']['T_peak'] for l in sizes]);inv=1/np.array(sizes);fit=np.polyfit(inv,t,1);fit2=np.polyfit(inv[1:],t[1:],1);results[method]['Tc_all']=float(fit[1]);results[method]['Tc_largest_two']=float(fit2[1]);ax[1].scatter(inv,t);x=np.linspace(0,max(inv)*1.05,100);ax[1].plot(x,np.polyval(fit,x),label='all sizes');ax[1].plot(x,np.polyval(fit2,x),label='largest two');ax[1].axhline(2/np.log(1+np.sqrt(2)),ls='--',color='gray',label='Onsager');ax[0].set(xlabel='Temperature T',ylabel='Specific heat per spin');ax[0].legend(fontsize=7);ax[1].set(xlabel='1/L',ylabel='Peak temperature');ax[1].legend(fontsize=7);fig.tight_layout();fig.savefig(f'artifacts/{method if method=="boundary" else "trg"}-critical.png',dpi=180);plt.close(fig)
 Path('artifacts/peak-results.json').write_text(json.dumps(results,indent=2)+'\n')
with open('artifacts/g-timings.csv','w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['method','L','K','chi','g','seconds']);writer.writeheader();writer.writerows(rows)
print(json.dumps(results,indent=2),flush=True)
