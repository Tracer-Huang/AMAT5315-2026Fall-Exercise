"""Optional periodic Ising peak-height study; retain all individual solver evidence."""
import csv
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SIZES = (64, 256, 1024)
CENTERS = {64: 437500, 256: 440000, 1024: 440000}  # inverse temperature in millionths
ORIGINAL = Path('runs/study-v2')
NEW = Path('runs/peak-height-r1')
OUT = Path('artifacts/peak-height-r1')

def specific_heat(k, g):
    k=np.asarray(k,float); g=np.asarray(g,float)
    delta=np.diff(k)
    if len(k)<3 or not np.allclose(delta,delta[0],rtol=1e-9,atol=1e-12):
        raise ValueError('A uniform grid of at least three points is required')
    return k[1:-1], k[1:-1]**2*(g[2:]-2*g[1:-1]+g[:-2])/delta[0]**2

def fit_peak(k, c):
    k=np.asarray(k,float);c=np.asarray(c,float);i=int(np.argmax(c))
    if not 0<i<len(k)-1: raise ValueError('Peak is not bracketed by curvature points')
    # Center the fit to avoid ill conditioning near K=0.44.
    a,b,z=np.polyfit(k[i-1:i+2]-k[i],c[i-1:i+2],2)
    if a>=0: raise ValueError('Parabolic maximum is not concave')
    offset=-b/(2*a)
    if abs(offset)>k[i+1]-k[i-1]: raise ValueError('Parabolic vertex is outside the local fit')
    return float(k[i]+offset), float(z-b*b/(4*a))

def fit_log(sizes, heights):
    return tuple(float(x) for x in np.polyfit(np.log(sizes),heights,1))

def read_run(l,chi,micro):
    name=f'trg-L{l}-chi{chi}-K{micro:06d}';old=ORIGINAL/name/'evidence.json';new=NEW/name/'evidence.json'
    source=old if old.exists() else new
    if not source.exists():
        NEW.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1'}
        subprocess.run(['target/release/thermo','--method','trg','--l',str(l),'--k',f'{micro/1e6:.6f}','--chi',str(chi),'--out',str(new.parent)],check=True,stdout=subprocess.DEVNULL,env=env)
    d=json.loads(source.read_text())
    if d['method']!='trg' or d['L']!=l or d['chi']!=chi or abs(d['K']-micro/1e6)>1e-10: raise ValueError(f'Incorrect run metadata: {source}')
    return {'L':l,'chi':chi,'K':micro/1e6,'g':d['g'],'seconds':d['seconds'],'source':str(source)}

def run_study():
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for l in SIZES:
        for chi in (16,32):
            # 0.00125 base grid and 0.000625 refined grid share points.
            micro=sorted(set([CENTERS[l]+i*1250 for i in range(-4,5)]+[CENTERS[l]+i*625 for i in range(-4,5)]))
            tasks.extend((l,chi,k) for k in micro)
    with ThreadPoolExecutor(max_workers=8) as ex:
        observations=list(ex.map(lambda a:read_run(*a),tasks))
    with (OUT/'g-values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','K','g','seconds','source']);w.writeheader();w.writerows(observations)
    heights=[]
    for l in SIZES:
        for chi in (16,32):
            d={round(row['K']*1e6):row['g'] for row in observations if row['L']==l and row['chi']==chi}
            for microstep,radius in [(2500,2),(1250,4),(625,4)]:
                ks=np.array([(CENTERS[l]+i*microstep)/1e6 for i in range(-radius,radius+1)])
                gs=[d[round(k*1e6)] for k in ks];x,c=specific_heat(ks,gs);kp,cp=fit_peak(x,c)
                heights.append({'L':l,'chi':chi,'h':microstep/1e6,'K_peak':kp,'T_peak':1/kp,'c_peak':cp})
    with (OUT/'heights.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','h','K_peak','T_peak','c_peak']);w.writeheader();w.writerows(heights)
    fits=[]
    for chi in (16,32):
        for h in (.0025,.00125,.000625):
            group=[r for r in heights if r['chi']==chi and r['h']==h];y=np.array([r['c_peak'] for r in group]);slope,intercept=fit_log(SIZES,y);pred=intercept+slope*np.log(SIZES)
            slope2,intercept2=fit_log(SIZES[1:],y[1:]);fits.append({'chi':chi,'h':h,'slope_all':slope,'intercept_all':intercept,'max_abs_residual_all':float(max(abs(pred-y))),'slope_largest_two':slope2,'intercept_largest_two':intercept2})
    (OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    print(json.dumps({'heights':heights,'fits':fits},indent=2))

if __name__=='__main__': run_study()
