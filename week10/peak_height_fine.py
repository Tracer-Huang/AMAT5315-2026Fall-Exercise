"""Refine h to 0.0003125, centered on the independently located h=0.000625 peaks."""
import csv,json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from peak_height import CENTERS, SIZES, specific_heat, fit_peak, fit_log

ROOT=Path('runs/peak-height-r2')
OUT=Path('artifacts/peak-height-r2')

def fine_grid(k_peak, coarse_center_micro):
    origin=coarse_center_micro*1000
    center=origin+round((k_peak*1e9-origin)/625000)*625000
    return [center+i*312500 for i in range(-4,5)]

def read_point(l,chi,nano,old):
    prior=old.get((l,chi,nano))
    if prior is not None:
        d=json.loads(Path(prior).read_text())
        return {'L':l,'chi':chi,'K':nano/1e9,'g':d['g'],'seconds':d['seconds'],'source':prior}
    path=ROOT/f'trg-L{l}-chi{chi}-K{nano:09d}'/'evidence.json'
    if not path.exists():
        ROOT.mkdir(parents=True,exist_ok=True)
        subprocess.run(['target/release/thermo','--method','trg','--l',str(l),'--k',f'{nano/1e9:.9f}','--chi',str(chi),'--out',str(path.parent)],check=True,stdout=subprocess.DEVNULL,env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1'})
    d=json.loads(path.read_text())
    if d['method']!='trg' or d['L']!=l or d['chi']!=chi or abs(d['K']-nano/1e9)>1e-12:raise ValueError(f'Incorrect run metadata: {path}')
    return {'L':l,'chi':chi,'K':nano/1e9,'g':d['g'],'seconds':d['seconds'],'source':str(path)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with open('artifacts/peak-height-r1/g-values.csv',newline='') as f:
        old={(int(r['L']),int(r['chi']),round(float(r['K'])*1e9)):r['source'] for r in csv.DictReader(f)}
    with open('artifacts/peak-height-r1/heights.csv',newline='') as f:
        peaks={(int(r['L']),int(r['chi'])):float(r['K_peak']) for r in csv.DictReader(f) if float(r['h'])==.000625}
    grids={(l,chi):fine_grid(peaks[l,chi],CENTERS[l]) for l in SIZES for chi in (16,32)}
    tasks=[(l,chi,n) for (l,chi),g in grids.items() for n in g]
    with ThreadPoolExecutor(max_workers=8) as ex:obs=list(ex.map(lambda t:read_point(*t,old),tasks))
    with (OUT/'g-values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','K','g','seconds','source']);w.writeheader();w.writerows(obs)
    heights=[]
    for l in SIZES:
        for chi in (16,32):
            group=sorted((r for r in obs if r['L']==l and r['chi']==chi),key=lambda r:r['K'])
            kp,cp=fit_peak(*specific_heat([r['K'] for r in group],[r['g'] for r in group]))
            heights.append({'L':l,'chi':chi,'h':.0003125,'K_peak':kp,'T_peak':1/kp,'c_peak':cp})
    with (OUT/'heights.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','h','K_peak','T_peak','c_peak']);w.writeheader();w.writerows(heights)
    fits=[]
    for chi in (16,32):
        y=np.array([r['c_peak'] for r in heights if r['chi']==chi]);s,b=fit_log(SIZES,y);s2,b2=fit_log(SIZES[1:],y[1:]);fits.append({'chi':chi,'h':.0003125,'slope_all':s,'intercept_all':b,'max_abs_residual_all':float(np.max(abs(b+s*np.log(SIZES)-y))),'slope_largest_two':s2,'intercept_largest_two':b2})
    (OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    print(json.dumps({'heights':heights,'fits':fits},indent=2))
if __name__=='__main__':main()
