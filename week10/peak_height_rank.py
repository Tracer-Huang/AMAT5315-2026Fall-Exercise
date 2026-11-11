"""Check intermediate bond dimension chi=24 on the chi=32 refined grids."""
import csv,json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from peak_height import SIZES,fit_peak,specific_heat,fit_log

ROOT=Path('runs/peak-height-rank24-r1')
OUT=Path('artifacts/peak-height-rank24-r1')

def run(l,nano):
    path=ROOT/f'trg-L{l}-chi24-K{nano:09d}'/'evidence.json'
    if not path.exists():
        ROOT.mkdir(parents=True,exist_ok=True)
        subprocess.run(['target/release/thermo','--method','trg','--l',str(l),'--k',f'{nano/1e9:.9f}','--chi','24','--out',str(path.parent)],check=True,stdout=subprocess.DEVNULL,env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1'})
    d=json.loads(path.read_text())
    if d['method']!='trg' or d['L']!=l or d['chi']!=24 or abs(d['K']-nano/1e9)>1e-12:raise ValueError(f'Incorrect run metadata: {path}')
    return {'L':l,'chi':24,'K':nano/1e9,'g':d['g'],'seconds':d['seconds'],'source':str(path)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with open('artifacts/peak-height-r2/g-values.csv',newline='') as f:
        tasks=sorted({(int(r['L']),round(float(r['K'])*1e9)) for r in csv.DictReader(f) if int(r['chi'])==32})
    with ThreadPoolExecutor(max_workers=8) as ex: obs=list(ex.map(lambda t:run(*t),tasks))
    with (OUT/'g-values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','K','g','seconds','source']);w.writeheader();w.writerows(obs)
    heights=[]
    for l in SIZES:
        rows=sorted((r for r in obs if r['L']==l),key=lambda r:r['K']);kp,cp=fit_peak(*specific_heat([r['K'] for r in rows],[r['g'] for r in rows]));heights.append({'L':l,'chi':24,'h':.0003125,'K_peak':kp,'T_peak':1/kp,'c_peak':cp})
    with (OUT/'heights.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['L','chi','h','K_peak','T_peak','c_peak']);w.writeheader();w.writerows(heights)
    y=[r['c_peak'] for r in heights];s,b=fit_log(SIZES,y);s2,b2=fit_log(SIZES[1:],y[1:]);(OUT/'fits.json').write_text(json.dumps({'chi':24,'h':.0003125,'slope_all':s,'intercept_all':b,'slope_largest_two':s2,'intercept_largest_two':b2},indent=2)+'\n');print(json.dumps(heights,indent=2))
if __name__=='__main__':main()
