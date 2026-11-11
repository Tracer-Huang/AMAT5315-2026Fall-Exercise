"""Verify peak fits and independently close saved TRG tensors at selected points."""
import csv,json
from pathlib import Path
import numpy as np
from peak_height import specific_heat,fit_peak
from audit_evidence import check

out=Path('artifacts/peak-height-r2')
selected=[]
for folder in [out,Path('artifacts/peak-height-rank24-r1')]:
 with (folder/'g-values.csv').open(newline='') as f: obs=list(csv.DictReader(f))
 with (folder/'heights.csv').open(newline='') as f: heights=list(csv.DictReader(f))
 for h in heights:
  rows=sorted([r for r in obs if r['L']==h['L'] and r['chi']==h['chi']],key=lambda r:float(r['K']))
  assert len(rows)==9
  k,c=specific_heat([float(r['K']) for r in rows],[float(r['g']) for r in rows]);kp,cp=fit_peak(k,c)
  assert abs(kp-float(h['K_peak']))<1e-12 and abs(cp-float(h['c_peak']))<1e-10
  nearest=min(rows,key=lambda r:abs(float(r['K'])-kp));ev=check(nearest['source']);ev.update({'L':int(h['L']),'chi':int(h['chi'])});selected.append(ev)
(out/'verification.json').write_text(json.dumps({'verified_peak_fits':len(selected),'independent_final_contractions':selected},indent=2)+'\n')
print(f'Verified {len(selected)} peak fits and independent final contractions.')
