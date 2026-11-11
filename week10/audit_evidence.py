import json,math
import numpy as np
from pathlib import Path

def check(path):
 d=json.loads(Path(path).read_text());l=d['L'];k=d['K']
 if d['method']=='boundary':
  v=np.array([1.]);b=1/math.sqrt(math.cosh(k))
  for t in d['sites']:
   a=np.array(t['data']).reshape(t['left'],2,t['right']);v=v @ (a[:,0,:]*b)
  reconstructed=(sum(d['log_scales'])+math.log(float(v[0])))/(l*l)
 else:
  count=l*l;logz=0.
  for s in d['steps']:
   logz+=count*s['log_scale'];count//=2
  q=d['steps'][-1]['new_dim'];a=np.array(d['final_tensor']).reshape(q,q,q,q);z=np.einsum('ijij->',a);reconstructed=(logz+math.log(z))/(l*l)
 err=abs(reconstructed-d['g']);assert err<1e-12,(path,err);return {'path':path,'g':d['g'],'reconstructed_g':reconstructed,'absolute_error':err}
if __name__=='__main__':
 out=[check('runs/boundary64-v2/evidence.json'),check('runs/trg1024-v2/evidence.json')];Path('artifacts/evidence-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
