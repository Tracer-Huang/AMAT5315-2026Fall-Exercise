import argparse,json,math
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
 parser=argparse.ArgumentParser(description='Recompute g from the saved sample tensors and scales')
 parser.add_argument('--boundary', default='runs/my-boundary64/evidence.json')
 parser.add_argument('--trg', default='runs/my-trg1024/evidence.json')
 args=parser.parse_args()
 out=[check(args.boundary),check(args.trg)];Path('artifacts/evidence-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
