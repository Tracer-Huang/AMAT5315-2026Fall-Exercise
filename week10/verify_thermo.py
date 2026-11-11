import itertools,json,math,subprocess
from pathlib import Path
import numpy as np

def exact(l,k,periodic):
 edges=[]
 for y in range(l):
  for x in range(l):
   i=y*l+x
   if periodic or x+1<l: edges.append((i,y*l+(x+1)%l))
   if periodic or y+1<l: edges.append((i,((y+1)%l)*l+x))
 H=np.array([-sum(s[a]*s[b] for a,b in edges) for s in itertools.product((-1,1),repeat=l*l)],float)
 weights=np.exp(-k*H);z=weights.sum();mean=weights@H/z;cv=k*k*(weights@(H*H)/z-mean*mean)/(l*l)
 return math.log(z)/(l*l),cv
root=Path('runs/verification-v2');root.mkdir(parents=True,exist_ok=True)
def calc(method,l,k,chi):
 d=root/f'{method}-{l}-{round(k*1000000):06d}'
 if not (d/'evidence.json').exists():subprocess.run(['target/release/thermo','--method',method,'--l',str(l),'--k',str(k),'--chi',str(chi),'--out',str(d)],check=True,stdout=subprocess.DEVNULL)
 return json.loads((d/'evidence.json').read_text())['g']
checks={}
for method,l,chi,periodic in [('boundary',4,8,False),('trg',2,16,True),('trg',4,8,True)]:
 for k in ([0.,.44,.7] if method=='boundary' else [.44]):
  actual=calc(method,l,k,chi);ref,_=exact(l,k,periodic);err=abs(actual-ref);tol=2e-3 if (method,l)==('trg',4) else 1e-8;assert err<tol,(method,l,k,actual,ref);checks[f'{method}_{l}_{k}']={'g':actual,'exact_g':ref,'abs_error':err}
 if (method,l) in [('boundary',4),('trg',2)]:
  h=.0025;k=.44;cv=k*k*(calc(method,l,k+h,chi)-2*calc(method,l,k,chi)+calc(method,l,k-h,chi))/h**2;_,ref=exact(l,k,periodic);assert abs(cv-ref)<1e-3,(cv,ref);checks[f'{method}_{l}_cv']={'finite_difference':cv,'energy_variance':ref,'abs_error':abs(cv-ref)}
Path('artifacts/small-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
