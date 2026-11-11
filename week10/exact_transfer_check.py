import numpy as np,math,json
from pathlib import Path
l=16
for k in [.45,.47,.48,.485]:
 states=np.arange(2**l,dtype=np.uint32);s=1-2*((states[:,None]>>np.arange(l))&1).astype(float);intra=np.exp(k*(s[:,:-1]*s[:,1:]).sum(axis=1));v=intra.copy();F=np.array([[np.exp(k),np.exp(-k)],[np.exp(-k),np.exp(k)]]);log=0.
 for _ in range(l-1):
  v=v.reshape((2,)*l)
  for axis in range(l):v=np.moveaxis(np.tensordot(F,v,axes=(1,axis)),0,axis)
  v=v.reshape(-1)*intra;scale=v.max();v/=scale;log+=math.log(scale)
 g=(math.log(v.sum())+log)/l**2
 p=Path(f'runs/study-v2/boundary-L16-chi16-K{round(k*1e6):06d}/evidence.json')
 actual=json.loads(p.read_text())['g'] if p.exists() else None
 print(k,g,actual)
 if actual is not None: assert abs(g-actual)<1e-9
