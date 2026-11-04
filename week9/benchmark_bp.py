import json,sys,time
from pathlib import Path
import numpy as np
from infer import read_uai,bp
root=Path('promedus')
tokens=list(map(int,(root/'Promedus_14.uai.evid').read_text().split())); ev=dict(zip(tokens[1::2],tokens[2::2]))
raw=(root/'Promedus_14.uai.MAR').read_text().split(); it=iter(raw); assert next(it)=='MAR'; n=int(next(it)); ref=[np.array([float(next(it)) for _ in range(int(next(it)))]) for i in range(n)]
start=time.monotonic(); beliefs,h=bp(read_uai(root/'Promedus_14.uai'),ev)
for step in h:
    step['selected_error']=float(max(np.max(np.abs(np.array(step['beliefs'][i])-ref[i])) for i in (59,102,139)))
    step['selected_beliefs']={str(i):step['beliefs'][i] for i in (59,102,139)}
    del step['beliefs']
result={'elapsed_seconds':time.monotonic()-start,'evidence':ev,'selected_reference':{i:ref[i].tolist() for i in (59,102,139)},'trace':h}
Path('baseline/promedus-bp.json').write_text(json.dumps(result))
print('elapsed',result['elapsed_seconds'],'sweeps',h[-1]['sweep'],'residual',h[-1]['residual'],'selected error',h[-1]['selected_error'],flush=True)
print('state-1', [beliefs[i][1] for i in (59,102,139)])
