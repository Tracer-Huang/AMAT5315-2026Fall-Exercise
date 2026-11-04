import json, time, pathlib
import numpy as np
from infer import read_uai, enumerate_asia, contract_asia, bp
out=pathlib.Path('baseline'); out.mkdir(exist_ok=True)
m=read_uai('asia.uai')
cases={'prior':{},'AD':{0:0,7:0},'ADX':{0:0,7:0,6:0}}
for name,ev in cases.items():
    z,mar,joint=enumerate_asia(m,ev); z2,mar2,joint2=contract_asia(m,ev)
    b,h=bp(m,ev)
    for step in h: step['marginal_error']=float(max(np.max(np.abs(np.array(x)-y)) for x,y in zip(step['beliefs'],mar)))
    result={'evidence':ev,'evidence_mass':z,'contraction_mass':z2,'marginals':[x.tolist() for x in mar], 'joint_T_L':joint.tolist(),'max_marginal_error':float(max(np.max(np.abs(x-y)) for x,y in zip(mar,mar2))), 'max_joint_error':float(np.max(np.abs(joint-joint2))),'bp_beliefs':[x.tolist() for x in b], 'bp_trace':h}
    (out/(name+'.json')).write_text(json.dumps(result,indent=2))
    print(name,'mass',z,'T L B',[mar[i][0] for i in [1,3,4]],'BP residual',h[-1]['residual'],'BP error',h[-1]['marginal_error'])
a=np.array([[1.,2.],[3.,4.]]); b=np.array([[2.,0.],[1.,2.]])
spins=np.array([-1.,1.]); bond=np.exp(.5*np.outer(spins,spins)); z=np.einsum('ab,bc,ca->',bond,bond,bond)
ze=sum(np.exp(.5*(x*y+y*w+w*x)) for x in spins for y in spins for w in spins)
print('AB', a@b, 'trace',np.trace(a@b),'Ising',z,'enum',ze,'error',abs(z-ze))
