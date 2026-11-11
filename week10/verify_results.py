import json,math
from pathlib import Path
from audit_evidence import check
result=json.loads(Path('artifacts/peak-results.json').read_text());tc=2/math.log(1+math.sqrt(2));report={};bad=[]
for method,sizes in [('boundary',[16,32,64]),('trg',[64,256,1024])]:
 d=result[method];r={'reference_Tc':tc,'Tc_all':d['Tc_all'],'Tc_largest_two':d['Tc_largest_two'],'relative_Tc_error':abs(d['Tc_all']-tc)/tc,'sizes':{}}
 if r['relative_Tc_error']>=.01 or abs(d['Tc_largest_two']-tc)/tc>=.01:bad.append(method+' Tc error')
 for l in sizes:
  a=d[str(l)];t=a['32']['T_peak'];rank=abs(a['16']['T_peak']-t)/t;step=abs(a['wide']['T_peak']-t)/t;r['sizes'][l]={'T_peak_chi32':t,'rank_relative_difference':rank,'step_relative_difference':step};
  if rank>=.01 or step>=.01:bad.append(f'{method} L={l} unstable rank or step')
 report[method]=r
 # Audit all nine chi=32 stencil runs for the largest L; center inferred from exact data table.
 files=sorted(Path('runs/study-v2').glob(f'{method}-L{sizes[-1]}-chi32-*/evidence.json'))
 assert len(files)==9,(method,len(files));r['peak_run_evidence']=[check(str(f)) for f in files]
Path('artifacts/acceptance.json').write_text(json.dumps({'checks':report,'failures':bad},indent=2)+'\n');print(json.dumps({'checks':report,'failures':bad},indent=2));assert not bad,bad
