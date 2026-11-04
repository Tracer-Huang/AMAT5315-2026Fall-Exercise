import json, math
from pathlib import Path
root=Path('promedus'); tokens=(root/'Promedus_14.uai.MAR').read_text().split(); it=iter(tokens); assert next(it)=='MAR'; n=int(next(it)); mar=[[float(next(it)) for _ in range(int(next(it)))] for _ in range(n)]
pr=float((root/'Promedus_14.uai.PR').read_text().split()[1]); a=json.load(open('artifacts/promedus-greedy/result.json')); b=json.load(open('artifacts/promedus-treesa/result.json')); bp=json.load(open('artifacts/promedus-bp/run.json')) if Path('artifacts/promedus-bp/run.json').exists() else json.load(__import__('gzip').open('artifacts/promedus-bp/run.json.gz','rt'))
assert abs(a['log10_z']-b['log10_z'])<1e-10 and abs(a['log10_z']-pr)<1e-4
for x,y in zip(a['queries'][1:],b['queries'][1:]):
 i=x['query'][0]; assert abs(x['log10_z']-a['log10_z'])<1e-10; assert max(abs(c-d) for c,d in zip(x['probabilities'],y['probabilities']))<1e-10; assert max(abs(c-d) for c,d in zip(x['probabilities'],mar[i]))<1e-5
selected=max(abs(bp['marginals'][i][j]-mar[i][j]) for i in (59,102,139) for j in (0,1)); assert not bp['converged'] and bp['residual']>1e-10 and selected>.38
for name,mass in [('prior',1),('AD',.004501375),('ADX',.00098822675)]:
 r=json.load(open(f'artifacts/{name}/run.json'));assert abs(r['evidence_mass']-mass)<1e-14;assert r['relative_mass_error']<1e-10 and r['max_marginal_error']<1e-10 and r['max_joint_error']<1e-10
out={'greedy_log10_z':a['log10_z'],'treesa_log10_z':b['log10_z'],'published_pr':pr,'selected_query_state1':{str(x['query'][0]):x['probabilities'][1] for x in a['queries'][1:]},'bp_residual':bp['residual'],'bp_selected_max_error_vs_published_mar':selected,'bp_sweeps':bp['sweeps'],'bp_converged':bp['converged'],'bp_cumulative_work':bp['history'][-1]['cumulative_update_work']}
Path('artifacts/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
