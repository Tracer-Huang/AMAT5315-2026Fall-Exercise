"""Finite requirement audit: checks measured outputs, separates domains."""
import json,hashlib
from pathlib import Path
import numpy as np
r=json.loads(Path('evidence/toy-results.json').read_text());im=json.loads(Path('evidence/image-results.json').read_text());vd=json.loads(Path('evidence/velocity-domain-diagnostics.json').read_text())
checks={}
def record(k,ok,detail):checks[k]=dict(passed=bool(ok),detail=detail)
record('P1.forward',np.max(np.abs(r['forward']['mean']))<.1 and np.max(np.abs(np.array(r['forward']['variance'])-1))<.1,r['forward'])
m=r['ddpm'];record('P1.ddpm_distribution',m['fraction_error']<.04 and m['centre_error']<.05 and m['width_error']<.03,m)
for kind in ['noise','flow']:
 z=np.load(f'artifacts/{kind}-training.npz');record(f'P1_3.{kind}_stages',len(z['stages'])==10 and z['stages'][0]==0 and z['stages'][-1]==len(z['loss']) and np.isfinite(z['loss']).all(),dict(stages=z['stages'].tolist(),finite_loss=bool(np.isfinite(z['loss']).all())))
record('P2.deterministic',r['determinism']['ddim_max_difference']==0,r['determinism'])
record('P3.velocity_supported_grid',all(v['supported_grid_relative_rms']<.1 for v in vd),vd)
record('P3.velocity_full_rectangle',all(v<.1 for v in r['velocity_relative_rms']),dict(relative_rms=r['velocity_relative_rms'],domain='21x21 [-4,4]^2 including negligible-density regions',status='original declared full-grid numerical acceptance test'))
f=next(v for v in r['comparison'] if v['method']=='flow' and v['steps']==20)
record('P3.flow20',f['energy']<.005 and f['width_error']<.03,f)
record('P3.sweep',len(r['comparison'])==23 and r['energy_null']['pairs']==20 and r['energy_null']['n']==4000,dict(rows=len(r['comparison']),null_q95=r['energy_null']['q95']))
photo=np.load('artifacts/photos.npz');record('P4.four_images',photo['samples'].shape==(4,3,32,32) and np.isfinite(photo['samples']).all(),dict(shape=list(photo['samples'].shape),dtype=str(photo['samples'].dtype),visual_review='dog/bird-like objects and scenery; low resolution limits semantic identification'))
record('P5.preservation',im['total_completions']==16 and im['observed_max_change']<1e-5 and im['outside_variance']==0,im)
actual=hashlib.sha256(Path('inputs/cifar-model/diffusion_pytorch_model.safetensors').read_bytes()).hexdigest()
record('P4_5.model_identity',actual=='1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f',dict(sha256=actual,source='pinned official LFS oid'))
expected=['forward.gif','training.gif','reverse.gif','forward.png','loss.png','ddpm.png','determinism.png','trajectories.png','velocity.png','velocity-supported.png','comparison.png','exact-vs-learned.png','photographs.png','real-vs-generated.png','inpainting.png']
record('evidence_files',all((Path('evidence')/f).exists() for f in expected),expected)
Path('evidence/audit.json').write_text(json.dumps(checks,indent=2))
for k,v in checks.items():print(('PASS' if v['passed'] else 'LIMITATION'),k)
summary=dict(all_outputs_present=checks['evidence_files']['passed'],all_recorded_numerical_targets_met=all(v['passed'] for v in checks.values()),failed_checks=[k for k,v in checks.items() if not v['passed']],status='All recorded numerical targets passed' if all(v['passed'] for v in checks.values()) else 'Not ready: at least one numerical target failed')
Path('evidence/audit-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
# A diagnostic report is not a successful submission gate when any target fails.
raise SystemExit(0 if summary['all_recorded_numerical_targets_met'] else 1)
