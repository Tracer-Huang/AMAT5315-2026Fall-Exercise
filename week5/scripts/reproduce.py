"""Reproduce runs in a new directory; existing output is never overwritten."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
out=Path(a.artifacts).resolve()
if out.exists() and any(out.iterdir()):p.error('choose a new empty --artifacts directory; existing evidence is preserved')
subprocess.run([str(ROOT/'scripts/cargo.sh'),'build','--release','--locked'],cwd=ROOT,check=True)
exe=ROOT/'target/release/seismic';out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'WEEK5_ARTIFACTS':str(out),'MPLCONFIGDIR':os.environ.get('MPLCONFIGDIR','/tmp/week5-mpl')}
subprocess.run([sys.executable,'scripts/ad.py'],cwd=ROOT,env=env,check=True)
def run(name,experiment,mode,*flags):
 command=[str(exe),'--experiment',f'inputs/{experiment}.json','--mode',mode,*flags,'--out',str(out/name)]
 with (out/(name+'.stdout.txt')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,check=True)
run('forward','reflector','forward','--every','3')
run('born','reflector','born')
weights=str(out/'born/born_data.npy')
run('adjoint','reflector','adjoint','--data',weights,'--every','3')
for b in [1,3,5,10]:run(f'checkpoint-{b}','reflector','adjoint','--data',weights,'--storage','treeverse','--checkpoints',str(b))
run('marmousi-born','marmousi','born')
run('marmousi-image','marmousi','adjoint','--data',str(out/'marmousi-born/born_data.npy'),'--storage','treeverse','--checkpoints','5')
subprocess.run([sys.executable,'scripts/figures.py'],cwd=ROOT,env=env,check=True)
print('Numerical artifacts complete. Export forward step 144 and adjoint step 132 with vendor/viewer.html Save PNG.')
