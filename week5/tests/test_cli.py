import subprocess,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'target/release/seismic'
def run(*a):return subprocess.run([str(BIN),*map(str,a)],capture_output=True,text=True)
def test_missing_required_options_rejected():
 assert run().returncode != 0

def test_forward_reference_and_refuse_overwrite(tmp_path):
 dest=tmp_path/'forward'
 r=run('--experiment',ROOT/'inputs/reflector.json','--mode','forward','--every','3','--out',dest)
 assert r.returncode==0,r.stderr
 assert (dest/'traces.npy').exists()
 data=np.load(dest/'traces.npy')
 assert data.shape==(3,240,14)
 assert abs(np.linalg.norm(data)/11.574770-1)<1e-4
 again=run('--experiment',ROOT/'inputs/reflector.json','--mode','forward','--out',dest)
 assert again.returncode != 0
