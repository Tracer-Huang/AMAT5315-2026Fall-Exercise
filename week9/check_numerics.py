import itertools, json, subprocess
from pathlib import Path
import numpy as np
r=json.loads(subprocess.check_output(['target/release/infer','--checks'],text=True)); a=np.array([[1.,2.],[3.,4.]]);b=np.array([[2.,0.],[1.,2.]])
mat=a@b;rust=np.array(r['matrix_col_major']).reshape((2,2),order='F'); trace=float(np.trace(mat));z=sum(np.exp(.5*(x*y+y*w+w*x)) for x,y,w in itertools.product((-1.,1.),repeat=3))
errors={'matrix':float(np.max(abs(rust-mat))),'trace':abs(r['trace']-trace),'ising':abs(r['ising']-z)}
print(f'AB tenferro:\n{rust}\nAB NumPy:\n{mat}\nmax absolute error: {errors["matrix"]:.17g}')
print(f'trace tenferro: {r["trace"]:.17g}; NumPy: {trace:.17g}; absolute error: {errors["trace"]:.17g}')
print(f'Ising tenferro: {r["ising"]:.17g}; enumeration: {z:.17g}; absolute error: {errors["ising"]:.17g}')
assert max(errors.values())<1e-12
Path('evidence/numerical-checks.json').write_text(json.dumps({'tenferro':r,'numpy_matrix':mat.tolist(),'numpy_trace':trace,'ising_enumeration':z,'absolute_errors':errors},indent=2))
