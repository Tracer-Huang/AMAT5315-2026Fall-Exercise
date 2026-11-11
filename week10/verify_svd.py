import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path('artifacts/svd'); n=512
A=np.fromfile(p/'matrix.f64',dtype='<f8').reshape((n,n),order='F')
s=np.fromfile(p/'singular.f64',dtype='<f8')
pred=np.sqrt(np.r_[np.cumsum((s[::-1]**2))[::-1],0]/np.sum(s*s))
measured={}
for k in (16,64,128,256,512):
 b=np.fromfile(p/f'rank{k}.f64',dtype='<f8').reshape((n,n),order='F')
 measured[k]=float(np.linalg.norm(A-b)/np.linalg.norm(A))
 assert abs(measured[k]-pred[k]) < 1e-8
assert measured[512]<1e-8
cross=int(np.flatnonzero(pred<=.1)[0])
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].semilogy(np.arange(1,n+1),s);ax[0].set(xlabel='Rank',ylabel='Singular value')
ax[1].plot(np.arange(n+1),pred,label='Discarded-spectrum prediction');ax[1].scatter(list(measured),list(measured.values()),color='C1',label='Measured reconstruction');ax[1].axvline(256,color='gray',ls='--',label='Storage equality');ax[1].axhline(.1,color='red',ls=':',label='10% error');ax[1].set(xlabel='Retained rank',ylabel='Relative Frobenius error');ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig('artifacts/svd.png',dpi=180)
result={'measured':measured,'first_rank_under_10_percent':cross,'storage_ratio_at_crossing':2*cross/n,'generator':'ChaCha8Rng, rand_chacha 0.9, rand_distr StandardNormal, seed 2026'}
(p/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
