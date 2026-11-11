"""Plot computed specific-heat peak heights and the three-size log fit."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
import matplotlib.pyplot as plt
import numpy as np

out=Path('artifacts/peak-height-r2')
with (out/'heights.csv').open(newline='') as f: rows=list(csv.DictReader(f))
with Path('artifacts/peak-height-r1/heights.csv').open(newline='') as f: rows.extend(csv.DictReader(f))
with Path('artifacts/peak-height-rank24-r1/heights.csv').open(newline='') as f: rows.extend(csv.DictReader(f))
fits=json.loads((out/'fits.json').read_text())
assert all(int(r['L']) > 0 for r in rows)
fig,ax=plt.subplots(figsize=(5.4,3.5),layout='constrained')
colors={16:'#847b6c',24:'#a46b38',32:'#245d85'}
for chi in (16,24,32):
    a=sorted((r for r in rows if int(r['chi'])==chi and abs(float(r['h'])-.0003125)<1e-10),key=lambda r:int(r['L']))
    x=np.array([int(r['L']) for r in a]); y=np.array([float(r['c_peak']) for r in a])
    ax.plot(x,y,'o-',color=colors[chi],label=fr'Fine: $\chi={chi}$, $h=0.0003125$')
coarse=sorted((r for r in rows if int(r['chi'])==32 and abs(float(r['h'])-.000625)<1e-10),key=lambda r:int(r['L']))
ax.plot([int(r['L']) for r in coarse],[float(r['c_peak']) for r in coarse],marker='s',markerfacecolor='white',linestyle='none',color=colors[32],label=r'Coarse: $\chi=32$, $h=0.000625$')
fit=next(r for r in fits if r['chi']==32 and abs(r['h']-.0003125)<1e-10)
x=np.geomspace(64,1024,150);ax.plot(x,fit['intercept_all']+fit['slope_all']*np.log(x),'--',color=colors[32],alpha=.6,label=r'Fit: $a+b\ln L$')
ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024],labels=['64','256','1024']);ax.set(xlabel='Periodic side length $L$',ylabel='Specific-heat peak per spin');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False,loc='best')
fig.savefig(out/'peak-height.pdf');fig.savefig(out/'peak-height.svg');fig.savefig(out/'peak-height.png',dpi=300)
