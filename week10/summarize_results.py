import csv,json
from pathlib import Path
p=json.loads(Path('artifacts/peak-results.json').read_text())
with open('artifacts/g-timings.csv',newline='') as f: rows=list(csv.DictReader(f))
with open('artifacts/peak-summary.csv','w',newline='') as f:
 fields=['method','L','T_peak_chi16','T_peak_chi32','T_peak_chi32_wide_step','seconds_all_recorded'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for method,sizes in [('boundary',[16,32,64]),('trg',[64,256,1024])]:
  for l in sizes:
   a=p[method][str(l)];unique={(round(float(r['K'])*1_000_000),r['chi']):float(r['seconds']) for r in rows if r['method']==method and int(r['L'])==l};sec=sum(unique.values())
   w.writerow(dict(method=method,L=l,T_peak_chi16=a['16']['T_peak'],T_peak_chi32=a['32']['T_peak'],T_peak_chi32_wide_step=a['wide']['T_peak'],seconds_all_recorded=sec))
