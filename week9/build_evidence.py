import json, math
from pathlib import Path
from PIL import Image,ImageDraw
E=Path('evidence');E.mkdir(exist_ok=True)
# Factor graph: each variable is a hyperedge; T remains open, A and D are selected.
pos={'A':(90,110),'T':(280,110),'S':(560,110),'L':(480,250),'B':(740,250),'E':(300,390),'X':(130,510),'D':(690,510)}
fac=[('p(A)',(90,30),['A']),('p(T|A)',(185,110),['A','T']),('p(S)',(560,30),['S']),('p(L|S)',(530,185),['L','S']),('p(B|S)',(690,185),['B','S']),('p(E|L,T)',(310,245),['E','L','T']),('p(X|E)',(200,450),['X','E']),('p(D|E,B)',(520,420),['D','E','B'])]
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="620" viewBox="0 0 860 620"><rect width="100%" height="100%" fill="white"/><text x="20" y="595" font-family="Arial" font-size="16">AD query: A=0, D=0; T open; other indices summed</text>']
for name,(x,y),axes in fac:
 for v in axes:
  a,b=pos[v];svg.append(f'<line x1="{x}" y1="{y}" x2="{a}" y2="{b}" stroke="#6b7280" stroke-width="2"/>')
for v,(x,y) in pos.items():
 color='#f3b469' if v in ('A','D') else ('#9fd9bd' if v=='T' else '#dce9f8');svg.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{color}" stroke="#222"/><text x="{x}" y="{y+5}" text-anchor="middle" font-family="Arial">{v}</text>')
for name,(x,y),axes in fac:
 svg.append(f'<rect x="{x-42}" y="{y-13}" width="84" height="26" fill="#fff" stroke="#222"/><text x="{x}" y="{y+5}" text-anchor="middle" font-family="Arial" font-size="13">{name}</text>')
# Evidence selectors, X summation, and the open T output leg are explicit.
for x1,y1,x2,y2,label in [(90,110,20,110,'[1,0]'),(690,510,800,510,'[1,0]'),(130,510,30,510,'[1,1]'),(280,110,280,40,'T open')]:
 svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="2"/><text x="{x2}" y="{y2-8}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
svg.append('</svg>');(E/'graphs.svg').write_text(''.join(svg))
# Two saved plans plotted directly from their trees.
im=Image.new('RGB',(1600,650),'white');d=ImageDraw.Draw(im)
for j,name in enumerate(['random','greedy']):
 tree=json.load(open(f'runs/{name}/order.json'))['tree']; cost=json.load(open(f'runs/{name}/complexity.json')); leaves=[]
 def layout(t,depth=0):
  if 'Leaf' in t:
   x=45+j*800+len(leaves)*90; leaves.append(x);y=560;d.text((x-10,y+12),str(t['Leaf']['tensor_index']),fill='black');return x,y
  children=[layout(c,depth+1) for c in t['Node']['args']]; x=sum(c[0] for c in children)/len(children);y=min(c[1] for c in children)-55
  for cx,cy in children:d.line((x,y,cx,cy),fill='#466',width=2)
  d.ellipse((x-4,y-4,x+4,y+4),fill='#c75336'); return x,y
 layout(tree);d.text((40+j*800,25),f'{name}: work {cost["work"]}, largest {cost["largest_tensor"]}',fill='black')
 d.text((40+j*800,50),'Leaves: 0..7 are the eight conditioned UAI factors',fill='black')
im.save(E/'orders.png')
# Benchmark traces, log scale; selected MAR error reconstructed at each state.
bp=json.load(open('artifacts/promedus-bp/run.json')) if Path('artifacts/promedus-bp/run.json').exists() else json.load(__import__('gzip').open('artifacts/promedus-bp/run.json.gz','rt')); tok=iter(Path('promedus/Promedus_14.uai.MAR').read_text().split());next(tok);n=int(next(tok)); mar=[[float(next(tok)) for _ in range(int(next(tok)))] for _ in range(n)]
im=Image.new('RGB',(1000,560),'white');d=ImageDraw.Draw(im);d.text((20,15),'Promedus BP: residual (blue), selected MAR error (orange)',fill='black');d.line((80,60,80,480,950,480),fill='black',width=2)
for color,key in [('#2466a0','res'),('#d46d2b','err')]:
 pts=[]
 for step in bp['history']:
  v=step['residual'] if key=='res' else max(abs(step['beliefs'][i][j]-mar[i][j]) for i in (59,102,139) for j in (0,1));x=80+int(step['sweep']*870/1000);y=480-int((max(-11,min(0,math.log10(max(v,1e-12))))+11)*420/11);pts.append((x,y))
 d.line(pts,fill=color,width=2)
for i in range(0,12,2):d.text((18,480-int(i*420/11)),str(i-11),fill='black')
d.text((460,510),'sweeps 0 to 1000; y = log10(value)',fill='black');im.save(E/'promedus-bp.png')
# Cost comparison for each same query. Counts obtained by independent tree traversal.
a=json.load(open('artifacts/promedus-greedy/independent-costs.json'));b=json.load(open('artifacts/promedus-treesa/independent-costs.json'))
im=Image.new('RGB',(1000,560),'white');d=ImageDraw.Draw(im);d.text((20,15),'Promedus: independently counted contraction work (log10 scale)',fill='black');d.line((80,60,80,470,950,470),fill='black',width=2)
for i,(x,y) in enumerate(zip(a,b)):
 for j,(v,c) in enumerate([(x['work'],'#d46d2b'),(y['work'],'#2466a0')]):
  px=130+200*i+60*j;top=470-int(math.log10(v)/11*400);d.rectangle((px,top,px+48,470),fill=c);d.text((px,top-18),f'{v:.2g}',fill='black')
 d.text((130+200*i,485),('Z' if not x['query'] else str(x['query'][0])),fill='black')
d.text((550,35),'orange: greedy  blue: TreeSA',fill='black');im.save(E/'promedus-costs.png')
