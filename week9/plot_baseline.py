import json, math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path('artifacts'); dest=Path('evidence');dest.mkdir(exist_ok=True)
files=[json.loads((root/n/'run.json').read_text()) for n in ['prior','AD','ADX']]
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14)
im=Image.new('RGB',(1200,680),'white');d=ImageDraw.Draw(im);d.text((40,20),'Disease probabilities: enumeration bars + tensor contraction circles',font=font,fill='black')
colors=['#3572b0','#d47736','#4a9568'];names=['Tuberculosis','Lung cancer','Bronchitis']
for j,(name,c) in enumerate(zip(names,colors)):
 d.text((110+j*365,80),name,font=font,fill=c)
 for i,r in enumerate(files):
  v=[1,3,4][j];p=r['marginals'][v][0];q=r['contraction_marginals'][v][0];x=120+j*365+i*95;y=560-int(p*440);y2=560-int(q*440)
  d.rectangle((x,y,x+50,560),fill=c);d.ellipse((x+19,y2-6,x+31,y2+6),fill='white',outline='black',width=2);d.text((x-2,y-26),f'{p:.3f}',font=small,fill='black');d.text((x-5,570),['prior','AD','ADX'][i],font=small,fill='black')
d.line((90,110,90,560,1170,560),fill='black',width=2);im.save(dest/'posteriors.png')
im=Image.new('RGB',(1200,750),'white');d=ImageDraw.Draw(im);d.text((35,20),'T,L joint vs product of marginals; difference = dependence',font=font,fill='black')
for case,r in enumerate(files):
 x0=55+case*390;d.text((x0,65),['prior','AD','ADX'][case],font=font,fill='black');joint=r['joint_T_L'];tm=r['marginals'][1];lm=r['marginals'][3]
 for k in range(4):
  t,l=divmod(k,2);v=joint[k];q=tm[t]*lm[l];y=125+k*140;d.text((x0,y),f'T={t}, L={l}',font=small,fill='black');d.rectangle((x0,y+30,x0+int(v*270),y+48),fill='#3572b0');d.rectangle((x0,y+56,x0+int(q*270),y+74),fill='#d47736');d.text((x0,y+80),f'joint {v:.4f}; product {q:.4f}; difference {v-q:+.4f}',font=small,fill='black')
d.text((750,25),'blue joint / orange product',font=small,fill='black');im.save(dest/'joint.png')
im=Image.new('RGB',(1000,600),'white');d=ImageDraw.Draw(im);d.text((25,20),'Rust BP: residual (solid) and marginal error (dashed), log10',font=font,fill='black');d.line((80,70,80,500,950,500),fill='black',width=2)
for j,(name,c) in enumerate(zip(['prior','AD','ADX'],colors)):
 h=json.loads((root/f'bp-{name}'/'run.json').read_text())['history'];d.text((100+j*180,45),name,font=small,fill=c)
 for key in ['residual','max_marginal_error']:
  pts=[]
  for t in h:
   x=80+int(t['sweep']/100*850);y=500-int((max(-11,min(0,math.log10(max(t[key],1e-12))))+11)/11*430);pts.append((x,y))
  for k in range(1,len(pts)):
   if key=='residual' or k%2==0:d.line((pts[k-1],pts[k]),fill=c,width=2)
for i in range(0,12,2):d.text((15,500-int(i/11*430)),str(i-11),font=small,fill='black')
d.text((460,540),'sweeps (0 to 100)',font=small,fill='black');im.save(dest/'bp-traces.png')
