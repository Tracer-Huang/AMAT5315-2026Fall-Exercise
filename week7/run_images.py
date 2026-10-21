"""Parts 4-5: fixed JAX predictor, no training, no hidden truth to sampler."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,time,hashlib,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import core as c
from unet import UNet
Path('evidence').mkdir(exist_ok=True);Path('artifacts').mkdir(exist_ok=True)
net=UNet();predict=lambda x,t:net.apply(x,t-1)

def rgb(x):return np.clip((np.asarray(x).transpose(1,2,0)+1)/2,0,1)
def save(name,fig):fig.tight_layout();fig.savefig(f'evidence/{name}.png',dpi=180);plt.close(fig)
t0=time.time();results={}
if not Path('artifacts/photos.npz').exists():
    # One image at a time limits compiler workspace and raw trajectory memory.
    outs=[];stages=[]
    for i in range(4):
        x=jax.random.normal(jax.random.key(7400+i),(1,3,32,32),dtype=j.float32)
        z,path=c.sample(predict,x,1000,'ddpm',7410+i)
        z.block_until_ready();outs.append(np.asarray(z[0]));stages.append(np.asarray(path)[[0,500,800,900,950,1000],0])
        print('photo',i,'seconds',time.time()-t0,flush=True)
    np.savez('artifacts/photos.npz',samples=np.array(outs),stages=np.array(stages))
photo=np.load('artifacts/photos.npz');fig,axs=plt.subplots(4,6,figsize=(12,8))
for i in range(4):
 for k,t in enumerate([1000,500,200,100,50,0]):axs[i,k].imshow(rgb(photo['stages'][i,k]),interpolation='nearest');axs[i,k].set_title(f't={t}');axs[i,k].axis('off')
save('photographs',fig)
# Reuse the Week 6 gallery's native CIFAR reader on its already local dataset.
sys.path.insert(0,str(Path('../week6').resolve()))
from data import cifar
real,labels=cifar('test');fig,axs=plt.subplots(2,8,figsize=(12,3.8))
for k in range(8):axs[0,k].imshow(real[k],interpolation='nearest');axs[0,k].set_title(f'Real label {labels[k]}');axs[0,k].axis('off')
for k in range(8):
 axs[1,k].axis('off')
 if k<4:axs[1,k].imshow(rgb(photo['samples'][k]),interpolation='nearest');axs[1,k].set_title('Generated')
save('real-vs-generated',fig)
# Four inputs x four random seeds; inference never receives the withheld centre.
pack=np.load('inputs/images.npz');print('input keys',pack.files,flush=True)
truth=pack['test']
assert truth.shape==(4,3,32,32) and truth.dtype==np.float32
mask=np.ones_like(truth);mask[:,:,6:26,6:26]=0;observed=truth*mask
completions=[];records=[]
for seed in [7201,7202,7203,7204]:
 path=Path(f'artifacts/inpaint-{seed}.npy')
 if path.exists():z=np.load(path)
 else:
  z,_=c.repaint(predict,j.array(observed),j.array(mask),seed);z=np.asarray(z);np.save(path,z)
 completions.append(z)
 for i in range(4):records.append(dict(image=i,seed=seed,hidden_mse=float(np.mean((z[i,:,6:26,6:26]-truth[i,:,6:26,6:26])**2)),observed_max_change=float(np.max(np.abs((z[i]-truth[i])*mask[i])))))
 print('inpaint',seed,'seconds',time.time()-t0,flush=True)
completions=np.array(completions);variance=completions.var(0).mean(1)
fig,axs=plt.subplots(4,7,figsize=(14,8))
for i in range(4):
 imgs=[truth[i],observed[i]]+[completions[k,i] for k in range(4)]
 titles=['Truth','Observed']+[f'{7201+k}\nMSE {records[k*4+i]["hidden_mse"]:.3f}' for k in range(4)]
 for k,(img,title) in enumerate(zip(imgs,titles)):axs[i,k].imshow(rgb(img),interpolation='nearest');axs[i,k].set_title(title);axs[i,k].axis('off')
 im=axs[i,6].imshow(variance[i],cmap='magma',vmin=0);axs[i,6].set_title('RGB variance');axs[i,6].axis('off');fig.colorbar(im,ax=axs[i,6],fraction=.046)
save('inpainting',fig)
results=dict(completions=records,mean_hidden_mse=float(np.mean([r['hidden_mse'] for r in records])),observed_max_change=float(max(r['observed_max_change'] for r in records)),outside_variance=float(np.max(variance*mask[:,0])),evaluations_per_seed=99,total_completions=16,seconds=time.time()-t0,photo_seeds=[7400,7401,7402,7403],photo_reverse_seeds=[7410,7411,7412,7413],model_revision='267b167dc01f0e4e61923ea244e8b988f84deb80',weights_sha256=hashlib.sha256(Path('inputs/cifar-model/diffusion_pytorch_model.safetensors').read_bytes()).hexdigest())
Path('evidence/image-results.json').write_text(json.dumps(results,indent=2));print('DONE',results,flush=True)
