"""Fixed CIFAR U-Net in JAX, matching diffusers v0.40.0 UNet2DModel.
Checkpoint tensors retain original names; convolutions transpose OIHW -> HWIO.
"""
import json,struct
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j

def read_weights(path):
    # Safetensors' documented format, float32 checkpoint only; no pickle.
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));raw=f.read()
    weights={}
    for name,info in header.items():
        if name=='__metadata__':continue
        assert info['dtype']=='F32',info
        a,b=info['data_offsets'];weights[name]=j.array(np.frombuffer(raw[a:b],dtype='<f4').reshape(info['shape']).copy())
    return weights

class UNet:
    def __init__(self,path='inputs/cifar-model'):
        path=Path(path);self.config=json.loads((path/'config.json').read_text());self.w=read_weights(path/'diffusion_pytorch_model.safetensors')
        assert self.config['block_out_channels']==[128,256,256,256]
        assert self.config['attention_head_dim'] is None
        self.apply=jax.jit(self.forward)
    def linear(self,x,name):return x@self.w[name+'.weight'].T+self.w[name+'.bias']
    def conv(self,x,name,stride=1,padding='SAME'):
        w=self.w[name+'.weight'].transpose(2,3,1,0)
        return jax.lax.conv_general_dilated(x,w,(stride,stride),padding,dimension_numbers=('NHWC','HWIO','NHWC'))+self.w[name+'.bias']
    def norm(self,x,name):
        b,h,w,c=x.shape;z=x.reshape(b,h,w,32,c//32)
        mean=j.mean(z,axis=(1,2,4),keepdims=True);var=j.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
        z=((z-mean)*jax.lax.rsqrt(var+1e-6)).reshape(x.shape)
        return z*self.w[name+'.weight']+self.w[name+'.bias']
    def res(self,x,emb,name):
        h=self.conv(jax.nn.silu(self.norm(x,name+'.norm1')),name+'.conv1')
        h=h+self.linear(jax.nn.silu(emb),name+'.time_emb_proj')[:,None,None,:]
        h=self.conv(jax.nn.silu(self.norm(h,name+'.norm2')),name+'.conv2')
        skip=self.conv(x,name+'.conv_shortcut') if name+'.conv_shortcut.weight' in self.w else x
        return skip+h
    def attn(self,x,name):
        b,h,w,c=x.shape;z=self.norm(x,name+'.group_norm').reshape(b,h*w,c)
        q=self.linear(z,name+'.query');k=self.linear(z,name+'.key');v=self.linear(z,name+'.value')
        a=jax.nn.softmax(j.einsum('bic,bjc->bij',q,k)/j.sqrt(float(c)),axis=-1)
        out=self.linear(a@v,name+'.proj_attn')
        return x+out.reshape(b,h,w,c)
    def forward(self,x,t):
        # Public NCHW -> internal NHWC.
        x=x.transpose(0,2,3,1);t=j.broadcast_to(j.asarray(t,dtype=j.float32),(x.shape[0],))
        freq=j.exp(-j.log(10000.)*j.arange(64,dtype=j.float32)/63)
        angles=t[:,None]*freq[None];emb=j.concatenate([j.sin(angles),j.cos(angles)],axis=-1)
        emb=self.linear(jax.nn.silu(self.linear(emb,'time_embedding.linear_1')),'time_embedding.linear_2')
        x=self.conv(x,'conv_in');skips=[x]
        for i in range(4):
            base=f'down_blocks.{i}'
            for k in range(2):
                x=self.res(x,emb,f'{base}.resnets.{k}')
                if i==1:x=self.attn(x,f'{base}.attentions.{k}')
                skips.append(x)
            if i<3:
                x=j.pad(x,((0,0),(0,1),(0,1),(0,0)))
                x=self.conv(x,base+'.downsamplers.0.conv',stride=2,padding='VALID');skips.append(x)
        x=self.res(x,emb,'mid_block.resnets.0');x=self.attn(x,'mid_block.attentions.0');x=self.res(x,emb,'mid_block.resnets.1')
        for i in range(4):
            base=f'up_blocks.{i}'
            for k in range(3):
                x=j.concatenate([x,skips.pop()],axis=-1);x=self.res(x,emb,f'{base}.resnets.{k}')
                if i==2:x=self.attn(x,f'{base}.attentions.{k}')
            if i<3:
                x=j.repeat(j.repeat(x,2,axis=1),2,axis=2);x=self.conv(x,base+'.upsamplers.0.conv')
        assert not skips
        return self.conv(jax.nn.silu(self.norm(x,'conv_norm_out')),'conv_out').transpose(0,3,1,2)
