"""Resumable ranged official checkpoint download; verify pinned LFS SHA256."""
import concurrent.futures,urllib.request,hashlib,json,time
from pathlib import Path
URL='https://huggingface.co/google/ddpm-cifar10-32/resolve/267b167dc01f0e4e61923ea244e8b988f84deb80/diffusion_pytorch_model.safetensors'
SIZE=143020084
out=Path('inputs/cifar-model');pieces=out/'chunks';pieces.mkdir(exist_ok=True)
def one(i):
    a=i*2**22;b=min(SIZE,a+2**22)-1;p=pieces/f'{i:03}'
    if p.exists() and p.stat().st_size==b-a+1:return p
    for trial in range(3):
        try:
            req=urllib.request.Request(URL+f'?piece={i}',headers={'Range':f'bytes={a}-{b}'})
            with urllib.request.urlopen(req,timeout=100) as r:
                data=r.read()
                if len(data)!=b-a+1:raise ValueError((i,len(data),r.headers.get('Content-Range')))
            p.write_bytes(data);print('chunk',i,'done',flush=True);return p
        except Exception as e:print('retry',i,str(e),flush=True)
    raise RuntimeError(i)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    paths=list(pool.map(one,range((SIZE+2**22-1)//2**22)))
p=out/'complete.safetensors'
with p.open('wb') as f:
    for path in paths:f.write(path.read_bytes())
sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha=='1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f', 'Pinned LFS SHA256 mismatch'
print('SHA256',sha,flush=True)
# The safetensors header/tensor lengths are also checked on load.
(out/'download.json').write_text(json.dumps(dict(url=URL,revision='267b167dc01f0e4e61923ea244e8b988f84deb80',bytes=p.stat().st_size,sha256=sha),indent=2))
p.replace(out/'diffusion_pytorch_model.safetensors')
