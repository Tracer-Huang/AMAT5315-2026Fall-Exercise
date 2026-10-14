"""Direct course downloads and native binary readers, no shuffled split reassignment."""
import gzip,struct,tarfile,json,hashlib,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'
URLS={f'mnist/{s}-{k}.gz':f'https://storage.googleapis.com/cvdf-datasets/mnist/{s}-{k}.gz' for s in ['train','t10k'] for k in ['images-idx3-ubyte','labels-idx1-ubyte']}
URLS['cifar10/cifar-10-binary.tar.gz']='https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz'

def download():
    records=[]
    for name,url in URLS.items():
        path=INPUT/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            print('Download',url,flush=True)
            with urllib.request.urlopen(url,timeout=120) as response,open(path.with_suffix(path.suffix+'.partial'),'wb') as out:
                while chunk:=response.read(1<<20):out.write(chunk)
            path.with_suffix(path.suffix+'.partial').rename(path)
        records.append(dict(file=name,url=url,bytes=path.stat().st_size,sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()))
    (ROOT/'evidence/provenance.json').write_text(json.dumps(records,indent=2))

def mnist(split):
    prefix='train' if split=='train' else 't10k'
    with gzip.open(INPUT/f'mnist/{prefix}-images-idx3-ubyte.gz','rb') as f:
        magic,n,h,w=struct.unpack('>IIII',f.read(16));assert magic==2051 and (h,w)==(28,28)
        images=np.frombuffer(f.read(),np.uint8).reshape(n,h,w)
    with gzip.open(INPUT/f'mnist/{prefix}-labels-idx1-ubyte.gz','rb') as f:
        magic,nl=struct.unpack('>II',f.read(8));assert magic==2049 and nl==n
        labels=np.frombuffer(f.read(),np.uint8);assert len(labels)==n
    return images,labels

def cifar(split):
    filenames=[f'data_batch_{i}.bin' for i in range(1,6)] if split=='train' else ['test_batch.bin']
    batches=[]
    with tarfile.open(INPUT/'cifar10/cifar-10-binary.tar.gz') as t:
        for name in filenames:
            member=next(m for m in t.getmembers() if m.name.endswith('/'+name))
            rows=np.frombuffer(t.extractfile(member).read(),np.uint8).reshape(-1,3073);batches.append(rows)
    rows=np.concatenate(batches)
    return rows[:,1:].reshape(-1,3,32,32).transpose(0,2,3,1),rows[:,0]
