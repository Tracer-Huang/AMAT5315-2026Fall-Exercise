"""Reproduce and retain the bigram sampling run from its original seed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ngrams import sample_check

ROOT=Path(__file__).resolve().parent
train=np.load(ROOT/'artifacts'/'train-tokens.npy',mmap_mode='r')
original=json.loads((ROOT/'evidence'/'part2-ngrams.json').read_text())
k=original['models']['2']['k']
counts=np.bincount(train[:-1].astype(np.int64)*512+train[1:],minlength=512**2).reshape(512,512)
p=(counts+k)/(counts.sum(axis=1,keepdims=True)+512*k)
path=ROOT/'artifacts'/'bigram-sample.npz'
result=sample_check(p,counts.sum(axis=1),output_path=path)
if abs(result['p_value']-original['bigram_sample']['p_value'])>1e-12:
    raise ValueError('seeded replay differs from the original run')
report=dict(replay=result,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            note='Exact seeded replay of the original 250000-token sample.')
(ROOT/'evidence'/'part2-bigram-sampler-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('bigram sample replay matches; p =',result['p_value'])
