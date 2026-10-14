"""Finite course acceptance checks on measured outputs, independent of narrative wording."""
import json,math
from pathlib import Path
import numpy as np
from data import ROOT

checks=[]
def check(name,ok,detail):
    checks.append(dict(check=name,passed=bool(ok),detail=detail))

def main():
    ds=json.loads((ROOT/'evidence/dataset-stats.json').read_text())
    check('Original train/test split sizes',ds['MNIST']['train_shape']==[60000,28,28] and ds['MNIST']['test_shape']==[10000,28,28] and ds['CIFAR-10']['train_shape']==[50000,32,32,3] and ds['CIFAR-10']['test_shape']==[10000,32,32,3],'MNIST 60000/10000; CIFAR10 50000/10000, original storage order')
    check('Full-label distributions',all(np.argmax(ds['MNIST'][s+'_counts'])==1 and np.argmin(ds['MNIST'][s+'_counts'])==5 for s in ['train','test']) and ds['CIFAR-10']['train_counts']==[5000]*10 and ds['CIFAR-10']['test_counts']==[1000]*10,'MNIST 1 most/5 least; CIFAR each class .1')
    rbm=json.loads((ROOT/'evidence/rbm-metrics.json').read_text());raw=ROOT/'artifacts/rbm-shared-seed'
    check('Full prescribed RBM budget',rbm['settings']['training']['images']==60000 and rbm['settings']['training']['epochs']==20 and rbm['hidden_units']==128,'60000 images, 20 epochs, batch 100, 128 hidden, SGD .1, float32')
    gens=[];recons=[]
    for k in ['1','10']:
        h=rbm['training'][k]['history'];e0=h[0]['reconstruction_error'];e1=h[-1]['reconstruction_error']
        check(f'CD-{k} full history and reconstruction',len(h)==21 and e1<e0/2,f'initial={e0:.8f}; final={e1:.8f}; threshold={e0/2:.8f}')
        adv=rbm['completion'][k]['advantage_at_1000']
        check(f'CD-{k} uses observed upper half',adv>=.1,f'true-minus-wrong lower-half Dice={adv:.8f}')
        sample=np.load(raw/f'cd{k}-sampling.npz');gens.append(sample['generation']);recons.append(sample['reconstruction'])
        c=np.load(raw/f'cd{k}-completion.npz');states=c['states']
        check(f'CD-{k} completion budget and exact clamps',states.shape==(5,1024,784) and np.array_equal(states[:,:,:392],np.broadcast_to(states[0,:,:392],states[:,:,:392].shape)),f'64 digits x 8 chains x 2 controls, steps {c["steps"].tolist()}, upper half exact')
        check(f'CD-{k} binary sampling',all(set(a.ravel())<={0.,1.} for a in [sample['generation'],sample['reconstruction'],states]),'sampled bits, never thresholded probabilities')
    check('Matched generation/reconstruction starts',np.array_equal(gens[0][0],gens[1][0]) and np.array_equal(recons[0][0],recons[1][0]),'Matched starts across CD1/CD10; eval key reset every epoch')
    hop=json.loads((ROOT/'evidence/hopfield-metrics.json').read_text());cross=hop['crossings_at_point9'];vals=[cross[str(n)] for n in [100,200,400,800]]
    check('Full Hopfield sweep',len(hop['records'])==4*21 and len(list((ROOT/'artifacts/hopfield').glob('n*-p*-r*.npz')))==4*21*20,'4 sizes x 21 loads x 20 independent sets; first min(10,P) patterns, random asynchronous sweeps')
    check('Recall crossings',all(a>b for a,b in zip(vals,vals[1:])) and all(abs(a-b)<.02 for a,b in zip(vals,[.186,.170,.156,.150])),str(cross))
    pt=next(r for r in hop['records'] if r['N']==800 and r['P']==160);pred=.5*math.erfc(1/math.sqrt(.4))
    check('Crosstalk at N800 alpha .2',abs(pt['unstable_mean']-pred)<max(.002,4*pt['unstable_se']),f'observed={pt["unstable_mean"]:.7f}; large-N prediction={pred:.7f}; SE={pt["unstable_se"]:.7f}')
    modern=json.loads((ROOT/'evidence/modern-metrics.json').read_text())
    check('Actual JAX attention equivalence',modern['attention_max_abs_difference']<1e-5,f'max over all P/beta/queries={modern["attention_max_abs_difference"]:.9g}')
    check('Full modern/classical experiment',len(modern['records'])==42 and modern['queries']==100,'6 P x 7 beta x 100 queries; same targets and starts within each P')
    at3=[r for r in modern['records'] if r['beta']==.3]
    check('Modern retrieval and classical failure',all(r['modern_success']>=.90 and r['classical_success']==0 for r in at3),str({r['P']:r['modern_success'] for r in at3}))
    expected=['mnist-distribution','cifar10-distribution','binarized-spins','spin-detail','rbm-architecture','rbm-training','rbm-cd1-sampling','rbm-cd10-sampling','rbm-cd1-completion-true','rbm-cd1-completion-wrong','rbm-cd10-completion-true','rbm-cd10-completion-wrong','rbm-completion-overlap','hopfield-unstable','hopfield-capacity','attention-weights','modern-beta-gallery','modern-success','modern-retrieval-gallery']
    check('Required evidence figures',all((ROOT/f'evidence/{f}.png').exists() for f in expected),f'{len(expected)} named figures cover all required VERIFY blocks')
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,optional_AIS='Not run: explicitly optional in course PDF p11',submission='User authorized Git publication after strict checks on 2026-10-14; course PDF requires no portal submission')
    (ROOT/'evidence/verification.json').write_text(json.dumps(report,indent=2))
    for c in checks:print(('PASS' if c['passed'] else 'FAIL')+' '+c['check']+': '+c['detail'])
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
