import json
import numpy as np
from data import download,mnist,cifar,ROOT
from plotting import save,distribution_figure
NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def main():
    download();stats={}
    for name,reader,names in [('MNIST',mnist,list(map(str,range(10)))),('CIFAR-10',cifar,NAMES)]:
        train,labels=reader('train');test,testlabels=reader('test')
        assert (len(train),len(test))==((60000,10000) if name=='MNIST' else (50000,10000))
        counts=[np.bincount(l,minlength=10) for l in [labels,testlabels]]
        stats[name]=dict(train_shape=list(train.shape),test_shape=list(test.shape),dtype=str(train.dtype),range=[int(train.min()),int(train.max())],train_counts=counts[0].tolist(),test_counts=counts[1].tolist())
        fig=distribution_figure(train,labels,testlabels,names,name)
        save(fig,name.lower().replace('-','')+'-distribution.png')
        if name=='CIFAR-10': assert np.all(counts[0]==5000) and np.all(counts[1]==1000)
        else: assert all(c.argmax()==1 and c.argmin()==5 for c in counts)
    (ROOT/'evidence/dataset-stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
