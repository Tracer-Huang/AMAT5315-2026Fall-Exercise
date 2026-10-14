import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
EVIDENCE=Path(__file__).resolve().parent/'evidence'
plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':150})

def save(fig,name):
    fig.savefig(EVIDENCE/name,bbox_inches='tight');plt.close(fig)

def gallery(images,labels=None,columns=8,title='',name='gallery.png',vmin=0,vmax=1):
    rows=(len(images)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(columns*1.2,rows*1.35),squeeze=False)
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if i<len(images):
            ax.imshow(images[i],cmap='gray',interpolation='nearest',vmin=vmin,vmax=vmax)
            if labels is not None:ax.set_title(str(labels[i]),fontsize=8)
    fig.suptitle(title);fig.tight_layout();save(fig,name)

def rows_gallery(rows,rowlabels,name,title,columns=6,vmin=0,vmax=1):
    fig,axes=plt.subplots(len(rows),columns,figsize=(columns*1.15,len(rows)*1.15),squeeze=False)
    for r,images in enumerate(rows):
        for c,ax in enumerate(axes[r]):
            ax.imshow(np.array(images[c]).reshape(28,28),cmap='gray',interpolation='nearest',vmin=vmin,vmax=vmax)
            ax.set_xticks([]);ax.set_yticks([])
            if c==0:ax.set_ylabel(rowlabels[r],fontsize=9)
            if r==0:ax.set_title(f'Image {c}')
    fig.suptitle(title);fig.tight_layout();save(fig,name)

def class_frequencies(labels,num_classes):
    """Full-split empirical fractions; independent of gallery subsampling."""
    labels=np.asarray(labels)
    if labels.size==0 or np.any((labels<0)|(labels>=num_classes)):
        raise ValueError('A nonempty split with valid class labels is required')
    return np.bincount(labels,minlength=num_classes)/len(labels)

def distribution_figure(images,train_labels,test_labels,class_names=None,title='Dataset',examples_per_class=5):
    """Reusable native grayscale/RGB gallery with optional named classes and split fractions."""
    classes=max(int(np.max(train_labels)),int(np.max(test_labels)))+1
    if class_names is not None:classes=len(class_names)
    names=list(map(str,range(classes))) if class_names is None else class_names
    fig=plt.figure(figsize=(max(6,classes*1.4),9));gs=fig.add_gridspec(2,1,height_ratios=[3,1]);sub=gs[0].subgridspec(examples_per_class,classes)
    for c in range(classes):
        for r,i in enumerate(np.flatnonzero(train_labels==c)[:examples_per_class]):
            ax=fig.add_subplot(sub[r,c]);ax.imshow(images[i],cmap='gray',interpolation='nearest',vmin=0,vmax=255);ax.axis('off')
            if r==0:ax.set_title(names[c],fontsize=9)
    ax=fig.add_subplot(gs[1]);x=np.arange(classes)
    for shift,labels,label in [(-.18,train_labels,'Training'),(.18,test_labels,'Test')]:
        ax.bar(x+shift,class_frequencies(labels,classes),width=.36,label=label)
    ax.set_xticks(x,names);ax.set_ylabel('Fraction of all split labels');ax.legend()
    ax.set_ylim(0,max(.13,1.2*max(class_frequencies(train_labels,classes).max(),class_frequencies(test_labels,classes).max())))
    fig.suptitle(f'{title}: first {examples_per_class} training images per class; all-label split frequencies')
    return fig
