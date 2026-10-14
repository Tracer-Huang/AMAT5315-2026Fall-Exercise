"""Enlarged native spin configuration for the Part 2 VERIFY(1) explanation."""
from data import mnist
from plotting import plt,save
import numpy as np

def main():
    images,labels=mnist('train');spins=2*(images[0]>=128).astype(np.int32)-1
    fig,ax=plt.subplots(figsize=(6,6))
    im=ax.imshow(spins,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
    ax.set_xticks([0,7,14,21,27]);ax.set_yticks([0,7,14,21,27]);ax.set_xlabel('Pixel column');ax.set_ylabel('Pixel row')
    ax.set_title(f'First training image: digit {labels[0]}, 28 x 28 visible spins')
    cb=fig.colorbar(im,ax=ax,ticks=[-1,1],shrink=.7);cb.ax.set_yticklabels(['-1: background (v=0)','+1: ink (v=1)'])
    save(fig,'spin-detail.png')
if __name__=='__main__':main()
