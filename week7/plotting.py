"""Plotting inputs preserve early history and every paired trajectory."""
import numpy as np

def loss_history(loss,stage):
    if stage==0:return np.array([],dtype=int),np.array([])
    indices=np.unique(np.rint(np.linspace(1,stage,min(stage,600))).astype(int))
    return indices,np.asarray(loss)[indices-1]

def draw_training_stage(axes,paths,stages,loss,index,centres):
    for ax in axes:ax.clear()
    for k in range(10):
        axes[0].plot(*paths[index][:,k].T,lw=.7)
        axes[0].scatter(*paths[index][-1,k],s=20)
    axes[0].scatter(*np.asarray(centres).T,marker='+',c='red')
    axes[0].set(xlim=(-5,5),ylim=(-5,5),title=f'Updates {stages[index]} (off-axis paths retained raw)')
    stage=int(stages[index]);x,y=loss_history(loss,stage)
    if stage:axes[1].plot(x,y,marker='.' if stage<100 else None,markersize=3)
    else:axes[1].text(.5,.5,'Initialization: no training loss yet',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlim=(0,max(2,stage)),ylim=(0,max(1.5,float(np.max(loss[:max(1,stage)]))*1.05)),xlabel='Updates (history through this stage)',ylabel='Noise MSE')
    return x,y

def draw_paired_paths(axes,ddim1,ddim2,ddpm1,ddpm2):
    from matplotlib import pyplot as plt
    for ax,path,repeat,title in zip(axes,[ddim1,ddpm1],[ddim2,ddpm2],['DDIM: repeats overlap','DDPM: two noise draws']):
        for k in range(10):
            color=plt.get_cmap('tab10')(k)
            ax.plot(*np.asarray(path)[:,k].T,lw=.8,color=color,label='Run 1' if k==0 else None)
            ax.plot(*np.asarray(repeat)[:,k].T,lw=.8,ls='--',color=color,label='Run 2' if k==0 else None)
        ax.set(xlim=(-5,5),ylim=(-5,5),title=title);ax.legend()
