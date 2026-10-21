import numpy as np
from plotting import loss_history

def test_early_stage_history_contains_all_available_losses():
    loss=np.arange(25000,dtype=float)
    for n in [1,4,13,45,158,561,1988,7050,25000]:
        x,y=loss_history(loss,n)
        assert len(x)>0 and x[0]==1 and x[-1]==n
        assert np.array_equal(y,loss[x-1])
    x,y=loss_history(loss,0)
    assert len(x)==len(y)==0

def test_actual_frame_artists_show_early_losses_and_both_repeats():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotting import draw_training_stage,draw_paired_paths
    paths=np.zeros((10,1001,10,2));stages=np.array([0,1,4,13,45,158,561,1988,7050,25000]);loss=np.ones(25000)
    fig,axes=plt.subplots(1,2)
    for i in range(1,10):
        draw_training_stage(axes,paths,stages,loss,i,np.zeros((4,2)))
        assert len(axes[1].lines)==1
        assert axes[1].lines[0].get_xdata()[-1]==stages[i]
    plt.close(fig)
    fig,axes=plt.subplots(1,2)
    first=np.zeros((1001,10,2));second=np.ones_like(first)
    draw_paired_paths(axes,first,first,first,second)
    assert len(axes[0].lines)==len(axes[1].lines)==20
    assert np.array_equal(axes[1].lines[1].get_xdata(),second[:,0,0])
    plt.close(fig)
