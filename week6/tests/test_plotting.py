import numpy as np
import plotting

def test_class_frequencies_use_all_labels_not_balanced_gallery():
    # Six images in class 0, two in class 1: gallery balance must not alter fractions.
    np.testing.assert_allclose(plotting.class_frequencies(np.array([0]*6+[1]*2),2),[.75,.25])

def test_reusable_native_grayscale_and_rgb_distribution():
    for shape in [(8,4,4),(8,4,4,3)]:
        images=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape)
        fig=plotting.distribution_figure(images,np.array([0]*6+[1]*2),np.array([0,1,1,1]),class_names=None,title='two classes')
        assert all(ax.images[0].get_interpolation()=='nearest' for ax in fig.axes if ax.images)
        heights=[p.get_height() for p in fig.axes[-1].patches]
        np.testing.assert_allclose(heights,[.75,.25,.25,.75])
        plotting.plt.close(fig)
