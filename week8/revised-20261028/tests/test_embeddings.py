import numpy as np
import pytest
from embeddings import loo_cosines


def test_each_offset_is_compared_to_the_other_offsets_only():
    directions = np.array([[1., 0.], [1., 0.], [0., 1.]])
    np.testing.assert_allclose(loo_cosines(directions), [2**-.5, 2**-.5, 0.])


def test_zero_offset_is_not_a_valid_direction():
    with pytest.raises(ValueError):
        loo_cosines(np.array([[0., 0.], [1., 0.]]))
