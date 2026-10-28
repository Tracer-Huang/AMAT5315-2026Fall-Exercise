import numpy as np
from interpret_mlp import side_concentration


def test_side_concentration_uses_only_the_requested_sign():
    values=np.array([3., 2., 1., -4., -1.])
    positive=side_concentration(values, 1, 2)
    negative=side_concentration(values, -1, 2)
    assert positive['indices'] == [0, 1]
    assert np.isclose(positive['share'], 13/14)
    assert negative['indices'] == [3, 4]
    assert np.isclose(negative['share'], 1.)
