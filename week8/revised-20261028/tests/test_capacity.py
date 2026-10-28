import numpy as np
from capacity import measure


def test_stores_recover_orthogonal_key_value_pairs():
    keys = np.eye(4)
    values = np.eye(4)
    result = measure(keys, values, np.arange(4))
    for name in ('outer', 'least_squares', 'relu', 'relu2', 'relu3'):
        np.testing.assert_allclose(result[name], 0., atol=1e-12)


def test_least_squares_uses_self_weight_as_its_signal():
    keys=np.array([[1.,0.],[0.,1.],[2**-.5,2**-.5]])
    values=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    result=measure(keys,values,np.array([0]))
    weights=(keys@np.linalg.pinv(keys))[0]
    reference=np.linalg.norm(weights@values-weights[0]*values[0])/np.linalg.norm(weights[0]*values[0])
    np.testing.assert_allclose(result['least_squares'], [reference], atol=1e-12)
