import numpy as np
import pytest

from memory import recall


def test_recall_uses_the_stored_keys_and_normalizes_by_own_weight():
    keys = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    values = np.array([[1., 1.], [-1., 1.], [1., -1.]])
    got, ratio = recall(keys, values, 0, 'outer')
    np.testing.assert_allclose(got, values[0] - values[2], atol=1e-14)
    assert ratio == pytest.approx(1.0)
    for method in ('relu', 'relu2'):
        got, ratio = recall(keys, values, 0, method)
        np.testing.assert_allclose(got, values[0], atol=1e-14)
        assert ratio == pytest.approx(0.0)
    got, ratio = recall(keys, values, 0, 'softmax', beta=8)
    expected = values[0] + np.exp(-8)*values[1] + np.exp(-16)*values[2]
    np.testing.assert_allclose(got, expected, atol=1e-14)
    assert ratio == pytest.approx(np.linalg.norm(expected-values[0])/np.linalg.norm(values[0]))


def test_recall_rejects_nonmatching_keys_and_values():
    with pytest.raises(ValueError):
        recall(np.ones((2, 3)), np.ones((3, 4)), 0, 'outer')
