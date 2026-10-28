import numpy as np
from sampling import distribution, broadest_context


def test_top_p_includes_the_token_that_crosses_the_threshold():
    p = np.array([.4, .35, .2, .05])
    actual = distribution(np.log(p), temperature=1., mode='top_p', value=.6)
    np.testing.assert_allclose(actual, [.4/.75, .35/.75, 0, 0])


def test_top_k_and_min_p_drop_only_the_specified_tokens():
    p = np.array([.4, .35, .2, .05])
    np.testing.assert_allclose(distribution(np.log(p), 1., 'top_k', 2), [.4/.75, .35/.75, 0, 0])
    np.testing.assert_allclose(distribution(np.log(p), 1., 'min_p', .5), [.4/.95, .35/.95, .2/.95, 0])


def test_small_temperature_is_numerically_greedy():
    p = distribution(np.array([0., 1., -1.]), .0001)
    np.testing.assert_allclose(p, [0, 1, 0])


def test_broad_context_is_chosen_by_entropy_before_sampling():
    index, entropy = broadest_context(np.array([[10., 0., 0.], [0., 0., 0.]]))
    assert index == 1
    assert np.isclose(entropy, np.log(3))
