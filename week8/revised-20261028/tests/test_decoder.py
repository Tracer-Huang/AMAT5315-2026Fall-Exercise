import jax
import jax.numpy as jnp
import numpy as np
from decoder import initialize, forward


def test_decoder_is_causal_and_attention_is_normalized():
    cfg = dict(vocab=16, context=8, width=8, heads=2, layers=2)
    params = initialize(cfg, jax.random.PRNGKey(1))
    first = jnp.array([[1,2,3,4,5,6,7,8]])
    second = jnp.array([[1,2,3,4,9,10,11,12]])
    a, weights = forward(params, first, cfg, return_attention=True)
    b, _ = forward(params, second, cfg, return_attention=True)
    assert a.shape == (1,8,16)
    np.testing.assert_allclose(np.asarray(a[:,:4]), np.asarray(b[:,:4]), atol=0, rtol=0)
    for attention in weights:
        assert np.max(np.abs(np.triu(np.asarray(attention), 1))) == 0
        np.testing.assert_allclose(np.asarray(attention).sum(-1), 1., atol=1e-6)
