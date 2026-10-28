import jax
import jax.numpy as jnp
import numpy as np
from mlp import initialize, logits, grouped_loss, direct_loss


def test_grouped_batch_matches_full_example_loss_and_gradient():
    cfg = dict(vocab=7, width=4, hidden=8)
    p = initialize(cfg, jax.random.PRNGKey(4))
    x = jnp.array([0, 1, 1, 3, 3, 3, 4]); y = jnp.array([1, 2, 2, 4, 4, 5, 0])
    assert logits(p, x).shape == (7, 7)
    np.testing.assert_allclose(grouped_loss(p, x, y, cfg), direct_loss(p, x, y), rtol=1e-6)
    g1 = jax.grad(grouped_loss)(p, x, y, cfg)
    g2 = jax.grad(direct_loss)(p, x, y)
    for name in p:
        np.testing.assert_allclose(g1[name], g2[name], rtol=1e-5, atol=1e-7)
