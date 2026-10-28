import jax
import jax.numpy as jnp
import numpy as np
from decoder import initialize, forward
from train_transformer import lm_loss


def test_training_loss_scores_shifted_targets_at_each_position():
    cfg = dict(vocab=16, context=8, width=8, heads=2, layers=2)
    p = initialize(cfg, jax.random.PRNGKey(5))
    inputs = jnp.array([[1, 2, 3, 4]])
    targets = jnp.array([[2, 3, 4, 5]])
    logits = forward(p, inputs, cfg)
    manual = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1).mean()
    np.testing.assert_allclose(lm_loss(p, inputs, targets, cfg), manual, rtol=1e-6)
