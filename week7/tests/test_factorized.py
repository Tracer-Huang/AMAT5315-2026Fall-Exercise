"""The target is separable in rotated coordinates; the network enforces its symmetry."""
import jax
import jax.numpy as j
import numpy as np
import core as c

def test_factorized_network_respects_square_symmetry_without_exact_predictor():
    p=c.init(jax.random.key(11),architecture='factorized')
    x=j.array([[1.,2.],[-.3,.8],[4.,-4.]])
    rotation=j.array([[0.,-1.],[1.,0.]])
    assert np.allclose(c.net(p,x@rotation,.3),c.net(p,x,.3)@rotation,atol=1e-5)
    assert np.allclose(c.net(p,-x,.7),-c.net(p,x,.7),atol=1e-5)
    assert c.net(p,x,j.array([.1,.5,.9])).shape==x.shape

def test_learned_linear_tanh_architecture_is_odd_and_trainable():
    p=c.init(jax.random.key(12),width=64,architecture='scalar_basis')
    x=j.array([[.2,.7],[-2.,3.]])
    assert np.allclose(c.net(p,-x,.4),-c.net(p,x,.4),atol=1e-6)
    gradient=jax.grad(lambda params:j.mean(c.net(params,x,.4)**2))(p)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(gradient))
    assert any(np.any(np.asarray(a)!=0) for a in jax.tree.leaves(gradient))
