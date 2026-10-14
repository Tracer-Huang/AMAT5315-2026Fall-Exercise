import itertools
import numpy as np
import jax
import jax.numpy as jnp
from models import free_energy, energy, cd_surrogate, gibbs, hebbian, recall, modern, attention, ink_overlap

def test_free_energy_matches_exact_hidden_sum():
    p=(jnp.array([[.2,-.5],[.8,.1]]),jnp.array([.3,-.1]),jnp.array([.2,.4]))
    v=jnp.array([[0.,1.],[1.,1.]])
    hs=np.array(list(itertools.product([0.,1.],repeat=2)))
    exact=np.array([-np.log(np.exp(-np.array([energy(p,x,h) for h in hs])).sum()) for x in v])
    np.testing.assert_allclose(free_energy(p,v),exact,atol=1e-6)

def test_cd_gradient_matches_independent_correlation_formula():
    p=(jnp.array([[.2,-.5],[.8,.1]]),jnp.array([.3,-.1]),jnp.array([.2,.4]))
    v=jnp.array([[0.,1.],[1.,1.]]); end=jnp.array([[1.,0.],[0.,0.]])
    gw,ga,gb=jax.grad(cd_surrogate)(p,v,end)
    w,a,b=map(np.array,p)
    sigmoid=lambda x:1/(1+np.exp(-x))
    h0=sigmoid(np.array(v)@w+b); hk=sigmoid(np.array(end)@w+b)
    np.testing.assert_allclose(gw,(-np.array(v).T@h0+np.array(end).T@hk)/2,atol=1e-6)
    np.testing.assert_allclose(ga,np.array(end).mean(0)-np.array(v).mean(0),atol=1e-6)
    np.testing.assert_allclose(gb,hk.mean(0)-h0.mean(0),atol=1e-6)

def test_gibbs_is_binary_and_clamp_is_exact():
    p=(jnp.zeros((4,2)),jnp.zeros(4),jnp.zeros(2)); v=jnp.array([[1.,0.,1.,0.]])
    out,key=gibbs(p,v,jax.random.key(6),20,v,jnp.array([True,True,False,False]))
    assert set(np.array(out).ravel())<= {0.,1.}
    np.testing.assert_array_equal(out[:,:2],v[:,:2])

def test_recall_energy_and_ties():
    patterns=jnp.array([[1.,1.,-1.,-1.],[-1.,1.,1.,-1.]])
    J=hebbian(patterns); np.testing.assert_array_equal(jnp.diag(J),0)
    starts=jnp.array([[1.,-1.,-1.,-1.],[-1.,1.,-1.,1.]])
    result,sweeps,converged=recall(J,starts,jax.random.key(3))
    assert np.all(converged)
    for x,y in zip(np.array(starts),np.array(result)):
        assert -y@np.array(J)@y/2<=-x@np.array(J)@x/2+1e-6
        assert np.all(y*(np.array(J)@y)>=-1e-6)
    tie,_,_=recall(jnp.zeros((4,4)),starts,jax.random.key(3))
    np.testing.assert_array_equal(tie,starts)

def test_attention_independent_api_and_overlap():
    X=jnp.array([[1.,-1.,1.,-1.],[-1.,1.,-1.,1.],[1.,1.,-1.,-1.]])
    q=X[:2]
    for beta in [.001,.03,1.]:
        np.testing.assert_allclose(modern(X,q,beta),attention(X,q,beta),atol=1e-6)
    np.testing.assert_allclose(ink_overlap(jnp.array([[1.,0.],[0.,0.]]),jnp.array([[1.,1.],[1.,0.]])),[2/3,0],atol=1e-6)

def test_sampled_endpoints_have_no_gradient():
    p=(jnp.ones((2,2))*.1,jnp.zeros(2),jnp.zeros(2));v=jnp.ones((3,2))
    np.testing.assert_array_equal(jax.grad(cd_surrogate,argnums=2)(p,v,v*.5),0)

def test_zero_couplings_sample_the_bias_probabilities():
    probs=jnp.array([.2,.5,.8,.95]);p=(jnp.zeros((4,2)),jnp.log(probs/(1-probs)),jnp.zeros(2))
    out,_=gibbs(p,jnp.zeros((20000,4)),jax.random.key(98),1)
    np.testing.assert_allclose(np.array(out).mean(0),probs,atol=.015)
