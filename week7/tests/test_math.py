import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c

def test_schedule_endpoints():
    assert c.A.shape==(1001,)
    assert c.A[0]==1
    assert abs(float(c.A[-1])-4.0358298e-5)<1e-10

def test_exact_predictors_gaussian_conditionals():
    x=jnp.array([[0.,0.],[2.,1.]])
    assert np.allclose(c.exact_noise(x,0),0)
    assert np.allclose(c.exact_flow(x,1),x)
    assert np.allclose(c.exact_flow(jnp.array([[0.,0.]]),.5),0)
    # Noise conditional mean equals minus sqrt(noise variance) times score.
    t=300; h=.001; xx=np.array([[.3,.8]],dtype=np.float32)
    var=float(.09*c.A[t]+1-c.A[t]); means=np.sqrt(float(c.A[t]))*np.array(c.CENTRES)
    def logp(v):
        z=np.exp(-np.sum((v-means)**2,axis=1)/(2*var)); return np.log(z.sum())
    score=np.array([(logp(xx+np.eye(2)[i]*h)-logp(xx-np.eye(2)[i]*h))/(2*h) for i in range(2)])
    assert np.allclose(c.exact_noise(jnp.array(xx),t),-np.sqrt(1-float(c.A[t]))*score,atol=1e-3)

def test_ddim_determinism_and_final_no_noise():
    x=jnp.array([[.2,.7],[1.,-.3]])
    a,_=c.sample(c.exact_noise,x,20,'ddim',1)
    b,_=c.sample(c.exact_noise,x,20,'ddim',999)
    assert np.array_equal(a,b)
    a,_=c.sample(c.exact_noise,x,1,'ddpm',1)
    b,_=c.sample(c.exact_noise,x,1,'ddpm',999)
    assert np.array_equal(a,b)

def test_repaint_only_observations_and_preservation():
    y=jnp.ones((1,3,4,4),dtype=jnp.float32)*.2
    m=jnp.ones_like(y).at[:,:,1:3,1:3].set(0)
    result,n=c.repaint(lambda x,t:jnp.zeros_like(x),y*m,m,7)
    assert n==99
    assert np.array_equal(np.asarray(result*m),np.asarray(y*m))
    assert np.isfinite(result).all()

def test_energy_unbiased_identical_point_masses():
    x=np.zeros((10,2)); y=np.ones((10,2))
    assert abs(c.energy(x,y)-2*np.sqrt(2))<1e-12

def test_pinned_unet_legacy_attention_names():
    from unet import UNet
    model=UNet()
    result=model.apply(jnp.zeros((1,3,32,32),dtype=jnp.float32),999)
    assert result.shape==(1,3,32,32)
    assert np.isfinite(result).all()
