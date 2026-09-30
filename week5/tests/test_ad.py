import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ad
import numpy as np
import jax

def test_manual_passes_and_shared_adjoint():
    out=ad.manual(1.3)
    assert abs(out['energy']+0.6570169144600471)<1e-12
    assert abs(out['adjoints']['a']+2.3425903117359734)<1e-12
    for x in [out['tangents']['U'],out['adjoints']['r']]:
        assert abs(x-2.239979929791143)<1e-12

def test_sampled_analytic_derivative():
    rs=np.linspace(.95,2.5,601)
    for r in rs:
        out=ad.manual(r)
        exact=24*(r**-7-2*r**-13)
        assert abs(out['tangents']['U']-exact)<1e-12
        assert abs(out['adjoints']['r']-exact)<1e-12
