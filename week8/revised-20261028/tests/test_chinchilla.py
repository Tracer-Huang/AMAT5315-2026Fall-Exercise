import numpy as np
from chinchilla import optimal_split, robust_objective


def test_equal_power_and_coefficient_split_has_equal_size_and_tokens():
    n, d = optimal_split((2., 3., 3., .5, .5), 6e12)
    assert np.isclose(n, 1e6)
    assert np.isclose(d, 1e6)
    assert np.isclose(6*n*d, 6e12)


def test_huber_objective_analytic_gradient_matches_finite_differences():
    theta = np.array([np.log(1.8), np.log(.4), np.log(.3), .34, .37])
    n = np.array([1e8, 3e9, 2e10]); d = np.array([1e10, 3e10, 2e11]); y = np.array([3.1, 2.7, 2.5])
    value, grad = robust_objective(theta, n, d, y)
    h = 1e-6
    finite = np.array([(robust_objective(theta+np.eye(5)[i]*h,n,d,y)[0] -
                        robust_objective(theta-np.eye(5)[i]*h,n,d,y)[0])/(2*h) for i in range(5)])
    assert np.isfinite(value)
    np.testing.assert_allclose(grad, finite, rtol=1e-5, atol=1e-8)
