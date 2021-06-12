import numpy as np
from e2cnn.diffops import store_cache, load_cache
from e2cnn.diffops.utils import discretize_homogeneous_polynomial

def test_cache():
    # generate a few diffops:
    coefficients = [
        np.array([2, 0]),
        np.array([0, 1, 0, 3]),
        np.array([1, -2, 1]),
    ]
    diffops = []
    points = [-2., -1., 0., 1., 2.]
    for c in coefficients:
        diffops.append(discretize_homogeneous_polynomial(points, c))
    
    store_cache()
    load_cache()
    for i, c in enumerate(coefficients):
        assert np.allclose(diffops[i], discretize_homogeneous_polynomial(points, c))