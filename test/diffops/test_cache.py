import numpy as np
from e2cnn.diffops import store_cache, load_cache
from e2cnn.diffops.utils import discretize_homogeneous_polynomial

import unittest
from unittest import TestCase


def make_grid(n):
    x = np.arange(-n, n + 1)
    return np.stack(np.meshgrid(x, -x)).reshape(2, -1)


class TestCache(TestCase):

    def test_cache(self):
        # generate a few diffops:
        coefficients = [
            np.array([2, 0]),
            np.array([0, 1, 0, 3]),
            np.array([1, -2, 1]),
        ]
        diffops = []
        points = make_grid(2)
        for c in coefficients:
            diffops.append(discretize_homogeneous_polynomial(points, c))
        
        store_cache()
        load_cache()
        for i, c in enumerate(coefficients):
            assert np.allclose(diffops[i], discretize_homogeneous_polynomial(points, c))
            

if __name__ == '__main__':
    unittest.main()
