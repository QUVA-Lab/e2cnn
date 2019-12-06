from unittest import TestCase
import unittest

from e2cnn.nn import *
from e2cnn.gspaces import *

import torch

from random import shuffle


class TestReshuffleLayer(TestCase):
    
    def test_indices_permutation(self):
        g = Rot2dOnR2(6)

        r = FieldType(g,
                      [g.representations['irrep_0']] * 3 +
                      [g.representations['irrep_1']] * 3 +
                      [g.representations['irrep_2']] * 3
                      )
        
        permutation = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        
        rl = ReshuffleModule(r, permutation)
        
        self.assertEqual(set(rl.indices.view(-1).numpy()), set(range(0, r.size)),
                         'Error! The indices in the ResuffleLayer are not a permutation of the input channels')

    def test_cyclic_equivariance(self):
        
        N = 9
        
        g = Rot2dOnR2(N)
    
        r = FieldType(g,
                      [g.representations['irrep_0']] * 2 +
                      [g.representations['irrep_1']] * 2 +
                      [g.representations['irrep_2']] * 2 +
                      [g.representations['irrep_3']] * 2
                      )
    
        x = torch.randn(200, r.size, 3, 4)

        x = GeometricTensor(x, r)
        
        for check_ in range(len(r.representations)**2):
        
            permutation = list(range(len(r.representations)))
            shuffle(permutation)
        
            rl = ReshuffleModule(r, permutation)
            
            for e in g.testing_elements:
            
                out1 = rl(x).transform_fibers(e).tensor
                out2 = rl(x.transform_fibers(e)).tensor
                self.assertTrue(torch.allclose(out1, out2, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
