import unittest
from unittest import TestCase

import e2cnn.nn.init as init
from e2cnn.nn import *
from e2cnn.gspaces import *

import numpy as np

import torch


class TestConvolution(TestCase):
    
    def test_cyclic(self):
        N = 8
        g = Rot2dOnR2(N)
        
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        s = 7
        sigma = None
        # fco = lambda r: 1. * r * np.pi
        fco = None
        
        cl = R2ConvTransposed(r1, r2, s, basisexpansion='blocks',
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        cl.bias.data = 20*torch.randn_like(cl.bias.data)

        for _ in range(1):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()
        
        cl.train()
        for _ in range(1):
            cl.check_equivariance()
        
        cl.eval()
        
        for _ in range(5):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            filter = cl.filter.clone()
            cl.check_equivariance()
            self.assertTrue(torch.allclose(filter, cl.filter))

    def test_so2(self):
        N = 7
        g = Rot2dOnR2(-1, N)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s, basisexpansion='blocks',
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = FlipRot2dOnR2(N, axis=np.pi/3)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
        # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
        # r2 = FieldType(g, [g.regular_repr])
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s, basisexpansion='blocks',
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 7
        g = FlipRot2dOnR2(-1, N)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s, basisexpansion='blocks',
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_flip(self):
        # g = Flip2dOnR2(axis=np.pi/3)
        g = Flip2dOnR2(axis=np.pi/2)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2ConvTransposed(r1, r2, s, basisexpansion='blocks',
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(32):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
