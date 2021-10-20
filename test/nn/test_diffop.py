import unittest
from unittest import TestCase

import e2cnn.nn.init as init
from e2cnn.nn import *
from e2cnn.gspaces import *

import numpy as np
import math

import torch


class TestDiffop(TestCase):
    
    def test_cyclic(self):
        N = 8
        g = Rot2dOnR2(N)
        
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        
        s = 7
        
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True)
        cl.bias.data = 20*torch.randn_like(cl.bias.data)

        cl.eval()
        cl.check_equivariance()
        
        cl.train()
        cl.check_equivariance()

    def test_cyclic_gauss(self):
        N = 8
        g = Rot2dOnR2(N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, smoothing=1., bias=True)
        cl.bias.data = 20 * torch.randn_like(cl.bias.data)
    
        cl.eval()
        cl.check_equivariance()
    
        cl.train()
        cl.check_equivariance()

    def test_cyclic_rbffd(self):
        N = 8
        g = Rot2dOnR2(N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, rbffd=True, bias=True)
        cl.bias.data = 20 * torch.randn_like(cl.bias.data)
    
        cl.eval()
        cl.check_equivariance()
    
        cl.train()
        cl.check_equivariance()

    def test_so2_gauss(self):
        N = 7
        g = Rot2dOnR2(-1, N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, smoothing=1., bias=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_so2_rbffd(self):
        N = 7
        g = Rot2dOnR2(-1, N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, rbffd=True, bias=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_so2(self):
        N = 7
        g = Rot2dOnR2(-1, N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = FlipRot2dOnR2(N, axis=np.pi/3)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
        
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True)

        cl.eval()
        cl.check_equivariance()

    def test_dihedral_gauss(self):
        N = 8
        g = FlipRot2dOnR2(N, axis=np.pi / 3)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, smoothing=1., bias=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_dihedral_rbffd(self):
        N = 8
        g = FlipRot2dOnR2(N, axis=np.pi / 3)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, rbffd=True, bias=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_o2(self):
        N = 7
        g = FlipRot2dOnR2(-1, N)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
        
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True)

        cl.eval()
        cl.check_equivariance()

    def test_o2_gauss(self):
        N = 7
        g = FlipRot2dOnR2(-1, N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True, smoothing=1.)
    
        cl.eval()
        cl.check_equivariance()

    def test_o2_rbffd(self):
        N = 7
        g = FlipRot2dOnR2(-1, N)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
    
        s = 7
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True, rbffd=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_flip(self):
        # g = Flip2dOnR2(axis=np.pi/3)
        g = Flip2dOnR2(axis=np.pi/2)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
        
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True)
        
        cl.eval()
        cl.check_equivariance()

    def test_flip_gauss(self):
        # g = Flip2dOnR2(axis=np.pi/3)
        g = Flip2dOnR2(axis=np.pi / 2)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True, smoothing=1.)
    
        cl.eval()
        cl.check_equivariance()

    def test_flip_rbffd(self):
        # g = Flip2dOnR2(axis=np.pi/3)
        g = Flip2dOnR2(axis=np.pi / 2)
    
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
    
        cl = R2Diffop(r1, r2, s, maximum_order=4, bias=True, rbffd=True)
    
        cl.eval()
        cl.check_equivariance()

    def test_padding_mode_reflect(self):
        g = Flip2dOnR2(axis=np.pi / 2)
    
        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])
    
        s = 3
        cl = R2Diffop(r1, r2, s, bias=True, padding=1, padding_mode='reflect', initialize=False)
    
        cl.eval()
        cl.check_equivariance()

    def test_padding_mode_circular(self):
        g = FlipRot2dOnR2(4, axis=np.pi / 2)
    
        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])
    
        for mode in ['circular', 'reflect', 'replicate']:
            for s in [3, 5, 7]:
                padding = s // 2
                cl = R2Diffop(r1, r2, s, bias=True, padding=padding, padding_mode=mode, initialize=False)
            
                cl.eval()
                cl.check_equivariance()
    
    def test_output_shape(self):
        g = FlipRot2dOnR2(4, axis=np.pi / 2)
    
        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])
        
        S = 17
        
        x = torch.randn(1, r1.size, S, S)
        x = GeometricTensor(x, r1)
        
        with torch.no_grad():
            for k in [3, 5, 7, 9, 4, 8]:
                for p in [0, 1, 2, 4]:
                    for s in [1, 2, 3]:
                        for mode in ['zeros', 'circular', 'reflect', 'replicate']:
                            cl = R2Diffop(r1, r2, k, maximum_order=2, padding=p, stride=s, padding_mode=mode, initialize=False).eval()
                            y = cl(x)
                            _S = math.floor((S + 2*p - k) / s + 1)
                            self.assertEqual(y.shape, (1, r2.size, _S, _S))
                            self.assertEqual(y.shape, cl.evaluate_output_shape(x.shape))


if __name__ == '__main__':
    unittest.main()
