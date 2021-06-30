import unittest
from unittest import TestCase

import numpy as np
from e2cnn.nn import *
from e2cnn.nn import init
from e2cnn.gspaces import *
from e2cnn.group import *

import torch

from random import shuffle


class TestGeneralizedHeInit(TestCase):
    
    def test_one_block(self):
        N = 8
        # gspace = FlipRot2dOnR2(6)
        gspace = Rot2dOnR2(N)
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(N//2 + 1)], name="irrepssum")
        t1 = FieldType(gspace, [irreps]*2)
        t2 = FieldType(gspace, [irreps]*3)
        # t1 = FieldType(gspace, [gspace.regular_repr]*2)
        # t2 = FieldType(gspace, [gspace.regular_repr]*3)
        self.check(t1, t2)
        self.check_caching(t1, t2)
        
    def test_many_block_discontinuous(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 2)
        t2 = FieldType(gspace, list(gspace.representations.values()) * 3)
        self.check(t1, t2)
        self.check_caching(t1, t2)
        
    def test_many_block_sorted(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 2).sorted()
        t2 = FieldType(gspace, list(gspace.representations.values()) * 3).sorted()
        self.check(t1, t2)
        self.check_caching(t1, t2)
        
    def test_different_irreps_ratio(self):
        N = 8
        gspace = Rot2dOnR2(N)
        irreps_in = directsum([gspace.fibergroup.irrep(np.random.randint(0, N//2+1)) for k in range(4)], name="irrepssum_in")
        irreps_out = directsum([gspace.fibergroup.irrep(np.random.randint(0, N//2+1)) for k in range(4)], name="irrepssum_out")
        t1 = FieldType(gspace, [irreps_in]*3)
        t2 = FieldType(gspace, [irreps_out]*3)
        self.check(t1, t2)
        self.check_caching(t1, t2)

    def test_caching(self):
    
        N = 7
        gspace = Rot2dOnR2(N)
    
        # try combinations of field types which are similar but should still be cached independently
    
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(N // 2 + 1)], name="irrepssum")
        t1 = FieldType(gspace, [irreps] * 2)
        t2 = FieldType(gspace, [irreps] * 3)
        self.check_caching(t1, t2)
    
        t1 = FieldType(gspace, [directsum([irreps]*2)])
        t2 = FieldType(gspace, [directsum([irreps]*3)])
        self.check_caching(t1, t2)
    
        irreps = [gspace.fibergroup.irrep(k) for k in range(N // 2 + 1)]
        t1 = FieldType(gspace, irreps * 2)
        t2 = FieldType(gspace, irreps * 3)
        self.check_caching(t1, t2)
    
        irreps = [gspace.fibergroup.irrep(k) for k in range(N // 2 + 1)]
        t1 = FieldType(gspace, irreps * 2).sorted()
        t2 = FieldType(gspace, irreps * 3).sorted()
        self.check_caching(t1, t2)

    def check(self, r1: FieldType, r2: FieldType):
        
        np.set_printoptions(precision=7, threshold=60000, suppress=True)
        
        assert r1.gspace == r2.gspace
        
        s = 7
        
        cl = R2Conv(r1, r2, s, basisexpansion='blocks',
                    sigma=[0.01] + [0.6]*int(s//2),
                    frequencies_cutoff=3.)
        
        ys = []
        for _ in range(1000):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            
            x = torch.randn(10, r1.size, s, s)
            
            xg = GeometricTensor(x, r1)
            y = cl(xg).tensor
            
            del xg
            del x
            
            ys.append(y.transpose(0, 1).reshape(r2.size, -1))
        
        ys = torch.cat(ys, dim=1)
        
        mean = ys.mean(1)
        std = ys.std(1)
        print(mean)
        print(std)
        
        self.assertTrue(torch.allclose(torch.zeros_like(mean), mean, rtol=2e-2, atol=5e-2))
        self.assertTrue(torch.allclose(torch.ones_like(std), std, rtol=1e-1, atol=6e-2))

    def check_caching(self, r1: FieldType, r2: FieldType):
    
        np.set_printoptions(precision=7, threshold=60000, suppress=True)
    
        assert r1.gspace == r2.gspace
    
        with torch.no_grad():
        
            s = 7
        
            cl = R2Conv(r1, r2, s,
                        # sigma=[0.01] + [0.6]*int(s//2),
                        frequencies_cutoff=3.)
        
            from datetime import datetime
        
            torch.manual_seed(42)
            weights1 = torch.zeros_like(cl.weights.data)
        
            start = datetime.now()
            init.generalized_he_init(weights1, cl.basisexpansion, cache=True)
            end = datetime.now()
            elapsed1 = (end - start).total_seconds()
        
            for i in range(10):
                torch.manual_seed(42)
                weights2 = torch.zeros_like(cl.weights.data)
            
                start = datetime.now()
                init.generalized_he_init(weights2, cl.basisexpansion, cache=True)
                end = datetime.now()
                elapsed2 = (end - start).total_seconds()
            
                self.assertTrue(torch.allclose(weights1, weights2))
            
                # this ensures that the first run was indeed the first time this was cached
                self.assertTrue(elapsed1 >= 4 * elapsed2)


if __name__ == '__main__':
    unittest.main()
