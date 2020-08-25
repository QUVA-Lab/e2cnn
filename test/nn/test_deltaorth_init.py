import unittest
from unittest import TestCase

import numpy as np
from e2cnn.nn import *
from e2cnn.nn import init
from e2cnn.gspaces import *
from e2cnn.group import *

import torch

from random import shuffle


class TestDeltaOrth(TestCase):
    
    def test_one_block(self):
        # gspace = FlipRot2dOnR2(6)
        gspace = Rot2dOnR2(8)
        irreps = directsum([gspace.fibergroup.irrep(k) for k in range(5)], name="irrepssum")
        # t1 = FieldType(gspace, [gspace.regular_repr]*1)
        # t2 = FieldType(gspace, [gspace.regular_repr]*1)
        t1 = FieldType(gspace, [irreps])
        t2 = FieldType(gspace, [irreps])
        self.check(t1, t2)
        
    def test_many_block_discontinuous(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 7)
        t2 = FieldType(gspace, list(gspace.representations.values()) * 7)
        self.check(t1, t2)
        
    def test_many_block_sorted(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        self.check(t1, t2)

    def check(self, r1: FieldType, r2: FieldType):
        
        np.set_printoptions(precision=7, threshold=60000, suppress=True)
        
        assert r1.gspace == r2.gspace
        
        assert r2.size >= r1.size
    
        s = 7
        
        c = int(s//2)
        
        cl = R2Conv(r1, r2, s, basisexpansion='blocks',
                    sigma=[0.01] + [0.6]*int(s//2),
                    frequencies_cutoff=3.)
        
        for _ in range(20):
            init.deltaorthonormal_init(cl.weights.data, cl.basisexpansion)
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            
            filter, _ = cl.expand_parameters()
            
            center = filter[..., c, c]
            
            # print(center.detach().numpy())

            id = torch.einsum("ab,bc->ac", center.t(), center)
            # print(id.detach().numpy())
            
            # we actually check that the matrix is a "multiple" of an orthonormal matrix because some energy might
            # be lost on surrounding cells
            id /= id.max()

            self.assertTrue(torch.allclose(id, torch.eye(r1.size), atol=5e-7))
            
            # filter /= (filter[..., c, c]**2 / filter.shape[1]).sum().sqrt()[..., None, None]
            
            filter[..., c, c].fill_(0)

            # max, _ = filter.reshape(-1, s, s,).abs().max(0)
            # print(max.detach().numpy())
            # print("\n")
            
            self.assertTrue(torch.allclose(filter, torch.zeros_like(filter), atol=1e-7))


if __name__ == '__main__':
    unittest.main()
