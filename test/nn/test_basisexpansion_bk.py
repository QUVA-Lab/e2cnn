import unittest
from unittest import TestCase

import numpy as np
from e2cnn.nn import *
from e2cnn.gspaces import *

import torch

from random import shuffle


class TestBasisExpansion(TestCase):
    
    def test_cyclicgroup_sorted(self):
        N = 8
        gc = Rot2dOnR2(N)
        
        reprs = [
            gc.representations['regular'],
            gc.representations['irrep_0'],
            gc.representations['irrep_0'],
            gc.representations['irrep_1'],
            gc.representations['irrep_2'],
            gc.representations['irrep_2'],
            gc.representations['irrep_3'],
            gc.representations['irrep_4']
        ]
        
        r1 = FieldType(gc, reprs).sorted()

        reprs = [
            gc.representations['regular'],
            gc.representations['regular'],
            gc.representations['irrep_0'],
            gc.representations['irrep_1'],
            gc.representations['irrep_2'],
            gc.representations['irrep_2'],
            gc.representations['irrep_3'],
            gc.representations['irrep_4']
        ]
        
        r2 = FieldType(gc, reprs).sorted()
        
        self.compare_methods(r1, r2)

    def test_cyclicgroup_shuffled(self):
        N = 8
        gc = Rot2dOnR2(N)
    
        reprs = [
            gc.representations['regular'],
            gc.representations['irrep_0'],
            gc.representations['irrep_0'],
            gc.representations['irrep_1'],
            gc.representations['irrep_2'],
            gc.representations['irrep_3'],
            gc.representations['irrep_4'],
            gc.representations['irrep_2'],
            gc.representations['irrep_4'],
        ]
    
        shuffle(reprs)
    
        r1 = FieldType(gc, reprs)
    
        reprs = [
            gc.representations['regular'],
            gc.representations['regular'],
            gc.representations['irrep_0'],
            gc.representations['irrep_1'],
            gc.representations['irrep_0'],
            gc.representations['irrep_2'],
            gc.representations['irrep_1'],
            gc.representations['irrep_4']
        ]
    
        shuffle(reprs)
    
        r2 = FieldType(gc, reprs)
    
        self.compare_methods(r1, r2)

    def test_dihedralgroup_sorted(self):
    
        N = 8
    
        g = FlipRot2dOnR2(N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
    
        reprs = [g.representations['regular']] * 2 + irreps
    
        r1 = FieldType(g, reprs).sorted()
    
        r2 = FieldType(g, reprs).sorted()
    
        self.compare_methods(r1, r2)

    def test_dihedralgroup_shuffled(self):
    
        N = 8
    
        g = FlipRot2dOnR2(N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
    
        reprs = [g.representations['regular']] * 2 + irreps
        
        shuffle(reprs)
    
        r1 = FieldType(g, reprs)
    
        r2 = FieldType(g, reprs)
    
        self.compare_methods(r1, r2)

    def test_so2_sorted(self):
    
        N = 4
    
        g = Rot2dOnR2(-1, N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
    
        r1 = FieldType(g, irreps).sorted()
    
        r2 = FieldType(g, irreps).sorted()
    
        self.compare_methods(r1, r2)

    def test_so2_shuffled(self):
    
        N = 4
    
        g = Rot2dOnR2(-1, N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
        
        shuffle(irreps)
        
        r1 = FieldType(g, irreps)
    
        r2 = FieldType(g, irreps)
    
        self.compare_methods(r1, r2)

    def test_o2_sorted(self):
    
        N = 4
    
        g = FlipRot2dOnR2(-1, N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
    
        r1 = FieldType(g, irreps).sorted()
    
        r2 = FieldType(g, irreps).sorted()
    
        self.compare_methods(r1, r2)

    def test_o2_shuffled(self):
    
        N = 4
    
        g = FlipRot2dOnR2(-1, N)
    
        irreps = [irr for irr in g.group.irreps.values()] * 2
        
        shuffle(irreps)
        
        r1 = FieldType(g, irreps)
    
        r2 = FieldType(g, irreps)
    
        self.compare_methods(r1, r2)

    def test_flips_sorted(self):
        g = Flip2dOnR2(np.pi / 4)
    
        r1 = FieldType(g, [
            g.representations['regular'],
            g.representations['regular'],
            g.representations['irrep_0'],
            g.representations['irrep_1'],
            g.representations['irrep_1'],
        ]).sorted()
    
        r2 = FieldType(g, [
            g.representations['regular'],
            g.representations['regular'],
            g.representations['irrep_0'],
            g.representations['irrep_0'],
            g.representations['irrep_1'],
        ]).sorted()
    
        self.compare_methods(r1, r2)

    def test_flips_shuffled(self):
        g = Flip2dOnR2(np.pi / 4)
    
        r1 = FieldType(g, [
            g.representations['regular'],
            g.representations['irrep_0'],
            g.representations['irrep_1'],
            g.representations['regular'],
            g.representations['irrep_1'],
        ])
    
        r2 = FieldType(g, [
            g.representations['regular'],
            g.representations['irrep_0'],
            g.representations['regular'],
            g.representations['irrep_0'],
            g.representations['irrep_1'],
        ])
    
        self.compare_methods(r1, r2)

    def compare_methods(self, r1: FieldType, r2: FieldType):
        
        assert r1.gspace == r2.gspace
    
        s = 7
        # sigma = 0.7
        # windows = [(float(r), (lambda x, r=r, sigma=sigma: np.exp(-((x - r) / sigma) ** 2))) for r in
        #            np.linspace(0, s // 2, 8)]
        #
        # frequencies_cutoff = {float(r): 0.65 * r * np.pi / 2 for r in np.linspace(0, s // 2, 8)}
        
        print(r1)
        print(r2)
        
        cl1 = R2Conv(r1, r2, s, basisexpansion='irreps')
        cl2 = R2Conv(r1, r2, s, basisexpansion='blocks')
    
        par1 = set(cl1.basisexpansion.get_parameters_names())
        par2 = set(cl2.basisexpansion.get_parameters_names())
        
        self.assertEqual(len(par1), len(par2), 'Error! The bases built by the two methods have 2 different lengths')
        
        self.assertEqual(par1, par2, 'Error! The bases built by the two methods have different names')
        
        for i, p in enumerate(sorted(par1)):
    
            # print('Attempt {}: Expanding with parameter {}'.format(i, p))
            
            params = torch.zeros_like(cl1.get_parameters())
            params[cl1.basisexpansion._ids_to_basis[p]] = 1.0
            cl1.basisexpansion.set_parameters(params)

            params = torch.zeros_like(cl2.get_parameters())
            params[cl2.basisexpansion._ids_to_basis[p]] = 1.0
            cl2.basisexpansion.set_parameters(params)

            filter1 = cl1.basisexpansion()
            filter2 = cl2.basisexpansion()
            
            np.set_printoptions(precision=2, threshold=2 * r2.size * r1.size * s ** 2, suppress=True,
                                linewidth=10 * (r1.size + s) + 3)

            if not torch.allclose(filter1, filter2, atol=1e-5):
                # print(filter1.detach().numpy())
                print((filter1 ** 2).sum(dim=[2, 3]).detach().numpy())
    
                # print(filter2.detach().numpy())
                print((filter2 ** 2).sum(dim=[2, 3]).detach().numpy())

            self.assertTrue(torch.allclose(filter1, filter2, atol=1e-5), 'Attempt {}: Expansion with parameter {} resulted in different filters'.format(i, p))

        def inv(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse
        
        ids = [(cl1.basisexpansion._ids_to_basis[par], cl2.basisexpansion._ids_to_basis[par]) for par in par1]
        ids = sorted(ids, key=lambda x: x[0])
        ids = [id[1] for id in ids]
        ids = inv(ids)
        
        for i in range(20):
            params = torch.randn_like(cl1.basisexpansion.get_parameters())
            
            par1 = params
            par2 = params[ids]
            
            cl1.basisexpansion.set_parameters(par1)
            cl2.basisexpansion.set_parameters(par2)

            filter1 = cl1.basisexpansion()
            filter2 = cl2.basisexpansion()
            
            self.assertTrue(torch.allclose(filter1, filter2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
