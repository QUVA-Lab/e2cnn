import unittest
from unittest import TestCase

from e2cnn.group import CyclicGroup
from e2cnn.group import DihedralGroup
from e2cnn.group import O2
from e2cnn.group import SO2
from e2cnn.group import Representation
from e2cnn.group import directsum
from e2cnn.group import disentangle

import numpy as np

from scipy.stats import ortho_group


class TestDisentangleRepresentation(TestCase):
    
    def test_regular_cyclic(self):
        g = CyclicGroup(15)
        rr = g.regular_representation
        N = 4
        size = rr.size * N
        
        p = np.eye(size, size)
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)

    def test_regular_dihedral(self):
        g = DihedralGroup(10)
        rr = g.regular_representation
        N = 4
        size = rr.size * N
        
        p = np.eye(size, size)
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)

    def test_mix_cyclic(self):
        g = CyclicGroup(15)
        rr = directsum(list(g.representations.values()))
        
        N = 3
        size = rr.size * N

        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)

        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)

    def test_mix_dihedral(self):
        g = DihedralGroup(10)
        rr = directsum(list(g.representations.values()))
    
        N = 3
        size = rr.size * N

        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)

        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)

    def test_mix_so2(self):
        g = SO2(6)
        rr = directsum(list(g.representations.values()))
    
        N = 3
        size = rr.size * N

        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)

        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)
        
    def test_mix_o2(self):
        g = O2(6)
        rr = directsum(list(g.representations.values()))
    
        N = 3
        size = rr.size * N

        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)

        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr]*N, change_of_basis=p)
        self.check_disentangle(repr)
    
    def test_restrict_irreps_dihedral_odd_dihedral_odd(self):
        dg = DihedralGroup(9)
        sg_id = (1, 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
    
    def test_restrict_irreps_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        sg_id = (0, 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
    
    def test_restrict_irreps_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        sg_id = (1, 1)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
    
    def test_restrict_irreps_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(9)
        sg_id = 3
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
            
    def test_restrict_irreps_dihedral_even_dihedral_even(self):
        dg = DihedralGroup(12)
        sg_id = (1, 6)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(12)
        sg_id = (0, 4)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_dihedral_even_dihedral_odd(self):
        dg = DihedralGroup(12)
        sg_id = (1, 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        sg_id = (0, 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
            
    def test_restrict_irreps_dihedral_even_flips(self):
        dg = DihedralGroup(12)
        sg_id = (1, 1)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(8)
        sg_id = 2
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
            
    def test_restrict_irreps_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(10)
        sg_id = 5
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_dihedral_even_cyclic(self):
        dg = DihedralGroup(12)
        sg_id = (0, 12)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
            
    def test_restrict_irreps_dihedral_odd_cyclic(self):
        dg = DihedralGroup(13)
        sg_id = (0, 13)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))
            
    def test_restrict_irreps_o2_dihedral_odd(self):
        dg = O2(10)
        sg_id = (0., 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_o2_cyclic_odd(self):
        dg = O2(10)
        sg_id = (None, 3)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_o2_flips(self):
        dg = O2(10)
        sg_id = (0., 1)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_so2_cyclic_odd(self):
        dg = SO2(10)
        sg_id = 3
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_o2_dihedral_even(self):
        dg = O2(10)
        sg_id = (0., 6)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_o2_cyclic_even(self):
        dg = O2(10)
        sg_id = (None, 4)
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_irreps_so2_cyclic_even(self):
        dg = SO2(10)
        sg_id = 4
        for name, irrep in dg.irreps.items():
            self.check_disentangle(dg.restrict_representation(sg_id, irrep))

    def test_restrict_rr_dihedral_even_flips(self):
        dg = DihedralGroup(10)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
    
    def test_restrict_rr_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_dihedral_even_dihedral_even(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 6)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_dihedral_even_dihedral_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_dihedral_odd_dihedral_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_rr_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(8)
        repr = dg.regular_representation
        sg_id = (0, 4)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (0, 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_rr_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (0, 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(16)
        repr = dg.regular_representation
        sg_id = 8
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_rr_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(14)
        repr = dg.regular_representation
        sg_id = 7
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_rr_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(15)
        repr = dg.regular_representation
        sg_id = 5
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_o2_flips(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (0., 1)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_o2_dihedral_even(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (0., 6)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_o2_dihedral_odd(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (0., 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_o2_so2(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, -1)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))
        
    def test_restrict_o2_cyclic_even(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, 4)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_o2_cyclic_odd(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, 3)
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_so2_cyclic_even(self):
        dg = SO2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = 8
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def test_restrict_so2_cyclic_odd(self):
        dg = SO2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = 7
        self.check_disentangle(dg.restrict_representation(sg_id, repr))

    def check_disentangle(self, repr: Representation):
    
        group = repr.group
        
        cob, reprs = disentangle(repr)
        
        self.assertEqual(repr.size, sum([r.size for r in reprs]))
        
        ds = directsum(reprs, name="directsum")

        for e in group.testing_elements():
            repr_a = repr(e)
            repr_b = cob.T @ ds(e) @ cob

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{e}:\n{repr_a}\ndifferent from\n {repr_b}\n")


if __name__ == '__main__':
    unittest.main()
