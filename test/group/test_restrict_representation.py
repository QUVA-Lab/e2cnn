import unittest
from unittest import TestCase

from e2cnn.group import CyclicGroup
from e2cnn.group import DihedralGroup
from e2cnn.group import O2
from e2cnn.group import SO2
from e2cnn.group import directsum

import numpy as np


class TestRestrictRepresentations(TestCase):
    
    def test_restrict_dihedral(self):
        dg = DihedralGroup(8)
        sg_id = (0, 4)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_dihedral(self):
        N = 9
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N // rot)):
                    sg_id = (axis, rot)
                    for name, irrep in dg.irreps.items():
                        self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        sg_id = (None, 3)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        for axis in range(11):
            sg_id = (axis, 1)
            for name, irrep in dg.irreps.items():
                self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(9)
        sg_id = 3
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_dihedral_even_dihedral(self):
        N = 12
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N//rot)):
                    sg_id = (axis, rot)
                    for name, irrep in dg.irreps.items():
                        self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(12)
        sg_id = (None, 4)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        sg_id = (None, 3)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_dihedral_even_flips(self):
        dg = DihedralGroup(12)
        sg_id = (1, 1)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(8)
        sg_id = 2
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(10)
        sg_id = 5
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic(self):
        dg = DihedralGroup(12)
        sg_id = (None, 12)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_dihedral_odd_cyclic(self):
        dg = DihedralGroup(13)
        sg_id = (None, 13)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_o2_dihedral_odd(self):
        dg = O2(10)
        
        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(5*S), 5)
            for name, irrep in dg.irreps.items():
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_cyclic_odd(self):
        dg = O2(10)
        sg_id = (None, 3)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_flips(self):
        dg = O2(10)

        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/S, 1)
            for name, irrep in dg.irreps.items():
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_so2_cyclic_odd(self):
        dg = SO2(10)
        sg_id = 3
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_dihedral_even(self):
        dg = O2(10)

        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(6*S), 6)
            for name, irrep in dg.irreps.items():
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_cyclic_even(self):
        dg = O2(10)
        sg_id = (None, 4)
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_so2_cyclic_even(self):
        dg = SO2(10)
        sg_id = 4
        for name, irrep in dg.irreps.items():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_rr_dihedral_even_flips(self):
        dg = DihedralGroup(10)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_restriction(dg, sg_id, repr)
    
    def test_restrict_rr_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_dihedral_even_dihedral_even(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 6)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_dihedral_even_dihedral_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_dihedral_odd_dihedral_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(8)
        repr = dg.regular_representation
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(16)
        repr = dg.regular_representation
        sg_id = 8
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_rr_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(14)
        repr = dg.regular_representation
        sg_id = 7
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(15)
        repr = dg.regular_representation
        sg_id = 5
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_flips(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (1., 1)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_dihedral_even(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (0., 6)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_dihedral_odd(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (0., 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_so2(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, -1)
        self.check_restriction(dg, sg_id, repr)
        
    def test_restrict_o2_cyclic_even(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_cyclic_odd(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_so2_cyclic_even(self):
        dg = SO2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = 8
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_so2_cyclic_odd(self):
        dg = SO2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = 7
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_o2(self):
        dg = O2(10)
        repr = directsum(list(dg.irreps.values()))
        sg_id = (1., -1)
        self.check_restriction(dg, sg_id, repr)

    def check_restriction(self, group, subgroup_id, repr):
    
        assert repr.group == group
    
        sg, parent_element, child_element = group.subgroup(subgroup_id)
    
        restrict_repr = group.restrict_representation(subgroup_id, repr)
        
        # def is_close(x, y):
        #     if isinstance(x, tuple):
        #         if isinstance(y, tuple):
        #             if len(x) == len(y):
        #                 return all([is_close(xi, yi) for xi, yi in zip(x, y)])
        #             else:
        #                 return False
        #         else:
        #             return False
        #     else:
        #         return np.fabs(x - y) < 1e-15
        
        for e in group.testing_elements():
            c = child_element(e)
            if c is not None:
                assert sg.is_element(c)
                assert group.equal(parent_element(c), e), f"Element {e} from subgroup {subgroup_id}: {parent_element(c)}, {e} | {c}"
        
        for e in sg.testing_elements():
            
            assert sg.equal(child_element(parent_element(e)), e), f"Element {e} from subgroup {sg.name}: {parent_element(e)}, {child_element(parent_element(e))}"
            
            repr_a = repr(parent_element(e))
            repr_b = restrict_repr(e)

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{group.name} | {repr.name} | {subgroup_id} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
            if not np.allclose(repr_a, repr_b):
                print(f"{repr.name} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")


if __name__ == '__main__':
    unittest.main()
