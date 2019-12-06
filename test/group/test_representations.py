import unittest
from unittest import TestCase

from e2cnn.group import CyclicGroup
from e2cnn.group import DihedralGroup
from e2cnn.group import O2
from e2cnn.group import SO2
from e2cnn.group import Representation
from e2cnn.group import directsum

import numpy as np


class TestRepresentation(TestCase):
    
    def test_regular_cyclic(self):
        g = CyclicGroup(15)
        rr = g.regular_representation
        self.check_representation(rr)
        self.check_character(rr)

    def test_regular_dihedral(self):
        g = DihedralGroup(10)
        rr = g.regular_representation
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_cyclic(self):
        g = CyclicGroup(15)
        rr = directsum(list(g.representations.values()))
        
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_dihedral(self):
        g = DihedralGroup(10)
        rr = directsum(list(g.representations.values()))

        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_so2(self):
        g = SO2(6)
        rr = directsum(list(g.representations.values()))
    
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_o2(self):
        g = O2(6)
        rr = directsum(list(g.representations.values()))

        self.check_representation(rr)
        self.check_character(rr)

    def check_representation(self, repr: Representation):
    
        group = repr.group
        
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
        
        P = directsum([group.irreps[irr] for irr in repr.irreps], name="irreps")
        
        self.assertTrue(np.allclose(repr.change_of_basis @ repr.change_of_basis.T, np.eye(repr.size)))
        self.assertTrue(np.allclose(repr.change_of_basis.T @ repr.change_of_basis, np.eye(repr.size)))
        
        for a in group.testing_elements():
            
            repr_1 = repr(a)
            repr_2 = repr.change_of_basis @ P(a) @ repr.change_of_basis_inv

            self.assertTrue(np.allclose(repr_1, repr_2),
                            msg=f"{a}:\n{repr_1}\ndifferent from\n {repr_2}\n")
            
            for b in group.testing_elements():
                repr_ab = repr(a) @ repr(b)
                c = group.combine(a, b)
                repr_c = repr(c)

                self.assertTrue(np.allclose(repr_ab, repr_c), msg=f"{a} x {b} = {c}:\n{repr_ab}\ndifferent from\n {repr_c}\n")

    def check_character(self, repr: Representation):
    
        group = repr.group
    
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
    
        for a in group.testing_elements():
        
            char_a_1 = repr.character(a)
            char_a_2 = np.trace(repr(a))
        
            self.assertAlmostEqual(char_a_1, char_a_2,
                                   msg=f"{a}: Character implemented in \n{repr}\n different from the trace")
        

if __name__ == '__main__':
    unittest.main()
