import unittest
from unittest import TestCase

from e2cnn.group import CyclicGroup
from e2cnn.group import DihedralGroup
from e2cnn.group import O2
from e2cnn.group import SO2
from e2cnn.group import Group

import math


class TestGroups(TestCase):
    
    def test_cyclic_odd(self):
        g = CyclicGroup(15)
        self.check_group(g)

    def test_cyclic_even(self):
        g = CyclicGroup(16)
        self.check_group(g)

    def test_dihedral_odd(self):
        g = DihedralGroup(15)
        self.check_group(g)

    def test_dihedral_even(self):
        g = DihedralGroup(16)
        self.check_group(g)

    def test_so2(self):
        g = SO2(4)
        self.check_group(g)

    def test_o2(self):
        g = O2(4)
        self.check_group(g)
        
    def check_group(self, group: Group):
        
        # def equal(x, y):
        #     if type(x) != type(y):
        #         return False
        #     elif hasattr(x, "__iter__"):
        #         if len(x) != len(y):
        #             return False
        #         else:
        #             for a, b in zip(x, y):
        #                 if type(a) != type(b):
        #                     return False
        #                 elif isinstance(a, float):
        #                     return math.isclose(a, b, abs_tol=1e-15)
        #                 else:
        #                     return a == b
        #     elif isinstance(x, float):
        #         return math.isclose(x, y, abs_tol=1e-15)
        #     else:
        #         return x == y
        
        e = group.identity
        
        for a in group.testing_elements():
            
            self.assertTrue(group.equal(group.combine(a, e), a))
            self.assertTrue(group.equal(group.combine(e, a), a))

            i = group.inverse(a)
            self.assertTrue(group.equal(group.combine(a, i), e))
            self.assertTrue(group.equal(group.combine(i, a), e))
            
            for b in group.testing_elements():
                for c in group.testing_elements():
                    
                    a_bc = group.combine(a, group.combine(b, c))
                    ab_c = group.combine(group.combine(a, b), c)

                    self.assertTrue(group.equal(a_bc, ab_c), f"{a_bc} != {ab_c}")


if __name__ == '__main__':
    unittest.main()
