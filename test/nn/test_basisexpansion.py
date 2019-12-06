import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *


class TestBasisExpansion(TestCase):
    
    def test_one_block(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, [gspace.regular_repr] * 5)
        t2 = FieldType(gspace, [gspace.regular_repr] * 5)
        self.compare(t1, t2)
        
    def test_many_block_discontinuous(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4)
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4)
        self.compare(t1, t2)
        
    def test_many_block_sorted(self):
        gspace = Rot2dOnR2(8)
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        self.compare(t1, t2)

    def compare(self, r1: FieldType, r2: FieldType):
        
        assert r1.gspace == r2.gspace
    
        s = 7
        # sigma = 0.7
        # windows = [(float(r), (lambda x, r=r, sigma=sigma: np.exp(-((x - r) / sigma) ** 2))) for r in
        #            np.linspace(0, s // 2, 8)]
        #
        # frequencies_cutoff = {float(r): 0.65 * r * np.pi / 2 for r in np.linspace(0, s // 2, 8)}
        
        cl = R2Conv(r1, r2, s, basisexpansion='blocks')
    
        par = cl.basisexpansion.get_basis_names()
        
        for i, attr1 in enumerate(cl.basisexpansion.get_basis_info()):
            attr2 = cl.basisexpansion.get_element_info(i)
            attr3 = cl.basisexpansion.get_element_info(par[i])
            
            self.assertEquals(attr1, attr2)
            self.assertEquals(attr2, attr3)


if __name__ == '__main__':
    unittest.main()
