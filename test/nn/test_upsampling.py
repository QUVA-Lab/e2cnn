import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import numpy as np


class TestUpsampling(TestCase):
    
    def test_cyclic_even_bilinear(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "bilinear")

    def test_cyclic_odd_bilinear(self):
        g = Rot2dOnR2(9)
        self.check_upsampling(g, "bilinear")

    def test_dihedral_even_bilinear(self):
        g = FlipRot2dOnR2(8)
        self.check_upsampling(g, "bilinear")

    def test_dihedral_odd_bilinear(self):
        g = Rot2dOnR2(9)
        self.check_upsampling(g, "bilinear")

    def test_so2_bilinear(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "bilinear")

    def test_o2_bilinear(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "bilinear")

    # "NEAREST" method is not equivariant!! As a result, all the following tests fail

    def test_cyclic_even_nearest(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "nearest")

    def test_cyclic_odd_nearest(self):
        g = Rot2dOnR2(9)
        self.check_upsampling(g, "nearest")

    def test_dihedral_even_nearest(self):
        g = FlipRot2dOnR2(8)
        self.check_upsampling(g, "nearest")

    def test_dihedral_odd_nearest(self):
        g = Rot2dOnR2(9)
        self.check_upsampling(g, "nearest")

    def test_so2_nearest(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "nearest")

    def test_o2_nearest(self):
        g = Rot2dOnR2(8)
        self.check_upsampling(g, "nearest")

    def check_upsampling(self, g, mode):
        for s in [2, 3, 5]:
            print(f"\nScale: {s}\n")
            for r in g.representations.values():
                r1 = FieldType(g, [r])
                ul = R2Upsampling(r1, mode=mode, scale_factor=s)
                ul.check_equivariance()
        
        
if __name__ == '__main__':
    unittest.main()
