import unittest
from unittest import TestCase

from e2cnn.gspaces import *
from e2cnn.nn import *


class TestRestriction(TestCase):
    
    def test_restrict_rotations(self):
        
        space = Rot2dOnR2(-1, maximum_frequency=10)
        cls = FieldType(space, list(space.representations.values()))
        
        rl = RestrictionModule(cls, 4)
        
        rl.check_equivariance()

    def test_restrict_rotations_to_trivial(self):
    
        space = Rot2dOnR2(-1, maximum_frequency=10)

        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, 1)

        rl.check_equivariance()

    def test_restrict_flipsrotations(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)

        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, (0., 10))

        rl.check_equivariance()

    def test_restrict_flipsrotations_to_rotations(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)
    
        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, (None, -1))

        rl.check_equivariance()

    def test_restrict_flipsrotations_to_flips(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)

        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, (0., 1))

        rl.check_equivariance()
    
    def test_restrict_fliprotations_to_trivial(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)
    
        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, (None, 1))

        rl.check_equivariance()

    def test_restrict_flips_to_trivial(self):
    
        space = Flip2dOnR2()

        cls = FieldType(space, list(space.representations.values()))

        rl = RestrictionModule(cls, 1)

        rl.check_equivariance()
    

if __name__ == '__main__':
    unittest.main()
