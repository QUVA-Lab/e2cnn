import unittest
from unittest import TestCase

from e2cnn.gspaces import *
import numpy as np


class TestRestrictGSpace(TestCase):
    
    def test_restrict_rotations(self):
        
        space = Rot2dOnR2(-1, maximum_frequency=10)
        
        subspace, mapping, _ = space.restrict(4)
        
        self.assertIsInstance(subspace, Rot2dOnR2)
        self.assertEqual(subspace.fibergroup.order(), 4)
        
        self.check_restriction(space, 4)

    def test_restrict_rotations_to_trivial(self):
    
        space = Rot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict(1)
    
        self.assertIsInstance(subspace, TrivialOnR2)
        self.assertEqual(subspace.fibergroup.order(), 1)

        self.check_restriction(space, 1)

    def test_restrict_flipsrotations(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)
        
        N=10
        for axis in range(13):
            axis = axis * 2 * np.pi / (13*N)
            subspace, mapping, _ = space.restrict((axis, N))
    
            self.assertIsInstance(subspace, FlipRot2dOnR2)
            self.assertEqual(subspace.fibergroup.order(), 2 * N)
    
            self.check_restriction(space, (axis, N))
        
    def test_restrict_flipsrotations_to_rotations(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict((None, -1))
    
        self.assertIsInstance(subspace, Rot2dOnR2)
        self.assertEqual(subspace.fibergroup.order(), -1)

        self.check_restriction(space, (None, -1))

    def test_restrict_flipsrotations_to_flips(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)

        for axis in range(13):
            axis = axis * 2*np.pi/13
            subspace, mapping, _ = space.restrict((axis, 1))
        
            self.assertIsInstance(subspace, Flip2dOnR2)
            self.assertEqual(subspace.fibergroup.order(), 2)

            self.check_restriction(space, (axis, 1))
        
    def test_restrict_fliprotations_to_trivial(self):
    
        space = FlipRot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict((None, 1))
    
        self.assertIsInstance(subspace, TrivialOnR2)
        self.assertEqual(subspace.fibergroup.order(), 1)
        
        self.check_restriction(space, (None, 1))

    def test_restrict_flips_to_trivial(self):
    
        space = Flip2dOnR2()
    
        subspace, mapping, _ = space.restrict(1)
    
        self.assertIsInstance(subspace, TrivialOnR2)
        self.assertEqual(subspace.fibergroup.order(), 1)

        self.check_restriction(space, 1)
    
    def check_restriction(self, space: GeneralOnR2, subgroup_id):
        subspace, parent, child = space.restrict(subgroup_id)
        
        # rho = space.trivial_repr
        
        for rho in space.fibergroup.irreps.values():
            sub_rho = rho.restrict(subgroup_id)
            
            x = np.random.randn(1, 1, 129, 129)
            
            for e in subspace.testing_elements:
                
                y1 = space.featurefield_action(x, rho, parent(e))
                y2 = subspace.featurefield_action(x, sub_rho, e)

                self.assertTrue(np.allclose(y1, y2), msg=f"{space.name} -> {subgroup_id}: {parent(e)} -> {e}")
        

if __name__ == '__main__':
    unittest.main()
