import unittest
from unittest import TestCase

from e2cnn.gspaces import *
from e2cnn.nn import *
from e2cnn.group import directsum

import numpy as np

from scipy.stats import ortho_group


class TestDisentangle(TestCase):
    
    def test_regular_cyclic(self):
        
        space = Rot2dOnR2(6)
        
        g = space.fibergroup
        rr = g.regular_representation
        N = 4
        size = rr.size * N
        
        p = np.eye(size, size)
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)

        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    
    def test_regular_dihedral(self):
        space = FlipRot2dOnR2(5)

        g = space.fibergroup
        rr = g.regular_representation
        N = 4
        size = rr.size * N
        
        p = np.eye(size, size)
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)

        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    
    def test_mix_cyclic(self):
        space = Rot2dOnR2(6)

        g = space.fibergroup
        rr = directsum(list(g.representations.values()))
        
        N = 3
        size = rr.size * N
        
        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)
        
        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)
        
        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    
    def test_mix_dihedral(self):
        space = FlipRot2dOnR2(5)

        g = space.fibergroup
        rr = directsum(list(g.representations.values()))
        
        N = 3
        size = rr.size * N
        
        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)
        
        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)
        
        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    
    def test_mix_so2(self):
        space = Rot2dOnR2(-1, maximum_frequency=4)

        g = space.fibergroup
        rr = directsum(list(g.representations.values()))
        
        N = 3
        size = rr.size * N
        
        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)
        
        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)

        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    
    def test_mix_o2(self):
        space = FlipRot2dOnR2(-1, maximum_frequency=4)

        g = space.fibergroup
        rr = directsum(list(g.representations.values()))
        
        N = 3
        size = rr.size * N
        
        bcob = ortho_group.rvs(dim=size//5)
        bsize = bcob.shape[0]
        p = np.eye(size, size)
        
        for i in range(size//bsize):
            p[i*bsize:(i+1)*bsize, i*bsize:(i+1)*bsize] = bcob
        p = p[:, np.random.permutation(size)]
        repr = directsum([rr] * N, change_of_basis=p)
        
        cls = FieldType(space, [repr] * 8)
        el = DisentangleModule(cls)
        el.check_equivariance()
    

if __name__ == '__main__':
    unittest.main()
