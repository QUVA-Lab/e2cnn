import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import torch
import torch.nn.functional as F
import numpy as np

import random


class TestDropout(TestCase):
    
    def test_pointwise_do_unsorted_inplace(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]*3)
        
        do = PointwiseDropout(r, inplace=True)
        
        self.check_do(do)
    
    def test_pointwise_do_unsorted(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]*3)
        
        do = PointwiseDropout(r)
        
        self.check_do(do)
    
    def test_pointwise_do_sorted_inplace(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]*3).sorted()
        
        do = PointwiseDropout(r, inplace=True)
        
        self.check_do(do)
    
    def test_pointwise_do_sorted(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]*3).sorted()

        do = PointwiseDropout(r)
        
        self.check_do(do)

    def test_field_do_sorted(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values())*3).sorted()
    
        bn = FieldDropout(r)
    
        self.check_do(bn)

    def test_field_do_unsorted(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values())*3)
    
        bn = FieldDropout(r)
    
        self.check_do(bn)

    def test_field_do_sorted_inplace(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values())*3).sorted()
    
        bn = FieldDropout(r, inplace=True)
    
        self.check_do(bn)

    def test_field_do_unsorted_inplace(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values())*3)
        
        bn = FieldDropout(r, inplace=True)
        
        self.check_do(bn)

    def check_do(self, do: EquivariantModule):
    
        x = 5 * torch.randn(3000, do.in_type.size, 20, 20) + 10
        x = torch.abs(x)
        x1 = x
        x2 = x.clone()
        x1 = GeometricTensor(x1, do.in_type)
        x2 = GeometricTensor(x2, do.in_type)

        do.train()
        
        y1 = do(x1)
        
        do.eval()
        
        y2 = do(x2)
        
        y1 = y1.tensor.permute(1, 0, 2, 3).reshape(do.in_type.size, -1)
        y2 = y2.tensor.permute(1, 0, 2, 3).reshape(do.in_type.size, -1)
        
        m1 = y1.mean(1)
        m2 = y2.mean(1)
        
        # print(m1)
        # print(m2)

        self.assertTrue(torch.allclose(m1, m2, rtol=5e-3, atol=5e-3))


if __name__ == '__main__':
    unittest.main()
