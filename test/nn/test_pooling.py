import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import torch

import numpy as np


class TestPooling(TestCase):
    
    def test_pointwise_maxpooling(self):
        
        N = 8
        g = Rot2dOnR2(N)
        
        r = FieldType(g, [repr for repr in g.representations.values() if 'pointwise' in repr.supported_nonlinearities] * 3)
        
        mpl = PointwiseMaxPool(r, kernel_size=(3, 1), stride=(2, 2))

        x = torch.randn(3, r.size, 10, 15)

        x = GeometricTensor(x, r)

        for el in g.testing_elements:
            
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_norm_maxpooling(self):
    
        N = 8
        g = Rot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values()) * 3)
        
        print(r.size)
    
        mpl = NormMaxPool(r, kernel_size=(3, 1), stride=(2, 2))
    
        x = torch.randn(3, r.size, 10, 15)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_orientation_pooling(self):
        
        N = 8
        g = Rot2dOnR2(N)
        
        r = FieldType(g, [repr for repr in g.representations.values() if 'pointwise' in repr.supported_nonlinearities] * 3)
        
        mpl = GroupPooling(r)

        x = torch.randn(3, r.size, 10, 15)

        x = GeometricTensor(x, r)

        for el in g.testing_elements:
            
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_norm_pooling(self):
    
        N = 8
        g = Rot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values()) * 3)
    
        mpl = NormPool(r)
    
        x = torch.randn(3, r.size, 10, 15)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_induced_norm_pooling(self):
    
        N = 8
        g = FlipRot2dOnR2(-1, 6)
        
        sgid = (None, -1)
        sg, _, _ = g.restrict(sgid)
    
        r = FieldType(g, list(g.induced_repr(sgid, r) for r in sg.representations.values() if not r.is_trivial()) * 3)
    
        mpl = InducedNormPool(r)
        W, H = 10, 15
        x = torch.randn(3, r.size, W, H)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            
            expected_out, _ = x.tensor.view(-1, len(r), 2, 2, W, H).norm(dim=3).max(dim=2)
            
            out1 = mpl(x)
            
            self.assertTrue(torch.allclose(expected_out, out1.tensor))
            
            out1 = out1.transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())


if __name__ == '__main__':
    unittest.main()
