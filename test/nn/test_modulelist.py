import unittest
from unittest import TestCase

import e2cnn.nn.init as init
from e2cnn.nn import *
from e2cnn.gspaces import *

import torch
from torch.optim import SGD
from torch import nn

import numpy as np


class TestModuleList(TestCase):
    
    def test_expand(self):
        
        for gs in [Rot2dOnR2(9), FlipRot2dOnR2(7), Flip2dOnR2(), TrivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values() if 'pointwise' in r.supported_nonlinearities]
            
            f_in = FieldType(gs, reprs)
            f_out = FieldType(gs, [gs.regular_repr] * 1)
            
            for i in range(20):
                net = [
                    R2Conv(f_in, f_in, 7, bias=True),
                    InnerBatchNorm(f_in, affine=True),
                    ReLU(f_in, inplace=True),
                    PointwiseMaxPool(f_in, 3, 2, 1),
                    R2Conv(f_in, f_out, 3, bias=True),
                    InnerBatchNorm(f_out, affine=False),
                    ELU(f_out, inplace=True),
                    GroupPooling(f_out),
                ]
                
                net1 = ModuleList(net[:5]).extend(net[5:])
                net2 = ModuleList(net)
                
                s1 = net1.state_dict()
                s2 = net2.state_dict()
                
                assert s1.keys() == s2.keys()
                for k in s1.keys():
                    assert torch.allclose(s1[k], s2[k])
    
    def test_export_modulelist(self):
    
        for gs in [Rot2dOnR2(9), FlipRot2dOnR2(7), Flip2dOnR2(), TrivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
            f_in = FieldType(gs, reprs)
            f_out = FieldType(gs, [gs.regular_repr]*1)
            
            for i in range(20):
                net = ModuleList([
                    R2Conv(f_in, f_in, 7, bias=True),
                    InnerBatchNorm(f_in, affine=True),
                    ReLU(f_in, inplace=True),
                    PointwiseMaxPool(f_in, 3, 2, 1),
                    R2Conv(f_in, f_out, 3, bias=True),
                    InnerBatchNorm(f_out, affine=False),
                    ELU(f_out, inplace=True),
                    GroupPooling(f_out),
                ])

                self.check_exported(net, atol=1e-6, rtol=5e-4)
                
                self.check_state_dict(net)

    def test_export_Sequential_example(self):
    
        s = Rot2dOnR2(8)
        c_in = FieldType(s, [s.trivial_repr] * 3)
        c_hid = FieldType(s, [s.regular_repr] * 3)
        c_out = FieldType(s, [s.regular_repr] * 1)
    
        net = ModuleList([
            R2Conv(c_in, c_hid, 5, bias=False),
            InnerBatchNorm(c_hid),
            ReLU(c_hid, inplace=True),
            PointwiseMaxPool(c_hid, kernel_size=3, stride=2, padding=1),
            R2Conv(c_hid, c_out, 3, bias=False),
            InnerBatchNorm(c_out),
            ELU(c_out, inplace=True),
            GroupPooling(c_out)
        ])
    
        print(net)
        print(net.export())
        
        self.check_exported(net)
        
        self.check_state_dict(net)

    def check_exported(self, modules: ModuleList, atol=1e-8, rtol=1e-5):
        
        equivariant = SequentialModule(*modules)
        equivariant = train(equivariant)
    
        in_size = equivariant.in_type.size

        conv_modules = modules.export()
        conventional = torch.nn.Sequential(*conv_modules)
        
        for _ in range(5):
            x = torch.randn(5, in_size, 31, 31)

            ye = equivariant(GeometricTensor(x, equivariant.in_type)).tensor
            yc = conventional(x)
            
            self.assertEqual(ye.shape, yc.shape, f"Tensor from the equivariant was shape {ye.shape}, but the tensor from the exported module was {yc.shape}")
            self.assertTrue(torch.allclose(ye, yc, atol=atol, rtol=rtol))

    def check_state_dict(self, modules: ModuleList, atol=1e-8, rtol=1e-5):
    
        equivariant = SequentialModule(*modules)
        equivariant = train(equivariant)

        conv_modules1 = modules.export()
        conventional1 = torch.nn.Sequential(*conv_modules1)

        equivariant = train(equivariant)

        conv_modules2 = modules.export()
        conventional2 = torch.nn.Sequential(*conv_modules2)
        state_dict = conv_modules2.state_dict()
        
        conv_modules1.load_state_dict(state_dict)
        
        # check the two versions are equivalent
        in_size = equivariant.in_type.size
        for _ in range(20):
            x = torch.randn(5, in_size, 31, 31)
        
            y1 = conventional1(x)
            y2 = conventional2(x)
        
            # print(torch.abs(y1-y2).max())
        
            self.assertTrue(torch.allclose(y1, y2, atol=atol, rtol=rtol))


def train(equivariant: EquivariantModule):

    in_size = equivariant.in_type.size

    equivariant.train()

    if len(list(equivariant.parameters())) > 0:
        sgd = SGD(equivariant.parameters(), lr=1e-3)
    else:
        sgd = None

    for i in range(5):
        x = torch.randn(5, in_size, 31, 31)
        x = GeometricTensor(x, equivariant.in_type)
    
        if sgd is not None:
            sgd.zero_grad()
    
        y = equivariant(x).tensor
        y = ((y - 1.) ** 2).mean()
    
        if sgd is not None:
            y.backward()
            sgd.step()
    
    return equivariant


if __name__ == '__main__':
    unittest.main()
