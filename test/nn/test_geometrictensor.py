import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *
import torch
import random


class TestGeometricTensor(TestCase):
    
    def test_sum(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr] * 3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor + t2.tensor
                    out2 = (t1 + t2).tensor
                    out3 = (t2 + t1).tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))
                    self.assertTrue(torch.allclose(out3, out2))
    
    def test_isum(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor + t2.tensor
                    t1 += t2
                    out2 = t1.tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))
    
    def test_sub(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr]*3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor - t2.tensor
                    out2 = (t1 - t2).tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))

    def test_isub(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                
                    out1 = t1.tensor - t2.tensor
                    t1 -= t2
                    out2 = t1.tensor
                
                    self.assertTrue(torch.allclose(out1, out2))

    def test_mul(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr] * 3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    s = 10*torch.randn(1)
                
                    out1 = t1.tensor * s
                    out2 = (s * t1).tensor
                    out3 = (t1 * s).tensor
                
                    self.assertTrue(torch.allclose(out1, out2))
                    self.assertTrue(torch.allclose(out3, out2))

    def test_imul(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    s = 10*torch.randn(1)

                    out1 = t1.tensor * s
                    t1 *= s
                    out2 = t1.tensor
                
                    self.assertTrue(torch.allclose(out1, out2))

    def test_slicing(self):
        for N in [2, 4, 7, 16]:
            gs = Rot2dOnR2(N)
            for irr in gs.irreps.values():
                F = 7
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    t = torch.randn(10, type.size, 11, 11)
                    gt = GeometricTensor(t, type)
        
                    # slice all dims except the channels
                    self.assertTrue(torch.allclose(
                        t[2:3, :, 2:7, 2:7],
                        gt[2:3, :, 2:7, 2:7].tensor,
                    ))
        
                    # slice only spatial dims
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[:, :, 2:7, 2:7].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[..., 2:7, 2:7].tensor,
                    ))
        
                    # slice only 1 spatial
                    self.assertTrue(torch.allclose(
                        t[..., 2:7],
                        gt[..., 2:7].tensor,
                    ))
        
                    # slice only batch
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4].tensor,
                    ))
        
                    # different ranges
                    self.assertTrue(torch.allclose(
                        t[:, :, 1:9:2, 0:8:3],
                        gt[..., 1:9:2, 0:8:3].tensor,
                    ))
        
                    # no slicing
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :, :].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[...].tensor,
                    ))
        
                    # raise error
                    with self.assertRaises(TypeError):
                        sliced = gt[2:5, 0:4, 1:7, 1:7, ...]
        
                    # slice channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 1 * irr.size:4 * irr.size:],
                        gt[:, 1:4, ...].tensor,
                    ))
                    # slice cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:7, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:7:1, ...].tensor,
                    ))
        
                    # with a larger step
                    start = 1
                    end = 6
                    step = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    start = 0
                    end = 7
                    step = 3
                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    # with negative step
                    start = 6
                    end = 1
                    step = -1
                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    start = 6
                    end = 1
                    step = -2
                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))

            for i in range(3):
                reprs = list(gs.representations.values())*3
                
                random.shuffle(reprs)
                type = FieldType(gs, reprs)
                F = len(type)
                
                t = torch.randn(3, type.size, 3, 4)
                gt = GeometricTensor(t, type)

                # no slicing
                self.assertTrue(torch.allclose(
                    t,
                    gt[:].tensor,
                ))

                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, :, :].tensor,
                ))

                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :].tensor,
                ))

                self.assertTrue(torch.allclose(
                    t,
                    gt[...].tensor,
                ))

                # slice channels with all fields of different types
                self.assertTrue(torch.allclose(
                    t[:, type.fields_start[1]:type.fields_end[3]:],
                    gt[:, 1:4, ...].tensor,
                ))
                
                # slice cover all channels
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, 0:F, ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, 0:F:1, ...].tensor,
                ))

                # with a larger step
                start = 1
                end = 6
                step = 2
                self.assertTrue(torch.allclose(
                    t[:,
                        [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))

                start = 0
                end = 7
                step = 3
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))

                # with negative step
                start = 6
                end = 1
                step = -1
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))

                start = 6
                end = 1
                step = -2
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))



if __name__ == '__main__':
    unittest.main()
