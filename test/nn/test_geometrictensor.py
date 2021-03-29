import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *
import torch
import random


class TestGeometricTensor(TestCase):
    
    def test_split(self):
    
        space = Rot2dOnR2(4)
        type = FieldType(space, [
            space.regular_repr,  # size = 4
            space.regular_repr,  # size = 4
            space.irrep(1),  # size = 2
            space.irrep(1),  # size = 2
            space.trivial_repr,  # size = 1
            space.trivial_repr,  # size = 1
            space.trivial_repr,  # size = 1
        ])  # sum = 15
    
        self.assertEqual(type.size, 15)
    
        geom_tensor = GeometricTensor(torch.randn(10, type.size, 7, 7), type)

        self.assertEqual(geom_tensor.shape, torch.Size([10, 15, 7, 7]))
    
        # split the tensor in 3 parts
        self.assertEqual(len(geom_tensor.split([4, 6])), 3)
    
        # the first contains
        # - the first 2 regular fields (2*4 = 8 channels)
        # - 2 vector fields (irrep(1)) (2*2 = 4 channels)
        # and, therefore, contains 12 channels
        self.assertEqual(geom_tensor.split([4, 6])[0].shape, torch.Size([10, 12, 7, 7]))
        self.assertEqual(geom_tensor.split([4, 6])[0].type, geom_tensor.type.index_select(list(range(4))))

        # the second contains only 2 scalar (trivial) fields (2*1 = 2 channels)
        self.assertEqual(geom_tensor.split([4, 6])[1].shape, torch.Size([10, 2, 7, 7]))
        self.assertEqual(geom_tensor.split([4, 6])[1].type, geom_tensor.type.index_select(list(range(4, 6))))

        # the last contains only 1 scalar (trivial) field (1*1 = 1 channels)
        self.assertEqual(geom_tensor.split([4, 6])[2].shape, torch.Size([10, 1, 7, 7]))
        self.assertEqual(geom_tensor.split([4, 6])[2].type, geom_tensor.type.index_select(list(range(6, len(geom_tensor.type)))))

        l1 = geom_tensor.split(list(range(1, len(geom_tensor.type))))
        l2 = geom_tensor.split(None)
        self.assertEqual(len(l1), len(l2))
        for t1, t2 in zip(l1, l2):
            self.assertEqual(t1.type, t2.type)
            self.assertTrue(torch.allclose(t1.tensor, t2.tensor))

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
            gs = FlipRot2dOnR2(N)
            for irr in gs.irreps.values():
                # with multiple fields
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
                    
                    # 1 single field

                    start = 1
                    end = 2
                    step = 1
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
                    
                    # index only one field
                    f = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f:f+1, ...].tensor,
                    ))
                    
                    # single index
                    f = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[1:2],
                        gt[1, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., 3:4],
                        gt[..., 3].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[..., 2:3, 3:4],
                        gt[..., 2, 3].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[3:4, ..., 2:3, 3:4],
                        gt[3, ..., 2, 3].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[1:2, :irr.size],
                        gt[1, 0, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[1:2, :irr.size, 4:5, 2:3],
                        gt[1, 0, 4, 2].tensor,
                    ))

                    # raise errors
                    with self.assertRaises(TypeError):
                        sliced = gt[2:5, 0:4, 1:7, 1:7, ...]
                        
                    with self.assertRaises(TypeError):
                        sliced = gt[[2, 4, 2], 0:4, ...]
                        
                    with self.assertRaises(TypeError):
                        sliced = gt[2, 0:4, range(3), range(3)]

                # with a single field
                F = 1
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
                    
                    # 1 single field
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:1, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0, ...].tensor,
                    ))

                    # negative index
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, -1, ...].tensor,
                    ))
                    
                    # with negative step
                    start = 0
                    end = -2
                    step = -1
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, start:end:step, ...].tensor,
                    ))

            for i in range(3):
                reprs = list(gs.representations.values())*3
    
                random.shuffle(reprs)
                type = FieldType(gs, reprs)
                F = len(type)
    
                t = torch.randn(3, type.size, 3, 4)
                gt = GeometricTensor(t, type)
                
                # assignment should not be allowed
                with self.assertRaises(TypeError):
                    gt[2, 1:3, ...] = torch.randn(gt[2, 1:3, ...].shape)
    
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

                # single index
                
                for f in range(F):
                
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[:, f, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[:, f].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[1:2,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[1, f, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[
                            1:2,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)],
                            3:4,
                            4:5
                        ],
                        gt[1, f, 3, 4].tensor,
                    ))

if __name__ == '__main__':
    unittest.main()
