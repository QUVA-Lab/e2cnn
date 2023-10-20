from typing import List, Tuple, Any, Union

import math
import numpy as np
import torch
from torch.nn import AdaptiveAvgPool2d
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn
from e2cnn.nn import init
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from argparse import ArgumentParser
from e2cnn.nn import EquivariantModule
from e2cnn.gspaces import *

from argparse import ArgumentParser

class VGG11(torch.nn.Module):
    
    def __init__(self, dropout_rate, num_classes=100, 
                 N: int = 4,
                 r: int = 0,
                 f: bool = False,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 ):

        r"""
        
        Build equivariant VGG-11.
        
        The parameter ``N`` controls rotation equivariance and the parameter ``f`` reflection equivariance.
        
        More precisely, ``N`` is the number of discrete rotations the model is initially equivariant to.
        ``N = 1`` means the model is only reflection equivariant from the beginning.
        
        ``f`` is a boolean flag specifying whether the model should be reflection equivariant or not.
        If it is ``False``, the model is not reflection equivariant.
        
        ``r`` is the restriction level:
        
        - ``0``: no restriction. The model is equivariant to ``N`` rotations from the input to the output
        - ``1``: restriction before the last block. The model is equivariant to ``N`` rotations before the last block
               (i.e. in the first 2 blocks). Then it is restricted to ``N/2`` rotations until the output.
        
        - ``2``: restriction after the first block. The model is equivariant to ``N`` rotations in the first block.
               Then it is restricted to ``N/2`` rotations until the output (i.e. in the last 3 blocks).
               
        - ``3``: restriction after the first and the second block. The model is equivariant to ``N`` rotations in the first
               block. It is restricted to ``N/2`` rotations before the second block and to ``1`` rotations before the last
               block.
        
        NOTICE: if restriction to ``N/2`` is performed, ``N`` needs to be even!
        
        """

        super(VGG11, self).__init__()
        self.nStages = [16, 32, 64, 64, 128, 128, 128, 128, 1024]

        # number of discrete rotations to be equivariant to
        self._N = N
        
        # if the model is [F]lip equivariant
        self._f = f
        if self._f:
            if self._N != 1:
                self.gspace = gspaces.FlipRot2dOnR2(N)
            else:
                self.gspace = gspaces.Flip2dOnR2()
        else:
            if self._N != 1:
                self.gspace = gspaces.Rot2dOnR2(N)
            else:
                self.gspace = gspaces.TrivialOnR2()

        self.input_type = nn.FieldType(self.gspace, 3*[self.gspace.trivial_repr])
        
        #ConvBlock-1 16x3x3
        in_type = self.input_type
        out_type = nn.FieldType(self.gspace, self.nStages[0]*[self.gspace.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #MaxPool-1 2x2
        self.pool1 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size  = 2, stride = 2)
        )

        #ConvBlock-2 32x3x3
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[1]*[self.gspace.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )    

        #MaxPool-2 2x2
        self.pool2 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size = 2, stride = 2)
        )

        #ConvBlock-3 64x3x3
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[2]*[self.gspace.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #ConvBlock-4 64x3x3
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[3]*[self.gspace.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #MaxPool-3 2x2
        self.pool3 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size  = 2, stride = 2)
        )

        #ConvBlock-5 128x3x3
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[4]*[self.gspace.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #ConvBlock-6 128x3x3
        in_type = self.block5.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[5]*[self.gspace.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #MaxPool-4 2x2
        self.pool4 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size  = 2, stride = 2)
        )

        #ConvBlock-7 128x3x3
        in_type = self.block6.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[6]*[self.gspace.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )

        #ConvBlock-8 128x3x3
        in_type = self.block7.out_type
        out_type = nn.FieldType(self.gspace, self.nStages[7]*[self.gspace.regular_repr])
        self.block8 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size = 3, padding = 1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ELU(out_type, inplace=True)
        )
        
        self.gpool = nn.GroupPooling(out_type)
        c = self.gpool.out_type.size
        self.pool5=AdaptiveAvgPool2d(1)

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, self.nStages[8]),
            torch.nn.BatchNorm1d(self.nStages[8]),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.nStages[8], num_classes),
        )
        
    
    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)
        out = self.block1(x)
        out = self.pool1(out)
        
        out = self.block2(out)
        out = self.pool2(out)
        
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool3(out)

        out = self.block5(out)
        out = self.block6(out)
        out = self.pool4(out)

        out = self.block7(out)
        out = self.block8(out)

        out = self.gpool(out)

        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        out = out.tensor
        gpool_out = out

        out = self.pool5(out)
        out = self.fully_net(out.reshape(out.shape[0], -1))
        
        return out, gpool_out


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--rot90', action='store_true', default=False, help='Makes the model equivariant to rotations of 90 degrees')
    parser.add_argument('--reflection', action='store_true', default=False, help='Makes the model equivariant to horizontal and vertical reflections')
    
    config = parser.parse_args()

    if config.rot90:
        if config.reflection:
            m = VGG11(0.3, N=4, f=True, r=0, num_classes=10)
        else:
            m = VGG11(0.3, N=4, f=False, r=0, num_classes=10)
    else:
        m = VGG11(0.3, N=4, f=True , r=3, num_classes=10)

    m.eval()
    
    # 3 random 64x64 RGB images (i.e. with 3 channel)
    x = torch.randn(3, 3, 64, 64)

    # the images flipped along the vertical axis
    x_fv = x.flip(dims=[3])
    # the images flipped along the horizontal axis
    x_fh = x.flip(dims=[2])
    # the images rotated by 90 degrees
    x90 = x.rot90(1, (2, 3))
    # the images flipped along the horizontal axis and rotated by 90 degrees
    x90_fh = x.flip(dims=[2]).rot90(1, (2, 3))

    # feed all inputs to the model
    y, gpool_out = m(x)

    y_fv, gpool_out_fv = m(x_fv)
    
    y_fh, gpool_out_fh = m(x_fh)
    
    y90, gpool_out90 = m(x90)
    
    y90_fh, gpool_out90_fh = m(x90_fh)
  
    # the outputs of group pooling layers should be (about) the same for all transformations the model is equivariant to
    print()
    print('TESTING G-POOL EQUIVARIANCE:                  ')
    print('REFLECTIONS along the VERTICAL axis:   ' + ('YES' if torch.allclose(gpool_out, gpool_out_fv.flip(dims=[3]), atol=1e-5) else 'NO'))
    print('REFLECTIONS along the HORIZONTAL axis: ' + ('YES' if torch.allclose(gpool_out, gpool_out_fh.flip(dims=[2]), atol=1e-5) else 'NO'))
    print('90 degrees ROTATIONS:                  ' + ('YES' if torch.allclose(gpool_out, gpool_out90.rot90(-1, (2, 3)), atol=1e-5) else 'NO'))
    print('REFLECTIONS along the 45 degrees axis: ' + ('YES' if torch.allclose(gpool_out, gpool_out90_fh.rot90(-1, (2, 3)).flip(dims=[2]), atol=1e-5) else 'NO'))

    # the final outputs (y) should be (about) the same for all transformations the model is invariant to
    print()
    print('TESTING FINAL INVARIANCE:                    ')
    print('REFLECTIONS along the VERTICAL axis:   ' + ('YES' if torch.allclose(y, y_fv, atol=1e-5) else 'NO'))
    print('REFLECTIONS along the HORIZONTAL axis: ' + ('YES' if torch.allclose(y, y_fh, atol=1e-5) else 'NO'))
    print('90 degrees ROTATIONS:                  ' + ('YES' if torch.allclose(y, y90, atol=1e-5) else 'NO'))
    print('REFLECTIONS along the 45 degrees axis: ' + ('YES' if torch.allclose(y, y90_fh, atol=1e-5) else 'NO'))
