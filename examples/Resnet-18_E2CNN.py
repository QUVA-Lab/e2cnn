from typing import Tuple, List, Any, Union

import e2cnn.nn as enn
from e2cnn import gspaces
from e2cnn.nn import init
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from e2cnn.nn import EquivariantModule
from e2cnn.gspaces import *

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from argparse import ArgumentParser

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )
    
def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    
    N = gspace.fibergroup.order()
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = planes / N
    planes = int(planes)
    
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""
    
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())
        
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)



FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}       

class BasicBlock(enn.EquivariantModule):
    
    def __init__(self,
                 in_type: enn.FieldType,
                 inner_type: enn.FieldType,
                 dropout_rate: float,
                 stride: int = 1,
                 out_type: enn.FieldType = None,
                 ):
        super(BasicBlock, self).__init__()
        
        if out_type is None:
            out_type = in_type
        
        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type
        
        conv = conv3x3
        
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_type)
        
        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type, inplace=True)
        
        self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)
        
        self.conv2 = conv(inner_type, self.out_type, stride=stride)
        
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False)
    
    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x

        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class ResNet18(torch.nn.Module):
    def __init__(self, dropout_rate, num_classes=100,
                 N: int = 4,
                 r: int = 0,
                 f: bool = False,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 ):
        r"""
        
        Build and equivariant ResNet-18.
        
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
        super(ResNet18, self).__init__()
        
        nStages = [16, 16, 32, 64, 128]
        
        self._fixparams = fixparams
        
        self._layer = 0
        
        # number of discrete rotations to be equivariant to
        self._N = N
        
        # if the model is [F]lip equivariant
        self._f = f
        if self._f:
            if N != 1:
                self.gspace = gspaces.FlipRot2dOnR2(N)
            else:
                self.gspace = gspaces.Flip2dOnR2()
        else:
            if N != 1:
                self.gspace = gspaces.Rot2dOnR2(N)
            else:
                self.gspace = gspaces.TrivialOnR2()

        # level of [R]estriction:
        #   r = 0: never do restriction, i.e. initial group (either DN or CN) preserved for the whole network
        #   r = 1: restrict before the last block, i.e. initial group (either DN or CN) preserved for the first
        #          2 blocks, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last block
        #   r = 2: restrict after the first block, i.e. initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last 2 blocks
        #   r = 3: restrict after each block. Initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the second block and to 1 rotation
        #          in the last one (D1 or C1)
        assert r in [0, 1, 2, 3]
        self._r = r
        
        # the input has 3 color channels (RGB).
        # Color channels are trivial fields and don't transform when the input is rotated or flipped
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        
        # input field type of the model
        self.in_type = r1
        
        # in the first layer we always scale up the output channels to allow for enough independent filters
        r2 = FIELD_TYPE["regular"](self.gspace, nStages[0], fixparams=self._fixparams)
        
        # dummy attribute keeping track of the output field type of the last submodule built, i.e. the input field type of
        # the next submodule to build
        self._in_type = r2
        
        # Number of blocks per layer
        n = 2

        self.conv1 = conv3x3(r1, r2)
        self.layer1 = self.basicLayer(BasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self.basicLayer(BasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self.basicLayer(BasicBlock, nStages[3], n, dropout_rate, stride=2)
        # last layer maps to a trivial (invariant) feature map
        self.layer4 = self.basicLayer(BasicBlock, nStages[4], n, dropout_rate, stride=2, totrivial=True)
        
        self.bn = enn.InnerBatchNorm(self.layer4.out_type, momentum=0.9)
        self.relu = enn.ReLU(self.bn.out_type, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gpool = enn.GroupPooling(self.bn.out_type)
        self.linear = torch.nn.Linear(self.gpool.out_type.size, num_classes)
        
        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(module.weights, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        for i, (name, mod) in enumerate(self.named_modules()):
            print(f"\t{i} - {name}")
    
    def basicLayer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    totrivial: bool = False
                    ) -> enn.SequentialModule:
    
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)
        inner_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)
        
        if totrivial:
            out_type = FIELD_TYPE["trivial"](self.gspace, planes, fixparams=self._fixparams)
        else:
            out_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(block(self._in_type, inner_type, dropout_rate, stride, out_type=out_f))
            self._in_type = out_f
            
        print("layer", self._layer, "built")
        return enn.SequentialModule(*layers)
    
    def features(self, x):
        
        x = enn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        
        x1 = self.layer1(out)
        
        x2 = self.layer2(x1)
        
        x3 = self.layer3(x2)

        x4 = self.layer4(x3)
        
        return x1, x2, x3, x4
    
    def forward(self, x):

        # wrap the input tensor in a GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.gpool(out)

        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        out = out.tensor
        gpool_out = out
        
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out, gpool_out
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--rot90', action='store_true', default=False, help='Makes the model equivariant to rotations of 90 degrees')
    parser.add_argument('--reflection', action='store_true', default=False, help='Makes the model equivariant to horizontal and vertical reflections')
    
    config = parser.parse_args()

    if config.rot90:
        if config.reflection:
            m = ResNet18(0.3, initial_stride=1, N=4, f=True, r=0, num_classes=10)
        else:
            m = ResNet18(0.3, initial_stride=1, N=4, f=False, r=0, num_classes=10)
    else:
        m = ResNet18(0.3, initial_stride=1, N=4, f=True , r=3, num_classes=10)

    m.eval()
    
    # 3 random 33x33 RGB images (i.e. with 3 channel)
    x = torch.randn(3, 3, 33, 33)

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
