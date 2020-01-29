from typing import Tuple

import torch
import torch.nn.functional as F

import math

import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces

from argparse import ArgumentParser


__all__ = [
    "wrn16_8_stl_d8d4d1",
    "wrn16_8_stl_d8d4d4",
    "wrn16_8_stl_d1d1d1",
    "wrn28_10_d8d4d1",
    "wrn28_7_d8d4d1",
    "wrn28_10_c8c4c1",
    "wrn28_10_d1d1d1",
]


########################################################################################################################
# Code adapted from:
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
########################################################################################################################

def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=2,
            dilation=1, bias=False):
    """5x5 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding, 
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


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


class WideBasic(enn.EquivariantModule):
    
    def __init__(self,
                 in_type: enn.FieldType,
                 inner_type: enn.FieldType,
                 dropout_rate: float,
                 stride: int = 1,
                 out_type: enn.FieldType = None,
                 ):
        super(WideBasic, self).__init__()
        
        if out_type is None:
            out_type = in_type
        
        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type
        
        if isinstance(in_type.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_type.gspace.fibergroup.rotation_order
        elif isinstance(in_type.gspace, gspaces.Rot2dOnR2):
            rotations = in_type.gspace.fibergroup.order()
        else:
            rotations = 0
        
        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5
        
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


class Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=100,
                 N: int = 8,
                 r: int = 1,
                 f: bool = True,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 ):
        r"""
        
        Build and equivariant Wide ResNet.
        
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
        super(Wide_ResNet, self).__init__()
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        
        print(f'| Wide-Resnet {depth}x{k}')
        
        nStages = [16, 16 * k, 32 * k, 64 * k]
        
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
        r2 = FIELD_TYPE["regular"](self.gspace, nStages[0], fixparams=True)
        
        # dummy attribute keeping track of the output field type of the last submodule built, i.e. the input field type of
        # the next submodule to build
        self._in_type = r2
        
        self.conv1 = conv5x5(r1, r2)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=initial_stride)
        if self._r >= 2:
            N_new = N//2
            id = (0, N_new) if self._f else N_new
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x
        
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        if self._r == 3:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        elif self._r == 1:
            N_new = N // 2
            id = (0, N_new) if self._f else N_new
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x
        
        # last layer maps to a trivial (invariant) feature map
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2, totrivial=True)
        
        self.bn = enn.InnerBatchNorm(self.layer3.out_type, momentum=0.9)
        self.relu = enn.ReLU(self.bn.out_type, inplace=True)
        self.linear = torch.nn.Linear(self.bn.out_type.size, num_classes)
        
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
    
    def _restrict_layer(self, subgroup_id) -> enn.SequentialModule:
        layers = list()
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer
    
    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
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
        
        x2 = self.layer2(self.restrict1(x1))
        
        x3 = self.layer3(self.restrict2(x2))
        
        return x1, x2, x3
    
    def forward(self, x):

        # wrap the input tensor in a GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        out = self.layer1(out)
        
        out = self.layer2(self.restrict1(out))
        
        out = self.layer3(self.restrict2(out))
        
        out = self.bn(out)
        out = self.relu(out)
        
        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        out = out.tensor
        
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
    

def wrn16_8_stl_d8d4d1(**kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2
    
    The model's block are respectively D8, D4 and D1 equivariant.
    
    """
    return Wide_ResNet(16, 8, 0.3, initial_stride=2, N=8, f=True, r=3, **kwargs)


def wrn16_8_stl_d8d4d4(**kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2

    The model's block are respectively D8, D4 and D4 equivariant.

    """
    return Wide_ResNet(16, 8, 0.3, initial_stride=2, N=8, f=True, r=2, **kwargs)


def wrn16_8_stl_d1d1d1(**kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2

    The model's block are respectively D1, D1 and D1 equivariant.

    """
    return Wide_ResNet(16, 8, 0.3, initial_stride=2, N=1, f=True, r=0, **kwargs)


def wrn28_10_d8d4d1(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D8, D4 and D1 equivariant.

    """
    return Wide_ResNet(28, 10, 0.3, initial_stride=1, N=8, f=True, r=3, **kwargs)


def wrn28_7_d8d4d1(**kwargs):
    """Constructs a Wide ResNet 28-10 model
    
    The model's block are respectively D8, D4 and D1 equivariant.
    
    """
    return Wide_ResNet(28, 7, 0.3, initial_stride=1, N=8, f=True, r=3, **kwargs)


def wrn28_10_c8c4c1(**kwargs):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no reflection equivariance)
    
    The model's block are respectively C8, C4 and C1 equivariant.
    
    """
    return Wide_ResNet(28, 10, 0.3, initial_stride=1, N=8, f=False, r=3, **kwargs)


def wrn28_10_d1d1d1(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D1, D1 and D1 equivariant.

    """
    return Wide_ResNet(28, 10, 0.3, initial_stride=1, N=1, f=True, r=0, **kwargs)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--rot90', action='store_true', default=False, help='Makes the model invariant to rotations of 90 degrees')
    
    config = parser.parse_args()

    if config.rot90:
        # build a 90 degrees rotation and reflection invariant model (includes both vertical and horizontal reflections)
        m = Wide_ResNet(10, 4, 0.3, initial_stride=1, N=4, f=True, r=0, num_classes=10)
    else:
        # build a reflection invariant model (only reflections along the vertical axis)
        m = Wide_ResNet(10, 4, 0.3, initial_stride=1, N=4, f=True, r=3, num_classes=10)
        
    # Alternative, wider model equivariant to N=8 rotations and reflection
    # m = Wide_ResNet(10, 6, 0.3, initial_stride=1, N=8, f=True, r=0, num_classes=10)

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
    y = m(x)
    y_fv = m(x_fv)
    y_fh = m(x_fh)
    y90 = m(x90)
    y90_fh = m(x90_fh)

    # the outputs should be (about) the same for all transformations the model is invariant to
    print()
    print('TESTING INVARIANCE:                    ')
    print('REFLECTIONS along the VERTICAL axis:   ' + ('YES' if torch.allclose(y, y_fv, atol=1e-6) else 'NO'))
    print('REFLECTIONS along the HORIZONTAL axis: ' + ('YES' if torch.allclose(y, y_fh, atol=1e-6) else 'NO'))
    print('90 degrees ROTATIONS:                  ' + ('YES' if torch.allclose(y, y90, atol=1e-6) else 'NO'))
    print('REFLECTIONS along the 45 degrees axis: ' + ('YES' if torch.allclose(y, y90_fh, atol=1e-6) else 'NO'))
    


