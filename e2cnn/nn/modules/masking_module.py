
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from .equivariant_module import EquivariantModule

from typing import Tuple

import torch

import math

__all__ = ["MaskModule"]


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(EquivariantModule):

    def __init__(self, in_type: FieldType, S: int, margin: float = 0.):
        r"""
        
        Performs an element-wise multiplication of the input with a *mask* of shape ``S x S``.
        
        The mask has value :math:`1` in all pixels with distance smaller than ``(S-1)/2 * (1 - margin)/100`` from the
        center of the mask and :math:`0` elsewhere. Values change smoothly between the two regions.
        
        This operation is useful to remove from an input image or feature map all the part of the signal defined on the
        pixels which lay outside the circle inscribed in the grid.
        Because a rotation would move these pixels outside the grid, this information would anyways be
        discarded when rotating an image. However, allowing a model to use this information might break the guaranteed
        equivariance as rotated and non-rotated inputs have different information content.
        
        
        .. note::
        
            In order to perform the masking, the module expects an input with the same spatial dimensions as the mask.
            Then, input tensors must have shape ``B x C x S x S``.
        
        
        Args:
            in_type (FieldType): input field type
            S (int): the shape of the mask and the expected inputs
            margin (float, optional): margin around the mask in percentage with respect to the radius of the mask
                                      (default ``0.``)
        
        """
        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)

        self.in_type = self.out_type = in_type

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        assert input.tensor.shape[2:] == self.mask.shape[2:]

        out = input.tensor * self.mask
        return GeometricTensor(out, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape


