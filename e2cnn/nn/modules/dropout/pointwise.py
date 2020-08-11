
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch.nn.functional as F
import torch
from typing import List, Tuple, Any

__all__ = ["PointwiseDropout"]


class PointwiseDropout(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 p: float = 0.5,
                 inplace: bool = False
                 ):
        r"""
        
        Applies dropout to individual *channels* independently.
        
        This class is just a wrapper for :func:`torch.nn.functional.dropout` in an :class:`~e2cnn.nn.EquivariantModule`.
        
        Only representations supporting pointwise non-linearities are accepted as input field type.
        
        Args:
            in_type (FieldType): the input field type
            p (float, optional): dropout probability
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
            
        """
    
        assert isinstance(in_type.gspace, GeneralOnR2)
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))

        super(PointwiseDropout, self).__init__()

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.p = p
        self.inplace = inplace
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        output = F.dropout(input.tensor, self.p, self.training, self.inplace)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(InnerBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Dropout` module and set to "eval" mode.

        """
    
        self.eval()
        
        return torch.nn.Dropout(self.p, self.inplace).eval()

