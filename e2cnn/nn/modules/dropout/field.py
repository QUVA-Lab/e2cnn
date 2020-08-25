
from collections import defaultdict


from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from typing import List, Tuple, Any


__all__ = ["FieldDropout"]


def dropout_field(input: torch.Tensor, p: float, training: bool, inplace: bool):
    
    if training:
        shape = list(input.size())
        shape[2] = 1
        
        if input.device == torch.device('cpu'):
            mask = torch.FloatTensor(*shape)
        else:
            device = input.device
            mask = torch.cuda.FloatTensor(*shape, device=device)
        
        mask = mask.uniform_() > p
        mask = mask.to(torch.float)
        
        if inplace:
            input *= mask / (1. - p)
            return input
        else:
            return input * mask / (1. - p)
    else:
        return input


class FieldDropout(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 p: float = 0.5,
                 inplace: bool = False
                 ):
        r"""
        
        Applies dropout to individual *fields* independently.
        
        Notice that, with respect to :class:`~e2cnn.nn.PointwiseDropout`, this module acts on a whole field instead
        of single channels.
        
        Args:
            in_type (FieldType): the input field type
            p (float, optional): dropout probability
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
            
        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        
        super(FieldDropout, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.p = p
        self.inplace = inplace
        
        self._nfields = None
        
        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each size
        self._nfields = defaultdict(int)
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        position = 0
        last_size = None
        for i, r in enumerate(self.in_type.representations):
            
            if r.size != last_size:
                if not r.size in self._contiguous:
                    self._contiguous[r.size] = True
                else:
                    self._contiguous[r.size] = False
            last_size = r.size
            
            _indices[r.size] += list(range(position, position + r.size))
            self._nfields[r.size] += 1
            position += r.size
        
        for s, contiguous in self._contiguous.items():
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[s] = torch.LongTensor([min(_indices[s]), max(_indices[s])+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[s] = torch.LongTensor(_indices[s])
                
            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(s), _indices[s])
        
        self._order = list(self._contiguous.keys())
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        if not self.training:
            return input
        
        input = input.tensor
        
        if not self.inplace:
            output = torch.empty_like(input)

        # iterate through all field sizes
        for s in self._order:
            
            indices = getattr(self, f"indices_{s}")
            
            shape = input.shape[:1] + (self._nfields[s], s) + input.shape[2:]
            
            if self._contiguous[s]:
                # if the fields are contiguous, we can use slicing
                out = dropout_field(input[:, indices[0]:indices[1], ...].view(shape), self.p, self.training, self.inplace)
                if not self.inplace:
                    shape = input.shape[:1] + (self._nfields[s] * s, ) + input.shape[2:]
                    output[:, indices[0]:indices[1], ...] = out.view(shape)
            else:
                # otherwise we have to use indexing
                out = dropout_field(input[:, indices, ...].view(shape), self.p, self.training, self.inplace)
                if not self.inplace:
                    shape = input.shape[:1] + (self._nfields[s] * s, ) + input.shape[2:]
                    output[:, indices, ...] = out.view(shape)
            
        if self.inplace:
            output = input
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        return input_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass
