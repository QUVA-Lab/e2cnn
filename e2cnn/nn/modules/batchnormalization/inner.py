
from typing import List, Tuple, Any

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import BatchNorm3d

from ..utils import indexes_from_labels

__all__ = ["InnerBatchNorm"]


class InnerBatchNorm(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 ):
        r"""
        
        Batch normalization for representations with permutation matrices.
        
        Statistics are computed both over the batch and the spatial dimensions and over the channels within
        the same field (which are permuted by the representation).
        
        Only representations supporting pointwise non-linearities are accepted as input field type.
        
        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional):  if ``True``, this module has learnable affine parameters. Default: ``True``
            
        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(InnerBatchNorm, self).__init__()

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        
        # group fields by their size and
        #   - check if fields with the same size are contiguous
        #   - retrieve the indices of the fields
        grouped_fields = indexes_from_labels(self.in_type, [r.size for r in self.in_type.representations])

        # number of fields of each size
        self._nfields = {}
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = {}
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        for s, (contiguous, fields, indices) in grouped_fields.items():
            self._nfields[s] = len(fields)
            self._contiguous[s] = contiguous
            
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[s] = torch.LongTensor([min(indices), max(indices)+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[s] = torch.LongTensor(indices)
                
            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(s), _indices[s])
        
        for s in _indices.keys():
            _batchnorm = BatchNorm3d(self._nfields[s], eps, momentum, affine=self.affine)
            self.add_module('batch_norm_[{}]'.format(s), _batchnorm)
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        b, c, h, w = input.tensor.shape
        
        output = torch.empty_like(input.tensor)
        
        # iterate through all field sizes
        for s, contiguous in self._contiguous.items():
            
            indices = getattr(self, f"indices_{s}")
            batchnorm = getattr(self, f'batch_norm_[{s}]')
            
            if contiguous:
                # if the fields were contiguous, we can use slicing
                output[:, indices[0]:indices[1], :, :] = batchnorm(
                    input.tensor[:, indices[0]:indices[1], :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] = batchnorm(
                    input.tensor[:, indices, :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(InnerBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass
