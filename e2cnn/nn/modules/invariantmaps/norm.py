
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from e2cnn.nn.modules.equivariant_module import EquivariantModule
from e2cnn.nn.modules.utils import indexes_from_labels

import torch

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

__all__ = ["NormPool"]


class NormPool(EquivariantModule):
    
    def __init__(self, in_type: FieldType, **kwargs):
        r"""
        
        Module that implements Norm Pooling.
        For each input field, an output one is built by taking the norm of that field; as a result, the output
        field transforms according to a trivial representation.
        
        Args:
            in_type (FieldType): the input field type
            
        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(NormPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        
        # build the output representation substituting each input field with a trivial representation
        self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))

        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(lambda: [])
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(lambda: [])

        # whether each group of fields is contiguous or not
        self._contiguous = {}

        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields
        indeces = indexes_from_labels(in_type, [r.size for r in in_type.representations])

        for s, (contiguous, fields, idxs) in indeces.items():
            self._contiguous[s] = contiguous
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _in_indices[s] = torch.LongTensor([min(idxs), max(idxs)+1])
                _out_indices[s] = torch.LongTensor([min(fields), max(fields)+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _in_indices[s] = torch.LongTensor(idxs)
                _out_indices[s] = torch.LongTensor(fields)
        
            # register the indices tensors as parameters of this module
            self.register_buffer('in_indices_{}'.format(s), _in_indices[s])
            self.register_buffer('out_indices_{}'.format(s), _out_indices[s])

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply the Norm Pooling to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type

        input = input.tensor
        b, c, h, w = input.shape
        
        output = torch.empty(self.evaluate_output_shape(input.shape), device=input.device, dtype=torch.float)
        
        for s, contiguous in self._contiguous.items():
            
            in_indices = getattr(self, f"in_indices_{s}")
            out_indices = getattr(self, f"out_indices_{s}")
            
            if contiguous:
                fm = input[:, in_indices[0]:in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]
                
            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, h, w)
            
            if contiguous:
                output[:, out_indices[0]:out_indices[1], ...] = fm.norm(dim=2)
            else:
                output[:, out_indices, ...] = fm.norm(dim=2)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
    
        x = torch.randn(3, c, 10, 10)
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors


