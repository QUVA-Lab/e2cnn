
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from e2cnn.nn.modules.equivariant_module import EquivariantModule
from e2cnn.nn.modules.utils import indexes_from_labels

import torch

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

__all__ = ["InducedNormPool"]


class InducedNormPool(EquivariantModule):
    
    def __init__(self, in_type: FieldType, **kwargs):
        r"""
        
        Module that implements Induced Norm Pooling.
        This module requires the input fields to be associated to an induced representation from a representation
        which supports 'norm' non-linearities.
        
        First, for each input field, an output one is built by taking the maximum norm of all its sub-fields.
        
        Args:
            in_type (FieldType): the input field type
            
        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(InducedNormPool, self).__init__()
        
        for r in in_type.representations:
            assert any(nl.startswith('induced_norm') for nl in r.supported_nonlinearities), \
                'Error! Representation "{}" does not support "induced_norm" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        
        # build the output representation substituting each input field with a trivial representation
        self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))

        # whether each group of fields is contiguous or not
        self._contiguous = {}

        # group fields by their size and the size of the subfields and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields
        
        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(lambda: [])
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(lambda: [])

        # number of fields of each size
        self._nfields = defaultdict(int)

        # whether each group of fields is contiguous or not
        self._contiguous = {}

        position = 0
        last_id = None
        for i, r in enumerate(self.in_type.representations):
            subfield_size = None
            for nl in r.supported_nonlinearities:
                if nl.startswith('induced_norm'):
                    assert subfield_size is None, "Error! The representation supports multiple " \
                                                  "sub-fields of different sizes"
                    subfield_size = int(nl.split('_')[-1])
                    assert r.size % subfield_size == 0
    
            id = (r.size, subfield_size)
    
            if id != last_id:
                self._contiguous[id] = not id in self._contiguous
    
            last_id = id
    
            _in_indices[id] += list(range(position, position + r.size))
            _out_indices[id] += [i]
            self._nfields[id] += 1
            position += r.size

        for id, contiguous in self._contiguous.items():
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _in_indices[id] = torch.LongTensor([min(_in_indices[id]), max(_in_indices[id])+1])
                _out_indices[id] = torch.LongTensor([min(_out_indices[id]), max(_out_indices[id])+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _in_indices[id] = torch.LongTensor(_in_indices[id])
                _out_indices[id] = torch.LongTensor(_out_indices[id])
        
            # register the indices tensors as parameters of this module
            self.register_buffer('in_indices_{}'.format(id), _in_indices[id])
            self.register_buffer('out_indices_{}'.format(id), _out_indices[id])

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
        
        for id, contiguous in self._contiguous.items():
            size, subfield_size = id
            n_subfields = size // subfield_size
            
            in_indices = getattr(self, f"in_indices_{id}")
            out_indices = getattr(self, f"out_indices_{id}")
            
            if contiguous:
                fm = input[:, in_indices[0]:in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]
                
            # split the channel dimension in 2 dimensions, separating fields
            fm, _ = fm.view(b, -1, n_subfields, subfield_size, h, w).norm(dim=3).max(dim=2)
            
            if contiguous:
                output[:, out_indices[0]:out_indices[1], ...] = fm
            else:
                output[:, out_indices, ...] = fm
        
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


