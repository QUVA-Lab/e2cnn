
from collections import defaultdict

from torch.nn import Parameter

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Any

import numpy as np

__all__ = ["InducedNormNonLinearity"]


class InducedNormNonLinearity(EquivariantModule):
    
    def __init__(self, in_type: FieldType, function: str = 'n_relu', bias: bool = True):
        r"""
        
        Induced Norm non-linearities.
        This module requires the input fields to be associated to an induced representation from a representation
        which supports 'norm' non-linearities.
        This module applies a bias and an activation function over the norm of each sub-field of an induced field.
        The bias is shared among all sub-field of the same induced field.
        
        The input representation of the fields is preserved by this operation.
        
        Args:
            in_type (FieldType): the input field type
            function (str, optional): the identifier of the non-linearity. It is used to specify which function to
                                      apply. By default (``'n_relu'``), ReLU is used.
            bias (bool, optional): add bias to norm of fields before computing the non-linearity. Default: ``True``

        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(InducedNormNonLinearity, self).__init__()

        for r in in_type.representations:
            assert any(nl.startswith('induced_norm') for nl in r.supported_nonlinearities), \
                'Error! Representation "{}" does not support "induced_norm" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self._nfields = None
        self.log_bias = None

        if function == 'n_relu':
            self._function = torch.relu
        elif function == 'n_sigmoid':
            self._function = torch.sigmoid
        elif function == "squash":
            self._function = lambda t: t / (1.0 + t)
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
        # group fields by their size and the size of the subfields and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each size
        self._nfields = defaultdict(int)
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
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
            
            _indices[id] += list(range(position, position + r.size))
            self._nfields[id] += 1
            position += r.size
        
        for id, contiguous in self._contiguous.items():
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[id] = torch.LongTensor([min(_indices[id]), max(_indices[id])+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[id] = torch.LongTensor(_indices[id])
                
            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(id), _indices[id])
        
        if bias:
            # build a bias for each field
            self.log_bias = Parameter(torch.zeros(len(self.in_type), dtype=torch.float), requires_grad=True)
        else:
            self.log_bias = None
    
        # build a sorted list of the fields groups, such that every time they are iterated through in the same order
        self._order = sorted(self._contiguous.keys())
        
        self.eps = Parameter(torch.tensor(1e-10), requires_grad=False)
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Apply norm non-linearities to the input feature map
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        input = input.tensor
        
        # scalar multipliers needed to turn the old norms into the newly computed ones
        multipliers = torch.empty_like(input)
        
        b, c, h, w = input.shape
        
        next_bias = 0
        
        if self.log_bias is not None:
            # build the bias
            # biases = torch.nn.functional.elu(self.log_bias)
            biases = torch.exp(self.log_bias)
            # biases = torch.nn.functional.elu(self.log_bias) + 1
        else:
            biases = None
        
        # iterate through all field sizes
        for id in self._order:
            size, subfield_size = id
            
            # retrieve the corresponding fiber indices
            indices = getattr(self, f"indices_{id}")
            
            if self._contiguous[id]:
                # if the fields were contiguous, we can use slicing
                # retrieve the fields
                fm = input[:, indices[0]:indices[1], :, :]
            else:
                # otherwise we have to use indexing
                # retrieve the fields
                fm = input[:, indices, :, :]

            # compute the norm of each field
            norms = fm.view(b, -1, subfield_size, h, w).norm(dim=2, keepdim=True)
            
            # compute the new norms
            if biases is not None:
                n_subfields = int(size // subfield_size)
                
                # retrieve the bias elements corresponding to the current fields
                bias = biases[next_bias:next_bias + self._nfields[id]]
                # the bias is shared across all sub-fields in the same field
                bias = bias.view(1, -1, 1, 1, 1)
                bias = bias.expand(1, -1, n_subfields, 1, 1)
                bias = bias.reshape(1, -1, 1, 1, 1)
                new_norms = self._function(norms - bias)
            else:
                new_norms = self._function(norms)

            # compute the scalar multipliers needed to turn the old norms into the newly computed ones
            # m = torch.zeros_like(new_norms)
            # in order to avoid division by 0
            # mask = norms > 0.
            # m[mask] = new_norms[mask] / norms[mask]
            
            m = new_norms / torch.max(norms, self.eps)
            m[norms <= self.eps] = 0.

            if self._contiguous[id]:
                # expand the multipliers tensor to all channels for each field
                multipliers[:, indices[0]:indices[1], :, :] = m.expand(b, -1, subfield_size, h, w).reshape(b, -1, h, w)
            
            else:
                # expand the multipliers tensor to all channels for each field
                multipliers[:, indices, :, :] = m.expand(b, -1, subfield_size, h, w).reshape(b, -1, h, w)
            
            # shift the position on the bias tensor
            next_bias += self._nfields[id]
        
        # multiply the input by the multipliers computed and wrap the result in a GeometricTensor
        return GeometricTensor(input * multipliers, self.out_type)

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
