

from typing import List, Tuple, Any

import numpy as np

from collections import defaultdict

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch

from torch.nn import Parameter

from .gated1 import GATED_ID, GATES_ID

__all__ = ["GatedNonLinearity2"]


class GatedNonLinearity2(EquivariantModule):
    
    def __init__(self,
                 in_type: Tuple[FieldType, FieldType],
                 **kwargs
                 ):
        r"""
        
        Gated non-linearities.
        This module applies a bias and a sigmoid function of the gates fields and, then, multiplies each gated
        field by one of the gates.
        
        The input representation of the gated fields is preserved by this operation while the gate fields are
        discarded.
        
        The gates and the gated fields are provided in two different tensors: :attr:`in_repr` is a tuple containing two
        representations: the representation of the tensor containing only the gates (which have to be trivial fields)
        and the representation of the tensor containing only the gated fields. Therefore, two tensors need to be
        provided as input to the ``forward`` method: the first contains the gates and the second the gated fields.
        Finally, notice that the number of gates and the number of gated fields have to match, i.e. the two
        representations need to have the same number of fields.
        
        .. todo ::
            This module has 2 input tensors and, so, two input field types.
            EquivariantModule only supports one input though.
            Fix this.
        
        Args:
            in_type (Tuple): a pair containing, in order, the field type of the gates and the field type
                             of the gated fields
            
        """

        assert len(in_type) == 2
        assert isinstance(in_type[0], FieldType)
        assert isinstance(in_type[1], FieldType)
        
        assert isinstance(in_type[0].gspace, GeneralOnR2)
        assert in_type[0].gspace == in_type[1].gspace

        # the first is the representation of the gates
        for r in in_type[0].representations:
            # assert GATES_ID in r.supported_nonlinearities, \
            assert r.is_trivial(), \
                "Error! Representation \"{}\" can't be a \"gate\"".format(r.name)

        # the second is the representation of the gated fields
        for r in in_type[1].representations:
            assert GATED_ID in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "gated" non-linearity'.format(r.name)

        assert len(in_type[0]) == len(in_type[1]), \
            'Error! Number of gates ({}) does not match the number of gated non-linearities required ({})' \
                .format(len(in_type[0]), len(in_type[1]))
        
        super(GatedNonLinearity2, self).__init__()

        self.space = in_type[0].gspace
        
        self._contiguous = {}
        _input_indices = defaultdict(lambda: [])
        
        self._nfields = defaultdict(int)
        
        self.branching = None

        self.in_type = in_type
        self.out_type = in_type[1]
        
        ngates = len(in_type[0])
        
        # group fields by their size, check if fields with the same size are consecutive
        # and retrieve the indices of the fields
        last_position = 0
        last_size = None
        for r in in_type[1].representations:

            if r.size != last_size:
                if not r.size in self._contiguous:
                    self._contiguous[r.size] = True
                else:
                    self._contiguous[r.size] = False
            last_size = r.size
            
            self._nfields[r.size] += 1
            _input_indices[r.size] += list(range(last_position, last_position + r.size))
            last_position += r.size

        _input_indices = dict(_input_indices)
        
        for s, contiguous in self._contiguous.items():
            if contiguous:
                # if the fields are contiguous, only the first and last indices are kept
                _input_indices[s] = torch.LongTensor([min(_input_indices[s]), max(_input_indices[s])+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _input_indices[s] = torch.LongTensor(_input_indices[s])
                
            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(s), _input_indices[s])
        
        # the gated fields are preserved and, therefore, the output representation is the same
        # as the gated fields input representation
        
        # build a sorted list of the fields groups, such that every time they are iterated through in the same order
        self._order = sorted(_input_indices.keys())
        
        # the bias for the gates
        self.bias = Parameter(torch.randn(1, ngates, 1, 1, dtype=torch.float), requires_grad=True)

    def forward(self, gates: GeometricTensor, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply the gated non-linearity to the input feature map.
        
        Args:
            gates (GeometricTensor): feature map corresponding to the gates
            input (GeometricTensor): input feature map corresponding to the gated fields

        Returns:
            the resulting feature map
            
        """
        
        assert isinstance(input, GeometricTensor)
        assert isinstance(gates, GeometricTensor)
        assert gates.type == self.in_type[0]
        assert input.type == self.in_type[1]
        
        gates = gates.tensor
        input = input.tensor
        
        # transform the gates
        gates = torch.sigmoid(gates - self.bias)
        
        b, c, h, w = input.shape
        
        # build the output tensor
        output = torch.empty(b, self.out_type.size, h, w, dtype=torch.float, device=self.bias.device)

        next_gate = 0
        
        # for each field size
        for size in self._order:
            
            indices = getattr(self, f"indices_{size}")
            
            # retrieve the needed gates
            g = gates[:, next_gate:next_gate + self._nfields[size], ...].view(b, -1, 1, h, w)
            
            if self._contiguous[size]:
                # if the fields were contiguous, we can use slicing
                output[:, indices[0]:indices[1], ...] =\
                    (
                            input[:, indices[0]:indices[1], ...]
                            .view(b, -1, size, h, w)
                            * g
                    ).view(b, -1, h, w)
            
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] = \
                    (
                            input[:, indices, ...]
                            .view(b, -1, size, h, w)
                            * g
                    ).view(b, -1, h, w)
            
            # shift the position on the gates fiber
            next_gate += self._nfields[size]
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, *input_shapes: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shapes) == 2
        gates_shape, input_shape = input_shapes
        
        assert len(gates_shape) == 4
        assert gates_shape[1] == self.in_type[0].size

        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type[1].size
        
        assert gates_shape[0] == input_shape[0]
        assert gates_shape[2:] == input_shape[2:]

        b, c, hi, wi = input_shape

        return b, self.out_type.size, hi, wi
        
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = [self.in_type[i].size for i in range(2)]

        x = [torch.randn(3, c[i], 10, 10) for i in range(2)]

        x = [GeometricTensor(x[i], self.in_type[i]) for i in range(2)]

        errors = []

        for el in self.space.testing_elements:
            
            out1 = self(*x).transform_fibers(el)
            out2 = self(*[x[i].transform_fibers(el) for i in range(2)])
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
    
            errors.append((el, errs.mean()))

        return errors

