

from typing import List, Tuple, Any

import numpy as np

from collections import defaultdict

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule
from .gated1 import GATED_ID, GATES_ID

import torch

from torch.nn import Parameter


__all__ = ["InducedGatedNonLinearity1"]


class InducedGatedNonLinearity1(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 gates: List = None,
                 drop_gates: bool = True,
                 **kwargs
                 ):
        r"""
        
        Induced Gated non-linearities.
        
        .. todo::
            complete documentation!
        
        .. note::
            Make sure all induced gate and gates have same subgroup
            
        
        Args:
            in_type (FieldType): the input field type
            gates (list, optional): list of strings specifying which field in input is a gate and which is a gated field
            drop_gates (bool, optional): if ``True`` (default), drop the trivial fields after using them to compute
                    the gates. If ``False``, the gates are stacked with the gated fields in the output
            
        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        if gates is None:
            assert len(in_type) % 2 == 0
            
            g = len(in_type) // 2
            gates = [GATES_ID]*g + [GATED_ID]*g
        
        assert len(gates) == len(in_type)
        
        super(InducedGatedNonLinearity1, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type

        self.drop_gates = drop_gates
        
        self._contiguous = {}
        _input_indices = defaultdict(lambda: [])
        _output_indices = defaultdict(lambda: [])

        self._nfields = defaultdict(int)
        
        self.branching = None
        
        for g, r in zip(gates, in_type.representations):
            if g == GATES_ID:
                assert "induced_gate" in r.supported_nonlinearities, \
                    "Error! Representation \"{}\" can't be a \"gate\"".format(r.name)
            elif g == GATED_ID:
                for nl in r.supported_nonlinearities:
                    if nl.startswith("induced_gated"):
                        break
                else:
                    raise ValueError('Error! Representation "{}" does not support "gated" non-linearity'.format(r.name))
            else:
                raise ValueError('Error! "{}" type not recognized'.format(g))

        ngates = len([g for g in gates if g == GATES_ID])
        ngated = len([g for g in gates if g == GATED_ID])

        assert ngates == ngated, \
            'Error! Number of gates ({}) does not match the number of gated non-linearities required ({})' \
            .format(ngates, ngated)

        quotient_size = None
        for g, r in zip(gates, in_type):
            if g == GATES_ID:
                if quotient_size is None:
                    quotient_size = r.size
                else:
                    assert r.size == quotient_size

        subfield_sizes = {}
        for g, r in zip(gates, in_type):
            if g == GATED_ID:
                subfield_size = None
                for nl in r.supported_nonlinearities:
                    if nl.startswith('induced_gated'):
                        assert subfield_size is None, "Error! The representation supports multiple " \
                                                      "sub-fields of different sizes"
                        subfield_size = int(nl.split('_')[-1])
                        assert r.size % subfield_size == 0
                        assert r.size // subfield_size == quotient_size
                        subfield_sizes[r.name] = subfield_size
        
        self.quotient_size = quotient_size
        
        if self.drop_gates:
            # only gated fields are preserved
            # therefore, the output representation is computed from the input one, removing the gates
            self.out_type = in_type.index_select([i for i, g in enumerate(gates) if g == GATED_ID])
        else:
            self.out_type = in_type

        in_last_position = 0
        out_last_position = 0
        last_type = None

        # group fields by their type (gated or gate) and their size, check if fields of the same type are
        # contiguous and retrieve the indices of the fields
        for g, r in zip(gates, in_type.representations):
            if g == GATES_ID:
                type = g
            else:
                type = r.size, subfield_sizes[r.name]
                self._nfields[type] += 1

            if type != last_type:
                if not type in self._contiguous:
                    self._contiguous[type] = True
                else:
                    self._contiguous[type] = False
            last_type = type
    
            _input_indices[type] += list(range(in_last_position, in_last_position + r.size))
            in_last_position += r.size
                
            if g != GATES_ID or not self.drop_gates:
                # since gates are discarded in output, the position on the output fiber is shifted
                # only when a gated field is met
                _output_indices[type] += list(range(out_last_position, out_last_position + r.size))
                out_last_position += r.size
    
        _input_indices = dict(_input_indices)
        # if self.drop_gates:
        _output_indices = dict(_output_indices)
        # else:
        #     self._output_indices = self._input_indices

        for t, contiguous in self._contiguous.items():
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _input_indices[t] = torch.LongTensor([min(_input_indices[t]), max(_input_indices[t]) + 1])
                if t != GATES_ID or not self.drop_gates:
                    _output_indices[t] = torch.LongTensor([min(_output_indices[t]), max(_output_indices[t]) + 1])
            else:
                # otherwise, transform the list of indices into a tensor
                _input_indices[t] = torch.LongTensor(_input_indices[t])
    
                if t != GATES_ID or not self.drop_gates:
                    _output_indices[t] = torch.LongTensor(_output_indices[t])
                    
            # register the indices tensors as parameters of this module
            self.register_buffer('input_indices_{}'.format(t), _input_indices[t])
            if t != GATES_ID or not self.drop_gates:
                self.register_buffer('output_indices_{}'.format(t), _output_indices[t])

        # gates need to be distinguished from gated fields
        _gates_indices = _input_indices.pop(GATES_ID)
        self.register_buffer('gates_indices', _gates_indices)
        
        # build a sorted list of the fields groups, such that every time they are iterated through in the same order
        self._order = sorted(_input_indices.keys())
        
        # the bias for the gates
        self.bias = Parameter(torch.randn(1, ngates, 1, 1, 1, dtype=torch.float), requires_grad=True)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply the gated non-linearity to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert isinstance(input, GeometricTensor)
        assert input.type == self.in_type
        
        # retrieve the gates
        
        if self._contiguous[GATES_ID]:
            gates = input.tensor[:, self.gates_indices[0]:self.gates_indices[1], ...]
        else:
            gates = input.tensor[:, self.gates_indices, ...]

        # retrieving only gated fields from the joint tensor is worthless
        input = input.tensor
        
        b, c, h, w = input.shape
        
        # transform the gates
        gates = torch.sigmoid(gates.view(b, -1, self.quotient_size, h, w ) - self.bias)

        # build the output tensor
        output = torch.empty(b, self.out_type.size, h, w, dtype=torch.float, device=self.bias.device)

        if not self.drop_gates:
            # copy the gates in the output
            if self._contiguous[GATES_ID]:
                output[:, self.gates_indices[0]:self.gates_indices[1], ...] = gates
            else:
                output[:, self.gates_indices, ...] = gates
        
        next_gate = 0
        
        # for each field size
        for type in self._order:
            size, subfield_size = type
            
            # retrieve the needed gates
            g = gates[:, next_gate:next_gate + self._nfields[type], ...].view(b, -1, 1, h, w)
            
            input_indices = getattr(self, f"input_indices_{type}")
            output_indices = getattr(self, f"output_indices_{type}")
            
            if self._contiguous[type]:
                # if the fields were contiguous, we can use slicing
                output[:, output_indices[0]:output_indices[1], ...] =\
                    (
                            input[:, input_indices[0]:input_indices[1], ...]
                            .view(b, -1, subfield_size, h, w)
                            * g
                    ).view(b, -1, h, w)
            
            else:
                # otherwise we have to use indexing
                output[:, output_indices, :, :] = \
                    (
                            input[:, input_indices, ...]
                            .view(b, -1, subfield_size, h, w)
                            * g
                    ).view(b, -1, h, w)
            
            # shift the position on the gates fiber
            next_gate += self._nfields[type]
        
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

