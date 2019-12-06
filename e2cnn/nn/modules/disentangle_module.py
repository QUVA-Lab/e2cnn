
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType

import torch
import numpy as np

from .equivariant_module import EquivariantModule
from .utils import indexes_from_labels
from e2cnn.gspaces import *
from e2cnn.group import disentangle

from collections import defaultdict

from typing import List, Tuple, Any

__all__ = ["DisentangleModule"]


class DisentangleModule(EquivariantModule):

    def __init__(self, in_type: FieldType):
        r"""
        
        Disentangles the representations in the field type of the input.
        
        This module only acts as a wrapper for :func:`e2cnn.group.disentangle`.
        In the constructor, it disentangles each representation in the input type to build the output type and
        pre-compute the change of basis matrices needed to transform each input field.
        
        During the forward pass, each field in the input tensor is transformed with the change of basis corresponding
        to its representation.
        
        Args:
            in_type (FieldType): the input field type
            
        """
        assert isinstance(in_type, FieldType)
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(EquivariantModule, self).__init__()
        
        self.in_type = in_type

        disentangled_representations = {}
        
        _change_of_basis_matrices = {}
        self._sizes = {}
        
        for r in self.in_type._unique_representations:
            self._sizes[r.name] = r.size
            cob, reprs = disentangle(r)
            disentangled_representations[r.name] = reprs
            _change_of_basis_matrices[r.name] = torch.FloatTensor(cob)
            self.register_buffer('change_of_basis_{}'.format(r.name), _change_of_basis_matrices[r.name])

        out_reprs = []
        self._nfields = defaultdict(int)
        for r in self.in_type.representations:
            self._nfields[r.name] += 1
            out_reprs += disentangled_representations[r.name]

        self.out_type = FieldType(self.in_type.gspace, out_reprs)

        grouped_indices = indexes_from_labels(self.in_type, [r.name for r in self.in_type.representations])
        
        self._order = []
        self._contiguous = {}
        
        for repr_name, (contiguous, fields_indices, fiber_indices) in grouped_indices.items():
            self._order.append(repr_name)
            self._contiguous[repr_name] = contiguous
            
            if contiguous:
                fiber_indices = (min(fiber_indices), max(fiber_indices)+1)
            
            fiber_indices = torch.LongTensor(fiber_indices)
            self.register_buffer(f"fiber_indices_{repr_name}", fiber_indices)
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type
        
        input = input.tensor
        
        b, c, w, h = input.shape
        
        output = torch.empty_like(input)
        
        # for each different representation in the fiber
        for repr_name in self._order:
            
            contiguous = self._contiguous[repr_name]
            fiber_indices = getattr(self, f"fiber_indices_{repr_name}")
            
            # retrieve the associated change of basis
            cob = getattr(self, f"change_of_basis_{repr_name}")
            
            # retrieve the associated fields from the input tensor
            if contiguous:
                input_fields = input[:, fiber_indices[0]:fiber_indices[1], ...]
            else:
                input_fields = input[:, fiber_indices, ...]
            
            # reshape to align all the fields in order to exploit broadcasting
            input_fields = input_fields.view(b, self._nfields[repr_name], self._sizes[repr_name], w, h)
            
            # TODO: can we exploit the fact the change of basis is a permutation matrix?
            # transform all the fields with the change of basis
            transformed_fields = torch.einsum("oi,bciwh->bcowh", (cob, input_fields)).reshape(b, -1, w, h)
            
            # insert the transformed fields in the output tensor
            if contiguous:
                output[:, fiber_indices[0]:fiber_indices[1], ...] = transformed_fields
            else:
                output[:, fiber_indices, ...] = transformed_fields
        
        return GeometricTensor(output, self.out_type)
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape
    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        
        c = self.in_type.size
    
        x = torch.randn(3, c, 10, 10)
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.out_type.testing_elements:
            print(el)
        
            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors


