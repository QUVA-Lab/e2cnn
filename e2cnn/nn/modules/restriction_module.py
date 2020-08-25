

import torch
import numpy as np

from .equivariant_module import EquivariantModule
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from e2cnn.gspaces import *

import torch
from typing import List, Tuple, Any

__all__ = ["RestrictionModule"]


class RestrictionModule(EquivariantModule):

    def __init__(self, in_type: FieldType, id):
        r"""
        
        Restricts the type of the input to the subgroup identified by ``id``.
        
        It computes the output type in the constructor and wraps the underlying tensor (:class:`torch.Tensor`) in input
        with the output type during the forward pass.
        
        This module only acts as a wrapper for :meth:`e2cnn.nn.FieldType.restrict`
        (or :meth:`e2cnn.nn.GeometricTensor.restrict`).
        The accepted values of ``id`` depend on the underlying ``gspace`` in the input type ``in_type``; check the
        documentation of the method :meth:`e2cnn.gspaces.GSpace.restrict` of the gspace used for
        further information.
        
        
        .. seealso::
            :meth:`e2cnn.nn.FieldType.restrict`,
            :meth:`e2cnn.nn.GeometricTensor.restrict`,
            :meth:`e2cnn.gspaces.GSpace.restrict`
        
        Args:
            in_type (FieldType): the input field type
            id: a valid id for a subgroup of the space associated with the input type
            
        """
        assert isinstance(in_type, FieldType)
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(EquivariantModule, self).__init__()
        
        self._id = id
        self.in_type = in_type
        self.out_type = in_type.restrict(id)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type
        return GeometricTensor(input.tensor, self.out_type)
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape
    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        
        _, parent_mapping, _ = self.in_type.gspace.restrict(self._id)
        
        c = self.in_type.size
    
        x = torch.randn(3, c, 10, 10)
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.out_type.testing_elements:
            print(el)
        
            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(parent_mapping(el))).tensor.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Identity` module and set to "eval" mode.

        .. warning ::
            Only working with PyTorch >= 1.2

        """
        self.eval()
        return torch.nn.Identity()


