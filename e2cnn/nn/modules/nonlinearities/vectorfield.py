
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Any

import numpy as np

__all__ = ["VectorFieldNonLinearity"]


class VectorFieldNonLinearity(EquivariantModule):
    
    def __init__(self, in_type: FieldType, **kwargs):
        r"""
        
        VectorField non-linearities.
        This non-linearity only supports the regular representation of cyclic group :math:`C_N`, i.e. the group of
        :math:`N` discrete rotations.
        For each input field, the output one is built by taking the rotation associated with the highest
        activation; then, a 2-dimensional vector with an angle with respect to the x-axis equal to that rotation and a
        length equal to its activation is set in the output field.
        
        Args:
            in_type (FieldType): the input field type
            
        """
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        assert in_type.gspace.fibergroup.order() > 1
        
        for r in in_type.representations:
            assert 'vectorfield' in r.supported_nonlinearities,\
                'Error! Representation "{}" does not support "vector-field" non-linearity'.format(r.name)
            
            assert r.name == 'regular' and r.size == in_type.gspace.fibergroup.order(), r.name

        super(VectorFieldNonLinearity, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        
        # build the output representation substituting each input field with a rotation representation with frequency 1
        self.out_type = FieldType(self.space, [self.space.representations['irrep_1']] * len(in_type))
        
        # the number of rotations associated with the group action
        self._rotations = self.space.fibergroup.order()
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply the VectorField non-linearity to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        b, c, h, w = input.tensor.shape
        
        # split the channel dimension in 2 dimensions, separating fields
        fm = input.tensor.view(b, -1, self._rotations, h, w)
        
        # evaluate the base rotation associated with the group action
        base_angle = 2 * np.pi / self._rotations
        
        # for each field, retrieve the maximum activation (and the argmax)
        max_activations, argmaxes = torch.max(fm, 2)
        max_activations = torch.relu_(max_activations)
        
        # compute the angles from the index of the maximum activation in the field
        max_angles = argmaxes.to(dtype=torch.float) * base_angle
        
        # build the output tensor
        output = torch.empty(b, self.out_type.size, h, w, dtype=torch.float, device=input.tensor.device)
        
        # to build the output vectors, take the cosine and the sine of the argmax angle
        # and multiply the 2-dimensional vector by the activation value
        output[:, ::2, ...] = torch.cos(max_angles) * max_activations
        output[:, 1::2, ...] = torch.sin(max_angles) * max_activations
        
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


