
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.group import Representation
from e2cnn.group.representation import build_from_discrete_group_representation

from ..equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Any

import numpy as np
import math

__all__ = ["ConcatenatedNonLinearity"]


class ConcatenatedNonLinearity(EquivariantModule):
    
    def __init__(self, in_type: FieldType, function: str = 'c_relu'):
        r"""
        
        Concatenated non-linearities.
        For each input channel, the module applies the specified activation function both to its value and its opposite
        (the value multiplied by -1).
        The number of channels is, therefore, doubled.
        
        Notice that not all the representations support this kind of non-linearity. Indeed, only representations
        with the same pattern of permutation matrices and containing only values in :math:`\{0, 1, -1\}` support it.
        
        
        Args:
            in_type (FieldType): the input field type
            function (str): the identifier of the non-linearity. It is used to specify which function to apply.
                    By default (``'c_relu'``), ReLU is used.
            
        """
        
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        for r in in_type.representations:
            assert 'concatenated' in r.supported_nonlinearities,\
                'Error! Representation "{}" does not support "concatenated" non-linearity'.format(r.name)

        super(ConcatenatedNonLinearity, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        
        # compute the output representation given the input one
        self.out_type = ConcatenatedNonLinearity._transform_fiber_representation(in_type)
        
        # retrieve the activation function to apply
        if function == 'c_relu':
            self._function = torch.relu
        elif function == 'c_sigmoid':
            self._function = torch.sigmoid
        elif function == 'c_tanh':
            self._function = torch.tanh
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        
        assert input.type == self.in_type
        
        b, c, w, h = input.tensor.shape
        
        # build the output tensor
        output = torch.empty(b, 2 * c, w, h, dtype=torch.float, device=input.tensor.device)
        
        # each channels is transformed to 2 channels:
        # first, apply the non-linearity to its value
        output[:, ::2, ...] = self._function(input.tensor)
        
        # then, apply the non-linearity to its values with the sign inverted
        output[:, 1::2, ...] = self._function(-1 * input.tensor)
        
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

    @staticmethod
    def _transform_fiber_representation(in_type: FieldType) -> FieldType:
        r"""
        
        Compute the output representation from the input one after applying the concatenated non-linearity.
        
        Args:
            in_type (FieldType): the input field type

        Returns:
            (FieldType): the new output field type
            
        """
        transformed = {}
        
        # transform each different input Representation
        for repr in in_type._unique_representations:
            transformed[repr] = ConcatenatedNonLinearity._transform_representation(repr)
        
        new_representations = []
        
        # concatenate the new representations
        for repr in in_type.representations:
            new_representations.append(transformed[repr])
        
        return FieldType(in_type.gspace, new_representations)
        
    @staticmethod
    def _transform_representation(representation: Representation) -> Representation:
        r"""
        
        Transform an input :class:`~e2cnn.group.Representation` according to the concatenated non-linearity.
        
        The input representation needs to have the pattern of a permutation matrix, with values -1 or 1.
        
        The output representation has double the size of the input one and is built by substituting the ``1`` s with 2x2
        identity matrices and the ``-1`` s with 2x2 antidiagonal matrix containing ``1`` s.
        
        Args:
            representation (Representation): the input representation

        Returns:
            (Representation): the new output representation

        """
        group = representation.group
        
        assert not group.continuous
        
        # the name of the new representation
        name = "concatenated_{}".format(representation.name)
        
        if name in group.representations:
            # if the representation has already been built, return it
            r = group.representations[name]
        else:
            # otherwise, build the new representation
            
            s = representation.size
            
            rep = {}
            # build the representation for each element
            for element in group.elements:
                
                # retrieve the input representation of the current element
                r = representation(element)
                
                # build the matrix for the output representation of the current element
                rep[element] = np.zeros((2*s, 2*s))
                
                # check if the input matrix has the pattern of a permutation matrix
                e = [-1]*s
                for i in range(s):
                    for j in range(s):
                        if not math.isclose(r[i, j], 0, abs_tol=1e-9):
                            if e[i] < 0:
                                e[i] = j
                            else:
                                raise ValueError(
                                    '''Error! the representation should have the pattern of a permutation matrix
                                    but 2 values have been found in a row for element "{}"'''
                                    .format(element)
                                )
                if len(set(e)) != len(e):
                    raise ValueError(
                        '''Error! the representation should have the pattern of a permutation matrix
                        but 2 values have been found in a column for element "{}"'''
                        .format(element)
                    )
                
                # parse the input representation matrix and fill the output representation accordingly
                for i in range(s):
                    for j in range(s):
                        
                        if math.isclose(r[i, j], 1, abs_tol=1e-9):
                            # if the current cell contains 1, fill the output submatrix with the 2x2 identity
                            rep[element][2 * i:2 * i + 2, 2 * j:2 * j + 2] = np.eye(2)
                        elif math.isclose(r[i, j], -1, abs_tol=1e-9):
                            # if the current cell contains -1, fill the output submatrix with the 2x2 antidigonal matrix
                            rep[element][2 * i:2 * i + 2, 2 * j:2 * j + 2] = np.flipud(np.eye(2))
                        elif not math.isclose(r[i, j], 0, abs_tol=1e-9):
                            # otherwise the cell has to contain a 0
                            raise ValueError(
                                '''Error! The representation should be a signed permutation matrix and, therefore,
                                 contain only -1, 1 or 0 values but {} found in position({}, {}) for element "{}"'''
                                .format(r[i, j], i, j, element)
                            )
            
            # the resulting representation is a quotient repreentation and, therefore,
            # it also supports pointwise non-linearities
            nonlinearities = representation.supported_nonlinearities.union(['pointwise'])
            
            # build the output representation
            r = build_from_discrete_group_representation(rep, name, group, supported_nonlinearities=nonlinearities)
        
        return r

