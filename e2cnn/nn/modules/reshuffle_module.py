
import torch

from e2cnn.gspaces import *
from .equivariant_module import EquivariantModule
from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType

from typing import List, Tuple, Any

import numpy as np

__all__ = ["ReshuffleModule"]


class ReshuffleModule(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 permutation: List[int]
                 ):
        r"""
        
        Permutes the fields of the input tensor according to the input ``permutation``.
        
        The parameter ``permutation`` should be a list containing a permutation of the integers ``{0, 1, ..., n-1}``,
        where ``n`` is the number of fields of ``in_type`` (see :meth:`e2cnn.nn.FieldType.__len__`).
        
        
        Args:
            in_type (FieldType): input field type
            permutation (list): permutation to apply
            
        """
    
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(ReshuffleModule, self).__init__()
        
        # check if it is actually a permutation of the fields
        assert set(permutation) == set(range(0, len(in_type.representations)))
        
        self.space = in_type.gspace
        self.in_type = in_type
        
        # permute the fields in the input representation to build the output representation
        self.out_type = in_type.index_select(permutation)
        
        # compute the starting position of each field in the input representation
        positions = []
        last_position = 0
        for r in in_type.representations:
            positions.append(last_position)
            last_position += r.size

        # compute the indices for the permutation
        
        indices = []
        for c in permutation:
            size = in_type.representations[c].size
            
            indices += list(range(positions[c], positions[c]+size))
        
        self.register_buffer("indices", torch.LongTensor(indices))
        
    def forward(self, input: GeometricTensor):
        
        assert input.type == self.in_type
        
        # retrieve the values from the input using the permutation of the indices computed before
        data = input.tensor[:, self.indices, ...]
        
        return GeometricTensor(data, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        b, c, hi, wi = input_shape
    
        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
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