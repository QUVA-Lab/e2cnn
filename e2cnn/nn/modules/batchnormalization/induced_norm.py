
from collections import defaultdict


from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import Parameter
from typing import List, Tuple, Any


__all__ = ["InducedNormBatchNorm"]


class InducedNormBatchNorm(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 ):
        r"""

        Batch normalization for induced isometric representations.
        This module requires the input fields to be associated to an induced representation from an isometric
        (i.e. which preserves the norm) non-trivial representation which supports 'norm' non-linearities.
        
        The module assumes the mean of the vectors is always zero so no running mean is computed and no bias is added.
        This is guaranteed as long as the representations do not include a trivial representation.
        
        Indeed, if :math:`\rho` does not include a trivial representation, it holds:
        
        .. math ::
        
             \forall \bold{v} \in \mathbb{R}^n, \ \ \frac{1}{|G|} \sum_{g \in G} \rho(g) \bold{v} = \bold{0}

        Hence, only the standard deviation is normalized.
        The same standard deviation, however, is shared by all the sub-fields of the same induced field.
        
        The input representation of the fields is preserved by this operation.
        
        
        Only representations which do not contain the trivial representation are allowed.
        You can check if a representation contains the trivial representation using
        :meth:`~e2cnn.group.Representation.contains_trivial`.
        To check if a trivial irrep is present in a representation in a :class:`~e2cnn.nn.FieldType`, you can use::
        
            for r in field_type:
                if r.contains_trivial():
                    print(f"field type contains a trivial irrep")

        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional): if ``True``, this module has learnable scale parameters. Default: ``True``

        """
    
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(InducedNormBatchNorm, self).__init__()

        for r in in_type.representations:
            assert any(nl.startswith('induced_norm') for nl in r.supported_nonlinearities), \
                'Error! Representation "{}" does not support "induced_norm" non-linearity'.format(r.name)
            # Norm batch-normalization assumes the fields to have mean 0. This is true as long as it doesn't contain
            # the trivial representation
            for irr in r.irreps:
                assert not in_type.fibergroup.irreps[irr].is_trivial()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        
        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each size
        self._nfields = defaultdict(int)
        
        # indices of the channales corresponding to fields belonging to each group
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
            self.register_buffer(f'{id}_indices', _indices[id])
            
            running_var = torch.ones((1, self._nfields[id], 1, 1, 1, 1), dtype=torch.float)
            self.register_buffer(f'{id}_running_var', running_var)
            
            if self.affine:
                weight = Parameter(torch.ones((1, self._nfields[id], 1, 1, 1, 1)), requires_grad=True)
                self.register_parameter(f'{id}_weight', weight)
        
        _indices = dict(_indices)
        
        self._order = list(_indices.keys())
        
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.eps = eps
        self.momentum = momentum

        # self.weight = Parameter(torch.ones(len(in_type)), requires_grad=True)

    def reset_running_stats(self):
        for s in self._order:
            running_var = getattr(self, f"{s}_running_var")
            running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        for s in self._order:
            weight = getattr(self, f"{s}_weight")
            weight.data.fill_(1)
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Apply norm non-linearities to the input feature map
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # compute the squares of the values of each channel
        # n = torch.mul(input.tensor, input.tensor)
        n = input.tensor.detach()**2
        
        b, c, h, w = input.tensor.shape
        
        output = input.tensor.clone()
        
        if self.training:
            
            # self.running_var *= 1 - exponential_average_factor
            
            next_var = 0
            # iterate through all field sizes
            for id in self._order:
                size, subfield_size = id
                n_subfields = int(size // subfield_size)
                
                indices = getattr(self, f"{id}_indices")
                running_var = getattr(self, f"{id}_running_var")
                
                # compute the norm squared of the fields
                
                if self._contiguous[id]:
                    # if the fields were contiguous, we can use slicing

                    # compute the norm of each field by summing the squares
                    norms = n[:, indices[0]:indices[1], :, :] \
                        .view(b, -1, n_subfields, subfield_size, h, w) \
                        .sum(dim=3, keepdim=False) #.sqrt()
                else:
                    # otherwise we have to use indexing
            
                    # compute the norm of each field by summing the squares
                    norms = n[:, indices, :, :] \
                        .view(b, -1,  n_subfields, subfield_size, h, w) \
                        .sum(dim=3, keepdim=False) #.sqrt()
                
                # Since the mean of the fields is 0, we can compute the variance as the mean of the norms squared
                # corrected with Bessel's correction
                norms = norms.transpose(0, 1).reshape(self._nfields[id], -1)
                correction = norms.shape[1]/(norms.shape[1]-1) if norms.shape[1] > 1 else 1
                vars = norms.mean(dim=1).view(1, -1, 1, 1, 1, 1) / subfield_size
                vars *= correction
                # vars = norms.transpose(0, 1).reshape(self._nfields[s], -1).var(dim=1)
        
                # self.running_var[next_var:next_var + self._nfields[s]] += exponential_average_factor * vars
                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars #.detach()

                next_var += self._nfields[id]

            # self.running_var = self.running_var.detach()
            
        next_var = 0
        
        # iterate through all field sizes
        for id in self._order:
            size, subfield_size = id
            n_subfields = int(size // subfield_size)
    
            indices = getattr(self, f"{id}_indices")
            
            # retrieve the running variances corresponding to the current fields
            vars = getattr(self, f"{id}_running_var")
            if self.affine:
                weight = getattr(self, f"{id}_weight")
            else:
                weight = 1.
            
            # compute the scalar multipliers needed
            multipliers = weight / (vars + self.eps).sqrt()
            
            # expand the multipliers tensor to all channels for each field
            multipliers = multipliers.expand(b, -1, n_subfields, subfield_size, h, w).reshape(b, -1, h, w)
            
            if self._contiguous[id]:
                # if the fields are contiguous, we can use slicing
                output[:, indices[0]:indices[1], :, :] *= multipliers
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] *= multipliers
            
            # shift the position on the running_var and weight tensors
            next_var += self._nfields[id]
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

