
from collections import defaultdict


from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import Parameter
from typing import List, Tuple, Any


__all__ = ["NormBatchNorm"]


class NormBatchNorm(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True
                 ):
        r"""

        Batch normalization for isometric (i.e. which preserves the norm) non-trivial representations.
        
        The module assumes the mean of the vectors is always zero so no running mean is computed and no bias is added.
        This is guaranteed as long as the representations do not include a trivial representation.
        
        Indeed, if :math:`\rho` does not include a trivial representation, it holds:
        
        .. math ::
        
             \forall \bold{v} \in \mathbb{R}^n, \ \ \frac{1}{|G|} \sum_{g \in G} \rho(g) \bold{v} = \bold{0}

        Hence, only the standard deviation is normalized.
        
        
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
        
        super(NormBatchNorm, self).__init__()

        for r in in_type.representations:
            assert 'norm' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "norm" non-linearity'.format(r.name)
            # Norm batch-normalization assumes the fields to have mean 0. This is true as long as it doesn't contain
            # the trivial representation
            for irr in r.irreps:
                assert not in_type.fibergroup.irreps[irr].is_trivial(), f"Input type contains trivial representation '{irr}'"

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        
        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each size
        self._nfields = defaultdict(int)
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        position = 0
        last_size = None
        for i, r in enumerate(self.in_type.representations):
            
            if r.size != last_size:
                if not r.size in self._contiguous:
                    self._contiguous[r.size] = True
                else:
                    self._contiguous[r.size] = False
            last_size = r.size
            
            _indices[r.size] += list(range(position, position + r.size))
            self._nfields[r.size] += 1
            position += r.size
        
        for s, contiguous in self._contiguous.items():
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[s] = torch.LongTensor([min(_indices[s]), max(_indices[s])+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[s] = torch.LongTensor(_indices[s])
                
            # register the indices tensors as parameters of this module
            self.register_buffer(f'{s}_indices', _indices[s])
            
            running_var = torch.ones((1, self._nfields[s], 1, 1, 1), dtype=torch.float)
            self.register_buffer(f'{s}_running_var', running_var)
            
            if self.affine:
                weight = Parameter(torch.ones((1, self._nfields[s], 1, 1, 1)), requires_grad=True)
                self.register_parameter(f'{s}_weight', weight)
        
        _indices = dict(_indices)
        
        self._order = list(_indices.keys())
        
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.eps = eps
        self.momentum = momentum

    def reset_running_stats(self):
        for s in self._order:
            running_var = getattr(self, f"{s}_running_var")
            running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        for s in self._order:
            weight = getattr(self, f"{s}_weight")
            weight.data.uniform_()
    
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
            for s in self._order:
                indices = getattr(self, f"{s}_indices")
                running_var = getattr(self, f"{s}_running_var")
                
                # compute the norm squared of the fields
                
                if self._contiguous[s]:
                    # if the fields were contiguous, we can use slicing

                    # compute the norm of each field by summing the squares
                    norms = n[:, indices[0]:indices[1], :, :] \
                        .view(b, -1, s, h, w) \
                        .sum(dim=2, keepdim=False) #.sqrt()
                else:
                    # otherwise we have to use indexing
            
                    # compute the norm of each field by summing the squares
                    norms = n[:, indices, :, :] \
                        .view(b, -1, s, h, w) \
                        .sum(dim=2, keepdim=False) #.sqrt()
                
                # Since the mean of the fields is 0, we can compute the variance as the mean of the norms squared
                # corrected with Bessel's correction
                norms = norms.transpose(0, 1).reshape(self._nfields[s], -1)
                correction = norms.shape[1]/(norms.shape[1]-1) if norms.shape[1] > 1 else 1
                vars = norms.mean(dim=1).view(1, -1, 1, 1, 1) / s
                vars *= correction
                # vars = norms.transpose(0, 1).reshape(self._nfields[s], -1).var(dim=1)
        
                # self.running_var[next_var:next_var + self._nfields[s]] += exponential_average_factor * vars
                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars #.detach()

                next_var += self._nfields[s]

            # self.running_var = self.running_var.detach()
            
        next_var = 0
        
        # iterate through all field sizes
        for s in self._order:
    
            indices = getattr(self, f"{s}_indices")
            
            # retrieve the running variances corresponding to the current fields
            # vars = self.running_var[next_var:next_var + self._nfields[s]].view(1, -1, 1, 1, 1)
            # weight = self.weight[next_var:next_var + self._nfields[s]].view(1, -1, 1, 1, 1)
            vars = getattr(self, f"{s}_running_var")
            
            if self.affine:
                weight = getattr(self, f"{s}_weight")
            else:
                weight = 1.
            
            # compute the scalar multipliers needed
            multipliers = weight / (vars + self.eps).sqrt()
            
            # expand the multipliers tensor to all channels for each field
            multipliers = multipliers.expand(b, -1, s, h, w).reshape(b, -1, h, w)
            
            if self._contiguous[s]:
                # if the fields are contiguous, we can use slicing
                output[:, indices[0]:indices[1], :, :] *= multipliers
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] *= multipliers
            
            # shift the position on the running_var and weight tensors
            next_var += self._nfields[s]
        
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

