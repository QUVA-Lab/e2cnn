
from collections import defaultdict


from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import Parameter
from typing import List, Tuple, Any, Union
import numpy as np

__all__ = ["GNormBatchNorm"]


class GNormBatchNorm(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 ):
        r"""

        Batch normalization for generic representations.

        .. todo ::
            Add more details about how stats are computed and how affine transformation is done.

        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``

        """
    
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(GNormBatchNorm, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        
        self._nfields = None
        
        # group fields by their type and
        #   - check if fields of the same type are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each type
        self._nfields = defaultdict(int)
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        ntrivials = 0
        position = 0
        last_field = None
        for i, r in enumerate(self.in_type.representations):
            
            for irr in r.irreps:
                if self.in_type.fibergroup.irreps[irr].is_trivial():
                    ntrivials += 1
            
            if r.name != last_field:
                if not r.name in self._contiguous:
                    self._contiguous[r.name] = True
                else:
                    self._contiguous[r.name] = False

            last_field = r.name
            _indices[r.name] += list(range(position, position + r.size))
            self._nfields[r.name] += 1
            position += r.size
        
        for name, contiguous in self._contiguous.items():
            
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[name] = [min(_indices[name]), max(_indices[name])+1]
                setattr(self, f"{name}_indices", _indices[name])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[name] = torch.LongTensor(_indices[name])
                
                # register the indices tensors as parameters of this module
                self.register_buffer(f"{name}_indices", _indices[name])
        
        # store the size of each field type
        self._sizes = []
        
        # store for each field type the indices of the trivial irreps in it
        self._trivial_idxs = {}

        # store for each field type the sizes and the indices of all its irreps, grouped by their size
        self._irreps_sizes = {}
        
        for r in self.in_type._unique_representations:
            p = 0
            irreps = defaultdict(lambda: [])
            trivials = []
            aggregator = torch.zeros(r.size, len(r.irreps))
            
            for i, irr in enumerate(r.irreps):
                irr = self.in_type.fibergroup.irreps[irr]
                if irr.is_trivial():
                    trivials.append(p)
                
                aggregator[p:p+irr.size, i] = 1. / irr.size
                
                irreps[irr.size] += list(range(p, p+irr.size))
                p += irr.size
            
            propagator = (aggregator > 0).clone().to(dtype=torch.float)
            
            name = r.name
            
            self._trivial_idxs[name] = torch.tensor(trivials, dtype=torch.long)
            self._irreps_sizes[name] = [(s, idxs) for s, idxs in irreps.items()]
            self._sizes.append((name, r.size))
            
            if not np.allclose(r.change_of_basis, np.eye(r.size)):
                self.register_buffer(f'{name}_change_of_basis', torch.tensor(r.change_of_basis, dtype=torch.float))
                self.register_buffer(f'{name}_change_of_basis_inv', torch.tensor(r.change_of_basis_inv, dtype=torch.float))
            
            self.register_buffer(f'vars_aggregator_{name}', aggregator)
            self.register_buffer(f'vars_propagator_{name}', propagator)
        
            running_var = torch.ones((self._nfields[r.name], len(r.irreps)), dtype=torch.float)
            running_mean = torch.zeros((self._nfields[r.name], len(trivials)), dtype=torch.float)
            self.register_buffer(f'{name}_running_var', running_var)
            self.register_buffer(f'{name}_running_mean', running_mean)
            
            if self.affine:
                weight = Parameter(torch.ones((self._nfields[r.name], len(r.irreps))), requires_grad=True)
                bias = Parameter(torch.zeros((self._nfields[r.name], len(trivials))), requires_grad=True)
                self.register_parameter(f'{name}_weight', weight)
                self.register_parameter(f'{name}_bias', bias)
            
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.eps = eps
        self.momentum = momentum

    def reset_running_stats(self):
        for name, size in self._sizes:
            running_var = getattr(self, f"{name}_running_var")
            running_mean = getattr(self, f"{name}_running_mean")
            running_var.fill_(1)
            running_mean.fill_(0)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for name, size in self._sizes:
                weight = getattr(self, f"{name}_weight")
                bias = getattr(self, f"{name}_bias")
                weight.data.fill_(1)
                bias.data.fill_(0)
    
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

        input = input.tensor
        b, c, h, w = input.shape
        
        output = torch.empty_like(input)
        
        # iterate through all field types
        for name, size in self._sizes:
            indices = getattr(self, f"{name}_indices")
            
            if self._contiguous[name]:
                slice = input[:, indices[0]:indices[1], ...]
            else:
                slice = input[:, indices, ...]

            slice = slice.view(b, -1, size, h, w)
            
            if hasattr(self, f"{name}_change_of_basis_inv"):
                cob_inv = getattr(self, f"{name}_change_of_basis_inv")
                slice = torch.einsum("ds,bcsxy->bcdxy", (cob_inv, slice))
                
            if self.training:
                
                # compute the mean and variance of the fields
                means, vars = self._compute_statistics(slice, name)
                
                running_var = getattr(self, f"{name}_running_var")
                running_mean = getattr(self, f"{name}_running_mean")
                
                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars
                
                running_mean *= 1 - exponential_average_factor
                running_mean += exponential_average_factor * means
                
                assert torch.allclose(running_mean, getattr(self, f"{name}_running_mean"))
                assert torch.allclose(running_var, getattr(self, f"{name}_running_var"))
                
            else:
                vars = getattr(self, f"{name}_running_var")
                means = getattr(self, f"{name}_running_mean")
                
            if self.affine:
                weight = getattr(self, f"{name}_weight")
            else:
                weight = 1.
                
            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()

            # compute the point shifts
            # shifts = bias - self._scale(means, scales, name=name)
            centered = self._shift(slice, -1*means, name=name, out=None)
            normalized = self._scale(centered, scales, name=name, out=None)
            
            if self.affine:
                bias = getattr(self, f"{name}_bias")
                normalized = self._shift(normalized, bias, name=name, out=None)
            
            if hasattr(self, f"{name}_change_of_basis"):
                cob = getattr(self, f"{name}_change_of_basis")
                normalized = torch.einsum("ds,bcsxy->bcdxy", (cob, normalized))
                
            if not self._contiguous[name]:
                output[:, indices, ...] = normalized.view(b, -1, h, w)
            else:
                output[:, indices[0]:indices[1], ...] = normalized.view(b, -1, h, w)

            # if self._contiguous[name]:
            #     slice2 = output[:, indices[0]:indices[1], ...]
            # else:
            #     slice2 = output[:, indices, ...]
            # assert torch.allclose(slice2.view(b, -1, size, h, w), slice), name
            
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

    def _compute_statistics(self, t: torch.Tensor, name: str):
        
        trivial_idxs = self._trivial_idxs[name]
        vars_aggregator = getattr(self, f"vars_aggregator_{name}")
        
        b, c, s, x, y = t.shape
        
        l = trivial_idxs.numel()
        
        # number of samples in the tensor used to estimate the statistics
        N = b * x * y
        
        # compute the mean of the trivial fields
        trivial_means = t[:, :, trivial_idxs, ...].view(b, c, l, x, y).sum(dim=(0, 3, 4), keepdim=False).detach() / N
        
        # compute the mean of squares of all channels
        vars = (t ** 2).view(b, c, s, x, y).sum(dim=(0, 3, 4), keepdim=False).detach() / N
        
        # For the non-trivial fields the mean of the fields is 0, so we can compute the variance as the mean of the
        # norms squared.
        # For trivial channels, we need to subtract the squared mean
        vars[:, trivial_idxs] -= trivial_means**2
        
        # aggregate the squared means of the channels which belong to the same irrep
        vars = torch.einsum("io,ci->co", (vars_aggregator, vars))

        # Correct the estimation of the variance with Bessel's correction
        correction = N/(N-1) if N > 1 else 1.
        vars *= correction
        
        return trivial_means, vars

    def _scale(self, t: torch.Tensor, scales: torch.Tensor, name: str, out: torch.Tensor = None):
        
        if out is None:
            out = torch.empty_like(t)
        
        vars_aggregator = getattr(self, f"vars_propagator_{name}")
        
        ndims = len(t.shape[3:])
        scale_shape = (1, scales.shape[0], vars_aggregator.shape[0]) + (1,)*ndims
        # scale all fields
        out[...] = t * torch.einsum("oi,ci->co", (vars_aggregator, scales)).reshape(scale_shape)
        
        return out
    
    def _shift(self, t: torch.Tensor, trivial_bias: torch.Tensor, name: str, out: torch.Tensor = None):
    
        if out is None:
            out = t.clone()
        else:
            out[:] = t
            
        trivial_idxs = self._trivial_idxs[name]
        
        bias_shape = (1,) + trivial_bias.shape + (1,)*(len(t.shape) - 3)
        
        # add bias to the trivial fields
        out[:, :, trivial_idxs, ...] += trivial_bias.view(bias_shape)

        return out
