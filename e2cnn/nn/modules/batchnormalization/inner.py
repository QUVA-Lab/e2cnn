
from typing import List, Tuple, Any

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
from torch.nn import BatchNorm3d

from ..utils import indexes_from_labels

__all__ = ["InnerBatchNorm"]


class InnerBatchNorm(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 ):
        r"""
        
        Batch normalization for representations with permutation matrices.
        
        Statistics are computed both over the batch and the spatial dimensions and over the channels within
        the same field (which are permuted by the representation).
        
        Only representations supporting pointwise non-linearities are accepted as input field type.
        
        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional):  if ``True``, this module has learnable affine parameters. Default: ``True``
            track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                                  when set to ``False``, it does not track such statistics but uses
                                                  batch statistics in both training and eval modes.
                                                  Default: ``True``
            
        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(InnerBatchNorm, self).__init__()

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        # group fields by their size and
        #   - check if fields with the same size are contiguous
        #   - retrieve the indices of the fields
        grouped_fields = indexes_from_labels(self.in_type, [r.size for r in self.in_type.representations])

        # number of fields of each size
        self._nfields = {}
        
        # indices of the channels corresponding to fields belonging to each group
        _indices = {}
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        for s, (contiguous, fields, indices) in grouped_fields.items():
            self._nfields[s] = len(fields)
            self._contiguous[s] = contiguous
            
            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[s] = torch.LongTensor([min(indices), max(indices)+1])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[s] = torch.LongTensor(indices)
                
            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(s), _indices[s])
        
        for s in _indices.keys():
            _batchnorm = BatchNorm3d(
                self._nfields[s],
                self.eps,
                self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats
            )
            self.add_module('batch_norm_[{}]'.format(s), _batchnorm)

    def reset_running_stats(self):
        for s, contiguous in self._contiguous.items():
            batchnorm = getattr(self, f'batch_norm_[{s}]')
            batchnorm.reset_running_stats()

    def reset_parameters(self):
        for s, contiguous in self._contiguous.items():
            batchnorm = getattr(self, f'batch_norm_[{s}]')
            batchnorm.reset_parameters()
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        b, c, h, w = input.tensor.shape
        
        output = torch.empty_like(input.tensor)
        
        # iterate through all field sizes
        for s, contiguous in self._contiguous.items():
            
            indices = getattr(self, f"indices_{s}")
            batchnorm = getattr(self, f'batch_norm_[{s}]')
            
            if contiguous:
                # if the fields were contiguous, we can use slicing
                output[:, indices[0]:indices[1], :, :] = batchnorm(
                    input.tensor[:, indices[0]:indices[1], :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
            else:
                # otherwise we have to use indexing
                output[:, indices, :, :] = batchnorm(
                    input.tensor[:, indices, :, :].view(b, -1, s, h, w)
                ).view(b, -1, h, w)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(InnerBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNorm2d` module and set to "eval" mode.

        """

        if not self.track_running_stats:
            raise ValueError('''
                Equivariant Batch Normalization can not be converted into conventional batch normalization when
                "track_running_stats" is False because the statistics contained in a single batch are generally
                not symmetric
            ''')
        
        self.eval()
        
        batchnorm = torch.nn.BatchNorm2d(
            self.in_type.size,
            self.eps,
            self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats
        )
        
        num_batches_tracked = None
        
        for s, contiguous in self._contiguous.items():
            if not contiguous:
                raise NotImplementedError(
                    '''Non-contiguous indices not supported yet when converting
                    inner-batch normalization into conventional BatchNorm2d'''
                )
            
            # indices = getattr(self, 'indices_{}'.format(s))
            start, end = getattr(self, 'indices_{}'.format(s))
            bn = getattr(self, 'batch_norm_[{}]'.format(s))
            
            n = self._nfields[s]
            
            batchnorm.running_var.data[start:end] = bn.running_var.data.view(n, 1).expand(n, s).reshape(-1)
            batchnorm.running_mean.data[start:end] = bn.running_mean.data.view(n, 1).expand(n, s).reshape(-1)
            batchnorm.num_batches_tracked.data = bn.num_batches_tracked.data

            if num_batches_tracked is None:
                num_batches_tracked = bn.num_batches_tracked.data
            else:
                assert num_batches_tracked == bn.num_batches_tracked.data
            
            if self.affine:
                batchnorm.weight.data[start:end] = bn.weight.data.view(n, 1).expand(n, s).reshape(-1)
                batchnorm.bias.data[start:end] = bn.bias.data.view(n, 1).expand(n, s).reshape(-1)

        batchnorm.eval()

        return batchnorm

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
    
        main_str = self._get_name() + '('
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(extra_lines) + '\n'
    
        main_str += ')'
        return main_str

    def extra_repr(self):
        return '{in_type}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'\
            .format(**self.__dict__)

