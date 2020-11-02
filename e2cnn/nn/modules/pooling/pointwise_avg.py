

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch.nn.functional as F
import torch

from typing import List, Tuple, Any, Union

import math

__all__ = ["PointwiseAvgPool", "PointwiseAvgPoolAntialiased"]


class PointwiseAvgPool(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False
                 ):
        r"""

        Channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AvgPool2D`, wrapping it in the
        :class:`~e2cnn.nn.EquivariantModule` interface.
        
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a average over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape

        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        # for r in in_type.representations:
        #     assert 'pointwise' in r.supported_nonlinearities, \
        #         """Error! Representation "{}" does not support pointwise non-linearities
        #         so it is not possible to pool each channel independently"""
        
        super(PointwiseAvgPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        self.ceil_mode = ceil_mode
            
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        # run the common max-pooling
        output = F.avg_pool2d(input.tensor,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              ceil_mode=self.ceil_mode)
                
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape

        # compute the output shape (see 'torch.nn.MaxPool2D')
        ho = (hi + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        wo = (wi + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        
        if self.ceil_mode:
            ho = math.ceil(ho)
            wo = math.ceil(wo)
        else:
            ho = math.floor(ho)
            wo = math.floor(wo)

        return b, self.out_type.size, ho, wo

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass
    
    
class PointwiseAvgPoolAntialiased(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 sigma: float,
                 stride: Union[int, Tuple[int, int]],
                 # kernel_size: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = None,
                 #dilation: Union[int, Tuple[int, int]] = 1,
                 ):
        r"""

        Antialiased channel-wise average-pooling: each channel is treated independently.
        It performs strided convolution with a Gaussian blur filter.
        
        The size of the filter is computed as 3 standard deviations of the Gaussian curve.
        By default, padding is added such that input size is preserved if stride is 1.
        
        Based on `Making Convolutional Networks Shift-Invariant Again <https://arxiv.org/abs/1904.11486>`_.
        
        Args:
            in_type (FieldType): the input field type
            sigma (float): standard deviation for the Gaussian blur filter
            stride: the stride of the window.
            padding: additional zero padding to be added on both sides

        """
        
        super(PointwiseAvgPoolAntialiased, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        assert sigma > 0.
        
        filter_size = 2*int(round(3*sigma))+1
        
        self.kernel_size = (filter_size, filter_size)
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if padding is None:
            padding = int((filter_size-1)//2)
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter

        grid_x = torch.arange(filter_size).repeat(filter_size).view(filter_size, filter_size)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)

        mean = (filter_size - 1) / 2.
        variance = sigma ** 2.

        # setting the dtype is needed, otherwise it becomes an integer tensor
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())

        # Build the gaussian kernel
        _filter = torch.exp(r / (2 * variance))

        # Normalize
        _filter /= torch.sum(_filter)

        # The filter needs to be reshaped to be used in 2d depthwise convolution
        _filter = _filter.view(1, 1, filter_size, filter_size).repeat((in_type.size, 1, 1, 1))

        self.register_buffer('filter', _filter)
        
        ################################################################################################################
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """
        
        assert input.type == self.in_type
        
        output = F.conv2d(input.tensor, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        # compute the output shape
        ho = (hi + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        wo = (wi + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
    
        ho = math.floor(ho)
        wo = math.floor(wo)
    
        return b, self.out_type.size, ho, wo

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        # this kind of pooling is not really equivariant so we can't test equivariance
        pass
