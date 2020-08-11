

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from .equivariant_module import EquivariantModule

from typing import Tuple

import torch
import numpy as np

import math

from torch.nn.functional import interpolate

__all__ = ["R2Upsampling"]


class R2Upsampling(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 scale_factor: int,
                 mode: str = "bilinear",
                 align_corners: bool = False
                 ):
        r"""
        
        Wrapper for :func:`torch.nn.functional.interpolate`. Check its documentation for further details.
        
        Only ``"bilinear"`` and ``"nearest"`` methods are supported.
        However, ``"nearest"`` is not equivariant; using this method may result in broken equivariance.
        For this reason, we suggest to use ``"bilinear"`` (default value).
        
        
        Args:
            in_type (FieldType): the input field type
            scale_factor (int): multiplier for spatial size
            mode (str): algorithm used for upsampling: ``nearest`` | ``bilinear``. Default: ``bilinear``
            align_corners (bool): if ``True``, the corner pixels of the input and output tensors are aligned, and thus
                    preserving the values at those pixels. This only has effect when mode is ``bilinear``.
                    Default: ``False``
            
        """

        assert isinstance(in_type.gspace, GeneralOnR2)

        super(R2Upsampling, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self._scale_factor = scale_factor
        self._mode = mode
        self._align_corners = align_corners if mode != "nearest" else None
        
        if mode not in ["nearest", "bilinear"]:
            raise ValueError(f'Error Upsampling mode {mode} not recognized! Mode should be `nearest` or `bilinear`.')
        
    def forward(self, input: GeometricTensor):
        r"""
        
        Args:
            input (torch.Tensor): input feature map

        Returns:
             the result of the convolution
             
        """
        
        assert input.type == self.in_type
        
        if self._align_corners is None:
            output = interpolate(input.tensor,
                                 scale_factor=self._scale_factor,
                                 mode=self._mode)
        else:
            output = interpolate(input.tensor,
                                 scale_factor=self._scale_factor,
                                 mode=self._mode,
                                 align_corners=self._align_corners)
        
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
        
        ho = math.floor(hi * self._scale_factor)
        wo = math.floor(wi * self._scale_factor)

        return b, self.out_type.size, ho, wo
        
    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1):
        
        initial_size = 55

        c = self.in_type.size

        # x = torch.randn(3, c, initial_size, initial_size)

        import matplotlib.image as mpimg
        from skimage.transform import resize

        x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]
        x = x / 255.0
        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size),
            anti_aliasing=True
        )

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = GeometricTensor(torch.FloatTensor(x), self.in_type)

        errors = []

        for el in self.space.testing_elements:

            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()

            b, c, h, w = out2.shape

            center_mask = np.zeros((2, h, w))
            center_mask[1, :, :] = np.arange(0, w) - w / 2
            center_mask[0, :, :] = np.arange(0, h) - h / 2
            center_mask[0, :, :] = center_mask[0, :, :].T
            center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 < (h * 0.4) ** 2

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]

            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)

            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum

            # print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

            # tol = rtol*(np.abs(out1) + np.abs(out2)) + atol
            tol = rtol * esum + atol

            if np.any(errs > tol):
                print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())
                # print(errs[errs > tol])
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])

            # assert np.all(np.abs(out1 - out2) < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
            assert np.all(errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Upsample` module and set to "eval" mode.

        """
    
        self.eval()
        
        if self._align_corners is not None:
            upsample = torch.nn.Upsample(
                scale_factor=self._scale_factor,
                mode=self._mode,
                align_corners=self._align_corners
            )
        else:
            upsample = torch.nn.Upsample(
                scale_factor=self._scale_factor,
                mode=self._mode,
            )
        
        return upsample.eval()
