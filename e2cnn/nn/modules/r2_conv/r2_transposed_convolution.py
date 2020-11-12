
from torch.nn.functional import conv_transpose2d

from e2cnn.nn import init
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.gspaces import *

from ..equivariant_module import EquivariantModule

from .basisexpansion import BasisExpansion
from .basisexpansion_blocks import BlocksBasisExpansion

from typing import Callable, Union, Tuple, List
from collections import defaultdict

import torch
from torch.nn import Parameter
import numpy as np
from scipy import stats
import math


__all__ = ["R2ConvTransposed"]


class R2ConvTransposed(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 output_padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 ):
        r"""
        
        Transposed G-steerable planar convolution layer.
        
        .. warning ::
            
            Transposed convolution can produce artifacts which can harm the overall equivariance of the model.
            We suggest using :class:`~e2cnn.nn.R2Upsampling` combined with :class:`~e2cnn.nn.R2Conv` to perform
            upsampling.
        
        .. seealso ::
            For additional information about the parameters and the methods of this class, see :class:`e2cnn.nn.R2Conv`.
            The two modules are essentially the same, except for the type of convolution used.
            
        Args:
            in_type (FieldType): the type of the input field
            out_type (FieldType): the type of the output field
            kernel_size (int): the size of the filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            output_padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            stride(int, optional): the stride of the convolving kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
            
        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)

        super(R2ConvTransposed, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        if groups > 1:
            # Check the input and output classes can be split in `groups` groups, all equal to each other
            # first, check that the number of fields is divisible by `groups`
            assert len(in_type) % groups == 0
            assert len(out_type) % groups == 0
            in_size = len(in_type) // groups
            out_size = len(out_type) // groups
            
            # then, check that all groups are equal to each other, i.e. have the same types in the same order
            assert all(in_type.representations[i] == in_type.representations[i % in_size] for i in range(len(in_type)))
            assert all(out_type.representations[i] == out_type.representations[i % out_size] for i in range(len(out_type)))
            
            # finally, retrieve the type associated to a single group in output.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from all input
            # groups to one output group. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            out_type = out_type.index_select(list(range(out_size)))
        
        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis
            
            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irreps[irr].is_trivial():
                        trivials += 1
            
            # if there is at least 1 trivial irrep
            if trivials > 0:
                
                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = torch.zeros(self.out_type.size, trivials)
                
                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irreps[irr]
                        if irr.is_trivial():
                            bias_expansion[p:p+r.size, c] = torch.tensor(r.change_of_basis[:, pi])
                            c += 1
                        pi += irr.size
                    p += r.size
                
                self.register_buffer("bias_expansion", bias_expansion)
                self.bias = Parameter(torch.zeros(trivials), requires_grad=True)
                self.register_buffer("expanded_bias", torch.zeros(out_type.size))
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(kernel_size,
                                                                                   frequencies_cutoff,
                                                                                   rings,
                                                                                   sigma,
                                                                                   dilation,
                                                                                   basis_filter)
        
        # BasisExpansion: submodule which takes care of building the filter
        self._basisexpansion = None
        
        # notice that `out_type` is used instead of `self.out_type` such that it works also when `groups > 1`
        if basisexpansion == 'blocks':
            self._basisexpansion = BlocksBasisExpansion(in_type, out_type,
                                                        grid,
                                                        sigma=sigma,
                                                        rings=rings,
                                                        maximum_offset=maximum_offset,
                                                        maximum_frequency=maximum_frequency,
                                                        basis_filter=basis_filter,
                                                        recompute=recompute)

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)
        
        self.weights = Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        self.register_buffer("filter", torch.zeros(in_type.size, out_type.size, kernel_size, kernel_size))
        
        if initialize:
            # by default, the weights are initialized with a generalized form of he's weight initialization
            init.generalized_he_init(self.weights.data, self.basisexpansion)
        
    @property
    def basisexpansion(self) -> BasisExpansion:
        return self._basisexpansion

    def expand_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)
        _filter = _filter.transpose(0, 1)
        
        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias
            
        return _filter, _bias

    def forward(self, input: GeometricTensor):
        assert input.type == self.in_type

        if not self.training:
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter, _bias = self.expand_parameters()
        
        # use it for convolution and return the result
        output = conv_transpose2d(
                        input.tensor, _filter,
                        padding=self.padding,
                        output_padding=self.output_padding,
                        stride=self.stride,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=_bias)
        
        return GeometricTensor(output, self.out_type)
    
    def train(self, mode=True):
        
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
            if hasattr(self, "expanded_bias"):
                del self.expanded_bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
    
            _filter, _bias = self.expand_parameters()
            
            self.register_buffer("filter", _filter)
            if _bias is not None:
                self.register_buffer("expanded_bias", _bias)
            else:
                self.expanded_bias = None

        return super(R2ConvTransposed, self).train(mode)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
        
        ho = math.floor((hi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo
        
    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True):
        
        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 33
        last_downsampling = 5
        first_downsampling = 5
        
        initial_size = (feature_map_size * last_downsampling - 1 + self.kernel_size) * first_downsampling
        
        c = self.in_type.size
    
        # x = torch.randn(3, c, 10, 10)
        
        import matplotlib.image as mpimg
        from skimage.measure import block_reduce
        from skimage.transform import resize

        x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size),
            anti_aliasing=True
        )
        
        x = x / 255.0 - 0.5
        
        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]
    
            x = np.concatenate(to_stack, axis=1)
    
        x = GeometricTensor(torch.FloatTensor(x), self.in_type)
        
        def shrink(t: GeometricTensor, s) -> GeometricTensor:
            return GeometricTensor(torch.FloatTensor(block_reduce(t.tensor.detach().numpy(), s, func=np.mean)), t.type)
        
        errors = []
    
        for el in self.space.testing_elements:
            
            out1 = self(shrink(x, (1, 1, 5, 5))).transform(el).tensor.detach().numpy()
            out2 = self(shrink(x.transform(el), (1, 1, 5, 5))).tensor.detach().numpy()
            
            out1 = block_reduce(out1, (1, 1, 5, 5), func=np.mean)
            out2 = block_reduce(out2, (1, 1, 5, 5), func=np.mean)
            
            b, c, h, w = out2.shape

            center_mask = np.zeros((2, h, w))
            center_mask[1, :, :] = np.arange(0, w) - w / 2
            center_mask[0, :, :] = np.arange(0, h) - h / 2
            center_mask[0, :, :] = center_mask[0, :, :].T
            center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 < (h / 4) ** 2

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]
            
            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)
            
            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum
    
            if verbose:
                print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())
        
            # tol = rtol*(np.abs(out1) + np.abs(out2)) + atol
            tol = rtol * esum + atol
            
            if np.any(errs > tol) and verbose:
                # print(errs[errs > tol])
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])
            
            if assertion:
                assert np.all(errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors
        
        # init.deltaorthonormal_init(self.weights.data, self.basisexpansion)
        # filter = self.basisexpansion()
        # center = self.s // 2
        # filter = filter[..., center, center]
        # assert torch.allclose(torch.eye(filter.shape[1]), filter.t() @ filter, atol=3e-7)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ConvTranspose2d` module and set to "eval" mode.

        """
    
        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias
    
        # build the PyTorch Conv2d module
        has_bias = self.bias is not None
        conv = torch.nn.ConvTranspose2d(self.in_type.size,
                                       self.out_type.size,
                                       self.kernel_size,
                                       padding=self.padding,
                                       stride=self.stride,
                                       dilation=self.dilation,
                                       groups=self.groups,
                                       bias=has_bias)
        
        # set the filter and the bias
        conv.weight.data[:] = _filter.data
        if has_bias:
            conv.bias.data[:] = _bias.data
    
        return conv

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
        s = ('{in_type}, {out_type}, kernel_size={kernel_size}, stride={stride}')
        if self.padding != 0 and self.padding != (0, 0):
            s += ', padding={padding}'
        if self.output_padding != 0 and self.output_padding != (0, 0):
            s += ', output_padding={output_padding}'
        if self.dilation != 1 and self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""

    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (True) or not (False)
    
    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. In thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.
    
    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    Args:
        frequency_cutoff (float): factor controlling the bandlimiting

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff
    
    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["frequency"]) <= frequency_cutoff(attributes["radius"])
    
    return bl_filter


def get_grid_coords(kernel_size: int, dilation: int = 1):
    
    actual_size = dilation * (kernel_size -1) + 1
    
    origin = actual_size / 2 - 0.5
    
    points = []
    
    for y in range(kernel_size):
        y *= dilation
        for x in range(kernel_size):
            x *= dilation
            p = (x - origin, -y + origin)
            points.append(p)
    
    points = np.array(points)
    assert points.shape == (kernel_size ** 2, 2), points.shape
    return points.T


def compute_basis_params(kernel_size: int,
                         frequencies_cutoff: Union[float, Callable[[float], float]] = None,
                         rings: List[float] = None,
                         sigma: List[float] = None,
                         dilation: int = 1,
                         custom_basis_filter: Callable[[dict], bool] = None,
                         ):
    
    # compute the coordinates of the centers of the cells in the grid where the filter is sampled
    grid = get_grid_coords(kernel_size, dilation)
    
    max_radius = np.sqrt((grid **2).sum(1)).max()
    # max_radius = kernel_size // 2
    
    # by default, the number of rings equals half of the filter size
    if rings is None:
        n_rings = math.ceil(kernel_size / 2)
        # if self.group.order() > 0:
        #     # compute the number of edges of the polygon inscribed in the filter (which is a square)
        #     # whose points stay inside the filter under the action of the group
        #     # the number of edges is lcm(group's order, 4)
        #     n_edges = self.group.order()
        #     while n_edges % 4 > 0:
        #         n_edges *= 2
        #     # the largest ring we can sample has radius equal to the circumradius of the polygon described above
        #     n_rings /= math.cos(math.pi/n_edges)
        
        # n_rings = s // 2 + 1
        
        # rings = torch.linspace(1 - s % 2, s // 2, n_rings)
        rings = torch.linspace(0, (kernel_size - 1) // 2, n_rings).tolist()
    
    assert all([max_radius >= r >= 0 for r in rings])
    
    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.:
                sigma[i] = 0.005
                
    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)
        
    # TODO - use a string name for this setting
    if frequencies_cutoff is None:
        frequencies_cutoff = -1.
    
    if isinstance(frequencies_cutoff, float):
        if frequencies_cutoff == -3:
            frequencies_cutoff = _manual_fco3(kernel_size // 2)
        elif frequencies_cutoff == -2:
            frequencies_cutoff = _manual_fco2(kernel_size // 2)
        elif frequencies_cutoff == -1:
            frequencies_cutoff = _manual_fco1(kernel_size // 2)
        else:
            frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r
    
    # check if the object is a callable function
    assert callable(frequencies_cutoff)
    
    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

    fco_filter = bandlimiting_filter(frequencies_cutoff)

    if custom_basis_filter is not None:
        basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (custom_basis_filter(d) and fco_filter(d))
    else:
        basis_filter = fco_filter
    
    return grid, basis_filter, rings, sigma, maximum_frequency


def _manual_fco3(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else 1 if r == max_radius else 2
        return max_freq
    
    return bl_filter


def _manual_fco2(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 1 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    
    return bl_filter


def _manual_fco1(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 2 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    
    return bl_filter



