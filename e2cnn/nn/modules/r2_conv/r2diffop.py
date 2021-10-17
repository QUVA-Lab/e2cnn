
import warnings

from e2cnn.diffops.utils import (
    load_cache,
    store_cache,
    required_points,
    largest_possible_order,
)

from torch.nn.functional import conv2d, pad

from e2cnn.nn import init
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.gspaces import *
from e2cnn.diffops import DiscretizationArgs

from ..equivariant_module import EquivariantModule

from .basisexpansion import BasisExpansion
from .basisexpansion_blocks import BlocksBasisExpansion

from typing import Callable, Union, Tuple, List

import torch
from torch.nn import Parameter
import numpy as np
import math

__all__ = ["R2Diffop"]


class R2Diffop(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int = None,
                 accuracy: int = None,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 maximum_order: int = None,
                 maximum_power: int = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 angle_offset: float = None,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 cache: Union[bool, str] = False,
                 rbffd: bool = False,
                 radial_basis_function: str = "ga",
                 smoothing: float = None,
                 ):
        r"""
        
        
        G-steerable planar partial differential operator mapping between the
        input and output :class:`~e2cnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^2\rtimes G` where :math:`G` is the
        :attr:`e2cnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.
        
        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~e2cnn.nn.R2Diffop` guarantees an equivariant mapping
        
        .. math::
            D [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [Df] \qquad\qquad \forall g \in G, u \in \R^2
            
        where the transformation of the input and output fields are given by
 
        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable PDOs is guaranteed by restricting the space of PDOs to an
        equivariant subspace.

        During training, in each forward pass the module expands the basis of G-steerable PDOs with learned weights
        before calling :func:`torch.nn.functional.conv2d`.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the PDO remains.
        
        .. warning ::
            
            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~e2cnn.nn.R2Diffop.filter` and
            :attr:`~e2cnn.nn.R2Diffop.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`e2cnn.nn.R2Diffop.train`.
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
 
 
        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`e2cnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~e2cnn.nn.init.generalized_he_init`.
        
        .. warning ::
            
            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~e2cnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.
        
        A reasonable default is to only set the ``kernel_size`` and leave all other options
        on their defaults. However, you might get considerable performance improvements
        by setting ``smoothing`` to something other than ``None`` (``kernel_size / 4`` is
        a sane default, see below for details).
        
        If you want to modify ``accuracy`` or ``maximum_order``, you will need to take
        into account how they are related to ``kernel_size``: it is possible to set any two
        of ``kernel_size``, ``accuracy`` and ``maximum_order``, in which case the third
        one will be determined automatically. Alternatively, you can set either ``kernel_size``
        or ``maximum_order``, in which case a sane default will be used for ``accuracy``. 
        The relation between the three is approximately :math:`\text{kernel size} \approx \text{accuracy} + \text{order}`,
        though this formula is off by one in some cases.
        A larger maximum order will lead to more basis filters and this more parameters.
        A larger accuracy (i.e. larger kernel size at constant order)
        might lead to lower equivariance errors, though whether this actually happens may
        depend on your exact setup.
        
        The parameters ``basisexpansion``, ``maximum_power``,  and ``maximum_offset`` are
        optional parameters used to control how the basis for the PDOs is built, how it is sampled on the filter
        grid and how it is expanded to build the filter. We suggest to keep these default values.
        
        .. warning::
            The discretization of the differential operators relies on two external packages: `sympy <https://docs.sympy.org/>`_ and
            `rbf <https://rbf.readthedocs.io>`_. If they are not available, an error is raised.
            
        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int, optional): the size of the (square) filter. This can be chosen automatically,
                see above for details.
            accuracy (int, optional): the desired asymptotic accuracy for the PDO discretization,
                affects the ``kernel_size``. See above for details.
            padding (int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            stride (int, optional): the stride of the kernel. Default: ``1``
            dilation (int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basisexpansion (str, optional): the basis expansion algorithm to use
            maximum_order (int, optional): the largest derivative order to allow
                as part of the basis. Larger maximum orders require larger kernel sizes,
                see above for details.
            maximum_power (int, optional): the maximum power of the Laplacian that will be used
                for constructing the basis. If this is not ``None``, it places a restriction on
                the basis elements, *in addition to* the restriction given by ``maximum_order``.
                We suggest to leave this setting on its default unless you have a good reason
                to change it.
            maximum_offset (int, optional): number of additional (aliased) frequencies in the intertwiners for finite
                    groups. By default (``None``), all additional frequencies allowed by the frequencies cut-off
                    are used.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant PDOs.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
            cache (bool or str, optional): Discretizing the PDOs can take a bit longer than
                for kernels, so we provide the option to cache PDOs on disk. Our suggestion is
                to keep the cache off (default) and only activate it if discretizing the PDOs
                is in fact a bottleneck for your setup (it often is not). Setting ``cache`` to
                ``True`` will load an existing cache before instantiating the layer and will
                write to the cache afterwards. You can also set ``cache`` to ``load`` or ``store``
                to only do one of these.

                All :class:`~e2cnn.nn.R2Diffop` layers share the PDO cache in memory.
                If you have several :class:`~e2cnn.nn.R2Diffop` layers inside your model,
                we therefore recommend to leave ``cache`` to ``False`` and instead call
                :func:`e2cnn.diffops.load_cache` before instantiating the model, and :func:`e2cnn.diffops.store_cache`
                afterwards to save the PDOs for the next run of the program.
                This will avoid unnecessary reads/writes from/to disk.
            rbffd (bool, optional): if set to ``True``, use RBF-FD discretization instead of
                finite differences (the default). We suggest leaving this to ``False`` unless
                you have a specific reason for wanting to use RBF-FD.
            radial_basis_function (str, optional): which RBF to use (only relevant for RBF-FD).
                Can be any of the abbreviations in `this list <https://rbf.readthedocs.io/en/latest/basis.html>`_.
                The default is to use Gaussian RBFs because this always avoids singularity issues.
                But other RBFs, such as polyharmonic splines, may work better if they are applicable.
            smoothing (float, optional): if not ``None``, discretization will be performed
                with derivatives of Gaussians as stencils. This is similar to smoothing
                with a Gaussian before applying the PDO, though there are slight technical
                differences. ``smoothing`` is the standard deviation (in pixels) of the Gaussian,
                meaning that larger values correspond to stronger smoothing.
                A reasonable value would be about ``kernel_size / 4`` but you might want to experiment
                a bit with this parameter.
        
        Attributes:
            
            ~.weights (torch.Tensor): the learnable parameters which are used to expand the PDO
            ~.filter (torch.Tensor): the convolutional stencil obtained by expanding the parameters
                                    in :attr:`~e2cnn.nn.R2Diffop.weights`
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~e2cnn.nn.R2Diffop.bias`
        
        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)

        super(R2Diffop, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type

        if cache and cache != "store":
            # Load the cached lambdas for RBFs if they exist
            load_cache()

        # out of kernel_size, accuracy and maximum_order, exactly two must be known,
        # the third one can then be determined automatically.
        # To provide sane defaults, we will also allow only kernel_size or maximum_order
        # to be set, in that case accuracy will become 2.
        if kernel_size is None:
            assert maximum_order is not None
            if accuracy is None:
                accuracy = 2 if (maximum_order > 0) else 1
            # TODO: Ideally, we should look at the basis, maybe the maximum_order isn't
            # reached (e.g. if it is odd but all basis diffops are even). In that case,
            # we could perhaps get away with a smaller kernel
            kernel_size = required_points(maximum_order, accuracy)
        elif maximum_order is None:
            assert kernel_size is not None
            if accuracy is None:
                accuracy = 2 if (kernel_size > 1) else 1
            maximum_order = largest_possible_order(kernel_size, accuracy)
            if maximum_order < 2:
                warnings.warn(f"Maximum order is only {maximum_order} for kernel size "
                              f"{kernel_size} and desired accuracy {accuracy}. This may "
                              "lead to a small basis. If this is unintentional, consider "
                              "increasing the kernel size.")
        elif accuracy is None:
            if kernel_size < required_points(maximum_order, 2):
                warnings.warn(f"Small kernel size: {kernel_size} x {kernel_size} kernel "
                              f"is used for differential operators of order up to {maximum_order}. "
                              "This may lead to bad approximations, consider using a larger kernel "
                              "or setting the desired accuracy instead of the kernel size.")
        else:
            # all three are set
            raise ValueError("At most two of kernel size, maximum order and accuracy can bet set, "
                             "see documentation for details.")
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.groups = groups

        if isinstance(padding, tuple) and len(padding) == 2:
            _padding = padding
        elif isinstance(padding, int):
            _padding = (padding, padding)
        else:
            raise ValueError('padding needs to be either an integer or a tuple containing two integers but {} found'.format(padding))
        
        padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in padding_modes:
            raise ValueError("padding_mode must be one of [{}], but got padding_mode='{}'".format(padding_modes, padding_mode))
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(_padding) for _ in range(2))
        
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
            
            # finally, retrieve the type associated to a single group in input.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from one input
            # group to all output groups. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            in_type = in_type.index_select(list(range(in_size)))
        
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

        grid, basis_filter, params = compute_basis_params(kernel_size,
                                                          dilation,
                                                          basis_filter,
                                                          maximum_order,
                                                          maximum_power,
                                                          rbffd,
                                                          radial_basis_function,
                                                          smoothing,
                                                          angle_offset)

        
        # BasisExpansion: submodule which takes care of building the filter
        self._basisexpansion = None
        
        # notice that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`
        if basisexpansion == 'blocks':
            self._basisexpansion = BlocksBasisExpansion(in_type, out_type,
                                                        self.space.build_diffop_basis,
                                                        points=grid,
                                                        maximum_offset=maximum_offset,
                                                        **params,
                                                        basis_filter=basis_filter,
                                                        recompute=recompute)

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)
        
        if self.basisexpansion.dimension() == 0:
            raise ValueError('''
                The basis for the steerable filter is empty!
                Tune the `frequencies_cutoff`, `kernel_size`, `rings`, `sigma` or `basis_filter` parameters to allow
                for a larger basis.
            ''')

        self.weights = Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        self.register_buffer("filter", torch.zeros(out_type.size, in_type.size, kernel_size, kernel_size))
        
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            init.generalized_he_init(self.weights.data, self.basisexpansion)

        if cache and cache != "load":
            store_cache()
    
    @property
    def basisexpansion(self) -> BasisExpansion:
        r"""
        Submodule which takes care of building the filter.
        
        It uses the learnt ``weights`` to expand a basis and returns a filter in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the PDO in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^2)`, where :math:`s` is the ``kernel_size``.
        
        """
        return self._basisexpansion
    
    def expand_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        
        Expand the filter in terms of the :attr:`e2cnn.nn.R2Diffop.weights` and the
        expanded bias in terms of :class:`e2cnn.nn.R2Diffop.bias`.
        
        Returns:
            the expanded filter and bias

        """
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)
        
        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias
            
        return _filter, _bias

    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.
        
        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``

        Returns:
            output feature field transforming according to ``out_type``
            
        """
        
        assert input.type == self.in_type

        if not self.training:
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter, _bias = self.expand_parameters()
        
        # use it for convolution and return the result
        
        if self.padding_mode == 'zeros':
            output = conv2d(input.tensor, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups,
                            bias=_bias)
        else:
            output = conv2d(pad(input.tensor, self._reversed_padding_repeated_twice, self.padding_mode),
                            _filter,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=(0,0),
                            groups=self.groups,
                            bias=_bias)

        return GeometricTensor(output, self.out_type)
    
    def train(self, mode=True):
        r"""
        
        If ``mode=True``, the method sets the module in training mode and discards the :attr:`~e2cnn.nn.R2Diffop.filter`
        and :attr:`~e2cnn.nn.R2Diffop.expanded_bias` attributes.
        
        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the filter and the bias using
        the current values of the trainable parameters and store them in :attr:`~e2cnn.nn.R2Diffop.filter` and
        :attr:`~e2cnn.nn.R2Diffop.expanded_bias` such that they are not recomputed at each forward pass.
        
        .. warning ::
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
        
        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.

        """
        
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

        return super(R2Diffop, self).train(mode)

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
        
            tol = rtol * esum + atol
            
            if np.any(errs > tol) and verbose:
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
        Export this module to a normal PyTorch :class:`torch.nn.Conv2d` module and set to "eval" mode.

        """
        
        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias

        if self.padding_mode not in ['zeros']:
            x, y = torch.__version__.split('.')[:2]
            if int(x) < 1 or int(y) < 5:
                if self.padding_mode == 'circular':
                    raise ImportError(
                        "'{}' padding mode had some issues in old `torch` versions. Therefore, we only support conversion from version 1.5 but only version {} is installed.".format(
                            self.padding_mode, torch.__version__
                        )
                    )

                else:
                    raise ImportError(
                        "`torch` supports '{}' padding mode only from version 1.5 but only version {} is installed.".format(
                            self.padding_mode, torch.__version__
                        )
                    )

        # build the PyTorch Conv2d module
        has_bias = self.bias is not None
        conv = torch.nn.Conv2d(self.in_type.size,
                               self.out_type.size,
                               self.kernel_size,
                               padding=self.padding,
                               padding_mode=self.padding_mode,
                               stride=self.stride,
                               dilation=self.dilation,
                               groups=self.groups,
                               bias=has_bias)

        # set the filter and the bias
        conv.weight.data = _filter.data
        if has_bias:
            conv.bias.data = _bias.data
        
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
        if self.dilation != 1 and self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
    

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
                         dilation: int,
                         custom_basis_filter: Callable[[dict], bool],
                         maximum_order: int,
                         maximum_power: int,
                         rbffd: bool,
                         radial_basis_function: str,
                         smoothing: float,
                         angle_offset: float,
                         ):

    # compute the coordinates of the centers of the cells in the grid where the filter is sampled
    grid = get_grid_coords(kernel_size, dilation)

    if custom_basis_filter is None:
        basis_filter = order_filter(maximum_order)
    else:
        basis_filter = lambda d: custom_basis_filter(d) and order_filter(maximum_order)(d)

    if maximum_power is not None:
        maximum_power = min(maximum_power, maximum_order // 2)
    else:
        maximum_power = maximum_order // 2
    
    if smoothing is not None and rbffd:
        raise ValueError("You can't use smoothing and RBF-FD at the same time.")
    if smoothing is not None:
        method = "gauss"
    elif rbffd:
        method = "rbffd"
    else:
        method = "fd"

    disc = DiscretizationArgs(
        method=method,
        smoothing=smoothing,
        angle_offset=angle_offset,
        phi=radial_basis_function,
    )
    params = {
        # to guarantee that all relevant tensor products
        # are generated, we need Laplacian powers up to
        # half the maximum order. Anything higher would be
        # discarded anyways by the basis_filter
        "max_power": maximum_power,
        # frequencies higher than than the maximum order will be discarded anyway
        "maximum_frequency": maximum_order,
        "discretization": disc,
    }

    return grid, basis_filter, params


def order_filter(maximum_order: int) -> Callable[[dict], bool]:
    return lambda attr: attr["order"] <= maximum_order
