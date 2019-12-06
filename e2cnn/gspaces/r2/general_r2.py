from __future__ import annotations

from e2cnn.gspaces import GSpace
from e2cnn import kernels
from e2cnn.group import Group
from e2cnn.group import Representation

from abc import abstractmethod
from typing import List, Union

from collections import defaultdict

__all__ = ["GeneralOnR2"]


class GeneralOnR2(GSpace):
    
    def __init__(self, fibergroup: Group, name: str):
        r"""
        
        Abstract class for the G-spaces which define the symmetries of the plane :math:`\R^2`.
        
        Args:
            fibergroup (Group): group of origin-preserving symmetries (fiber group)
            name (str): identification name
            
        """
        super(GeneralOnR2, self).__init__(fibergroup, 2, name)
        
        # in order to not recompute the basis for the same intertwiner as many times as it appears, we store the basis
        # in these dictionaries the first time we compute it
        
        # Store the computed intertwiners between irreps
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_irrep, output_irrep) pairs to the corresponding basis
        self._irreps_intertwiners_basis_memory = defaultdict(lambda: dict())

        # Store the computed intertwiners between general representations
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_repr, output_repr) pairs to the corresponding basis
        self._fields_intertwiners_basis_memory = defaultdict(dict)

    def build_kernel_basis(self,
                           in_repr: Representation,
                           out_repr: Representation,
                           sigma: Union[float, List[float]],
                           rings: List[float],
                           **kwargs) -> kernels.KernelBasis:
        r"""
        
        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~e2cnn.gspaces.GSpace`.
        
        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^2
        
        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.
        
        
        Because the equivariance constraints only restrict the angular part of the kernels, any radial profile is
        permitted.
        The basis for the radial profile used here contains rings with different radii (``rings``)
        associated with (possibly different) widths (``sigma``).
        A ring is implemented as a Gaussian function over the radial component, centered at one radius
        (see also :class:`~e2cnn.kernels.GaussianRadialProfile`).
        
        .. note ::
            This method is a wrapper for the functions building the bases which are defined in :doc:`e2cnn.kernels`:
            
            - :meth:`e2cnn.kernels.kernels_O2_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_SO2_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_DN_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_CN_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_Flip_act_R2`,
            
            - :meth:`e2cnn.kernels.kernels_Trivial_act_R2`
            
            
        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            sigma (list or float): parameters controlling the width of each ring of the radial profile.
                    If only one scalar is passed, it is used for all rings
            rings (list): radii of the rings defining the radial profile
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            the analytical basis
        
        """
        
        assert isinstance(in_repr, Representation)
        assert isinstance(out_repr, Representation)
        
        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup
        
        if isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        assert all([s > 0. for s in sigma])
        assert len(sigma) == len(rings)
        
        # build the key
        key = dict(**kwargs)
        key["sigma"] = tuple(sigma)
        key["rings"] = tuple(rings)
        key = tuple(sorted(key.items()))

        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[key]:
            # TODO - we could use a flag in the args to choose whether to store it or not
            
            basis = self._basis_generator(in_repr, out_repr, rings, sigma, **kwargs)
       
            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)] = basis

        # return the dictionary with all the basis built for this filter size
        return self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)]
    
    @abstractmethod
    def _basis_generator(self,
                         in_repr: Representation,
                         out_repr: Representation,
                         rings: List[float],
                         sigma: List[float],
                         **kwargs):
        pass

