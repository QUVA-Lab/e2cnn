
from __future__ import annotations

from e2cnn import gspaces
from e2cnn import kernels

from .general_r2 import GeneralOnR2
from .utils import rotate_array

from typing import Union, Tuple, Callable, List

from e2cnn.group import Representation
from e2cnn.group import Group
from e2cnn.group import CyclicGroup
from e2cnn.group import SO2
from e2cnn.group import cyclic_group
from e2cnn.group import so2_group

import numpy as np


__all__ = ["Rot2dOnR2"]


class Rot2dOnR2(GeneralOnR2):

    def __init__(self, N: int = None, maximum_frequency: int = None, fibergroup: Group = None):
        r"""

        Describes rotation symmetries of the plane :math:`\R^2`.

        If ``N > 1``, the class models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
        (:class:`~e2cnn.group.CyclicGroup`).
        Otherwise, if ``N=-1``, the class models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
        In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
        :class:`~e2cnn.group.SO2` (see its documentation for more details)

        Args:
            N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
            maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``
            fibergroup (Group, optional): use an already existing instance of the symmetry group.
                   In that case, the other parameters should not be provided.

        """
        
        assert N is not None or fibergroup is not None, "Error! Either use the parameter `N` or the parameter `group`!"
    
        if fibergroup is not None:
            assert isinstance(fibergroup, CyclicGroup) or isinstance(fibergroup, SO2)
            assert maximum_frequency is None, "Maximum Frequency can't be set when the group is already provided in input"
            N = fibergroup.order()
            
        assert isinstance(N, int)
        
        if N > 1:
            assert maximum_frequency is None, "Maximum Frequency can't be set for finite cyclic groups"
            name = '{}-Rotations'.format(N)
        elif N == -1:
            name = 'Continuous-Rotations'
        else:
            raise ValueError(f'Error! "N" has to be an integer greater than 1 or -1, but got {N}')

        if fibergroup is None:
            if N > 1:
                fibergroup = cyclic_group(N)
            elif N == -1:
                fibergroup = so2_group(maximum_frequency)

        super(Rot2dOnR2, self).__init__(fibergroup, name)

    def restrict(self, id: int) -> Tuple[gspaces.GSpace, Callable, Callable]:
        r"""

        Build the :class:`~e2cnn.group.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.
        
        ``id`` is a positive integer :math:`M` indicating the number of rotations in the subgroup.
        If the current fiber group is :math:`C_N` (:class:`~e2cnn.group.CyclicGroup`), then :math:`M` needs to divide
        :math:`N`. Otherwise, :math:`M` can be any positive integer.
        
        Args:
            id (int): the number :math:`M` of rotations in the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)


        """
        subgroup, mapping, child = self.fibergroup.subgroup(id)

        if id > 1:
            return gspaces.Rot2dOnR2(fibergroup=subgroup), mapping, child
        elif id == 1:
            return gspaces.TrivialOnR2(fibergroup=subgroup), mapping, child
        else:
            raise ValueError(f"id {id} not recognized!")

    def _basis_generator(self,
                         in_repr: Representation,
                         out_repr: Representation,
                         rings: List[float],
                         sigma: List[float],
                         **kwargs,
                         ) -> kernels.KernelBasis:
        r"""
        Method that builds the analitical basis that spans the space of equivariant filters which
        are intertwiners between the representations induced from the representation ``in_repr`` and ``out_repr``.

        If this :class:`~e2cnn.gspaces.GSpace` includes only a discrete number of rotations (``N > 1``), either ``maximum_frequency``
        or ``maximum_offset``  must be set in the keywords arguments.

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            rings (list): radii of the rings where to sample the bases
            sigma (list): parameters controlling the width of each ring where the bases are sampled.

        Keyword Args:
            maximum_frequency (int): the maximum frequency allowed in the basis vectors
            maximum_offset (int): the maximum frequencies offset for each basis vector with respect to its base ones (sum and difference of the frequencies of the input and the output representations)

        Returns:
            the basis built

        """
    
        if self.fibergroup.order() > 0:
            maximum_frequency = None
            maximum_offset = None
    
            if 'maximum_frequency' in kwargs and kwargs['maximum_frequency'] is not None:
                maximum_frequency = kwargs['maximum_frequency']
                assert isinstance(maximum_frequency, int) and maximum_frequency >= 0
    
            if 'maximum_offset' in kwargs and kwargs['maximum_offset'] is not None:
                maximum_offset = kwargs['maximum_offset']
                assert isinstance(maximum_offset, int) and maximum_offset >= 0
    
            assert (maximum_frequency is not None or maximum_offset is not None), \
                'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'
            
            return kernels.kernels_CN_act_R2(in_repr, out_repr, rings, sigma,
                                             maximum_frequency,
                                             max_offset=maximum_offset)
        else:
            return kernels.kernels_SO2_act_R2(in_repr, out_repr, rings, sigma)

    def _basespace_action(self, input: np.ndarray, element: Union[float, int]) -> np.ndarray:
    
        assert self.fibergroup.is_element(element)
        
        if self.fibergroup.order() > 1:
            n = self.fibergroup.order()
            
            rotation = element * 2.0 * np.pi / n
        else:
            rotation = element
            
        output = rotate_array(input, rotation)
            
        return output

    def __eq__(self, other):
        if isinstance(other, Rot2dOnR2):
            return self.fibergroup == other.fibergroup
        else:
            return False

    def __hash__(self):
        return hash(self.name)
