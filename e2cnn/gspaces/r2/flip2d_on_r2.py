
from __future__ import annotations

from e2cnn import gspaces
from e2cnn import kernels

from .general_r2 import GeneralOnR2
from .utils import rotate_array

from e2cnn.group import Group
from e2cnn.group import Representation
from e2cnn.group import CyclicGroup
from e2cnn.group import cyclic_group

import numpy as np


from typing import Tuple, Callable, List

__all__ = ["Flip2dOnR2"]


class Flip2dOnR2(GeneralOnR2):
    
    def __init__(self,
                 axis: float = np.pi/2,
                 fibergroup: Group = None):
        r"""
        
        Describes reflectional symmetries of the plane :math:`\R^2`.
        
        Reflections are applied along the line through the origin with an angle ``axis`` degrees with respect to
        the *X*-axis.
        
        
        Args:
            axis (float, optional): the slope of the axis of the reflection (in radians).
                                    By default, the vertical axis is used (:math:`\pi/2`).
            fibergroup (Group, optional): use an already existing instance of the symmetry group
        
        Attributes:
            ~.axis (float):  Angle with respect to the horizontal axis which defines the reflection axis.
            
        """
        
        self.axis = axis
        
        if fibergroup is None:
            fibergroup = cyclic_group(2)
        else:
            assert isinstance(fibergroup, CyclicGroup) and fibergroup.order() == 2
        
        name = 'Flips'
        
        super(Flip2dOnR2, self).__init__(fibergroup, name)

    def restrict(self, id: int) -> Tuple[gspaces.GSpace, Callable, Callable]:
        r"""

        Build the :class:`~e2cnn.gspaces.GSpace` associated with the subgroup of the current fiber group identified
        by the input ``id``.

        As the reflection group contains only two elements, it has only one subgroup: the trivial group.
        The only accepted input values are ``id=1`` which returns an instance of :class:`~e2cnn.gspaces.TrivialOnR2` and
        ``id=2`` which returns a new instance of the current group.

        Args:
            id (tuple): the id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        """
        group, mapping, child = self.fibergroup.subgroup(id)
        if id == 1:
            return gspaces.TrivialOnR2(fibergroup=group), mapping, child
        else:
            return gspaces.Flip2dOnR2(axis=self.axis, fibergroup=group), mapping, child
    
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

        Either ``maximum_frequency`` or ``maximum_offset`` must be set in the keywords arguments.

        Args:
            in_repr: the input representation
            out_repr: the output representation
            rings: radii of the rings where to sample the bases
            sigma: parameters controlling the width of each ring where the bases are sampled.

        Keyword Args:
            maximum_frequency (int): the maximum frequency allowed in the basis vectors
            maximum_offset (int): the maximum frequencies offset for each basis vector with respect to its base ones
                                  (sum and difference of the frequencies of the input and the output representations)

        Returns:
            the basis built

        """
    
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
    
        return kernels.kernels_Flip_act_R2(in_repr, out_repr, rings, sigma,
                                           axis=self.axis,
                                           max_frequency=maximum_frequency,
                                           max_offset=maximum_offset)

    def _basespace_action(self, input: np.ndarray, element: int) -> np.ndarray:
    
        assert self.fibergroup.is_element(element)
        
        output = input.copy()
    
        if element:
            output = output[..., ::-1, :]
        
            if self.axis != 0:
                output = rotate_array(output, 2*self.axis)
            
        return output

    def __eq__(self, other):
        if isinstance(other, Flip2dOnR2):
            return self.fibergroup == other.fibergroup
        else:
            return False

    def __hash__(self):
        return hash(self.name)
