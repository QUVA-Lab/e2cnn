from __future__ import annotations

from e2cnn import gspaces
from e2cnn import kernels

from .general_r2 import GeneralOnR2
from .utils import rotate_array

from e2cnn.group import Representation
from e2cnn.group import Group
from e2cnn.group import DihedralGroup
from e2cnn.group import O2
from e2cnn.group import dihedral_group
from e2cnn.group import o2_group

import numpy as np


from typing import Tuple, Union, Callable, List

__all__ = ["FlipRot2dOnR2"]


class FlipRot2dOnR2(GeneralOnR2):
    
    def __init__(self, N: int = None, maximum_frequency: int = None, axis: float = np.pi / 2, fibergroup: Group = None):
        r"""
        
        Describes reflectional and rotational symmetries of the plane :math:`\R^2`.
        
        Reflections are applied with respect to the line through the origin with an angle ``axis`` degrees with respect
        to the *X*-axis.
        
        If ``N > 1``, the class models reflections and *discrete* rotations by angles multiple of :math:`\frac{2\pi}{N}`
        (:class:`~e2cnn.group.DihedralGroup`).
        Otherwise, if ``N=-1``, the class models reflections and *continuous* planar rotations
        (:class:`~e2cnn.group.O2`).
        In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
        :class:`~e2cnn.group.O2` (see its documentation for more details)
        
        .. note ::
            
            All axes obtained from the axis defined by ``axis`` with a rotation in the symmetry group are equivalent.
            For instance, if ``N = 4``, an axis :math:`\beta` is equivalent to the axis :math:`\beta + \pi/2`.
            It follows that for ``N = -1``, i.e. in case the symmetry group contains all continuous rotations, any
            reflection axis is theoretically equivalent.
            In practice, though, a basis for equivariant convolutional filter sampled on a grid is affected by the
            specific choice of the axis. In general, choosing an axis aligned with the grid (an horizontal or a
            vertical axis, i.e. :math:`0` or :math:`\pi/2`) is suggested.
        
        Args:
            N (int): number of discrete rotations (integer greater than 1) or -1 for continuous rotations
            maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.O2` 's irreps if ``N = -1``
            axis (float, optional): the slope of the axis of the flip (in radians)
            fibergroup (Group, optional): use an already existing instance of the symmetry group.
                    In that case only the parameter ``axis`` should be used.
        
        Attributes:
            ~.axis (float): Angle with respect to the horizontal axis which defines the reflection axis.
            
        """

        assert N is not None or fibergroup is not None, "Error! Either use the parameter `N` or the parameter `group`!"

        if fibergroup is not None:
            assert isinstance(fibergroup, DihedralGroup) or isinstance(fibergroup, O2)
            assert maximum_frequency is None, "Maximum Frequency can't be set when the group is already provided in input"
            N = fibergroup.rotation_order
    
        assert isinstance(N, int)

        self.axis = axis
        
        if N > 1:
            assert maximum_frequency is None, "Maximum Frequency can't be set for finite cyclic groups"
            name = 'Flip_{}-Rotations(f={:.5f})'.format(N, self.axis)
        elif N == -1:
            name = 'Flip_Continuous-Rotations(f={:.5f})'.format(self.axis)
            # self.axis = np.pi/2
        else:
            raise ValueError(f'Error! "N" has to be an integer greater than 1 or -1, but got {N}')
    
        if fibergroup is None:
            if N > 1:
                fibergroup = dihedral_group(N)
            elif N == -1:
                fibergroup = o2_group(maximum_frequency)
    
        super(FlipRot2dOnR2, self).__init__(fibergroup, name)

    def restrict(self, id: Tuple[Union[None, float, int], int]) -> Tuple[gspaces.GSpace, Callable, Callable]:
        r"""

        Build the :class:`~e2cnn.group.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``, which is a tuple :math:`(k, M)`.
        
        Here, :math:`M` is a positive integer indicating the number of discrete rotations in the subgroup while
        :math:`k` is either ``None`` (no reflections) or an angle indicating the axis of reflection.
        If the current fiber group is :math:`D_N` (:class:`~e2cnn.group.DihedralGroup`), then :math:`M` needs to divide
        :math:`N` and :math:`k` needs to be an integer in :math:`\{0, \dots, \frac{N}{M}-1\}`.
        Otherwise, :math:`M` can be any positive integer while :math:`k` needs to be a real number in
        :math:`[0, \frac{2\pi}{M}]`.
        
        Valid combinations are:
        
        - (``None``, :math:`1`): restrict to no reflection and rotation symmetries
        
        - (``None``, :math:`M`): restrict to only the :math:`M` rotations generated by :math:`r_{2\pi/M}`.
        
        - (:math:`0`, :math:`1`): restrict to only reflections :math:`\langle f \rangle` around the same axis as in the current group
        
        - (:math:`0`, :math:`M`): restrict to reflections and :math:`M` rotations generated by :math:`r_{2\pi/M}` and :math:`f`
        
        If the current fiber group is :math:`D_N` (an instance of :class:`~e2cnn.group.DihedralGroup`):
        
        - (:math:`k`, :math:`M`): restrict to reflections :math:`\langle r_{k\frac{2\pi}{N}} f \rangle` around the axis of the current G-space rotated by :math:`k\frac{\pi}{N}` and :math:`M` rotations generated by :math:`r_{2\pi/M}`
        
        If the current fiber group is :math:`O(2)` (an instance of :class:`~e2cnn.group.O2`):
        
        - (:math:`\theta`, :math:`M`): restrict to reflections :math:`\langle r_{\theta} f \rangle` around the axis of the current G-space rotated by :math:`\frac{\theta}{2}` and :math:`M` rotations generated by :math:`r_{2\pi/M}`
        
        - (``None``, :math:`-1`): restrict to all (continuous) rotations
        
        Args:
            id (tuple): the id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)


        """
    
        subgroup, mapping, child = self.fibergroup.subgroup(id)
        
        if id[0] is not None:
            # the new flip axis is the previous one rotated by the new chosen axis for the flip
            # notice that the actual group element used to generate the subgroup does not correspond to the flip axis
            # but to 2 times that angle
            
            if self.fibergroup.order() > 1:
                n = self.fibergroup.rotation_order
                rotation = id[0] * 2.0 * np.pi / n
            else:
                rotation = id[0]
                
            new_axis = divmod(self.axis + 0.5*rotation, 2*np.pi)[1]

        if id[0] is None and id[1] == 1:
            return gspaces.TrivialOnR2(fibergroup=subgroup), mapping, child
        elif id[0] is None and (id[1] > 1 or id[1] == -1):
            return gspaces.Rot2dOnR2(fibergroup=subgroup), mapping, child
        elif id[0] is not None and id[1] == 1:
            return gspaces.Flip2dOnR2(fibergroup=subgroup, axis=new_axis), mapping, child
        elif id[0] is not None:
            return gspaces.FlipRot2dOnR2(fibergroup=subgroup, axis=new_axis), mapping, child
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

        If this :class:`~e2cnn.group.GSpace` includes only a discrete number of rotations (``n > 1``), either
        ``maximum_frequency`` or ``maximum_offset``  must be set in the keywords arguments.

        Args:
            in_repr: the input representation
            out_repr: the output representation
            rings: radii of the rings where to sample the bases
            sigma: parameters controlling the width of each ring where the bases are sampled.

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
        
            return kernels.kernels_DN_act_R2(in_repr, out_repr, rings, sigma,
                                             axis=self.axis,
                                             max_frequency=maximum_frequency,
                                             max_offset=maximum_offset)
        else:
            return kernels.kernels_O2_act_R2(in_repr, out_repr, rings, sigma, axis=self.axis)

    def _basespace_action(self, input: np.ndarray, element: Tuple[int, Union[float, int]]) -> np.ndarray:
    
        assert self.fibergroup.is_element(element)
        
        if self.fibergroup.order() > 1:
            n = self.fibergroup.rotation_order
            
            rotation = element[1] * 2.0 * np.pi / n
    
        else:
            rotation = element[1]
        
        output = input
        
        if element[0]:
            output = output[..., ::-1, :]
            rotation += 2*self.axis
            
        if rotation != 0.:
            output = rotate_array(output, rotation)
        else:
            output = output.copy()
    
        return output

    def __eq__(self, other):
        if isinstance(other, FlipRot2dOnR2):
            return self.fibergroup == other.fibergroup
        else:
            return False
    
    def __hash__(self):
        return hash(self.name)
