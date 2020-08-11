from __future__ import annotations

from e2cnn.group import Group
from .cyclicgroup import CyclicGroup
from e2cnn.group import IrreducibleRepresentation, Representation
from e2cnn.group import utils

import numpy as np
import math

from typing import List, Tuple, Callable, Iterable

__all__ = ["DihedralGroup"]

_cached_group_instances = {}


class DihedralGroup(Group):
    
    def __init__(self, N: int):
        r"""
        Build an instance of the dihedral group :math:`D_N` which contains reflections and ``N`` discrete planar
        rotations.
        The order of the group :math:`D_N` is :math:`2N`.
        
        The group elements are
        :math:`\{e,\, r,\, r^2,\, r^3,\, \dots,\, r^{N-1},\, f,\, rf,\, r^2f,\, r^3f,\, \dots,\, r^{N-1}f\}`,
        so an element of this group is either a rotation :math:`r^k` or a reflection :math:`r^kf` along an axis.
        
        Any group element is either a discrete rotation :math:`r^k` by an angle :math:`k\frac{2\pi}{N}` or a
        reflection :math:`f` followed by a rotation, i.e. :math:`r^kf`.
        As in :class:`~e2cnn.group.CyclicGroup`, combination of rotations behaves like arithmetic modulo ``N``, so
        :math:`r^a \cdot r^b = r^{\ a + b \!\! \mod \!\! N}\ `.
        Two reflections gives the identity :math:`f \cdot f = e` and a reflection commutes with a rotation by inverting
        it, i.e. :math:`r^k \cdot f = f \cdot r^{-k}`.
        
        A group element :math:`r^kf^j` is implemented as a pair :math:`(j, k)` with :math:`j \in \{0, 1\}` and
        and :math:`k \in \{0, \dots, N-1\}`.
        
        
        Args:
            N (int): number of discrete rotations in the group
            
        Attributes:
            
            ~.reflection: the reflection element
            ~.rotation_order (int): the number of discrete rotations in this group (equal to the parameter ``N``)
            
        """
        
        assert (isinstance(N, int) and N > 0)
        
        super(DihedralGroup, self).__init__("D%d" % N, False, False)
        
        self.rotation_order = N
        
        # self.elements = [(0, i * 2 * np.pi / N) for i in range(N)] + [(1, i * 2 * np.pi / N) for i in range(N)]
        self.elements = [(0, i) for i in range(N)] + [(1, i) for i in range(N)]
        
        self.elements_names = ['e'] + ['r%d' % i for i in range(1, N)]
        self.elements_names += ['f'] + ['r%df' % i for i in range(1, N)]

        self.identity = (0, 0)
        self.reflection = (1, 0)
        
        self._build_representations()
    
    def inverse(self, element: Tuple[int, int]) -> Tuple[int, int]:
        r"""
        Returns the inverse element of the input element.
        Given the element :math:`r^kf^j` as a pair :math:`(j, k)`,
        the method returns :math:`r^{-k}` (as :math:`(0, -k)`) if :math:`f = 0` and
        :math:`fr^{-k}=r^kf` (as :math:`(1, k)`) otherwise.
        
        Args:
            element (tuple): a group element :math:`r^kf^j` as a pair :math:`(j, k)`

        Returns:
            its inverse
           
        """
        return element[0], (-element[1] * (-1 if element[0] else 1)) % self.rotation_order
    
    def combine(self, e1: Tuple[int, int], e2: Tuple[int, float]) -> Tuple[int, float]:
        r"""
        Return the combination of the two input elements.
        Given two input element :math:`r^af^b` and :math:`r^cf^d`, the method returns
        :math:`r^af^b \cdot r^cf^d`.

        Args:
            e1 (tuple): a group element :math:`r^af^b` as a pair :math:`(b, a)`
            e2 (tuple): another element :math:`r^cf^d` as a pair :math:`(d, c)`

        Returns:
            their combination :math:`r^af^b \cdot r^cf^d`
           
        """
        return (e1[0] + e2[0]) % 2, (e1[1] + (-1 if e1[0] else 1) * e2[1]) % self.rotation_order
    
    def is_element(self, element: Tuple[int, int]) -> bool:
        if isinstance(element, tuple) and len(element) == 2 and isinstance(element[0], int) and isinstance(element[1],
                                                                                                           int):
            return element[0] in {0, 1} and 0 <= element[1] < self.rotation_order
        else:
            return False

    def equal(self, e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
        r"""

        Check if the two input values corresponds to the same element.

        Args:
            e1 (tuple): an element
            e2 (tuple): another element

        Returns:
            whether they are the same element

        """
        return e1[0] == e2[0] and e1[1] == e2[1]

    def testing_elements(self) -> Iterable[Tuple[int, int]]:
        r"""
        A finite number of group elements to use for testing.
        """
        return iter(self.elements)
    
    def __eq__(self, other):
        if not isinstance(other, DihedralGroup):
            return False
        else:
            return self.name == other.name and self.rotation_order == other.rotation_order
    
    def subgroup(self, id: Tuple[int, int]) -> Tuple[Group, Callable, Callable]:
        r"""
        Restrict the current group :math:`D_{2N}` to the subgroup identified by the input ``id``, where ``id`` is a
        tuple :math:`(k, M)`.
        
        Here, :math:`M` is a positive integer indicating the number of discrete rotations in the subgroup while
        :math:`k` is either ``None`` or an integer in :math:`\{0, \dots, \frac{N}{M}-1\}`. If :math:`k` is ``None``,
        the subgroup does not contain any reflections. Otherwise, the subgroup contains the reflection :math:`r^k f`
        along the axis of the current group rotated by :math:`k\frac{\pi}{N}`.
        The order :math:`M` has to divide the rotation order :math:`N` of the current group (:math:`D_{2N}`).
        
        Valid combinations are:
        
        - (``None``, :math:`M`): restrict to the cyclic subgroup :math:`C_M` generated by :math:`\langle r^{N/M} \rangle`.
        
        - (:math:`k`, :math:`M`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r^{N/M}, r^{k}f \rangle`
        
        In particular:
        
        - (``None``, :math:`1`): restrict to the cyclic subgroup of order 1 containing only the identity
        
        - (:math:`0`, :math:`1`): restrict to the reflection group generated by :math:`\langle f \rangle`
        
        - (:math:`0`, :math:`M`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r^{N/M}, f \rangle`
        
        - (:math:`k`, :math:`1`): restrict to the reflection group generated by :math:`\langle r^{k}f \rangle = \{e, r^{k}f\}`
        
        
        Args:
            id (tuple): the identification of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
              
        """
        
        assert isinstance(id, tuple) and len(id) == 2
        assert id[0] is None or isinstance(id[0], int)
        assert isinstance(id[1], int)
        assert id[1] >= 1

        axis = id[0]
        order = id[1]
        
        assert self.rotation_order % order == 0, \
            "Error! The rotations order of the subgroups of a dihedral group has to divide the rotations order of the overgroup." \
            " %d does not divide %d " % (order, self.rotation_order)
        
        assert axis is None or 0 <= axis < self.rotation_order // order

        if id not in self._subgroups:
    
            ratio = self.rotation_order // order
            
            if id[0] is not None and id[1] == 1:
                
                # take the elements of the group generated by "r^axis f"
                sg = CyclicGroup(2)

                parent_mapping = lambda e, axis=axis: (e, axis*e)
                child_mapping = lambda e, axis=axis: None if e[1] != e[0]*axis else e[0]
            
            elif id[0] is None:
                # take the elements of the group generated by "r^ratio"
                sg = CyclicGroup(order)
                parent_mapping = lambda e, ratio=ratio: (0, e * ratio)
                child_mapping = lambda e, ratio=ratio: None if (e[0] != 0 or e[1] % ratio > 0) else int(e[1] / ratio)
                
            else:
                # take the elements of the group generated by "r^ratio" and "r^axis f"
                sg = DihedralGroup(order)
                parent_mapping = lambda e, ratio=ratio, axis=axis: (e[0], e[1] * ratio + e[0]*axis)
                child_mapping = lambda e, ratio=ratio, axis=axis: None if (e[1] - e[0]*axis) % ratio > 0 else (e[0], int((e[1] - e[0]*axis) / ratio))

            self._subgroups[id] = sg, parent_mapping, child_mapping
        
        return self._subgroups[id]

    def _restrict_irrep(self, irrep: str, id: Tuple[int, int]) -> Tuple[np.matrix, List[str]]:
        r"""
        Restrict the input irrep of current group :math:`D_{2n}` to the subgroup identified by "id".
        More precisely, "id" is a tuple :math:`(k, m)`, where :math:`m` is a positive integer indicating the number of
        rotations in the subgroup while :math:`k` is either None (no flips in the subgroup) or an integer in
        :math:`[0, \frac{n}{m}-1]` (indicating the axis of flip in the subgroup).
        The order :math:`m` has to divide the rotation order :math:`n` of the current group (:math:`D_{2n}`).

        Valid combinations are:
        - (None, m): restrict to the cyclic subgroup with order "m" :math:`C_m` generated by :math:`\langle r^{(n/m)} \rangle`.
        - (1, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle r^{n/m}, f \rangle`
        - (k, 1): restrict to the cyclic subgroup of order 2 :math:`C_2` generated by the flip :math:`\langle r^{k}f \rangle = \{e, r^{k}f\}`
        - (None, 1): restrict to the cyclic subgroup of order 1 :math:`C_1` containing only the identity
        - (k, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle r^{n/m}, r^{k}f \rangle`
        
        Args:
            irrep (str): the name/identifier of the irrep to restrict
            id (tuple): the identification of the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """

        irr = self.irreps[irrep]
        
        sg, _, _ = self.subgroup(id)
        
        irreps = []
        change_of_basis = None

        psi = lambda e, k: utils.psi(e * 2 * np.pi / self.rotation_order, k)
        
        if id[0] is not None and id[1] == 1:
            j = irr.attributes["flip_frequency"]
            k = irr.attributes["frequency"]
            if k == self.rotation_order/2:
                j = (j+id[0]) % 2
            change_of_basis = np.eye(irr.size)
            if irr.size > 1:
                irreps.append("irrep_0")
                change_of_basis = psi(0.5 * id[0], k)
            irreps.append(f"irrep_{j}")
        
        elif id[0] is None:
            
            order = id[1]
            
            f = irr.attributes["frequency"] % order
            if f > order / 2:
                f = order - f
                # change_of_basis = np.array([[1, 0], [0, -1]])
                change_of_basis = utils.chi(1)
            else:
                change_of_basis = np.eye(irr.size)
            
            r = f"irrep_{f}"
            irreps.append(r)
            if sg.irreps[r].size < irr.size:
                irreps.append(r)
            
        elif id[0] is not None and id[1] > 1:
            
            order = id[1]
            f = irr.attributes["frequency"]
            j = irr.attributes["flip_frequency"]
            if f == self.rotation_order/2:
                j = (j+id[0]) % 2
            k = f % order
            
            if k > order / 2:
                k = order - k
                # change_of_basis = np.array([[1, 0], [0, -1]])
                change_of_basis = utils.chi(1)
            else:
                change_of_basis = np.eye(irr.size)
                
            r = f"irrep_{j},{k}"
            
            if sg.irreps[r].size < irr.size:
                irreps.append(f"irrep_0,{k}")
                irreps.append(r)
            else:
                irreps.append(r)
            
            if irr.size == 2:
                # change_of_basis = phi(0.5 * id[0], f) @ change_of_basis
                change_of_basis = psi(0.5 * id[0], f) @ change_of_basis

        else:
            raise ValueError(f"id '{id}' not recognized")
        
        return change_of_basis, irreps
    
    def _build_representations(self):
        r"""
        Build the irreps and the regular representation for this group

        """
        
        n = self.rotation_order
        
        # Build all the Irreducible Representations
        
        # add Trivial representation
        j, k = 0, 0
        self.irrep(j, k)
        
        j = 1
        
        for k in range(0, int(n//2)+1):
            self.irrep(j, k)
        
        if n % 2 == 0:
            j = 0
            self.irrep(j, k)
            
        # Build all Representations
        
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**self.irreps)

        # build the regular representation
        
        # N.B.: it represents the LEFT-ACTION of the elements
        self.representations['regular'] = self.regular_representation

    def _build_quotient_representations(self):
        r"""
        Build all the quotient representations for this group

        """
        for n in range(2, int(math.ceil(math.sqrt(self.rotation_order)))):
            if self.rotation_order % n == 0:
                for f in range(2):
                    sg_id = (f, n)
                    self.quotient_representation(sg_id)
        
    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0,0']

    def irrep(self, j: int, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with reflecion and rotation frequencies :math:`j` (reflection) and :math:`k` (rotation) of the
        current dihedral group. Note: the frequencies has to be non-negative integers, i.e.
        :math:`j \in [0, 1]` and :math:`k \in \{0, \dots, \left\lfloor N/2 \right\rfloor \}`,
        where :math:`N` is the rotational order of the group (the number of rotations, i.e. the half the group order).
        
        If :math:`N` is odd, valid parameters are :math:`(0, 0)`, :math:`(1, 0)`, :math:`(1, 1)` ... :math:`(1, \left\lfloor N/2 \right\rfloor)`.
        If :math:`N` is even, the group also has the irrep :math:`(0, N/2)`.
        
        Args:
            j (int): the frequency of the reflections in the irrep
            k (int): the frequency of the rotations in the irrep

        Returns:
            the corresponding irrep

        """
        
        N = self.rotation_order
        
        assert j in [0, 1]
        assert 0 <= k <= N//2
    
        name = f"irrep_{j},{k}"
    
        if name not in self.irreps:
        
            base_angle = 2.0 * np.pi / N

            if j == 0:
                
                if k == 0:
                    # Trivial representation
                    irrep = lambda element, identity=np.eye(1): identity
                    character = lambda e: 1
                    supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate', 'concatenated']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  # trivial=True,
                                                                  character=character,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )

                elif N % 2 == 0 and k == N//2:
                    
                    irrep = lambda element, k=k, base_angle=base_angle: np.array([[np.cos(k * element[1] * base_angle)]])
                    character = lambda element, k=k: np.cos(k * element[1])
                    supported_nonlinearities = ['norm', 'gated', 'concatenated']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  # character=character,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )
                else:
                    raise ValueError(f"Error! Flip frequency {j} and rotational frequency {k} don't correspond to any irrep of the group {self.name}!")

            else:
                
                if k == 0:
                    # Trivial on Cyclic subgroup Representation

                    irrep = lambda element: np.array([[-1 if element[0] else 1]])
                    character = lambda element: (-1 if element[0] else 1)
                    supported_nonlinearities = ['norm', 'gated', 'concatenated']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  # character=character,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )

                elif N % 2 == 0 and k == N / 2:
    
                    # 1 dimensional Irreducible representation (only for groups with an even number of rotations)
                    irrep = lambda element, k=k, base_angle=base_angle: np.array([[np.cos(k * element[1] * base_angle) * (-1 if element[0] else 1)]])
                    character = lambda element, k=k: np.cos(k * element[1]) * (-1 if element[0] else 1)
                    supported_nonlinearities = ['norm', 'gated', 'concatenated']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  # character=character,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )
                else:
                    # 2 dimensional Irreducible Representations
                
                    # build the rotation matrix with rotation order 'k'
                    irrep = lambda element, k=k, base_angle=base_angle:\
                        utils.psichi(element[1] * base_angle, element[0], k=k)
            
                    # build the trace of this matrix
                    character = lambda element, k=k: 0 if element[0] else 2*np.cos(k * element[1])
            
                    supported_nonlinearities = ['norm', 'gated']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 2, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  # character=character,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )

        return self.irreps[name]

    @staticmethod
    def _generator(N: int) -> 'DihedralGroup':
        global _cached_group_instances
        if N not in _cached_group_instances:
            _cached_group_instances[N] = DihedralGroup(N)
    
        return _cached_group_instances[N]

