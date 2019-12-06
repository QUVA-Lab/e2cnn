from __future__ import annotations

from e2cnn.group import Group

from .so2group import SO2
from .cyclicgroup import CyclicGroup
from .dihedralgroup import DihedralGroup

from e2cnn.group import IrreducibleRepresentation
from e2cnn.group import Representation
from e2cnn.group import directsum
from e2cnn.group import utils

import numpy as np

from typing import Tuple, List, Callable, Iterable

__all__ = ["O2"]

_cached_group_instance = None


class O2(Group):

    def __init__(self, maximum_frequency: int):
        r"""
        Build an instance of the orthogonal group :math:`O(2)` which contains reflections and continuous planar
        rotations.
        
        Any group element is either a rotation :math:`r_{\theta}` by an angle :math:`\theta \in [0, 2\pi)` or a
        reflection :math:`f` followed by a rotation, i.e. :math:`r_{\theta}f`.
        Two reflections gives the identity :math:`f \cdot f = e` and a reflection commutes with a rotation by
        inverting it, i.e. :math:`r_\theta \cdot f = f \cdot r_{-\theta}`.
        A group element :math:`r_{\theta}f^j` is implemented as a pair :math:`(j, \theta)` with :math:`j \in \{0, 1\}`
        and :math:`\theta \in [0, 2\pi)`.
        
        .. note ::
        
            Since the group has infinitely many irreducible representations, it is not possible to build all of them.
            Each irrep is associated to one unique integer frequency and the parameter ``maximum_frequency`` specifies
            the maximum frequency of the irreps to build.
            New irreps (associated to higher frequencies) can be manually created by calling the method
            :meth:`~e2cnn.group.O2.irrep` (see the method's documentation).
        
        
        Args:
            maximum_frequency (int): the maximum frequency to consider when building the irreps of the group
        
        Attributes:
            
            ~.reflection: the reflection element :math:`(j, \theta) = (1, 0.)`

        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(O2, self).__init__("O(2)", True, False)
        
        self.rotation_order = -1
        
        self._maximum_frequency = maximum_frequency
        
        self.identity = (0, 0.)
        self.reflection = (1, 0.)
        
        self._build_representations()
    
    def inverse(self, element: Tuple[int, float]) -> Tuple[int, float]:
        r"""
        Return the inverse element of the input element.
        Given the element :math:`r_\theta f^j` as a pair :math:`(j, \theta)`,
        the method returns :math:`r_{-\theta}` (as :math:`(0, -\theta)`) if :math:`f = 0` and
        :math:`r_\theta f` (as :math:`(1, \theta)`) otherwise.
        
        Args:
            element (tuple): a group element :math:`r_{\theta}f^j` a pair :math:`(j, \theta)`

        Returns:
            its inverse
            
        """
        return element[0], -element[1] * (-1 if element[0] else 1)

    def combine(self, e1: Tuple[int, float], e2: Tuple[int, float]) -> Tuple[int, float]:
        r"""
        Return the combination of the two input elements.
        Given two input element :math:`r_\alpha f^a` and :math:`r_\beta f^b`, the method returns
        :math:`r_\alpha f^a \cdot r_\beta f^b`.

        Args:
            e1 (tuple): a group element :math:`r_\alpha f^a` as a pair :math:`(a, \alpha)`
            e2 (tuple): another element :math:`r_\beta f^b` as a pair :math:`(b, \beta)`

        Returns:
            their combination :math:`r_\alpha f^a \cdot r_\beta f^b`
        """
        return (e1[0] + e2[0]) % 2, e1[1] + (-1 if e1[0] else 1)*e2[1]

    def equal(self, e1: Tuple[int, float], e2: Tuple[int, float]) -> bool:
        r"""

        Check if the two input values corresponds to the same element.

        See :meth:`e2cnn.group.SO2.equal` for more details.

        Args:
            e1 (tuple): an element
            e2 (tuple): another element
            
        Returns:
            whether they are the same element

        """
        return e1[0] == e2[0] and utils.cycle_isclose(e1[1], e2[1], 2 * np.pi)

    def is_element(self, element: Tuple[int, float]) -> bool:
        if isinstance(element, tuple) and len(element) == 2 and isinstance(element[0], int) and isinstance(element[1], float):
            return element[0] in {0, 1}
        else:
            return False

    def testing_elements(self) -> Iterable[Tuple[int, float]]:
        r"""
        A finite number of group elements to use for testing.
        """
        N = 4*13
        return iter([(0, i * 2. * np.pi / N) for i in range(N)] + [(1, i * 2. * np.pi / N) for i in range(N)])
    
    def __eq__(self, other):
        if not isinstance(other, O2):
            return False
        else:
            return self.name == other.name and self._maximum_frequency == other._maximum_frequency

    def subgroup(self, id: Tuple[float, int]) -> Tuple[Group, Callable, Callable]:
        r"""
        Restrict the current group :math:`O(2)` to the subgroup identified by the input ``id``, where ``id`` is a
        tuple :math:`(\theta, M)`.
        
        Here, :math:`M` can be either a positive integer indicating the number of rotations in the subgroup or
        :math:`-1`, indicating that the subgroup contains all continuous rotations.
        :math:`\theta` is either ``None`` or an angle in :math:`[0, \frac{2\pi}{M})`.
        If :math:`\theta` is ``None``, the subgroup does not contain any reflections.
        Otherwise, the subgroup contains the reflection :math:`r_{\theta}f` along the axis of the current group rotated
        by :math:`\frac{\theta}{2}`.
        
        Valid combinations are:
        
        - (``None``, :math:`M>0`): restrict to the cyclic subgroup :math:`C_M` generated by :math:`\langle r_{2\pi/M} \rangle`.
        
        - (``None``, :math:`-1`): restrict to the subgroup :math:`SO(2)` containing only the rotations
        
        - (:math:`\theta`, :math:`M>0`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r_{2\pi/M}, r_{\theta} f \rangle`
        
        In particular:
        
        - (:math:`0`, :math:`1`): restrict to the reflection group generated by :math:`\langle f \rangle`
        
        - (:math:`0`, :math:`M`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r_{2\pi/M}, f \rangle`
        
        - (``None``, :math:`1`): restrict to the cyclic subgroup of order 1 containing only the identity
        
        
        Args:
            id (tuple): the identification of the subgroup

        Returns:
            a tuple containing
            
                - the subgroup
                
                - a function which maps an element of the subgroup to its inclusion in the original group and
                
                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
                
        """
        
        assert isinstance(id, tuple) and len(id) == 2
        assert id[0] is None or isinstance(id[0], float)
        assert isinstance(id[1], int)

        order = id[1]
        axis = id[0]
        # assert (id[0] is None and (id[1] > 0 or id[1] == -1)) or (id[0] is not None and id[1] > 0)
        assert axis is None or 0 <= axis < 2*np.pi/order or order < 0
        
        if id not in self._subgroups:
            if id[0] is not None and id[1] == -1:
                sg = O2(self._maximum_frequency)
                parent_mapping = lambda e, axis=axis: (e[0], e[1] + e[0]*axis)
                child_mapping = lambda e, axis=axis: (e[0], e[1] - e[0]*axis)

            elif id[0] is None and id[1] == -1:
                sg = SO2(self._maximum_frequency)
                parent_mapping = lambda e, axis=axis: (0, e)
                child_mapping = lambda e: None if e[0] != 0 else e[1]
                
            elif id[0] is not None and id[1] == 1:
                # take the elements of the group generated by "2pi/k f"
                sg = CyclicGroup(2)
                parent_mapping = lambda e, axis=axis: (e, axis*e)
                # child_mapping = lambda e, axis=axis: None if not is_close(e[1], axis*e[0]) else e[0]
                child_mapping = lambda e, axis=axis: None if not utils.cycle_isclose(e[1], axis*e[0], 2*np.pi) else e[0]
                
            elif id[0] is None:
                # take the elements of the group generated by "2pi/order"
                sg = CyclicGroup(order)
                parent_mapping = lambda e, order=order: (0, e * 2.*np.pi/order)
                # child_mapping = lambda e, order=order: None if (e[0] != 0 or not divides(2.*np.pi/order, e[1])) else \
                #                                        int(round(e[1] * order / (2.*np.pi)))
                child_mapping = lambda e, order=order: None if (e[0] != 0 or not utils.cycle_isclose(e[1], 0., 2.*np.pi/order)) else \
                    int(round(e[1] * order / (2.*np.pi)))

            elif id[0] is not None and id[1] > 1:
                # take the elements of the group generated by "2pi/order" and "2pi/k f"
                sg = DihedralGroup(order)

                parent_mapping = lambda e, order=order, axis=axis: (e[0], e[1] * 2. * np.pi / order + e[0]*axis)
                child_mapping = lambda e, order=order, axis=axis: None if not utils.cycle_isclose(e[1] - e[0]*axis, 0., 2.*np.pi/order) else \
                    (e[0], int(round((e[1] - e[0]*axis) * order / (2. * np.pi))))
            else:
                raise ValueError(f"id '{id}' not recognized")

            self._subgroups[id] = sg, parent_mapping, child_mapping

        return self._subgroups[id]

    def _restrict_irrep(self, irrep: str, id: Tuple[int, int]) -> Tuple[np.matrix, List[str]]:
        r"""
        Restrict the input irrep of current group to the subgroup identified by "id".
        More precisely, "id" is a tuple :math:`(k, m)`, where :math:`m` is a positive integer indicating the number of
        rotations in the subgroup while :math:`k` is either None (no flips in the subgroup) or an angle in
        :math:`[0, \frac{2\pi}{m})` (indicating the axis of flip in the subgroup).
        Valid combinations are:
        - (None, -1): restrict to the subgroup :math:`SO(2)` containing only the rotations
        - (None, m): restrict to the cyclic subgroup with order "m" :math:`C_m` generated by :math:`\langle 2\pi/m \rangle`.
        - (0, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle 2\pi/m, f \rangle`
        - (0, 1): restrict to the cyclic subgroup of order 2 :math:`C_2` generated by the flip :math:`\langle f \rangle`
        - (None, 1): restrict to the cyclic subgroup of order 1 :math:`C_1` containing only the identity
        - (k, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle 2\pi/m, 2\pi/k f \rangle`
        
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

        if id[0] is not None and id[1] == -1:
            
            change_of_basis = np.eye(irr.size)
            irreps.append(irr.name)

            if irr.size == 2:
                f = irr.attributes["frequency"]
                change_of_basis = utils.psi(0.5 * id[0], f) @ change_of_basis

        elif id[0] is None and id[1] == -1:
            f = irr.attributes["frequency"]
            irreps.append(f"irrep_{f}")
            change_of_basis = np.eye(irr.size)
            
        elif id[0] is not None and id[1] == 1:
            j = irr.attributes["flip_frequency"]
            k = irr.attributes["frequency"]
            change_of_basis = np.eye(irr.size)
            if irr.size > 1:
                irreps.append("irrep_0")
                change_of_basis = utils.psi(0.5 * id[0], k)
            irreps.append(f"irrep_{j}")
        elif id[0] is None and id[1] > 0:
        
            order = id[1]

            f = irr.attributes["frequency"] % order
            if f > order/2:
                f = order - f
                change_of_basis = utils.chi(1)
            else:
                change_of_basis = np.eye(irr.size)

            r = f"irrep_{f}"
            if sg.irreps[r].size < irr.size:
                irreps.append(f"irrep_{f}")
            irreps.append(r)
    
        elif id[0] is not None and id[1] > 1:
        
            order = id[1]
            j = irr.attributes["flip_frequency"]
            f = irr.attributes["frequency"]
            k = f % order
            
            if k > order/2:
                k = order - k
                change_of_basis = np.array([[1, 0], [0, -1]])
            else:
                change_of_basis = np.eye(irr.size)
            
            r = f"irrep_{j},{k}"
            if sg.irreps[r].size < irr.size:
                irreps.append(f"irrep_0,{k}")
            irreps.append(r)
                
            if irr.size == 2:
                change_of_basis = utils.psi(0.5 * id[0], f) @ change_of_basis

        else:
            raise ValueError(f"id '{id}' not recognized")
        
        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
        
        # Build all the Irreducible Representations
    
        j, k = 0, 0
    
        # add Trivial representation
        self.irrep(j, k)
    
        j = 1
        for k in range(self._maximum_frequency + 1):
            self.irrep(j, k)

        # Build all Representations
        
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**self.irreps)

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0,0']

    def irrep(self, j: int, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with reflection and rotation frequencies :math:`j` (reflection) and :math:`k` (rotation) of the
        current group.
        Notice: the frequencies has to be non-negative integers: :math:`j \in \{0, 1\}` and :math:`k \in \mathbb{N}`
        
        Valid parameters are :math:`(0, 0)` and :math:`(1, 0)`, :math:`(1, 1)`, :math:`(1, 2)`, :math:`(1, 3)`, ...
        
        Args:
            j (int): the frequency of the reflection in the irrep
            k (int): the frequency of the rotations in the irrep

        Returns:
            the corresponding irrep

        """
    
        assert j in [0, 1]
        assert k >= 0
    
        name = f"irrep_{j},{k}"
    
        if name not in self.irreps:
            if j == 0:
                if k == 0:
                    # Trivial representation
                    irrep = lambda element, identity=np.eye(1): identity
                    character = lambda e: 1
                    supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                    self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  character=character,
                                                                  # trivial=True,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )
                else:
                    raise ValueError(f"Error! Flip frequency {j} and rotational frequency {k} don't correspond to any irrep of the group {self.name}!")
                
            elif k == 0:

                # add Trivial on SO(2) subgroup Representation
                irrep = lambda element: np.array([[-1 if element[0] else 1]])
                character = lambda element: (-1 if element[0] else 1)
                supported_nonlinearities = ['norm', 'gated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k,
                                                              flip_frequency=j
                                                              )
            else:

                # 2 dimensional Irreducible Representations
        
                # build the rotation matrix with rotation order 'k'
                irrep = lambda element, k=k: utils.psichi(element[1], element[0], k=k)
        
                # build the trace of this matrix
                character = lambda element, k=k: 0 if element[0] else 2*np.cos(k * element[1])
        
                supported_nonlinearities = ['norm', 'gated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 2, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k,
                                                              flip_frequency=j
                                                              )

        return self.irreps[name]

    def _induced_from_irrep(self, subgroup_id: Tuple[float, int],
                            repr: IrreducibleRepresentation,
                            ) -> Tuple[List[IrreducibleRepresentation], np.ndarray, np.ndarray]:
    
        if subgroup_id == (None, -1):
            # Induced representation from SO(2)
            # As the quotient set is finite, a finite dimensional representation of SO(2)
            # defines a finite dimensional induced representation of O(2)
            
            subgroup, parent, child = self.subgroup(subgroup_id)
            assert repr.group == subgroup

            name = f"induced[{subgroup_id}][{repr.name}]"
            
            frequency = repr.attributes["frequency"]
            
            if frequency > 0:
                multiplicities = [(self.irrep(1, frequency), 2)]
            else:
                multiplicities = [(self.irrep(0, 0), 1), (self.irrep(1, 0), 1)]
            
            irreps = []
            for irr, multiplicity in multiplicities:
                irreps += [irr] * multiplicity
            
            P = directsum(irreps, name=f"{name}_irreps")
            
            size = P.size
            
            v = np.zeros((repr.size, size), dtype=np.float)

            def build_commuting_matrix(rho, t):
                k = rho.attributes["frequency"]
                
                if rho.size == 1:
                    E = np.eye(1)
                    M = 2*np.pi*np.eye(1)
                else:
                    E = np.array([[1, -1], [1, 1]])
                    if t % 2 == 0:
                        E = E.T
                    I = np.eye(4)
                    A = np.fliplr(np.eye(4)) * np.array([1, -1, -1, 1])
                    M = np.pi * (A + I)

                # compute the averaging of rho(g).T @ E @ rho(g)
                # i.e. X = 1/2pi Integral_{0, 2pi} rho(theta).T @ E @ rho(theta) d theta
                # as vec(X) = 1/2pi Integral_{0, 2pi} (rho *tensor* rho)(theta) @ vec(E) d theta
                # where M = Integral_{0, 2pi} (rho *tensor* rho)(theta) d theta
                X = M @  E.reshape(-1, 1)
                X /= 2*np.pi
                
                # normalization
                X /= np.sqrt(np.sum(X @ X.T) / rho.size)
                
                X = X.reshape(rho.size, rho.size)
                
                return X

            p = 0
            for irr, m in multiplicities:
                assert irr.size >= m
    
                if m > 0:
                    restricted_irr = self.restrict_representation(subgroup_id, irr)
        
                    n_repetitions = len([name for name in restricted_irr.irreps if name == repr.name])
                    assert repr.size * n_repetitions >= m, (
                        f"{self.name}\{subgroup.name}:{repr.name}", irr.name, m, n_repetitions)
        
                    for shift in range(m):
                        commuting_matrix = build_commuting_matrix(repr, shift // n_repetitions)
                        x = p
                        i = 0
                        for r_irrep in restricted_irr.irreps:
                            if r_irrep == repr.name:
                                if i == shift % n_repetitions:
                                    v[:, x:x + repr.size] = commuting_matrix
                                i += 1
                            x += subgroup.irreps[r_irrep].size
            
                        v[:, p:p + irr.size] = v[:, p:p + irr.size] @ restricted_irr.change_of_basis_inv
                        v[:, p:p + irr.size] *= np.sqrt(irr.size)
            
                        p += irr.size

            v /= np.sqrt(size)

            change_of_basis = np.zeros((size, size))

            change_of_basis[:repr.size, :] = v @ P(self.identity)
            change_of_basis[repr.size:, :] = v @ P(self.reflection)

            change_of_basis_inv = change_of_basis.T
            
            return irreps, change_of_basis, change_of_basis_inv

        else:
            raise ValueError(f"Induction from discrete subgroups of O(2) leads to infinite dimensional induced "
                             f"representations. Hence, induction from the subgroup identified "
                             f"by {subgroup_id} is not allowed.")

    @staticmethod
    def _generator(maximum_frequency: int = 10) -> 'O2':
        global _cached_group_instance
        if _cached_group_instance is None:
            _cached_group_instance = O2(maximum_frequency)
        elif _cached_group_instance._maximum_frequency < maximum_frequency:
            _cached_group_instance._maximum_frequency = maximum_frequency
            _cached_group_instance._build_representations()
    
        return _cached_group_instance


