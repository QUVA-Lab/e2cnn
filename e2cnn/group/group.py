
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Iterable, List, Any, Dict

import e2cnn.group
import numpy as np
from scipy import sparse

__all__ = ["Group"]


class Group(ABC):
    
    def __init__(self, name: str, continuous: bool, abelian: bool):
        r"""
        Abstract class defining the interface of a group.
        
        Args:
            name (str): name identifying the group
            continuous (bool): whether the group is non-finite or finite
            abelian (bool): whether the group is *abelian* (commutative)
            
        Attributes:
            ~.name (str): Name identifying the group
            ~.continuous (bool): Whether it is a non-finite or a finite group
            ~.abelian (bool): Whether it is an *abelian* group (i.e. if the group law is commutative)
            ~.identity : Identity element of the group. The identity element :math:`e` satisfies the
                following property :math:`\forall\ g \in G,\ g \cdot e = e \cdot g= g`

        """
        
        self.name = name
        
        self.continuous = continuous
        
        self.abelian = abelian
        
        self._irreps = {}
        
        self._representations = {}
        
        if self.continuous:
            self.elements = None
            self.elements_names = None
        else:
            self.elements = []
            self.elements_names = []
        
        self.identity = None
        
        self._subgroups = {}
            
    def order(self) -> int:
        r"""
        Returns the number of elements in this group if it is a finite group, otherwise -1 is returned
        
        Returns:
            the size of the group or ``-1`` if it is a continuous group

        """
        if self.elements is not None:
            return len(self.elements)
        else:
            return -1
    
    @abstractmethod
    def combine(self, e1, e2):
        r"""

        Method that returns the combination of two group elements according to the *group law*.
        
        Args:
            e1: an element of the group
            e2: another element of the group
    
        Returns:
            the group element :math:`e_1 \cdot e_2`
            
        """
        pass

    @abstractmethod
    def inverse(self, element):
        r"""
        Method that returns the inverse in the group of the element given as input

        Args:
            element: an element of the group

        Returns:
            its inverse
        """
        pass

    @abstractmethod
    def is_element(self, element) -> bool:
        r"""
        Check whether the input is an element of this group or not.

        Args:
            element: input object to test

        Returns:
            if the input is an element of the group

        """
        pass

    @abstractmethod
    def equal(self, e1, e2) -> bool:
        r"""
        Method that checks whether the two inputs are the same element of the group.

        This is especially useful for continuous groups with periodicity; see for instance
        :meth:`e2cnn.group.SO2.equal`.

        Args:
            e1: an element of the group
            e2: another element of the group

        Returns:
            if they are equal

        """
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def subgroup(self, id) -> Tuple[e2cnn.group.Group, Callable, Callable]:
        r"""
        Restrict the current group to the subgroup identified by the input ``id``.

        Args:
            id: the identifier of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                -a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)

        """
        pass

    @property
    def irreps(self) -> Dict[str, e2cnn.group.IrreducibleRepresentation]:
        r"""
        Dictionary containing all irreducible representations (:class:`~e2cnn.group.IrreducibleRepresentation`)
        instantiated for this group.

        Returns:
            a dictionary containing all irreducible representations built

        """
        return self._irreps

    @property
    def representations(self) -> Dict[str, e2cnn.group.Representation]:
        r"""
        Dictionary containing all representations (:class:`~e2cnn.group.Representation`)
        instantiated for this group.

        Returns:
            a dictionary containing all representations built

        """
        return self._representations

    @property
    @abstractmethod
    def trivial_representation(self) -> e2cnn.group.IrreducibleRepresentation:
        r"""
        Builds the trivial representation of the group.
        The trivial representation is a 1-dimensional representation which maps any element to 1,
        i.e. :math:`\forall g \in G,\ \rho(g) = 1`.
        
        Returns:
            the trivial representation of the group

        """
        pass

    @abstractmethod
    def irrep(self, *id) -> e2cnn.group.IrreducibleRepresentation:
        r"""

        Builds the irreducible representation (:class:`~e2cnn.group.IrreducibleRepresentation`) of the group which is
        specified by the input arguments.

        .. seealso ::

            Check the documentation of the specific group subclass used for more information on the valid ``id`` values.

        Args:
            *id: parameters identifying the specific irrep.

        Returns:
            the irrep built

        """
        pass

    @property
    def regular_representation(self) -> e2cnn.group.Representation:
        r"""
        Builds the regular representation of the group if the group has a *finite* number of elements;
        returns ``None`` otherwise.
        
        The regular representation of a finite group :math:`G` acts on a vector space :math:`\R^{|G|}` by permuting its
        axes.
        Specifically, associating each axis :math:`e_g` of :math:`\R^{|G|}` to an element :math:`g \in G`, the
        representation of an element :math:`\tilde{g}\in G` is a permutation matrix which maps :math:`e_g` to
        :math:`e_{\tilde{g}g}`.
        For instance, the regular representation of the group :math:`C_4` with elements
        :math:`\{r^k | k=0,\dots,3 \}` is instantiated by:
        
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        |    :math:`g`                      |          :math:`e`                                                                                         |          :math:`r`                                                                                         |        :math:`r^2`                                                                                         |        :math:`r^3`                                                                                         |
        +===================================+============================================================================================================+============================================================================================================+============================================================================================================+============================================================================================================+
        |  :math:`\rho_\text{reg}^{C_4}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\  0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\  0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ \end{bmatrix}` |
        +-----------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------+
        
        A vector :math:`v=\sum_g v_g e_g` in :math:`\R^{|G|}` can be interpreted as a scalar function
        :math:`v:G \to \R,\, g \mapsto v_g` on :math:`G`.
        
        Returns:
            the regular representation of the group

        """
        if self.order() < 0:
            raise ValueError(f"Regular representation is supported only for finite groups but "
                             f"the group {self.name} has an infinite number of elements")
        else:
            if "regular" not in self.representations:
                irreps, change_of_basis, change_of_basis_inv = e2cnn.group.representation.build_regular_representation(self)
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'concatenated']
                self.representations["regular"] = e2cnn.group.Representation(self,
                                                                             "regular",
                                                                             [r.name for r in irreps],
                                                                             change_of_basis,
                                                                             supported_nonlinearities,
                                                                             change_of_basis_inv=change_of_basis_inv,
                                                                             )
            return self.representations["regular"]

    def quotient_representation(self, subgroup_id) -> e2cnn.group.Representation:
        r"""
        Builds the quotient representation of the group with respect to the subgroup identified by the
        input ``subgroup_id``.
        
        Similar to :meth:`~e2cnn.group.Group.regular_representation`, the quotient representation
        :math:`\rho_\text{quot}^{G/H}` of :math:`G` w.r.t. a subgroup :math:`H` acts on :math:`\R^{|G|/|H|}` by
        permuting its axes.
        Labeling the axes by the cosets :math:`gH` in the quotient space :math:`G/H`, it can be defined via its action
        :math:`\rho_\text{quot}^{G/H}(\tilde{g})e_{gH}=e_{\tilde{g}gH}`.

        Regular and trivial representations are two specific cases of quotient representations obtained by choosing
        :math:`H=\{e\}` or :math:`H=G`, respectively.
        Vectors in the representation space :math:`\R^{|G|/|H|}` can be viewed as scalar functions on the quotient
        space :math:`G/H`.
        
        The quotient representation :math:`\rho_\text{quot}^{G/H}` can also be defined as the
        :meth:`~e2cnn.group.Group.induced_representation` from the trivial representation of the subgroup :math:`H`.
        
        Args:
            subgroup_id: identifier of the subgroup
        
        Returns:
            the quotient representation of the group

        """
        
        name = f"quotient[{subgroup_id}]"
        
        if name not in self.representations:
            subgroup, _, _ = self.subgroup(subgroup_id)
            
            supported_nonlinearities = _induced_nonlinearities(subgroup.trivial_representation)
            
            irreps, change_of_basis, change_of_basis_inv = self._induced_from_irrep(subgroup_id,
                                                                                    subgroup.trivial_representation)
            self.representations[name] = e2cnn.group.Representation(self,
                                                                    name,
                                                                    [r.name for r in irreps],
                                                                    change_of_basis,
                                                                    supported_nonlinearities,
                                                                    change_of_basis_inv=change_of_basis_inv,
                                                                    )

        return self.representations[name]

    def induced_representation(self, subgroup_id, repr: e2cnn.group.Representation) -> e2cnn.group.Representation:
        r"""
        Builds the induced representation from the input representation ``repr`` of the subgroup identified by
        the input ``subgroup_id``.
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup
            

        Returns:
            the induced representation of the group

        """
        
        assert repr.irreducible, "Induction from general representations is not supported yet"
        
        name = f"induced[{subgroup_id}][{repr.name}]"
        if name not in self.representations:

            supported_nonlinearities = _induced_nonlinearities(repr)

            irreps, change_of_basis, change_of_basis_inv = self._induced_from_irrep(subgroup_id, repr)
            self.representations[name] = e2cnn.group.Representation(self,
                                                                    name,
                                                                    [r.name for r in irreps],
                                                                    change_of_basis,
                                                                    supported_nonlinearities,
                                                                    change_of_basis_inv=change_of_basis_inv,
                                                                    )

        return self.representations[name]

    def _induced_from_irrep(self, subgroup_id: Tuple[float, int],
                            repr: e2cnn.group.IrreducibleRepresentation,
                            ) -> Tuple[List[e2cnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    
        r"""
        Builds the induced representation from the input *irreducible* representation ``repr`` of the subgroup
        identified by the input ``subgroup_id``.
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup
            

        Returns:
            a tuple containing the list of irreps, the change of basis and the inverse change of basis of
            the induced representation

        """
    
        assert repr.irreducible
    
        # The method only supports *finite* group
        if self.order() < 0:
            raise ValueError(f"Only finite group are supported for induction but you tried to induce to the group "
                             f"{self.name} which has an infinite number of elements")
        else:
            return e2cnn.group.representation.build_induced_representation(self, subgroup_id, repr)

    def restrict_representation(self, id, repr: e2cnn.group.Representation) -> e2cnn.group.Representation:
        r"""

        Restrict the input :class:`~e2cnn.group.Representation` to the subgroup identified by ``id``.
        
        Any representation :math:`\rho : G \to \GL{\R^n}` can be uniquely restricted to a representation
        of a subgroup :math:`H < G` by restricting its domain of definition:

        .. math ::

            \Res{H}{G}(\rho): H \to \GL{{\R}^n},\ h \mapsto \rho\big|_H(h)
        
        .. seealso ::

            Check the documentation of the method :meth:`~e2cnn.group.Group.subgroup()` of the group used to see
            the available subgroups and accepted ids.

        Args:
            id: identifier of the subgroup
            repr (Representation): the representation to restrict

        Returns:
            the restricted representation

        """
    
        assert repr.group == self
    
        sg, _, _ = self.subgroup(id)
    
        # First, restrict each irrep in the representation
    
        irreps_changes_of_basis = []
        irreps = []
    
        for irr in repr.irreps:
            irrep_cob, reduced_irreps = self._restrict_irrep(irr, id)
            size = self.irreps[irr].size
            assert irrep_cob.shape == (size, size)
        
            irreps_changes_of_basis.append(irrep_cob)
            irreps += reduced_irreps
    
        # concatenate the restricted irreps and merge the representation's change of basis with the
        # restricted irreps' change of basis matrices
        irreps_changes_of_basis = sparse.block_diag(irreps_changes_of_basis, format='csc')
        change_of_basis = repr.change_of_basis @ irreps_changes_of_basis
    
        name = f"{self.name}:{repr.name}"
    
        resr = e2cnn.group.Representation(sg,
                                          name,
                                          irreps,
                                          change_of_basis,
                                          repr.supported_nonlinearities)
    
        if resr.is_trivial() and 'pointwise' not in repr.supported_nonlinearities:
            resr.supported_nonlinearities.add("pointwise")
    
        return resr

    @abstractmethod
    def _restrict_irrep(self, irrep: str, id) -> Tuple[np.matrix, List[str]]:
        pass

    @abstractmethod
    def testing_elements(self) -> Iterable[Any]:
        r"""
        A finite number of group elements to use for testing.
        """
        pass

    @staticmethod
    @abstractmethod
    def _generator(*inputs) -> 'Group':
        pass


def _induced_nonlinearities(repr: e2cnn.group.Representation):
    
    supported_nonlinearities = []
    
    if 'pointwise' in repr.supported_nonlinearities:
        supported_nonlinearities.append('pointwise')
    if 'concatenated' in repr.supported_nonlinearities:
        supported_nonlinearities.append('concatenated')
    if 'gated' in repr.supported_nonlinearities:
        supported_nonlinearities.append('gated')
        for nl in repr.supported_nonlinearities:
            if nl.startswith('induced_gated'):
                supported_nonlinearities.append(nl)
                break
        else:
            supported_nonlinearities.append(f'induced_gated_{repr.size}')
    if 'norm' in repr.supported_nonlinearities:
        supported_nonlinearities.append('norm')
        for nl in repr.supported_nonlinearities:
            if nl.startswith('induced_norm'):
                supported_nonlinearities.append(nl)
                break
        else:
            supported_nonlinearities.append(f'induced_norm_{repr.size}')
    if 'gate' in repr.supported_nonlinearities or 'induced_gate' in repr.supported_nonlinearities:
        supported_nonlinearities.append('induced_gate')

    # 'vectorfield' not always supported by the induced representation so they are not added

    return supported_nonlinearities
