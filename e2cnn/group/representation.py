from __future__ import annotations

import e2cnn.group
from e2cnn.group import Group

from collections import defaultdict

from typing import Callable, Any, List, Tuple, Dict, Union, Set

import numpy as np
import scipy as sp
import math
from scipy.sparse.csgraph import connected_components


__all__ = [
    "Representation",
    "build_from_discrete_group_representation",
    "directsum",
    "disentangle",
    "change_basis",
    "build_regular_representation",
    "build_quotient_representation",
    "build_induced_representation"
]


class Representation:
    
    def __init__(self,
                 group: Group,
                 name: str,
                 irreps: List[str],
                 change_of_basis: np.ndarray,
                 supported_nonlinearities: Union[List[str], Set[str]],
                 representation: Union[Dict[Any, np.ndarray], Callable[[Any], np.ndarray]] = None,
                 character: Union[Dict[Any, float], Callable[[Any], float]] = None,
                 change_of_basis_inv: np.ndarray = None,
                 **kwargs):
        r"""
        Class used to describe a group representation.
        
        A (real) representation :math:`\rho` of a group :math:`G` on a vector space :math:`V=\mathbb{R}^n` is a map
        (a *homomorphism*) from the group elements to invertible matrices of shape :math:`n \times n`, i.e.:
        
        .. math::
            \rho : G \to \GL{V}
            
        such that the group composition is modeled by a matrix multiplication:
        
        .. math::
            \rho(g_1 g_2) = \rho(g_1) \rho(g_2) \qquad  \forall \ g_1, g_2 \in G \ .
        
        Any representation (of a compact group) can be decomposed into the *direct sum* of smaller, irreducible
        representations (*irreps*) of the group up to a change of basis:
        
        .. math::
            \forall \ g \in G, \ \rho(g) = Q \left( \bigoplus\nolimits_{i \in I} \psi_i(g) \right) Q^{-1} \ .
        
        Here :math:`I` is an index set over the irreps of the group :math:`G` which are contained in the
        representation :math:`\rho`.
        
        This property enables one to study a representation by its irreps and it is used here to work with arbitrary
        representations.
        
        :attr:`e2cnn.group.Representation.change_of_basis` contains the change of basis matrix :math:`Q` while
        :attr:`e2cnn.group.Representation.irreps` is an ordered list containing the names of the irreps :math:`\psi_i`
        indexed by the index set :math:`I`.
        
        A ``Representation`` instance can be used to describe a feature field in a feature map.
        It is the building block to build the representation of a feature map, by "stacking" multiple representations
        (taking their *direct sum*).
        
        .. note ::
            In most of the cases, it should not be necessary to manually instantiate this class.
            Indeed, the user can build the most common representations or some custom representations via the following
            methods and functions:
            
            - :meth:`e2cnn.group.Group.irrep`,
            
            - :meth:`e2cnn.group.Group.regular_representation`,
            
            - :meth:`e2cnn.group.Group.quotient_representation`,
            
            - :meth:`e2cnn.group.Group.induced_representation`,
            
            - :meth:`e2cnn.group.Group.restrict_representation`,
            
            - :func:`e2cnn.group.directsum`,
            
            - :func:`e2cnn.group.change_basis`
            
            
        
        If ``representation`` is ``None`` (default), it is automatically inferred by evaluating each irrep, stacking
        their results (through direct sum) and then applying the changes of basis. Warning: the representation of an
        element is built at run-time every time this object is called (through ``__call__``) and this approach might
        become computationally expensive with large representations.
        
        Analogously, if the ``character`` of the representation is ``None`` (default), it is automatically inferred
        evaluating ``representation`` and computing its trace.
        
        .. todo::
            improve the interface for "supported non-linearities" and write somewhere the available options
        
        Args:
            group (Group): the group to be represented.
            name (str): an identification name for this representation.
            irreps (list): a list of strings. Each string represents the name of one of the *irreps* of the
                    group (see :attr:`e2cnn.group.Group.irreps`).
            change_of_basis (~numpy.ndarray): the matrix which transforms the direct sum of the irreps
                    in this representation.
            supported_nonlinearities (list or set): a list or set of nonlinearity types supported by this
                    representation.
            representation (dict or callable, optional): a callable implementing this representation or a dictionary
                    mapping each of the group's elements to its representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dictionary mapping each element to its character.
            change_of_basis_inv (~numpy.ndarray, optional): the inverse of the ``change_of_basis`` matrix; if not
                    provided (``None``), it is computed from ``change_of_basis``.
            **kwargs: custom attributes the user can set and, then, access from the dictionary in
                    :attr:`e2cnn.group.Representation.attributes`
            
        Attributes:
            ~.group (Group): The group which is being represented.
            ~.name (str): A string identifying this representation.
            ~.size (int): Dimensionality of the vector space of this representation. In practice, this is the size of the
                matrices this representation maps the group elements to.
            ~.change_of_basis (~numpy.ndarray): Change of basis matrix for the irreps decomposition.
            ~.change_of_basis_inv (~numpy.ndarray): Inverse of the change of basis matrix for the irreps decomposition.
            ~.representation (callable): Method implementing the map from group elements to their representation matrix.
            ~.supported_nonlinearities (set): A set of strings identifying the non linearities types supported by this representation.
            ~.irreps (list): List of irreps into which this representation decomposes.
            ~.attributes (dict): Custom attributes set when creating the instance of this class.
            ~.irreducible (bool): Whether this is an irreducible representation or not (i.e. if it can't be decomposed into further invariant subspaces).

        
        """
        
        assert len(change_of_basis.shape) == 2 and change_of_basis.shape[0] == change_of_basis.shape[1]
        
        # can't have the name of an already existing representation
        assert name not in group.representations, f"A representation for {group.name} with name {name} already exists!"
        
        if change_of_basis_inv is None:
            change_of_basis_inv = sp.linalg.inv(change_of_basis)

        assert len(change_of_basis_inv.shape) == 2
        assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
        assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
        assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
        assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
        
        # Group: A string identifying this representation.
        self.group = group
        
        # str: The group this is a representation of.
        self.name = name
        
        # int: Dimensionality of the vector space of this representation.
        # In practice, this is the size of the matrices this representation maps the group elements to.
        self.size = change_of_basis.shape[0]
        
        # np.ndarray: Change of basis matrix for the irreps decomposition.
        self.change_of_basis = change_of_basis

        # np.ndarray: Inverse of the change of basis matrix for the irreps decomposition.
        self.change_of_basis_inv = change_of_basis_inv

        if representation is None:
            irreps_instances = [group.irreps[n] for n in irreps]
            representation = direct_sum_factory(irreps_instances, change_of_basis, change_of_basis_inv)
        elif isinstance(representation, dict):
            assert set(representation.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._stored_representations = representation
            representation = lambda e, repr=self: repr._stored_representations[e]
            
        elif not callable(representation):
            raise ValueError('Error! "representation" is neither a dictionary nor callable')
        
        # Callable: Method implementing the map from group elements to matrix representations.
        self.representation = representation

        if isinstance(character, dict):
            
            assert set(character.keys()) == set(self.group.elements), "Error! Keys don't match group's elements"
            
            self._characters = character

        elif callable(character):
            self._characters = character
        elif character is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            self._characters = None
        else:
            raise ValueError('Error! "character" must be a dictionary, a callable or "None"')

        # TODO - assert size matches size of the matrix returned by the callable
        
        # list(str): List of irreps this representation decomposes into
        self.irreps = irreps
        
        self.supported_nonlinearities = set(supported_nonlinearities)
        
        # dict: Custom attributes set when creating the instance of this class
        self.attributes = kwargs

        # TODO : remove the condition of an identity change of basis?
        # bool: Whether this is an irreducible representation or not (i.e.: if it can't be decomposed further)
        self.irreducible = len(self.irreps) == 1 and np.allclose(self.change_of_basis, np.eye(self.change_of_basis.shape[0]))
    
    def character(self, e) -> float:
        r"""

        The *character* of a finite-dimensional real representation is a function mapping a group element
        to the trace of its representation:

        .. math::

            \chi_\rho: G \to \mathbb{C}, \ \ g \mapsto \chi_\rho(g) := \operatorname{tr}(\rho(g))

        It is useful to perform the irreps decomposition of a representation using *Character Theory*.
        
        Args:
            e: an element of the group of this representation
        
        Returns:
            the character of the element
        
        """
        
        if self._characters is None:
            # if the character is not given as input, it is automatically inferred from the given representation
            # taking its trace
            repr = self(e)
            return np.trace(repr)
        elif isinstance(self._characters, dict):
            return self._characters[e]

        elif callable(self._characters):
            return self._characters(e)
        else:
            raise RuntimeError('Error! Character not recognized!')

    def is_trivial(self) -> bool:
        r"""
        
        Whether this representation is trivial or not.
        
        Returns:
            if the representation is trivial

        """
        return self.irreducible and self.group.trivial_representation.name == self.irreps[0]
    
    def contains_trivial(self) -> bool:
        r"""

        Whether this representation contains the trivial representation among its irreps.
        This is an alias for::
            
            any(self.group.irreps[irr].is_trivial() for irr in self.irreps)

        Returns:
           if it contains the trivial representation

        """
        for irrep in self.irreps:
            if self.group.irreps[irrep].is_trivial():
                return True
        return False

    def restrict(self, id) -> e2cnn.group.Representation:
        r"""
        
        Restrict the current representation to the subgroup identified by ``id``.
        Check the documentation of the :meth:`~e2cnn.group.Group.subgroup` method in the underlying group to see the
        available subgroups and accepted ids.

        Args:
            id: identifier of the subgroup

        Returns:
            the restricted representation
        """
        return self.group.restrict_representation(id, self)
    
    def __call__(self, element) -> np.ndarray:
        """
        An instance of this class can be called and it implements the mapping from an element of a group to its
        representation.
        
        This is equivalent to calling :meth:`e2cnn.group.Representation.representation`,
        though ``__call__`` first checks ``element`` is a valid input (i.e. an element of the group).
        It is recommended to use this call.

        Args:
            element: an element of the group

        Returns:
            A matrix representing the input element

        """
        
        assert self.group.is_element(element), f"{self.group.name}, {element}: {self.group.is_element(element)}"
        
        return self.representation(element)

    def __add__(self, other: e2cnn.group.Representation) -> e2cnn.group.Representation:
        r"""

        Compute the *direct sum* of two representations of a group.

        The two representations need to belong to the same group.

        Args:
            other (Representation): another representation

        Returns:
            the direct sum

        """
        
        return directsum([self, other])
    
    def __eq__(self, other: e2cnn.group.Representation) -> bool:
        if not isinstance(other, Representation):
            return False
        
        return (self.name == other.name
                and self.group == other.group
                and np.allclose(self.change_of_basis, other.change_of_basis)
                and self.irreps == other.irreps
                and self.supported_nonlinearities == other.supported_nonlinearities)
    
    def __repr__(self) -> str:
        return f"{self.group.name}|{self.name}:{self.size},{len(self.irreps)},{self.change_of_basis.sum()}"
    
    def __hash__(self):
        return hash(repr(self))
    
    # TODO when built from "directsum" we can "optimize" the representation by sorting the internal irreps
    #      and permuting the change of basis matrix's columns accordingly. Could be useful when one uses GNORM batchnorm
   
    
def build_from_discrete_group_representation(representation: Dict[Any, np.array],
                                             name: str,
                                             group: e2cnn.group.Group,
                                             supported_nonlinearities: List[str]
                                             ) -> e2cnn.group.Representation:
    r"""
    Given a representation of a finite group as a dictionary of matrices, the method decomposes it as a direct sum
    of the irreps of the group and computes the change-of-basis matrix. Then, a new instance of
    :class:`~e2cnn.group.Representation` is built using the direct sum of irreps and the change-of-basis matrix as
    representation taking as input elements from the continuous parent group.
    
    For instance, given a regular representation of a cyclic group of order :math:`n` implemented as a
    list of permutations matrices, the method builds a representation of SO(2) whose values are these permutation
    matrices when evaluated to the angles corresponding to the elements of the cyclic group (i.e. any angle in the
    form :math:`k 2 \pi / n` with :math:`k` in :math:`[0, \dots, n-1]`)
    
    Args:
        representation (dict): a dictionary mapping an element of ``group`` to a numpy array (must be a squared matrix)
        name (str): an identification name of the representation
        group (Group): the group whose representation has to be built
        supported_nonlinearities (list): list of non linearities types supported by this representation.
    
    Returns:
        a new representation
        
    """

    assert set(representation.keys()) == set(group.elements), "Error! Keys don't match group's elements"

    # decompose the representation
    cob, multiplicities = decompose_representation(representation, group)

    # build a list of representation instances with their multiplicities
    irreps_with_multiplicities = [(group.irreps[name], m) for (name, m) in multiplicities]

    # build the character of this representation
    new_character = lambda element, irreps=irreps_with_multiplicities: sum([m * irrep.character(element) for (irrep, m) in irreps])

    irreps = []
    for irr, m in multiplicities:
       irreps += [irr] * m

    # build the representation object
    return Representation(group,
                          name,
                          irreps,
                          cob,
                          supported_nonlinearities,
                          representation=representation,
                          character=new_character)


def directsum(reprs: List[e2cnn.group.Representation],
              change_of_basis: np.ndarray = None,
              name: str = None
              ) -> e2cnn.group.Representation:
    r"""

    Compute the *direct sum* of a list of representations of a group.
    
    The direct sum of two representations is defined as follow:
    
    .. math::
        \rho_1(g) \oplus \rho_2(g) = \begin{bmatrix} \rho_1(g) & 0 \\ 0 & \rho_2(g) \end{bmatrix}
    
    This can be generalized to multiple representations as:
    
    .. math::
        \bigoplus_{i=1}^I \rho_i(g) = (\rho_1(g) \oplus (\rho_2(g) \oplus (\rho_3(g) \oplus \dots = \begin{bmatrix}
            \rho_1(g) &         0 &  \dots &      0 \\
                    0 & \rho_2(g) &  \dots & \vdots \\
               \vdots &    \vdots & \ddots &      0 \\
                    0 &     \dots &      0 & \rho_I(g) \\
        \end{bmatrix}
    

    .. note::
        All the input representations need to belong to the same group.

    Args:
        reprs (list): the list of representations to sum.
        change_of_basis (~numpy.ndarray, optional): an invertible square matrix to use as change of basis after computing the direct sum.
                By default (``None``), an identity matrix is used, such that only the direct sum is evaluated.
        name (str, optional): a name for the new representation.

    Returns:
        the direct sum

    """
    
    group = reprs[0].group
    for r in reprs:
        assert group == r.group
    
    if name is None:
        name = "_".join([f"[{r.name}]" for r in reprs])
    
    irreps = []
    for r in reprs:
        irreps += r.irreps
    
    size = sum([r.size for r in reprs])
    
    cob = np.zeros((size, size))
    cob_inv = np.zeros((size, size))
    p = 0
    for r in reprs:
        cob[p:p + r.size, p:p + r.size] = r.change_of_basis
        cob_inv[p:p + r.size, p:p + r.size] = r.change_of_basis_inv
        p += r.size

    if change_of_basis is not None:
        change_of_basis = change_of_basis @ cob
        change_of_basis_inv = sp.linalg.inv(change_of_basis)
    else:
        change_of_basis = cob
        change_of_basis_inv = cob_inv

    supported_nonlinearities = set.intersection(*[r.supported_nonlinearities for r in reprs])
    
    return Representation(group, name, irreps, change_of_basis, supported_nonlinearities, change_of_basis_inv=change_of_basis_inv)


def disentangle(repr: Representation) -> Tuple[np.ndarray, List[Representation]]:
    r"""
    
    If possible, disentangle the input representation by decomposing it into the direct sum of smaller representations
    and a change of basis acting as a permutation matrix.
    
    This method is useful to decompose a feature vector transforming with a complex representation into multiple feature
    vectors which transform independently with simpler representations.
    
    Note that this method only decomposes a representation by applying a permutation of axes.
    A more general decomposition using any invertible matrix is possible but is just a decomposition into
    irreducible representations (see :class:`~e2cnn.group.Representation`).
    However, since the choice of change of basis is relevant for the kind of operations which can be performed
    (e.g. non-linearities), it is often not desirable to discard any change of basis and completely disentangle a
    representation.
    
    Considering only change of basis matrices which are permutation matrices is sometimes more useful.
    For instance, the restriction of the regular representation of a group to a subgroup results in a representation containing
    multiple regular representations of the subgroup (one for each `coset`).
    However, depending on how the original representation is built, the restricted representation might not be
    block-diagonal and, so, the subgroup's regular representations might not be clearly separated.
    
    For example, this happens when restricting the regular representation of :math:`\D3`
    
    +-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |    :math:`g`                      |          :math:`e`                                                                                                                                                                       |          :math:`r`                                                                                                                                                                       |        :math:`r^2`                                                                                                                                                                       |          :math:`f`                                                                                                                                                                       |         :math:`rf`                                                                                                                                                                       |       :math:`r^2f`                                                                                                                                                                       |
    +===================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+
    |  :math:`\rho_\text{reg}^{\D3}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}` |
    +-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    
    to the reflection group :math:`\C2`
    
    +--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |    :math:`g`                                     |          :math:`e`                                                                                                                                                                       |          :math:`f`                                                                                                                                                                       |
    +==================================================+==========================================================================================================================================================================================+==========================================================================================================================================================================================+
    |  :math:`\Res{\C2}{\D3} \rho_\text{reg}^{\D3}(g)` | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}` | :math:`\begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}` |
    +--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    
    Indeed, in :math:`\Res{\C2}{\D3} \rho_\text{reg}^{\D3}(g)` the three pairs of entries (1, 4), (2, 6) and (3, 5)
    never mix with each other but only permute internally.
    Moreover, each pair transforms according to the regular representation of :math:`\C2`.
    Through a permutation of the entries, it is possible to make all the entries belonging to the same pair contiguous.
    This reshuffled representation is then equal to
    :math:`\rho_\text{reg}^{\C2} \oplus \rho_\text{reg}^{\C2} \oplus \rho_\text{reg}^{\C2}`.
    Though theoretically equivalent, an implementation of this representation where the entries are contiguous is
    convenient when computing functions over single fields like batch normalization.
    
    Notice that applying the change of basis returned to the input representation (e.g. through
    :func:`e2cnn.group.change_basis`) will result in a representation containing the direct sum of the representations
    in the list returned.
    
    .. seealso::
        :func:`~e2cnn.group.directsum`,
        :func:`~e2cnn.group.change_basis`
    
    Args:
        repr (Representation): the input representation to disentangle

    Returns:
        a tuple containing
        
            - **change of basis**: a (square) permutation matrix of the size of the input representation
            
            - **representation**: the list of representations the input one is decomposed into
        
    """
    
    rsize = repr.size
    nirreps = len(repr.irreps)
    
    cob_mask = np.isclose(repr.change_of_basis, np.zeros_like(repr.change_of_basis))
    cob_mask = np.invert(cob_mask)
    
    irreps = [repr.group.irreps[irr] for irr in repr.irreps]
    irreps_pos = np.cumsum([0] + [irr.size for irr in irreps])
    
    masks = []
    i_pos = 0
    for i, irr in enumerate(irreps):
        mask = cob_mask[:, i_pos:i_pos + irr.size].any(axis=1)
        masks.append(mask)
        i_pos += irr.size
    
    cob_mask = np.array(masks, dtype=bool)
    
    graph = np.zeros((nirreps + rsize, nirreps + rsize), dtype=bool)
    graph[:nirreps, nirreps:] = cob_mask
    graph[nirreps:, :nirreps] = cob_mask.T
    
    n_blocks, labels = connected_components(graph, directed=False, return_labels=True)
    
    irreps_labels = labels[:nirreps]
    field_labels = labels[nirreps:]
    
    blocks = [([], []) for _ in range(n_blocks)]
    
    for i in range(nirreps):
        blocks[irreps_labels[i]][0].append(i)
    for i in range(rsize):
        blocks[field_labels[i]][1].append(i)
    
    change_of_basis = np.zeros_like(repr.change_of_basis)
    
    representations = []
    current_position = 0
    for block, (irreps_indices, row_indices) in enumerate(blocks):
        
        irreps_indices = sorted(irreps_indices)
        row_indices = sorted(row_indices)
        
        total_size = len(row_indices)
        
        assert sum([irreps[irr].size for irr in irreps_indices]) == total_size
        
        col_indices = []
        for irr in irreps_indices:
            col_indices += list(range(irreps_pos[irr], irreps_pos[irr]+irreps[irr].size))
        
        assert len(col_indices) == total_size
        
        new_cob = repr.change_of_basis[np.ix_(row_indices, col_indices)]
        
        field_repr = Representation(repr.group,
                                      f"{repr.name}_{block}",
                                      [irreps[id].name for id in irreps_indices],
                                      new_cob,
                                      repr.supported_nonlinearities)
        representations.append(field_repr)

        next_position = current_position + len(row_indices)
        change_of_basis[current_position:next_position, row_indices] = np.eye(len(row_indices))
        
        current_position = next_position
        
    return change_of_basis, representations


def change_basis(repr: Representation,
                 change_of_basis: np.ndarray,
                 name: str,
                 supported_nonlinearities: List[str] = None
                 ) -> Representation:
    r"""
    Build a new representation from an already existing one by applying a change of basis.
    In other words, if :math:`\rho(\cdot)` is the representation and :math:`Q` the change of basis in input, the
    resulting representation will evaluate to :math:`Q \rho(\cdot) Q^{-1}`.
    
    Notice that the change of basis :math:`Q` has to be invertible.
    
    
    Args:
        repr (Representation): the input representation
        change_of_basis (~numpy.ndarray): the change of basis to apply
        name (str, optional): the name to use to identify the new representation
        supported_nonlinearities (list, optional): a list containing the ids of the supported non-linearities
            for the new representation

    Returns:
        the new representation

    """
    assert len(change_of_basis.shape) == 2
    assert change_of_basis.shape[0] == change_of_basis.shape[1]
    assert change_of_basis.shape[0] == repr.size
    
    if supported_nonlinearities is None:
        # by default, no non-linearities are supported
        supported_nonlinearities = []
    
    # compute the new change of basis
    new_cob = change_of_basis @ repr.change_of_basis
    new_cob_inv = repr.change_of_basis_inv @ sp.linalg.inv(change_of_basis)
    
    return Representation(repr.group, name, repr.irreps, new_cob,
                          supported_nonlinearities=supported_nonlinearities,
                          change_of_basis_inv=new_cob_inv)


def build_regular_representation(group: e2cnn.group.Group) -> Tuple[List[e2cnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""
    
    Build the regular representation of the input ``group``.
    As the regular representation has size equal to the number of elements in the group, only
    finite groups are accepted.
    
    Args:
        group (Group): the group whose representations has to be built

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the regular representation

    """
    assert group.order() > 0
    assert group.elements is not None and len(group.elements) > 0
    
    size = group.order()

    index = {e: i for i, e in enumerate(group.elements)}
    
    representation = {}
    character = {}
    
    for e in group.elements:
        # print(index[e], e)
        r = np.zeros((size, size), dtype=float)
        for g in group.elements:
            
            eg = group.combine(e, g)

            i = index[g]
            j = index[eg]
            
            r[j, i] = 1.0
        
        representation[e] = r
        # the character maps an element to the trace of its representation
        character[e] = np.trace(r)

    # compute the multiplicities of the irreps from the dot product between
    # their characters and the character of the representation
    irreps = []
    multiplicities = []
    for irrep_name, irrep in group.irreps.items():
        # for each irrep
        multiplicity = 0.0
    
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(group.inverse(element))
    
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
    
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep %s is not an integer: %f" % (irrep_name, multiplicity)
        # print(irrep_name, multiplicity)

        multiplicity = int(round(multiplicity))
        irreps += [irrep]*multiplicity
        multiplicities += [(irrep, multiplicity)]
    
    P = directsum(irreps, name="irreps")
    
    v = np.zeros((size, 1), dtype=float)
    
    p = 0
    for irr, m in multiplicities:
        assert irr.size >= m
        s = irr.size
        v[p:p+m*s, 0] = np.eye(m, s).reshape(-1) * np.sqrt(s)
        p += m*s
        
    change_of_basis = np.zeros((size, size))
    
    np.set_printoptions(precision=4, threshold=10*size**2, suppress=False, linewidth=25*size + 5)
    
    for e in group.elements:
        ev = P(e) @ v
        change_of_basis[index[e], :] = ev.T
    
    change_of_basis /= np.sqrt(size)
    
    # the computed change of basis is an orthonormal matrix
    
    # change_of_basis_inv = sp.linalg.inv(change_of_basis)
    change_of_basis_inv = change_of_basis.T
    
    return irreps, change_of_basis, change_of_basis_inv
    
    # return Representation(group,
    #                       "regular",
    #                       [r.name for r in irreps],
    #                       change_of_basis,
    #                       ['pointwise', 'norm', 'gated', 'concatenated'],
    #                       representation=representation,
    #                       change_of_basis_inv=change_of_basis_inv)


def build_quotient_representation(group: e2cnn.group.Group,
                                  subgroup_id
                                  ) -> Tuple[List[e2cnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""

    Build the quotient representation of the input ``group`` with respect to the subgroup identified by ``subgroup_id``.
    
    .. seealso::
        See the :class:`~e2cnn.group.Group` instance's implementation of the method :meth:`~e2cnn.group.Group.subgroup`
        for more details on ``subgroup_id``.
    
    .. warning ::
        Only finite groups are supported
    
    Args:
        group (Group): the group whose representation has to be built
        subgroup_id: identifier of the subgroup

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the quotient representation

    """
    subgroup, _, _ = group.subgroup(subgroup_id)
    return build_induced_representation(group,
                                        subgroup_id,
                                        subgroup.trivial_representation)


def build_induced_representation(group: e2cnn.group.Group,
                                 subgroup_id,
                                 repr: e2cnn.group.IrreducibleRepresentation,
                                 ) -> Tuple[List[e2cnn.group.IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""

    Build the induced representation of the input ``group`` from the representation ``repr`` of the subgroup
    identified by ``subgroup_id``.

    .. seealso::
        See the :class:`~e2cnn.group.Group` instance's implementation of the method :meth:`~e2cnn.group.Group.subgroup`
        for more details on ``subgroup_id``.

    .. warning ::
        Only irreducible representations are supported as the subgroup representation.

    .. warning ::
        Only finite groups are supported.

    Args:
        group (Group): the group whose representation has to be built
        subgroup_id: identifier of the subgroup
        repr (IrreducibleRepresentation): the representation of the subgroup

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the induced representation

    """
    
    assert repr.irreducible, "Induction from general representations is not supported yet"
    assert group.order() > 0, "Induction from non-discrete groups is not supported yet"
    assert group.elements is not None and len(group.elements) > 0
    
    subgroup, parent, child = group.subgroup(subgroup_id)
    
    assert repr.group == subgroup
    
    # compute the "index" of the subgroup H in the group G
    quotient_size = int(group.order() / subgroup.order())
    
    # the size of the induced representation
    size = repr.size * quotient_size
    
    # the coset each element belongs to
    cosets = {}
    
    # map from a representative to the elements of its coset
    representatives = defaultdict(lambda: [])
    
    for e in group.elements:
        if e not in cosets:
            representatives[e] = []
            for g in subgroup.elements:
                eg = group.combine(e, parent(g))
                
                cosets[eg] = e
                
                representatives[e].append(eg)
    
    index = {e: i for i, e in enumerate(representatives)}
    
    # compute the matrix and the character associated to each group element by the induced representation
    
    representation = {}
    character = {}
    
    for g in group.elements:
        repr_g = np.zeros((size, size), dtype=float)
        for r in representatives:
            gr = group.combine(g, r)
            
            g_r = cosets[gr]
            
            i = index[r]
            j = index[g_r]
            
            hp = group.combine(group.inverse(g_r), gr)
            
            h = child(hp)
            assert h is not None, (g, r, gr, g_r, group.inverse(g_r), hp)
            
            repr_g[j * repr.size:(j + 1) * repr.size, i * repr.size:(i + 1) * repr.size] = repr(h)
        
        representation[g] = repr_g
        
        # the character maps an element to the trace of its representation
        character[g] = np.trace(repr_g)
    
    # compute the multiplicities of the G-irreps in the induced representation using the
    # orthogonality theorem from Character Theory over the real field
    
    irreps = []
    multiplicities = []
    for irrep_name, irrep in group.irreps.items():
        # for each irrep
        multiplicity = 0.0
        
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(group.inverse(element))
        
        # adapt the multiplicities in the case of a splitting field (e.g. the real numbers for SO(2))
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
        
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep %s is not an integer: %f" % (irrep_name, multiplicity)
        
        multiplicity = int(round(multiplicity))
        irreps += [irrep] * multiplicity
        multiplicities += [(irrep, multiplicity)]
    
    def build_commuting_matrix(rho: e2cnn.group.IrreducibleRepresentation, t):
        # In a splitting field, the intertwiner between copies of the same irrep does not
        # always need to be a multiple of the identity
        
        if rho.sum_of_squares_constituents == 1:
            # In a non-splitting field, a basis for the intertwiners contains only the identity
            E = np.eye(rho.size)
        else:
            # In a splitting field, there are more solutions
            if t % 2 == 0:
                E = np.eye(2)
            else:
                E = np.array([[0, -1], [1, 0]])

        return E

    P = directsum(irreps, name="irreps")

    # rectangular matrix contained in one of the blocks (associated with the coset containing the trivial element)
    # of the change of basis matrix
    v = np.zeros((size, repr.size), dtype=float)

    # position in the matrix
    p = 0
    
    # norm of the column vectors in the matrix `v`
    norm_squared = 0
    
    # iterate over all G-irreps in the induced representation
    for irr, m in multiplicities:
        assert irr.size >= m
        
        if m > 0:
            restricted_irr = group.restrict_representation(subgroup_id, irr)
            
            # indices and positions of the subgroup irrep in the restricted irrep in the induced representation
            J = []
            x = 0
            for j, name in enumerate(restricted_irr.irreps):
                if name == repr.name:
                    J.append((j, x))
                x += subgroup.irreps[name].size
                
            # using Frobenius reciprocity for induced characters on the real field
            assert repr.sum_of_squares_constituents * len(J) == m * irr.sum_of_squares_constituents, \
                (f"{group.name}\{subgroup.name}:{repr.name}", irr.name, m, len(J), irr.sum_of_squares_constituents)
            
            # number of vectors from the basis to use
            if m == len(J):
                N = m
                dn = 1
                dr = 1
            else:
                N = repr.sum_of_squares_constituents * len(J)
                dn = repr.sum_of_squares_constituents
                dr = irr.sum_of_squares_constituents
            
            for shift in range(m):
                for i in range(dr):
                    idx = shift * irr.sum_of_squares_constituents + i
                    
                    j, x = J[idx % len(J)]
                    v[p+x:p+x + repr.size, :] = build_commuting_matrix(repr, idx // len(J))
                    
                v[p:p + irr.size, :] = restricted_irr.change_of_basis @ v[p:p + irr.size, :]
                
                # scale the vector to ensure the matrix is orthogonal
                v[p:p + irr.size, :] *= np.sqrt(irr.size/irr.sum_of_squares_constituents)
                
                p += irr.size
                
            # accumulate the square of norms of each subvector
            norm_squared += N * irr.size/irr.sum_of_squares_constituents
        
    # normalize the column vectors to ensure the columns have unit length
    v /= np.sqrt(norm_squared)

    # build the complete change of basis
    # fill the blocks associated with each coset by transforming the block `v` with the representative of the cosets
    change_of_basis_inv = np.zeros((size, size))
    for r in representatives:
        i = index[r]
        change_of_basis_inv[:, i * repr.size:(i + 1) * repr.size] = P(r) @ v

    # invert the change of basis
    
    change_of_basis = change_of_basis_inv.T
    # change_of_basis = linalg.inv(change_of_basis_inv)
    
    assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(size)), f"{group.name}\{subgroup.name}:{repr.name} - change of basis not orthonormal\n"
    assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(size)), f"{group.name}\{subgroup.name}:{repr.name} - change of basis not orthonormal\n"

    for g, r in representation.items():
        ir = change_of_basis @ P(g) @ change_of_basis_inv
        assert np.allclose(ir, representation[g]), f"{group.name}\{subgroup.name}:{repr.name} - {g}:\n{ir}\n{representation[g]}\n"
    
    return irreps, change_of_basis, change_of_basis_inv


########################################################################################################################
# Utils methods for decomposing or composing representations ###########################################################
########################################################################################################################

from scipy import linalg, sparse
import scipy.sparse.linalg as slinalg
from scipy.sparse import find


try:
    import pymanopt
    from pymanopt.manifolds import Euclidean
    from pymanopt.solvers import TrustRegions
    
except ImportError:
    pymanopt = None
    
try:
    import autograd.numpy as anp
except ImportError:
    anp = None


def direct_sum_factory(irreps: List[e2cnn.group.IrreducibleRepresentation],
                       change_of_basis: np.ndarray,
                       change_of_basis_inv: np.ndarray = None
                       ) -> Callable[[Any], np.ndarray]:
    """
    The method builds and returns a function implementing the direct sum of the "irreps" transformed by the given
    "change_of_basis" matrix.

    More precisely, the built method will take as input a value accepted by all the irreps, evaluate the irreps on that
    input and return the direct sum of the produced matrices left and right multiplied respectively by the
    change_of_basis matrix and its inverse.

    Args:
        irreps (list): list of irreps
        change_of_basis: the matrix transforming the direct sum of the irreps
        change_of_basis_inv: the inverse of the change of basis matrix

    Returns:
        function taking an input accepted by the irreps and returning the direct sum of the irreps evaluated
        on that input
    """
    
    shape = change_of_basis.shape
    assert len(shape) == 2 and shape[0] == shape[1]
    
    if change_of_basis_inv is None:
        # pre-compute the inverse of the change-of-_bases matrix
        change_of_basis_inv = linalg.inv(change_of_basis)
    else:
        assert len(change_of_basis_inv.shape) == 2
        assert change_of_basis_inv.shape[0] == change_of_basis.shape[0]
        assert change_of_basis_inv.shape[1] == change_of_basis.shape[1]
        assert np.allclose(change_of_basis @ change_of_basis_inv, np.eye(change_of_basis.shape[0]))
        assert np.allclose(change_of_basis_inv @ change_of_basis, np.eye(change_of_basis.shape[0]))
    
    unique_irreps = list({irr.name: irr for irr in irreps}.items())
    irreps_names = [irr.name for irr in irreps]
    
    def direct_sum(element,
                   irreps_names=irreps_names, change_of_basis=change_of_basis,
                   change_of_basis_inv=change_of_basis_inv, unique_irreps=unique_irreps):
        reprs = {}
        for n, irr in unique_irreps:
            reprs[n] = irr(element)
        
        blocks = []
        for irrep_name in irreps_names:
            repr = reprs[irrep_name]
            blocks.append(repr)
        
        P = sparse.block_diag(blocks, format='csc')
        
        return change_of_basis @ P @ change_of_basis_inv
    
    return direct_sum


def null(A: Union[np.matrix, sparse.linalg.LinearOperator],
         use_sparse: bool,
         eps: float = 1e-12
         ) -> np.ndarray:
    """
    Compute _bases for the Kernel space of the matrix A.

    If ``use_sparse`` is ``True``, :meth:`scipy.sparse.linalg.svds` is used;
    otherwise, :meth:`scipy.linalg.svd` is used.

    Moreover, if the input is a sparse matrix, ``use_sparse`` has to be set to ``True``.

    Args:
        A: input matrix
        use_sparse: whether to use spare methods or not
        eps: threshold to consider a value zero. The default value is ``1e-12``

    Returns:
        A matrix whose columns are a basis of the kernel space

    """
    if use_sparse:
        u, s, vh = slinalg.svds(A, k=min(A.shape) - 1)
    else:
        u, s, vh = linalg.svd(A)
    
    # print(u.shape, s.shape, vh.shape)
    # print(min(s))
    null_space = np.compress((s <= eps), vh, axis=0)
    return np.transpose(null_space)


def find_orthogonal_matrix(basis: np.ndarray, shape):
    
    if pymanopt is None:
        raise ImportError("Missing optional 'pymanopt' dependency. Install 'pymanopt' to use this function")
    
    if anp is None:
        raise ImportError("Missing optional 'autograd' dependency. Install 'autograd' to use this function")

    manifold = Euclidean(basis.shape[1])
    
    def cost(X):
        d = anp.dot(basis, X).reshape(shape, order='F')
        return anp.sum(
            anp.square(anp.dot(d, d.T) - anp.eye(*shape)) +
            anp.square(anp.dot(d.T, d) - anp.eye(*shape))
        )
    
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    
    # solver = TrustRegions(use_rand=True, miniter=10, mingradnorm=1e-10)
    # solver = ParticleSwarm(populationsize=500, maxcostevals=10000, logverbosity=0)
    # solver = ParticleSwarm(logverbosity=0)
    
    import os, sys
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    #
    # Xopt = solver.solve(problem)
    # c = cost(Xopt)
    # print('PSO, Final Error:', c)
    #
    # x = Xopt
    
    solver = TrustRegions(mingradnorm=1e-10, logverbosity=0)
    
    Xopt = solver.solve(problem)  # , x=x) #, Delta_bar=np.sqrt(basis.shape[1])*2)
    
    c = cost(Xopt)
    
    sys.stdout = old_stdout  # sys.__stdout__
    
    # print('TrustRegions, Final Error:', c)
    # print('Weights:', Xopt)
    
    D = np.dot(basis, Xopt).reshape(shape, order='F')
    
    return D, c


def compute_change_of_basis(representation: Dict[Any, np.matrix],
                            irreps: List[Tuple[Callable[[Any], np.matrix], int]]
                            ) -> np.matrix:
    r"""
    This method computes the change-of-_bases matrix that decompose a representation of a finite group
    in a direct sum of irreps.

    Notice that the irreps are "stacked" in the same order as they are in the "irreps" list and consecutive copies
    of each irrep are added accordingly to the multiplicities specified.

    Args:
        representation: a dictionary mapping an element of "group" to a matrix
        irreps: a list of pairs (callable, integer). The callable implements an representation (takes an element as input and returns a matrix)
        and the integer is the multiplicity of this representation (i.e. how many times it has to appear in the decomposition)

    Returns:
        the change of _bases matrix

    """
    
    # Contains a list of Sylvester Equations, one for each group element
    equations = []
    # for each group element build the corresponding equation and append it to the list
    for element, rho in representation.items():
        # Build the direct sum of the irreps for this element
        blocks = []
        for (irrep, m) in irreps:
            repr = irrep(element)
            for i in range(m):
                blocks.append(repr)
        P = sparse.block_diag(blocks, format='csc')
        
        # build the linear system corresponding to the Sylvester Equation with the current group element
        equation = sparse.kronsum(rho, -1 * P.T, format='csc')
        
        equations.append(equation)
    
    # stack all equations in one unique matrix
    M = sparse.vstack(equations, format='csc')
    
    # the kernel space of this matrix contains the solutions of our problem
    
    if M.shape[1] == 1:
        assert np.count_nonzero(M.todense()) == 0
        return np.ones([1, 1])
    else:
        
        # compute the basis of the kernel
        if len(representation) > 10:
            basis = null(M, True)
        else:
            basis = null(M.todense(), False)
        
        assert np.allclose(M @ basis, np.zeros([M.shape[0], basis.shape[1]]))
        
        # reshape it to get the Change of Basis matrix
        shape = list(representation.values())[0].shape
        
        # np.set_printoptions(precision=2, threshold=2 * len(representation)**2, suppress=True,
        #                     linewidth=len(representation) * 10 + 3)
        
        basis = linalg.orth(basis)
        
        # we could take any linear combination of the basis vectors to get the vectorized form of the Change of Basis matrix
        # d = basis @ np.random.randn(basis.shape[1], 1)
        
        # in case of CyclicGroup, if we have all the basis (i.e. we don't use the SparseSVD algorithm),
        # the sum of all basis vectors seems to always lead to an orthonormal matrix
        # d = basis @ np.ones((basis.shape[1], 1))
        # D = np.reshape(d, shape, order='F')
        
        # however, for large groups we can't use the dense SVD, so we need to find another orthonormal matrix in the
        # smaller space of solutions
        D, err = find_orthogonal_matrix(basis, shape)
        
        # print(D)
        # print(D @ D.T)
        # print(D.T @ D)
        
        # assert np.allclose(D @ D.T, np.eye(*shape))
        # assert np.allclose(D.T @ D, np.eye(*shape))
        
        # in case we take a random combination of the basis vectors, it is possible that the generated matrix is
        # singular. To be sure it is not we sample a few matrices and pick the one with the largest smallest singular
        # value. Anyway, the event of sampling a singular matrix should be unlikely enough to assume it never happens
        
        # max_sv = min(linalg.svd(D, compute_uv=False))
        # for i in range(50):
        #     # take any linear combination of them to get the vectorized form of the Change of Basis matrix
        #     d = _bases @ np.random.randn(_bases.shape[1], 1)
        #
        #     d = np.reshape(d, shape, order='F')
        #
        #     s = min(linalg.svd(d, compute_uv=False))
        #
        #     if s > max_sv:
        #         max_sv = s
        #         D = d
        
        # Check the change of basis found is right
        D_inv = linalg.inv(D)
        for element, rho in representation.items():
            # Build the direct sum of the irreps for this element
            blocks = []
            for (irrep, m) in irreps:
                repr = irrep(element)
                for i in range(m):
                    blocks.append(repr)
            
            P = sparse.block_diag(blocks, format='csc')
            
            # if not np.allclose(rho, D @ P @ D_inv):
            #     print(element)
            #     print(rho)
            #     print(D @ P @ D_inv)
            
            assert (np.allclose(rho, D @ P @ D_inv)), "Error at element {}".format(element)
        
        return D


def decompose_representation(representation: Dict[Any, np.matrix],
                             group: e2cnn.group.Group
                             ) -> Tuple[np.matrix, List[Tuple[str, int]]]:
    r"""
    Decompose the input ``representation`` in a direct sum of irreps of the input ``group``.
    First, the method computes the multiplicities of each irrep in the representation using the inner product of their
    characters. Then, it computes the change-of-basis matrix which transforms the block-diagonal matrix coming from
    the direct sum of the irreps in the input representation.

    It returns the decomposition in irreps as a change-of-basis matrix and a list of "(irrep-name, multiplicity)" pairs,
    where "irrep-name" is the name of one of the irreps in ``group`` (a key in the :attr:`e2cnn.group.Group.irreps`
    dictionary) and "multiplicity" is the number of times this irrep appears in the decomposition.
    The order of this list follows the alphabetic order of the names and it represents the order in which the irreps
    have to be summed to build the block-diagonal representation.

    Args:
        representation: a dictionary associating to each group element a matrix representation
        group: the group whose irreps have to be used

    Returns:
        a tuple containing:

                - the change-of-basis matrix,

                - an ordered list of pairs (irrep-name, multiplicity)

    """
    
    # TODO - check elements of the dictionary are all and only the elements of the group
    
    # compute the character of the representation w.r.t. the discrete group given
    character = {}
    for element, repr in representation.items():
        # the character maps an element to the trace of its representation
        character[element] = np.trace(repr)
    
    # compute the multiplicities of the irreps from the dot product between
    # their characters and the character of the representation
    multiplicities = []
    for irrep_name, irrep in group.irreps.items():
        # for each irrep
        multiplicity = 0.0
        
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(group.inverse(element))
        
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
        
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep %s is not an integer: %f" % (irrep_name, multiplicity)
        
        multiplicities.append((irrep_name, int(round(multiplicity))))
    
    # sort irreps by their name
    multiplicities = sorted(multiplicities, key=lambda x: x[0])
    
    # build a list of representation instaces with their multiplicities
    irreps = [(group.irreps[name], m) for (name, m) in multiplicities]
    
    # compute te Change-Of-Basis matrix that transform the direct sum of irreps in the representation
    cob = compute_change_of_basis(representation, irreps)
    
    return cob, multiplicities


def sparse_allclose(A, B, atol=1e-8):
    diff = abs(A - B)
    _, _, v = find(diff)
    
    return np.less_equal(v, atol).all()
