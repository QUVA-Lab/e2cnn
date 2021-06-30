
from typing import List, Dict

from collections import defaultdict

from e2cnn.group import Group
from e2cnn.group import Representation
from e2cnn.gspaces import GSpace
from e2cnn.group import directsum

import numpy as np
from scipy import sparse

import torch

__all__ = ["FieldType"]


class FieldType:
    
    def __init__(self,
                 gspace: GSpace,
                 representations: List[Representation]):
        r"""
        
        An ``FieldType`` can be interpreted as the *data type* of a feature space. It describes:
        
        - the base space on which a feature field is living and its symmetries considered
        
        - the transformation law of feature fields under the action of the fiber group
        
        The former is formalize by a choice of ``gspace`` while the latter is determined by a choice of group
        representations (``representations``), passed as a list of :class:`~e2cnn.group.Representation` instances.
        Each single representation in this list corresponds to one independent feature field contained in the feature
        space.
        The input ``representations`` need to belong to ``gspace``'s fiber group
        (:attr:`e2cnn.gspaces.GSpace.fibergroup`).
        
        .. note ::
            
            Mathematically, this class describes a *(trivial) vector bundle*, *associated* to the symmetry group
            :math:`(\R^D, +) \rtimes G`.
            
            Given a *principal bundle* :math:`\pi: (\R^D, +) \rtimes G \to \R^D, tg \mapsto tG`
            with fiber group :math:`G`, an *associated vector bundle* has the same base space
            :math:`\R^D` but its fibers are vector spaces like :math:`\mathbb{R}^c`.
            Moreover, these vector spaces are associated to a :math:`c`-dimensional representation :math:`\rho` of the
            fiber group :math:`G` and transform accordingly.
            
            The representation :math:`\rho` is defined as the *direct sum* of the representations :math:`\{\rho_i\}_i`
            in ``representations``. See also :func:`~e2cnn.group.directsum`.
            
        
        Args:
            gspace (GSpace): the space where the feature fields live and its symmetries
            representations (list): a list of :class:`~e2cnn.group.Representation` s of the ``gspace``'s fiber group,
                            determining the transformation laws of the feature fields
        
        Attributes:
            ~.gspace (GSpace)
            ~.representations (list)
            ~.size (int): dimensionality of the feature space described by the :class:`~e2cnn.nn.FieldType`.
                          It corresponds to the sum of the dimensionalities of the individual feature fields or
                          group representations (:attr:`e2cnn.group.Representation.size`).
 
            
        """
        assert len(representations) > 0
        
        for repr in representations:
            assert repr.group == gspace.fibergroup
        
        # GSpace: Space where data lives and its (abstract) symmetries
        self.gspace = gspace
        
        # list: List of representations of each feature field composing the feature space of this type
        self.representations = representations
        
        # int: size of the field associated to this type.
        # as the representation associated to the field is the direct sum of the representations
        # in :attr:`e2cnn.nn.fieldtype.representations`, its size is the sum of each of these
        # representations' size
        self.size = sum([repr.size for repr in representations])

        self._unique_representations = set(self.representations)
        
        self._representation = None
        
        self._field_start = None
        self._field_end = None

        self._hash = hash(self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}')

    @property
    def fibergroup(self) -> Group:
        r"""
        The fiber group of :attr:`~e2cnn.nn.FieldType.gspace`.

        Returns:
            the fiber group

        """
        return self.gspace.fibergroup

    @property
    def representation(self) -> Representation:
        r"""
        The (combined) representations of this field type.
        They describe how the feature vectors transform under the fiber group action, that is, how the channels mix.
 
        It is the direct sum (:func:`~e2cnn.group.directsum`) of the representations in
        :attr:`e2cnn.nn.FieldType.representations`.
        
        Because a feature space can contain a very large number of feature fields, computing this representation as
        the direct sum of many small representations can be expensive.
        Hence, this representation is only built the first time it is explicitly used, in order to avoid unnecessary
        overhead when not needed.
        
        Returns:
            the :class:`~e2cnn.group.Representation` describing the whole feature space
            
        """
        if self._representation is None:
            uniques_fields_names = sorted([r.name for r in self._unique_representations])
            self._representation = directsum(self.representations, name=f"FiberRepresentation:[{self.size}], [{uniques_fields_names}]")

        return self._representation

    @property
    def irreps(self):
        r"""
        Ordered list of irreps contained in the :attr:`~e2cnn.nn.FieldType.representation` of the field type.
        It is the concatenation of the irreps in each representation in :attr:`e2cnn.nn.FieldType.representations`.

        Returns:
            list of irreps

        """
        irreps = []
        for repr in self.representations:
            irreps += repr.irreps
        return irreps

    @property
    def change_of_basis(self) -> sparse.coo_matrix:
        r"""
        
        The change of basis matrix which decomposes the field types representation into irreps, given as a sparse
        (block diagonal) matrix (:class:`scipy.sparse.coo_matrix`).
        
        It is the direct sum of the change of basis matrices of each representation in
        :attr:`e2cnn.nn.FieldType.representations`.
        
        .. seealso ::
            :attr:`e2cnn.group.Representation.change_of_basis`
 
        
        Returns:
            the change of basis
        
        """
        change_of_basis = []
        for repr in self.representations:
            change_of_basis.append(repr.change_of_basis)
        return sparse.block_diag(change_of_basis)

    @property
    def change_of_basis_inv(self):
        r"""
        Inverse of the (sparse) change of basis matrix. See :attr:`e2cnn.nn.FieldType.change_of_basis` for more details.
        
        Returns:
            the inverted change of basis

        """
        change_of_basis_inv = []
        for repr in self.representations:
            change_of_basis_inv.append(repr.change_of_basis_inv)
        return sparse.block_diag(change_of_basis_inv)

    def get_dense_change_of_basis(self) -> torch.FloatTensor:
        """
        The method returns a dense :class:`torch.Tensor` containing a copy of the change-of-basis matrix.
        
        .. seealso ::
            See :attr:`e2cnn.nn.FieldType.change_of_basis` for more details.

        """
        return torch.FloatTensor(self.change_of_basis.todense())

    def get_dense_change_of_basis_inv(self) -> torch.FloatTensor:
        """
        The method returns a dense :class:`torch.Tensor` containing a copy of the inverse of the
        change-of-basis matrix.
        
        .. seealso ::
            See :attr:`e2cnn.nn.FieldType.change_of_basis` for more details.

        """
        return torch.FloatTensor(self.change_of_basis_inv.todense())

    def transform(self, input: torch.Tensor, element) -> torch.Tensor:
        r"""

        The method takes a PyTorch's tensor, compatible with this type (i.e. whose spatial dimensions are supported
        by the base space and whose number of channels equals the :attr:`e2cnn.nn.FieldType.size`
        of this type), and an element of the fiber group of this type.

        Transform the input tensor according to the group representation associated with the input element
        and its (induced) action on the base space.

        .. warning ::
            This method is internally implemented using ```numpy```.
            This means that the input tensor is detached (and moved to CPU) before the transformation, therefore no
            gradient is propagated back through this operation.

        .. seealso ::

            See :meth:`e2cnn.nn.GeometricTensor.transform_fibers` to transform only the fibers, i.e. not transform
            the base space.

            See :meth:`e2cnn.gspaces.GSpace.featurefield_action` for more details.


        Args:
            input (torch.Tensor): input tensor
            element: element of the fiber group

        Returns:
            transformed tensor

        """
        if input.is_cuda:
            import warnings
            warnings.warn('The input tensor is on GPU. The `FieldType.transform()` operation is based on `numpy` and,'
                          ' therefore, must temporarily move the tensor on CPU. This can cause performance issues.')
            
        transformed = self.gspace.featurefield_action(input.detach().cpu().numpy(), self.representation, element)
        transformed = np.ascontiguousarray(transformed)
        return torch.from_numpy(transformed.astype(np.float32)).to(device=input.device)

    def restrict(self, id) -> 'FieldType':
        r"""
        
        Reduce the symmetries modeled by the :class:`~e2cnn.nn.FieldType` by choosing a subgroup of its fiber group as
        specified by ``id``. This implies a restriction of each representation in
        :attr:`e2cnn.nn.FieldType.representations` to this subgroup.
 
        .. seealso ::
        
            Check the documentation of the :meth:`~e2cnn.gspaces.GSpace.restrict` method in the subclass of
            :class:`~e2cnn.gspaces.GSpace` used for a description of the parameter ``id``.

        Args:
            id: identifier of the subgroup to which the :class:`~e2cnn.nn.FieldType` and its
                :attr:`e2cnn.nn.FieldType.representations` should be restricted

        Returns:
            the restricted type

        """
    
        # build the subgroup
        subspace, _, _ = self.gspace.restrict(id)
    
        # restrict each different base representation in the fiber representation
        restricted_reprs = {}
        for r in self._unique_representations:
            restricted_reprs[r.name] = self.gspace.fibergroup.restrict_representation(id, r)
    
        # for each field, retrieve the corresponding restricted representation
        fields = [restricted_reprs[r.name] for r in self.representations]
    
        # build the restricted fiber representation
        rrepr = FieldType(subspace, fields)
    
        return rrepr

    def sorted(self) -> 'FieldType':
        r"""

        Return a new field type containing the fields of the current one sorted by their dimensionalities.
        It is built from the :attr:`e2cnn.nn.FieldType.representations` of this field type sorted.

        Returns:
            the sorted field type

        """
        keys = [(r.size, i) for i, r in enumerate(self.representations)]
    
        keys = sorted(keys)
    
        permutation = [k[1] for k in keys]
    
        return self.index_select(permutation)

    def __add__(self, other: 'FieldType') -> 'FieldType':
        r"""

        Returns a field type associate with the *direct sum* :math:`\rho = \rho_1 \oplus \rho_2` of the representations
        :math:`\rho_1` and :math:`\rho_2` of two field types.
        
        In practice, the method builds a new :class:`~e2cnn.nn.FieldType` using the concatenation of the lists
        :attr:`e2cnn.nn.FieldType.representations` of the two field types.
        
        The two field types need to be associated with the same :class:`~e2cnn.gspaces.GSpace`.

        Args:
            other (FieldType): the other addend

        Returns:
            the direct sum

        """
    
        assert self.gspace == other.gspace
    
        return FieldType(self.gspace, self.representations + other.representations)

    def __len__(self) -> int:
        r"""

        Return the number of feature fields in this :class:`~e2cnn.nn.FieldType`, i.e. the length of
        :attr:`e2cnn.nn.FieldType.representations`.
        
        .. note ::
            This is in general different from :attr:`e2cnn.nn.FieldType.size`.

        Returns:
            the number of fields in this type

        """
        return len(self.representations)

    def fields_names(self) -> List[str]:
        r"""
        Return an ordered list containing the names of the representation associated with each field.

        Returns:
            the list of fields' representations' names

        """
        return [r.name for r in self.representations]

    def index_select(self, index: List[int]) -> 'FieldType':
        r"""
        
        Build a new :class:`~e2cnn.nn.FieldType` from the current one by taking the
        :class:`~e2cnn.group.Representation` s selected by the input ``index``.
        
        Args:
            index (list): a list of integers in the range ``{0, ..., N-1}``, where ``N`` is the number of representations
                          in the current field type

        Returns:
            the new field type
            
            
        """
        assert max(index) < len(self.representations)
        assert min(index) >= 0

        # retrieve the fields in the input representation to build the output representation
        representations = [self.representations[i] for i in index]
        return FieldType(self.gspace, representations)

    @property
    def fields_end(self) -> np.ndarray:
        r"""
        
            Array containing the index of the first channel following each field.
            More precisely, the integer in the :math:`i`-th position is equal to the index of the last channel of
            the :math:`i`-th field plus :math:`1`.
        
        """
        if self._field_end is None:
            field_idx = []
            p = 0
            for r in self.representations:
                p += r.size
                field_idx.append(p)
            self._field_end = np.array(field_idx, dtype=np.uint64)
    
        return self._field_end

    @property
    def fields_start(self) -> np.ndarray:
        r"""

            Array containing the index of the first channel of each field.
            More precisely, the integer in the :math:`i`-th position is equal to the index of the first channel of
            the :math:`i`-th field.

        """
        if self._field_start is None:
            field_idx = []
            p = 0
            for r in self.representations:
                field_idx.append(p)
                p += r.size
            self._field_start = np.array(field_idx, dtype=np.uint64)
            
        return self._field_start

    def group_by_labels(self, labels: List[str]) -> Dict[str, 'FieldType']:
        r"""
        
        Associate a label to each feature field (or representation in :attr:`e2cnn.nn.FieldType.representations`)
        and group them accordingly into new :class:`~e2cnn.nn.FieldType` s.
 
        Args:
            labels (list): a list of strings with length equal to the number of representations in
                           :attr:`e2cnn.nn.FieldType.representations`

        Returns:
            a dictionary mapping each different input label to a new field type

        """
        assert len(labels) == len(self)
        
        fields = defaultdict(lambda: [])
        
        for c, l in enumerate(labels):
            # append the index of the current field to the list of fields belonging to this label
            fields[l].append(c)
        
        # for each label, build the field type of the sub-fiber on which it acts
        types = {}
    
        for l in labels:
            # retrieve the sub-fiber corresponding to this label
            types[l] = self.index_select(fields[l])
        
        return types

    def __iter__(self):
        r"""
        
        It is possible to iterate over all :attr:`~e2cnn.nn.FieldType.representations` in a field type by using
        :class:`~e2cnn.nn.FieldType` as an *iterable* object.

        """
        return iter(self.representations)

    def __eq__(self, other):
        if isinstance(other, FieldType):
            return self.gspace == other.gspace and self.representations == other.representations
        else:
            return False
    
    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        return '[' + self.gspace.name + ': {' + ', '.join([r.name for r in self.representations]) + '}]'

    @property
    def testing_elements(self):
        r"""
        Alias for ``self.gspace.testing_elements``.
        
        .. seealso::
            :attr:`e2cnn.gspaces.GSpace.testing_elements` and
            :attr:`e2cnn.group.Group.testing_elements`
        
        """
        return self.gspace.testing_elements

    # def __call__(self, element) -> np.matrix:
    # #precompute the representation of the input element according to all the base representations
    # representation = {}
    # for r in self._unique_representations:
    #     representation[r.name] = r(element)
    #
    # #build the fiber representation by merging the base representations in one unique block diagonal matrix
    # blocks = []
    # for base_repr in self.representations:
    #     blocks.append(representation[base_repr.name])
    #
    # return sparse.block_diag(blocks, format='csc').todense()
