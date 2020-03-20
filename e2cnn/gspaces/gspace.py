
from __future__ import annotations

import e2cnn.kernels
import e2cnn.group

from abc import ABC, abstractmethod
from typing import Tuple, Callable

import numpy as np


__all__ = ["GSpace"]


class GSpace(ABC):
    
    def __init__(self, fibergroup: e2cnn.group.Group, dimensionality: int, name: str):
        r"""
        Abstract class for G-spaces.
        
        A ``GSpace`` describes the space where a signal lives (e.g. :math:`\R^2` for planar images) and its symmetries
        (e.g. rotations or reflections).
        As an `Euclidean` base space is assumed, a G-space is fully specified by the ``dimensionality`` of the space
        and a choice of origin-preserving symmetry group (``fibergroup``).
        
        .. seealso::
            
            :class:`~e2cnn.gspaces.FlipRot2dOnR2`,
            :class:`~e2cnn.gspaces.Rot2dOnR2`,
            :class:`~e2cnn.gspaces.Flip2dOnR2`,
            :class:`~e2cnn.gspaces.TrivialOnR2`
        
        .. note ::
        
            Mathematically, this class describes a *Principal Bundle*
            :math:`\pi : (\R^D, +) \rtimes G \to \mathbb{R}^D, tg \mapsto tG`,
            with the Euclidean space :math:`\mathbb{R}^D` (where :math:`D` is the ``dimensionality``) as `base space`
            and :math:`G` as `fiber group` (``fibergroup``).
            For more details on this interpretation we refer to
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.
        
        
        Args:
            fibergroup (Group): the fiber group
            dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            name (str): an identification name
        
        Attributes:
            ~.fibergroup (Group): the fiber group
            ~.dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            ~.name (str): an identification name
            ~.basespace (str): the name of the space whose symmetries are modeled. It is an Euclidean space :math:`\R^D`.
        
        """

        self.name = name
        self.dimensionality = dimensionality
        self.fibergroup = fibergroup
        self.basespace = f"R^{self.dimensionality}"

    @abstractmethod
    def restrict(self, id) -> Tuple[GSpace, Callable, Callable]:
        r"""

        Build the :class:`~e2cnn.gspaces.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.
        This reduces the level of symmetries of the base space to be considered.

        Check the ``restrict`` method's documentation in the non-abstract subclass used for a description of the
        parameter ``id``.

        Args:
            id: id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        """
        pass

    def featurefield_action(self, input: np.ndarray, repr: e2cnn.group.Representation, element) -> np.ndarray:
        r"""
        
        This method implements the action of the symmetry group on a feature field defined over the basespace of this
        G-space.
        It includes both an action over the basespace (e.g. a rotation of the points on the plane) and a transformation
        of the channels by left-multiplying them with a representation of the fiber group.

        The method takes as input a tensor (``input``), a representation (``repr``) and an ``element`` of the fiber
        group. The tensor ``input`` is the feature field to be transformed and needs to be compatible with this G-space
        and the representation (i.e. its number of channels equals the size of that representation).
        ``element`` needs to belong to the fiber group: check :meth:`e2cnn.group.Group.is_element`.
        This method returns a transformed tensor through the action of ``element``.

        More precisely, given an input tensor, interpreted as an :math:`c`-dimensional signal
        :math:`f: \R^D \to \mathbb{R}^c` defined over the base space :math:`\R^D`, a representation
        :math:`\rho: G \to \mathbb{R}^{c \times c}` of :math:`G` and an element :math:`g \in G` of the fiber group,
        the method returns the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := \rho(g) f(g^{-1} x)

        .. note ::

            Mathematically, this method transforms the input with the **induced representation** from the input ``repr``
            (:math:`\rho`) of the symmetry group (:math:`G`) to the *total space* (:math:`P`), i.e.
            with :math:`Ind_{G}^{P} \rho`.
            For more details on this, see
            `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_ or
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.

        Args:
            input (~numpy.ndarray): input tensor
            repr (Representation): representation of the fiber group
            element: element of the fiber group

        Returns:
            the transformed tensor

        """
        assert repr.group == self.fibergroup
    
        rho = repr(element)
    
        output = np.einsum("oi,bi...->bo...", rho, input)
    
        return self._basespace_action(output, element)

    @abstractmethod
    def _basespace_action(self, input: np.ndarray, element) -> np.ndarray:
        r"""

        Defines how the fiber group transforms the base space.

        The methods takes a tensor compatible with this space (i.e. whose spatial dimensions are supported by the
        base space) and returns the transformed tensor.

        More precisely, given an input tensor, interpreted as an :math:`n`-dimensional signal
        :math:`f: X \to \mathbb{R}^n` defined over the base space :math:`X`, and an element :math:`g \in G` of the
        fiber group, the methods return the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := f(g^{-1} x)

        This method is specific of the particular GSpace and defines how :math:`g^{-1}` transforms a point
        :math:`x \in X` of the base space.


        Args:
            input (~numpy.ndarray): input tensor
            element: element of the fiber group

        Returns:
            the transformed tensor

        """
        pass

    @abstractmethod
    def build_kernel_basis(self,
                           in_repr: e2cnn.group.Representation,
                           out_repr: e2cnn.group.Representation,
                           **kwargs) -> e2cnn.kernels.KernelBasis:
        r"""

        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~e2cnn.gspaces.GSpace`.

        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^D
        
        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.
        
        This method relies on the functionalities implemented in :mod:`e2cnn.kernels` and returns an instance of
        :class:`~e2cnn.kernels.KernelBasis`.


        Args:
            in_repr (Representation): the representation associated with the input field
            out_repr (Representation): the representation associated with the output field
            **kwargs: additional keyword arguments for the equivariance contraint solver

        Returns:

            a basis for space of equivariant convolutional kernels


        """
        pass

    @property
    def irreps(self):
        r"""
        Dictionary containing all the already built irreducible representations of the fiber group of this space.

        .. seealso::

            See :attr:`e2cnn.group.Group.irreps` for more details

        """
        return self.fibergroup.irreps

    @property
    def representations(self):
        r"""
        Dictionary containing all the already built representations of the fiber group of this space.

        .. seealso::

            See :attr:`e2cnn.group.Group.representations` for more details

        """
        return self.fibergroup.representations

    @property
    def trivial_repr(self) -> e2cnn.group.Representation:
        r"""
        The trivial representation of the fiber group of this space.

        .. seealso::

            :attr:`e2cnn.group.Group.trivial_representation`

        """
        return self.fibergroup.trivial_representation

    def irrep(self, *id) -> e2cnn.group.IrreducibleRepresentation:
        r"""
        Builds the irreducible representation (:class:`~e2cnn.group.IrreducibleRepresentation`) of the fiber group
        identified by the input arguments.

        .. seealso::

            This method is a wrapper for :meth:`e2cnn.group.Group.irrep`. See its documentation for more details.
            Check the documentation of :meth:`~e2cnn.group.Group.irrep` of the specific fiber group used for more
            information on the valid ``id``.


        Args:
            *id: parameters identifying the irrep.

        """
        return self.fibergroup.irrep(*id)

    @property
    def regular_repr(self) -> e2cnn.group.Representation:
        r"""
        The regular representation of the fiber group of this space.

        .. seealso::

            :attr:`e2cnn.group.Group.regular_representation`

        """
        return self.fibergroup.regular_representation

    def quotient_repr(self, subgroup_id) -> e2cnn.group.Representation:
        r"""
        Builds the quotient representation of the fiber group of this space with respect to the subgroup identified
        by ``subgroup_id``.
        
        Check the :meth:`~e2cnn.gspaces.GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::
            
            See :attr:`e2cnn.group.Group.quotient_representation` for more details on the representation.
        
        Args:
            subgroup_id: identifier of the subgroup

        """
        return self.fibergroup.quotient_representation(subgroup_id)

    def induced_repr(self, subgroup_id, repr: e2cnn.group.Representation) -> e2cnn.group.Representation:
        r"""
        Builds the induced representation of the fiber group of this space from the representation ``repr`` of
        the subgroup identified by ``subgroup_id``.

        Check the :meth:`~e2cnn.gspaces.GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::
            
            See :attr:`e2cnn.group.Group.induced_representation` for more details on the representation.
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup to induce

        """
        return self.fibergroup.induced_representation(subgroup_id, repr)

    @property
    def testing_elements(self):
        return self.fibergroup.testing_elements()


