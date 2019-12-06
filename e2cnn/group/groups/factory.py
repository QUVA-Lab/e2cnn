
from .cyclicgroup import CyclicGroup
from .dihedralgroup import DihedralGroup
from .so2group import SO2
from .o2group import O2

__all__ = [
    "cyclic_group",
    "dihedral_group",
    "so2_group",
    "o2_group",
    "trivial_group",
]


def trivial_group():
    r"""
    
    Builds the trivial group :math:`C_1` which contains only the identity element :math:`e`.
    
    You should use this factory function to build an instance of the trivial group.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built,
    this unique instance is updated with the new representations and, therefore, all its references will see the new
    representations.
    
    Returns:
        the trivial group

    """
    return CyclicGroup._generator(1)


def cyclic_group(N: int):
    r"""

    Builds a cyclic group :math:`C_N`of order ``N``, i.e. the group of ``N`` discrete planar rotations.
    
    You should use this factory function to build an instance of :class:`e2cnn.group.CyclicGroup`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~e2cnn.group.Group.quotient_representation`), this unique instance is updated with
    the new representations and, therefore, all its references will see the new representations.

    Args:
        N (int): number of discrete rotations in the group

    Returns:
        the cyclic group of order ``N``

    """
    return CyclicGroup._generator(N)


def dihedral_group(N: int):
    r"""

    Builds a dihedral group :math:`D_{2N}`of order ``2N``, i.e. the group of ``N`` discrete planar rotations
    and reflections.
    
    You should use this factory function to build an instance of :class:`e2cnn.group.DihedralGroup`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`~e2cnn.group.Group.quotient_representation`), this unique instance is updated with
    the new representations and, therefore, all its references will see the new representations.

    Args:
        N (int): number of discrete rotations in the group
        
    Returns:
        the dihedral group of order ``2N``

    """
    return DihedralGroup._generator(N)


def so2_group(maximum_frequency: int = 10):
    r"""

    Builds the group :math:`SO(2)`, i.e. the group of continuous planar rotations.
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`e2cnn.group.SO2.irrep` (see the method's documentation).
    
    You should use this factory function to build an instance of :class:`e2cnn.group.SO2`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`e2cnn.group.SO2.irrep`), this unique instance is updated with the new representations
    and, therefore, all its references will see the new representations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`SO(2)`

    """
    return SO2._generator(maximum_frequency)


def o2_group(maximum_frequency: int = 10):
    r"""

    Builds the group :math:`O(2)`, i.e. the group of continuous planar rotations and reflections.
    Since the group has infinitely many irreducible representations, it is not possible to build all of them.
    Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
    the maximum frequency of the irreps to build.
    New irreps (associated to higher frequencies) can be manually created by calling the method
    :meth:`e2cnn.group.O2.irrep` (see the method's documentation).
    
    You should use this factory function to build an instance of :class:`e2cnn.group.O2`.
    Only one instance is built and, in case of multiple calls to this function, the same instance is returned.
    In case of multiple calls of this function with different parameters or in case new representations are built
    (eg. through the method :meth:`e2cnn.group.O2.irrep`), this unique instance is updated with the new representations
    and, therefore, all its references will see the new representations.
    
    Args:
        maximum_frequency (int): maximum frequency of the irreps

    Returns:
        the group :math:`O(2)`

    """
    return O2._generator(maximum_frequency)


