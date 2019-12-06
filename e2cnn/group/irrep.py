from __future__ import annotations

import e2cnn.group
from e2cnn.group import Representation

from typing import Callable, Any, List, Union, Dict

import numpy as np

__all__ = ["IrreducibleRepresentation"]


class IrreducibleRepresentation(Representation):
    
    def __init__(self,
                 group: e2cnn.group.Group,
                 name: str,
                 representation: Union[Dict[Any, np.ndarray], Callable[[Any], np.ndarray]],
                 size: int,
                 sum_of_squares_constituents: int,
                 supported_nonlinearities: List[str],
                 character: Union[Dict[Any, float], Callable[[Any], float]] = None,
                 **kwargs
                 ):
        """
        Describes an "*irreducible representation*" (*irrep*).
        It is a subclass of a :class:`~e2cnn.group.Representation`.
        
        Irreducible representations are the building blocks into which any other representation decomposes under a
        change of basis.
        Indeed, any :class:`~e2cnn.group.Representation` is internally decomposed into a direct sum of irreps.
        
        Args:
            group (Group): the group which is being represented
            name (str): an identification name for this representation
            representation (dict or callable): a callable implementing this representation or a dictionary
                    mapping each of the group's elements to its representation.
            size (int): the size of the vector space where this representation is defined (i.e. the size of the matrices)
            sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                        irreducible constituents of the character of this representation over a non-splitting field
            supported_nonlinearities (list): list of nonlinearitiy types supported by this representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dictionary mapping each element to its character.
            **kwargs: custom attributes the user can set and, then, access from the dictionary
                    in :attr:`e2cnn.group.Representation.attributes`
        
        Attributes:
            sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                    irreducible constituents of the character of this representation over a non-splitting field (see
                    `Character Orthogonality Theorem <https://groupprops.subwiki.org/wiki/Character_orthogonality_theorem#Statement_over_general_fields_in_terms_of_inner_product_of_class_functions>`_
                    over general fields)
            
        """
        
        super(IrreducibleRepresentation, self).__init__(group,
                                                        name,
                                                        [name],
                                                        np.eye(size),
                                                        supported_nonlinearities,
                                                        representation=representation,
                                                        character=character,
                                                        **kwargs)

        self.irreducible = True
        self.sum_of_squares_constituents = sum_of_squares_constituents
