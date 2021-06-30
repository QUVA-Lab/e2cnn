
from abc import ABC, abstractmethod

from torch.nn import Module
import torch
import numpy as np

from typing import List, Iterable, Dict, Union


__all__ = ["BasisExpansion"]


class BasisExpansion(ABC, Module):
    
    def __init__(self):
        r"""
        Abstract class defining the interface for the different basis expansion algorithms.
       
        """
        super(BasisExpansion, self).__init__()
        
    @abstractmethod
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the module which expands the basis and returns the filter built

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis elements
            
        Returns:
            the filter built
            
        """
        
        pass

    @abstractmethod
    def get_basis_names(self) -> List[str]:
        """
        Method that returns the list of identification names of the basis elements
        
        Returns:
            list of names
        """
        pass

    @abstractmethod
    def get_element_info(self, name: Union[str, int]) -> Dict:
        """
        Method that returns the information associated to a basis element
        
        Parameters:
            name (str or int): identifier of the basis element or its index
        
        Returns:
            dictionary containing the information
        """
        pass
    
    @abstractmethod
    def get_basis_info(self) -> Iterable:
        """
        Method that returns an iterable over all basis elements' attributes.

        Returns:
            an iterable over all the basis elements' attributes
            
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        r"""
        The dimensionality of the basis and, so, the number of weights needed to expand it.
        
        Returns:
            the dimensionality of the basis
            
        """
        pass

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()
