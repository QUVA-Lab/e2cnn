
from .equivariant_module import EquivariantModule

from torch._six import container_abcs

import torch

from typing import List, Iterable


__all__ = ["ModuleList"]


class ModuleList(torch.nn.ModuleList):
    
    def __init__(self,
                 modules: Iterable[EquivariantModule] = None,
                 ):
        r"""
        
        Module similar to :class:`~torch.nn.ModuleList` containing a list of :class:`~e2cnn.nn.EquivariantModule` s.
        
        This class works like :class:`~torch.nn.ModuleList` except for the fact it only accepts instances of
        :class:`~e2cnn.nn.EquivariantModule`.
        
        Additionally, this class provides a `.export()` method.
        This method calls the :meth:`~e2cnn.nn.EquivariantModule.export` method of each module contained in this
        :class:`~e2cnn.nn.ModuleList` and returns a :class:`~torch.nn.ModuleList` containing the exported modules.
        
        
        Args:
            modules (iterable, optional): an iterable of equivariant modules to add
            

        """
        super(ModuleList, self).__init__(modules)
        
    def __setitem__(self, idx: int, module: EquivariantModule):
        assert isinstance(module, EquivariantModule)
        super(ModuleList, self).__setitem__(idx, module)

    def insert(self, index: int, module: EquivariantModule) -> None:
        assert isinstance(module, EquivariantModule)
        super(ModuleList, self).insert(index, module)

    def append(self, module: EquivariantModule) -> 'ModuleList':
        r"""Appends an  :class:`~e2cnn.nn.EquivariantModule` to the end of the list.

        Args:
            module (EquivariantModule): equivariant module to append

        """
        assert isinstance(module, EquivariantModule)
        return super(ModuleList, self).append(module)

    def extend(self, modules: Iterable[EquivariantModule]) -> 'ModuleList':
        r"""Appends multiple :class:`~e2cnn.nn.EquivariantModule` instances from a Python
        iterable to the end of the list.

        Args:
            modules (iterable): iterable of equivariant modules to append
            
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend expects an iterable object, but found " + type(modules).__name__)
        
        for module in modules:
            assert isinstance(module, EquivariantModule)
            self.append(module)
        return self

    def export(self) -> torch.nn.ModuleList:
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ModuleList` module and set to "eval" mode.

        """
    
        self.eval()
    
        submodules = []
        # convert all the submodules
        for module in self:
            module = module.export()
            submodules.append(module)

        m = torch.nn.ModuleList(submodules)
        m.eval()
        return m
    
    def forward(self):
        raise NotImplementedError()

