
from e2cnn.nn import GeometricTensor
from .equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Union, Any

from collections import OrderedDict

__all__ = ["SequentialModule"]


class SequentialModule(EquivariantModule):
    
    def __init__(self,
                 *args: EquivariantModule,
                 ):
        r"""
        
        A sequential container similar to :class:`torch.nn.Sequential`.
        
        The constructor accepts both a list or an ordered dict of :class:`~e2cnn.nn.EquivariantModule` instances.
        
        Example::
        
            # Example of SequentialModule
            s = e2cnn.gspaces.Rot2dOnR2(8)
            c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*16)
            model = e2cnn.nn.SequentialModule(
                      e2cnn.nn.R2Conv(c_in, c_out, 5),
                      e2cnn.nn.InnerBatchNorm(c_out),
                      e2cnn.nn.ReLU(c_out),
            )

            # Example with OrderedDict
            s = e2cnn.gspaces.Rot2dOnR2(8)
            c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*16)
            model = e2cnn.nn.SequentialModule(OrderedDict([
                      ('conv', e2cnn.nn.R2Conv(c_in, c_out, 5)),
                      ('bn', e2cnn.nn.InnerBatchNorm(c_out)),
                      ('relu', e2cnn.nn.ReLU(c_out)),
            ]))
        
        """
        
        super(SequentialModule, self).__init__()

        self.in_type = None
        self.out_type = None
        
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                assert isinstance(module, EquivariantModule)
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                assert isinstance(module, EquivariantModule)
                self.add_module(str(idx), module)
        
        # for i in range(1, len(self._modules.values())):
        #     assert self._modules.values()[i-1].out_type == self._modules.values()[i].in_type
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input GeometricTensor

        Returns:
            the output tensor
            
        """
        
        assert input.type == self.in_type
        x = input
        for m in self._modules.values():
            x = m(x)

        assert x.type == self.out_type
        
        return x
    
    def add_module(self, name: str, module: EquivariantModule):
        r"""
        Append ``module`` to the sequence of modules applied in the forward pass.
        
        """
        
        if len(self._modules) == 0:
            assert self.in_type is None
            assert self.out_type is None
            self.in_type = module.in_type
        else:
            assert module.in_type == self.out_type
            
        self.out_type = module.out_type
        super(SequentialModule, self).add_module(name, module)
    
    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        out_shape = input_shape

        for m in self._modules.values():
            out_shape = m.evaluate_output_shape(out_shape)
        
        return out_shape

    def check_equivariance(self, atol: float = 2e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        return super(SequentialModule, self).check_equivariance(atol=atol, rtol=rtol)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Sequential` module and set to "eval" mode.

        """
    
        self.eval()
    
        submodules = []
        
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if isinstance(module, EquivariantModule):
                module = module.export()
                
            submodules.append(
                (name, module)
            )

        return torch.nn.Sequential(OrderedDict(submodules))
