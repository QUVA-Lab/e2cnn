
import torch

from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from .equivariant_module import EquivariantModule


from typing import List, Tuple, Any, Union, Dict

__all__ = ["MergeModule"]


class MergeModule(EquivariantModule):
    
    def __init__(self,
                 modules: List[Tuple[EquivariantModule, Union[str, List[str]]]],
                 ):
        r"""
        
        Applies different modules to multiple tensors in input.
        
        ``modules`` contains a list of pairs, each containing an :class:`~e2cnn.nn.EquivariantModule` and a label (or a
        list of labels).
        
        This module takes as input a dictionary mapping labels to tensors.
        Then, each module in ``modules`` is applied to the tensors associated to its labels.
        Finally, output tensors are stacked together.
        
        .. todo ::
            Technically this is not an EquivariantModule as the input is not a single tensor.
            Either fix EquivariantModule to support multiple inputs and outputs or set this as just subclass of
            torch.nn.Module.
        
        Args:
            modules (list): list of modules to apply to the labeled input tensors
            
        """

        super(MergeModule, self).__init__()
        
        labels = []
        
        for i in range(len(modules)):
            if isinstance(modules[i][1], str):
                modules[i] = (modules[i][0], [modules[i][1]])
            else:
                assert isinstance(modules[i][1], list)
                for s in modules[i][1]:
                    assert isinstance(s, str)

            labels += modules[i][1]
        
        self.in_type = None
        self.gspace = modules[0][0].in_type.gspace
        
        out_repr = []
        
        for module, labels in modules:
            if isinstance(module.in_type, tuple):
                assert all(t.gspace == self.gspace for t in module.in_type)
            else:
                assert module.in_type.gspace == self.gspace
                
            out_repr += module.out_type.representations
        
        self.out_type = FieldType(self.gspace, out_repr)
        
        self._labels = []
        # add the input modules as sub-modules
        for i, (module, labels) in enumerate(modules):
            self._labels.append(labels)
            self.add_module('submodule_{}'.format(i), module)
        
    def forward(self, input: Dict[str, GeometricTensor]) -> GeometricTensor:
        r"""
        
        Apply each module to the corresponding input tensors and stack the results
        
        Args:
            input (dict): a dictionary mapping each label to a GeometricTensor

        Returns:
            the concatenation of the output of each module
            
        """
        
        # compute the output shape
        out_shape = self.evaluate_output_shape(**{l: t.tensor.shape for l, t in input.items()})
        
        device = list(input.values())[0].tensor.device
        
        # pre-allocate the output tensor
        output = torch.empty(out_shape, dtype=torch.float, device=device)
        
        last_channel = 0
        # iterate through the modules
        for i, labels in enumerate(self._labels):
            module = getattr(self, f"submodule_{i}")
            # retrieve the corresponding sub-tensor
            output[:, last_channel:last_channel + module.out_type.size, ...] = module(*[input[l] for l in labels]).tensor
            last_channel += module.out_type.size
        
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, **input_shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    
        output_shapes = []
    
        # iterate through the modules
        for i, labels in enumerate(self._labels):
            module = getattr(self, f"submodule_{i}")
            # evaluate the corresponding output shape
            # output_shapes.append(module.evaluate_output_shape(*[input_shapes[l] for l in labels]))
            os = module.evaluate_output_shape(*[input_shapes[l] for l in labels])
            output_shapes.append(list(os))
    
        out_shape = list(output_shapes[0])
    
        for os in output_shapes[1:]:
            assert out_shape[0] == os[0]
            assert out_shape[2:] == os[2:]
            
            out_shape[1] += os[1]
        
        return out_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        pass