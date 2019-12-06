from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from e2cnn.gspaces import *

from .equivariant_module import EquivariantModule
from .branching_module import BranchingModule
from .merge_module import MergeModule

from typing import List, Tuple, Union, Any

import torch

import numpy as np

__all__ = ["MultipleModule"]


class MultipleModule(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 labels: List[str],
                 modules: List[Tuple[EquivariantModule, Union[str, List[str]]]],
                 reshuffle: int = 0
                 ):
        r"""
        
        Split the input tensor in multiple branches identified by the input ``labels`` and apply to each of them the
        corresponding module in ``modules``
        
        A label is associated to each field in the input type, while ``modules`` assigns a module to apply to
        each label (or set of labels).
        ``modules`` should be a list of pairs, each containing an :class:`~e2cnn.nn.EquivariantModule` and a label (or a
        list of labels).
        
        During forward, fields are grouped by the labels and the input tensor is split accordingly.
        Then, each subtensor is passed to the corresponding module in ``modules``.
        
        If ``reshuffle`` is set to a positive integer, a copy of the input tensor is first built sorting the
        fields according to the value set:
        
        - 1: fields are sorted by their labels
        
        - 2: fields are sorted by their labels and, then, by their size
        
        - 3: fields are sorted by their labels, by their size and, then, by their type
        
        In this way, fields that need to be retrieved together are contiguous and it is possible to exploit slicing
        to split the tensor.
        By default, ``reshuffle = 0`` which means that no sorting is performed and, so, if input
        fields are not contiguous this layer will use indexing to retrieve sub-tensors.
        
        This modules wraps a :class:`~e2cnn.nn.BranchingModule` followed by a :class:`~e2cnn.nn.MergeModule`.
        
        Args:
            in_type (FieldType): the input field type
            labels (list): the list of labels to group the fields
            modules (list): list of modules to apply to the labeled fields
            reshuffle (int, optional): set how to reshuffle the input fields before splitting the tensor.
                                       By default (``0``) no reshuffling is done
            
        """
        
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(MultipleModule, self).__init__()
        
        self.gspace = in_type.gspace
        
        self.in_type = in_type
        
        all_labels = set(labels)
        
        modules_labels = []
        
        for _, l in modules:
            if isinstance(l, list):
                modules_labels += l
            else:
                modules_labels.append(l)
        
        modules_labels = set(modules_labels)
        
        assert (modules_labels in all_labels) or (modules_labels == all_labels), "Error! Some labels assigned to the modules don't appear among the channels labels"
        
        # print(labels)
        
        reshuffle_level = int(reshuffle)
        self.branching = BranchingModule(in_type, labels, reshuffle=reshuffle_level)

        for module, l in modules:
            if isinstance(l, str):
                assert module.in_type == self.branching.out_type[l], f"Label {l}, branch class and module ({module}) class don't match:\n [{module.in_type}] \n [{self.branching.out_type[l]}]\n"
            else:
                for i, lb in enumerate(l):
                    assert module.in_type[i] == self.branching.out_type[lb], f"Label {lb}, branch class and module ({module}) class [{i}] don't match: \n [{module.in_type[i]}] \n [{self.branching.out_type[lb]}]\n"
        
        self.merging = MergeModule(modules)
        
        self.out_type = self.merging.out_type
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Split the input tensor according to the labels, apply each module to the corresponding input sub-tensors and
        stack the results.
        
        Args:
            input (GeometricTensor): the input GeometricTensor

        Returns:
            the concatenation of the output of each module
            
        """
        
        assert input.type == self.in_type
        
        sub_tensors = self.branching(input)
        
        return self.merging(sub_tensors)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        branches_shapes = self.branching.evaluate_output_shape(input_shape)
        
        out_shape = self.merging.evaluate_output_shape(branches_shapes)
        
        return out_shape

    def check_equivariance(self, atol: float = 2e-6, rtol: float = 1e-5, full_space_action: bool = True) -> List[Tuple[Any, float]]:
        
        if full_space_action:
            
            return super(MultipleModule, self).check_equivariance(atol=atol, rtol=rtol)
        
        else:
            c = self.in_type.size
        
            x = torch.randn(10, c, 9, 9)
            print(c, self.out_type.size)
            print([r.name for r in self.in_type.representations])
            print([r.name for r in self.out_type.representations])
            x = GeometricTensor(x, self.in_type)
        
            errors = []
        
            for el in self.gspace.testing_elements:
                out1 = self(x).transform_fibers(el)
                out2 = self(x.transform_fibers(el))
            
                errs = (out1.tensor - out2.tensor).detach().numpy()
                errs = np.abs(errs).reshape(-1)
                print(el, errs.max(), errs.mean(), errs.var())
                
                if not torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol):
                    tmp = np.abs((out1.tensor - out2.tensor).detach().numpy())
                    tmp = tmp.reshape(out1.tensor.shape[0], out1.tensor.shape[1], -1).max(axis=2)#.mean(axis=0)
                    
                    np.set_printoptions(precision=2, threshold=200000000, suppress=True, linewidth=500)
                    print(tmp.shape)
                    print(tmp)
            
                assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                        .format(el, errs.max(), errs.mean(), errs.var())
            
                errors.append((el, errs.mean()))
        
            return errors