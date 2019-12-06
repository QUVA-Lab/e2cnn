

from collections import defaultdict

import torch

from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from .equivariant_module import EquivariantModule

from .reshuffle_module import ReshuffleModule

from e2cnn.gspaces import *

from typing import List, Tuple, Any, Dict

import numpy as np

__all__ = ["BranchingModule"]


class BranchingModule(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 labels: List[str],
                 reshuffle: int = 0
                 ):
    
        r"""
        
        Splits the input tensor in multiple branches identified by the input ``labels``.
        A label is associated to each field in the input class.
        During forward, fields are grouped by the labels and the input tensor is split accordingly,
        returning a dictionary mapping labels to tensors.
        
        If ``reshuffle`` is set to a positive integer, this module first builds a copy of the input tensor sorting the
        fields according to the value set:
        
        - 1: fields are sorted by their labels
        
        - 2: fields are sorted by their labels and, then, by their size
        
        - 3: fields are sorted by their labels, by their size and, then, by their type
        
        In this way, fields that need to be retrieved together are contiguous and it is possible to exploit slicing
        to split the tensor.
        By default, ``reshuffle = 0`` which means that no sorting is performed and, so, if input
        fields are not contiguous this layer will use indexing to retrieve sub-tensors.
        
        .. todo ::
            Technically this is not an EquivariantModule as the output is not a single tensor.
            Either fix EquivariantModule to support multiple inputs and outputs or set this as just subclass of
            torch.nn.Module.
        
        Args:
            in_type (FieldType): the input class
            labels (list): the list of labels to group the fields
            reshuffle (int, optional): set how to reshuffle the input fields before splitting the tensor.
                                       By default (``0``) no reshuffling is done.
            
        """
        
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        assert 0 <= reshuffle < 3
        
        super(BranchingModule, self).__init__()
        
        self.space = in_type.gspace
        self.in_type = in_type
        self.reshuffle_layer = None
        
        self._labels = set(labels)
        
        total_fields = len(in_type.representations)
        
        assert total_fields == len(labels), \
            'Error! Number of labels ({}) does not match number of fields ({})'.format(len(labels), total_fields)
        
        # If the user required to sort the input representation build
        # a ReshuffleLayer to apply at the beginning of the forward
        if reshuffle > 0:
            
            # fields are sorted, in order of priority, by the non-linearity applied, their size and their name
            # according to the reshuffle set
            keys = []
            c = 0
            for l in labels:
                # build an array containing the sorting keys and the fields' positions
                keys.append((l, in_type.representations[c].size, in_type.representations[c].name, c))
                c += 1
            
            # sort the keys list to build the fields permutation
            keys = sorted(keys, key=lambda x: x[:reshuffle])
            permutation = [k[3] for k in keys]
            
            # if the fields were already sorted, it is useless to add the ReshuffleLayer
            if permutation != list(range(len(in_type.representations))):
                # build the ReshuffleLayer
                self.reshuffle_layer = ReshuffleModule(self.in_type, permutation)
                
                # add the reshuffle layer to the sub-modules
                self.add_module('reshuffle', self.reshuffle_layer)
                
                # set the input representation to consider for the non-linearities to the sorted one
                # (i.e. the output of the ReshuffleLayer)
                in_type = self.reshuffle_layer.out_type
                
                # permute the non-linearities list accordingly
                labels = [labels[p] for p in permutation]

        # for each label, build the representation of the sub-fiber on which it acts
        self.out_type = in_type.group_by_labels(labels)
        
        # check which non-linearity has all its fields consecutive
        self._contiguous = {}
        
        last_label = None
        for l in labels:
            if l != last_label:
                if not l in self._contiguous:
                    self._contiguous[l] = True
                else:
                    self._contiguous[l] = False
                
            last_label = l
        
        _input_indices = defaultdict(lambda: [])
        
        fields = defaultdict(lambda: [])
        
        # for each label, compute:
        #   - the set of indices on the fiber of its fields and
        #   - the the indices of the fields belonging to it
        c = 0
        last_position = 0
        for l in labels:
            # append the indices of the current field
            _input_indices[l] += list(range(last_position, last_position + in_type.representations[c].size))
            
            # append the index of the current field to the list of fields belonging to this label
            fields[l].append(c)
            
            # move on the fiber
            last_position += in_type.representations[c].size
            
            # move to the next field
            c += 1
        
        for l, contiguous in self._contiguous.items():
            if contiguous:
                # for labels with contiguous fields, only the first and the last indices are preserved
                _input_indices[l] = torch.LongTensor([min(_input_indices[l]), max(_input_indices[l])+1])
                
            else:
                # for the others, the indices list is trasformed into a PyTorch's Tensor
                _input_indices[l] = torch.LongTensor(_input_indices[l])

            # register the indices tensors as parameters of this module
            self.register_buffer('indices_{}'.format(l), _input_indices[l])
        
    def _retrieve_subfiber(self, input: GeometricTensor, l: str) -> GeometricTensor:
        r"""
        
        Return a new GeometricTensor containg the portion of memory of the input tensor corresponding to the fields
        the input non-linearity acts on. The method automatically deals with the continuity of these fields, using
        either indexing or slicing.
        
        The resulting tensor is returned wrapped in a GeometricTensor with the proper representation
        
        Args:
            input (GeometricTensor): the input tensor
            l (str): the label to consider

        Returns:
            (GeometricTensor): the sub-tensor containing the fields belonging to the input label
            
        """
        indices = getattr(self, f"indices_{l}")
        
        if self._contiguous[l]:
            # if the fields are contiguous, use slicing
            data = input.tensor[:, indices[0]:indices[1], ...]
        else:
            # otherwise, use indexing
            data = input.tensor[:, indices, ...]
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(data, self.out_type[l])
        
    def forward(self, input: GeometricTensor) -> Dict[str, GeometricTensor]:
        r"""
        
        Group input fields by their labels
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            a dictionary mapping each label to a portion of the input tensor containing the
            fields assigned to that label
            
        """
        
        assert input.type == self.in_type
        
        # if reshuffling is required, apply the ReshuffleLayer
        if self.reshuffle_layer is not None:
            input = self.reshuffle_layer(input)
        
        output = {}
        
        # iterate through the labels
        for l in self._labels:
            # retrieve the corresponding sub-tensor
            output[l] = self._retrieve_subfiber(input, l)
            
        return output

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Dict[Any, Tuple[int, ...]]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
    
        return {l: (b, repr.size, hi, wi) for l, repr in self.out_type.items()}

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
    
        x = torch.randn(3, c, 1, 2)
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.space.testing_elements:
            
            outs2 = self(x.transform_fibers(el))
            
            outs1 = self(x)
            
            for l in self._labels:
                out1 = outs1[l].transform_fibers(el)
                out2 = outs2[l]
            
                errs = (out1.tensor - out2.tensor).detach().numpy()
                errs = np.abs(errs).reshape(-1)
                print(el, errs.max(), errs.mean(), errs.var())
            
                assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                        .format(el, errs.max(), errs.mean(), errs.var())
            
                errors.append((el, errs.mean()))
    
        return errors