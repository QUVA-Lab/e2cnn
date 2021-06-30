
from e2cnn.kernels import KernelBasis, EmptyBasisException
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from .. import utils

from .basisexpansion import BasisExpansion
from .basisexpansion_singleblock import block_basisexpansion

from collections import defaultdict

from typing import Callable, List, Iterable, Dict, Union

import torch
import numpy as np


__all__ = ["BlocksBasisExpansion"]


class BlocksBasisExpansion(BasisExpansion):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 points: np.ndarray,
                 sigma: List[float],
                 rings: List[float],
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 **kwargs
                 ):
        r"""
        
        With this algorithm, the expansion is done on the intertwiners of the fields' representations pairs in input and
        output.
        
        Args:
            in_type (FieldType): the input field type
            out_type (FieldType): the output field type
            points (~numpy.ndarray): points where the analytical basis should be sampled
            sigma (list): width of each ring where the bases are sampled
            rings (list): radii of the rings where to sample the bases
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.
            **kwargs: keyword arguments specific to the groups and basis used
        
        Attributes:
            S (int): number of points where the filters are sampled
            
        """

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(BlocksBasisExpansion, self).__init__()
        self._in_type = in_type
        self._out_type = out_type
        self._input_size = in_type.size
        self._output_size = out_type.size
        self.points = points
        
        # int: number of points where the filters are sampled
        self.S = self.points.shape[1]

        space = in_type.gspace

        # we group the basis vectors by their input and output representations
        _block_expansion_modules = {}
        
        # iterate through all different pairs of input/output representationions
        # and, for each of them, build a basis
        for i_repr in in_type._unique_representations:
            for o_repr in out_type._unique_representations:
                reprs_names = (i_repr.name, o_repr.name)
                try:
                    
                    basis = space.build_kernel_basis(i_repr, o_repr,
                                                     sigma=sigma,
                                                     rings=rings,
                                                     **kwargs)
                    
                    block_expansion = block_basisexpansion(basis, points, basis_filter, recompute=recompute)
                    _block_expansion_modules[reprs_names] = block_expansion
                    
                    # register the block expansion as a submodule
                    self.add_module(f"block_expansion_{reprs_names}", block_expansion)
                    
                except EmptyBasisException:
                    # print(f"Empty basis at {reprs_names}")
                    pass
        
        if len(_block_expansion_modules) == 0:
            print('WARNING! The basis for the block expansion of the filter is empty!')

        self._n_pairs = len(in_type._unique_representations) * len(out_type._unique_representations)

        # the list of all pairs of input/output representations which don't have an empty basis
        self._representations_pairs = sorted(list(_block_expansion_modules.keys()))
        
        # retrieve for each representation in both input and output fields:
        # - the number of its occurrences,
        # - the indices where it occurs and
        # - whether its occurrences are contiguous or not
        self._in_count, _in_indices, _in_contiguous = _retrieve_indices(in_type)
        self._out_count, _out_indices, _out_contiguous = _retrieve_indices(out_type)
        
        # compute the attributes and an id for each basis element (and, so, of each parameter)
        # attributes, basis_ids = _compute_attrs_and_ids(in_type, out_type, _block_expansion_modules)
        basis_ids = _compute_attrs_and_ids(in_type, out_type, _block_expansion_modules)
        
        self._weights_ranges = {}

        last_weight_position = 0

        self._ids_to_basis = {}
        self._basis_to_ids = []
        
        self._contiguous = {}
        
        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        for io_pair in self._representations_pairs:
    
            self._contiguous[io_pair] = _in_contiguous[io_pair[0]] and _out_contiguous[io_pair[1]]
    
            # build the indices tensors
            if self._contiguous[io_pair]:
                # in_indices = torch.LongTensor([
                in_indices = [
                    _in_indices[io_pair[0]].min(),
                    _in_indices[io_pair[0]].max() + 1,
                    _in_indices[io_pair[0]].max() + 1 - _in_indices[io_pair[0]].min()
                ]# )
                # out_indices = torch.LongTensor([
                out_indices = [
                    _out_indices[io_pair[1]].min(),
                    _out_indices[io_pair[1]].max() + 1,
                    _out_indices[io_pair[1]].max() + 1 - _out_indices[io_pair[1]].min()
                ] #)
                
                setattr(self, 'in_indices_{}'.format(io_pair), in_indices)
                setattr(self, 'out_indices_{}'.format(io_pair), out_indices)

            else:
                out_indices, in_indices = torch.meshgrid([_out_indices[io_pair[1]], _in_indices[io_pair[0]]])
                in_indices = in_indices.reshape(-1)
                out_indices = out_indices.reshape(-1)
                
                # register the indices tensors and the bases tensors as parameters of this module
                self.register_buffer('in_indices_{}'.format(io_pair), in_indices)
                self.register_buffer('out_indices_{}'.format(io_pair), out_indices)
                
            # count the actual number of parameters
            total_weights = len(basis_ids[io_pair])

            for i, id in enumerate(basis_ids[io_pair]):
                self._ids_to_basis[id] = last_weight_position + i
            
            self._basis_to_ids += basis_ids[io_pair]
            
            # evaluate the indices in the global weights tensor to use for the basis belonging to this group
            self._weights_ranges[io_pair] = (last_weight_position, last_weight_position + total_weights)
    
            # increment the position counter
            last_weight_position += total_weights
            
    def get_basis_names(self) -> List[str]:
        return self._basis_to_ids
    
    def get_element_info(self, name: Union[str, int]) -> Dict:
        if isinstance(name, str):
            idx = self._ids_to_basis[name]
        else:
            idx = name
        
        reprs_names = None
        relative_idx = None
        for pair, idx_range in self._weights_ranges.items():
            if idx_range[0] <= idx < idx_range[1]:
                reprs_names = pair
                relative_idx = idx - idx_range[0]
                break
        assert reprs_names is not None and relative_idx is not None
        
        block_expansion = getattr(self, f"block_expansion_{reprs_names}")
        block_idx = relative_idx // block_expansion.dimension()
        relative_idx = relative_idx % block_expansion.dimension()
        
        attr = block_expansion.get_element_info(relative_idx).copy()
        
        block_count = 0
        out_irreps_count = 0
        for o, o_repr in enumerate(self._out_type.representations):
            in_irreps_count = 0
            for i, i_repr in enumerate(self._in_type.representations):
            
                if reprs_names == (i_repr.name, o_repr.name):
                    
                    if block_count == block_idx:

                        # retrieve the attributes of each basis element and build a new list of
                        # attributes adding information specific to the current block
                        attr.update({
                            "in_irreps_position": in_irreps_count + attr["in_irrep_idx"],
                            "out_irreps_position": out_irreps_count + attr["out_irrep_idx"],
                            "in_repr": reprs_names[0],
                            "out_repr": reprs_names[1],
                            "in_field_position": i,
                            "out_field_position": o,
                        })
                    
                        # build the ids of the basis vectors
                        # add names and indices of the input and output fields
                        id = '({}-{},{}-{})'.format(i_repr.name, i, o_repr.name, o)
                        # add the original id in the block submodule
                        id += "_" + attr["id"]
                    
                        # update with the new id
                        attr["id"] = id
                        
                        attr["idx"] = idx
                        
                        return attr
                        
                    block_count += 1
                
                in_irreps_count += len(i_repr.irreps)
            out_irreps_count += len(o_repr.irreps)
        
        raise ValueError(f"Parameter with index {idx} not found!")

    def get_basis_info(self) -> Iterable:
        
        out_irreps_counts = [0]
        out_block_counts = defaultdict(list)
        for o, o_repr in enumerate(self._out_type.representations):
            out_irreps_counts.append(out_irreps_counts[-1] + len(o_repr.irreps))
            out_block_counts[o_repr.name].append(o)
            
        in_irreps_counts = [0]
        in_block_counts = defaultdict(list)
        for i, i_repr in enumerate(self._in_type.representations):
            in_irreps_counts.append(in_irreps_counts[-1] + len(i_repr.irreps))
            in_block_counts[i_repr.name].append(i)

        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        idx = 0
        for reprs_names in self._representations_pairs:

            block_expansion = getattr(self, f"block_expansion_{reprs_names}")
            
            for o in out_block_counts[reprs_names[1]]:
                out_irreps_count = out_irreps_counts[o]
                for i in in_block_counts[reprs_names[0]]:
                    in_irreps_count = in_irreps_counts[i]
    
                    # retrieve the attributes of each basis element and build a new list of
                    # attributes adding information specific to the current block
                    for attr in block_expansion.get_basis_info():
                        attr = attr.copy()
                        attr.update({
                            "in_irreps_position": in_irreps_count + attr["in_irrep_idx"],
                            "out_irreps_position": out_irreps_count + attr["out_irrep_idx"],
                            "in_repr": reprs_names[0],
                            "out_repr": reprs_names[1],
                            "in_field_position": i,
                            "out_field_position": o,
                        })
            
                        # build the ids of the basis vectors
                        # add names and indices of the input and output fields
                        id = '({}-{},{}-{})'.format(reprs_names[0], i, reprs_names[1], o)
                        # add the original id in the block submodule
                        id += "_" + attr["id"]
                
                        # update with the new id
                        attr["id"] = id
                        
                        attr["idx"] = idx
                        idx += 1
                
                        yield attr

    def dimension(self) -> int:
        return len(self._ids_to_basis)

    def _expand_block(self, weights, io_pair):
        # retrieve the basis
        block_expansion = getattr(self, f"block_expansion_{io_pair}")

        # retrieve the linear coefficients for the basis expansion
        coefficients = weights[self._weights_ranges[io_pair][0]:self._weights_ranges[io_pair][1]]
    
        # reshape coefficients for the batch matrix multiplication
        coefficients = coefficients.view(-1, block_expansion.dimension())
        
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        _filter = block_expansion(coefficients)
        k, o, i, p = _filter.shape
        
        _filter = _filter.view(
            self._out_count[io_pair[1]],
            self._in_count[io_pair[0]],
            o,
            i,
            self.S,
        )
        _filter = _filter.transpose(1, 2)
        return _filter
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the Module which expands the basis and returns the filter built

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters

        Returns:
            the filter built

        """
        assert weights.shape[0] == self.dimension()
        assert len(weights.shape) == 1
    
        if self._n_pairs == 1:
            # if there is only one block (i.e. one type of input field and one type of output field),
            #  we can return the expanded block immediately, instead of copying it inside a preallocated empty tensor
            io_pair = self._representations_pairs[0]
            in_indices = getattr(self, f"in_indices_{io_pair}")
            out_indices = getattr(self, f"out_indices_{io_pair}")
            _filter = self._expand_block(weights, io_pair).reshape(out_indices[2], in_indices[2], self.S)
            
        else:
        
            # build the tensor which will contain te filter
            _filter = torch.zeros(self._output_size, self._input_size, self.S, device=weights.device)

            # iterate through all input-output field representations pairs
            for io_pair in self._representations_pairs:
                
                # retrieve the indices
                in_indices = getattr(self, f"in_indices_{io_pair}")
                out_indices = getattr(self, f"out_indices_{io_pair}")
                
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter
                expanded = self._expand_block(weights, io_pair)
                
                if self._contiguous[io_pair]:
                    _filter[
                        out_indices[0]:out_indices[1],
                        in_indices[0]:in_indices[1],
                        :,
                    ] = expanded.reshape(out_indices[2], in_indices[2], self.S)
                else:
                    _filter[
                        out_indices,
                        in_indices,
                        :,
                    ] = expanded.reshape(-1, self.S)

        # return the new filter
        return _filter

    def __hash__(self):
    
        _hash = 0
        for io in self._representations_pairs:
            n_pairs = self._in_count[io[0]] * self._out_count[io[1]]
            _hash += hash(getattr(self, f"block_expansion_{io}")) * n_pairs
    
        return _hash

    def __eq__(self, other):
        if not isinstance(other, BlocksBasisExpansion):
            return False
    
        if self.dimension() != other.dimension():
            return False
    
        if self._representations_pairs != other._representations_pairs:
            return False
    
        for io in self._representations_pairs:
            if self._contiguous[io] != other._contiguous[io]:
                return False
        
            if self._weights_ranges[io] != other._weights_ranges[io]:
                return False
        
            if self._contiguous[io]:
                if getattr(self, f"in_indices_{io}") != getattr(other, f"in_indices_{io}"):
                    return False
                if getattr(self, f"out_indices_{io}") != getattr(other, f"out_indices_{io}"):
                    return False
            else:
                if torch.any(getattr(self, f"in_indices_{io}") != getattr(other, f"in_indices_{io}")):
                    return False
                if torch.any(getattr(self, f"out_indices_{io}") != getattr(other, f"out_indices_{io}")):
                    return False
        
            if getattr(self, f"block_expansion_{io}") != getattr(other, f"block_expansion_{io}"):
                return False
    
        return True


def _retrieve_indices(type: FieldType):
    fiber_position = 0
    _indices = defaultdict(list)
    _count = defaultdict(int)
    _contiguous = {}
    
    for repr in type.representations:
        _indices[repr.name] += list(range(fiber_position, fiber_position + repr.size))
        fiber_position += repr.size
        _count[repr.name] += 1
    
    for name, indices in _indices.items():
        # _contiguous[o_name] = indices == list(range(indices[0], indices[0]+len(indices)))
        _contiguous[name] = utils.check_consecutive_numbers(indices)
        _indices[name] = torch.LongTensor(indices)
    
    return _count, _indices, _contiguous


def _compute_attrs_and_ids(in_type, out_type, block_submodules):
    
    basis_ids = defaultdict(lambda: [])
    
    # iterate over all blocks
    # each block is associated to an input/output representations pair
    out_fiber_position = 0
    out_irreps_count = 0
    for o, o_repr in enumerate(out_type.representations):
        in_fiber_position = 0
        in_irreps_count = 0
        for i, i_repr in enumerate(in_type.representations):
            
            reprs_names = (i_repr.name, o_repr.name)
            
            # if a basis for the space of kernels between the current pair of representations exists
            if reprs_names in block_submodules:
                
                # retrieve the attributes of each basis element and build a new list of
                # attributes adding information specific to the current block
                ids = []
                for attr in block_submodules[reprs_names].get_basis_info():
                    # build the ids of the basis vectors
                    # add names and indices of the input and output fields
                    id = '({}-{},{}-{})'.format(i_repr.name, i, o_repr.name, o)
                    # add the original id in the block submodule
                    id += "_" + attr["id"]
                    
                    ids.append(id)

                # append the ids of the basis vectors
                basis_ids[reprs_names] += ids
            
            in_fiber_position += i_repr.size
            in_irreps_count += len(i_repr.irreps)
        out_fiber_position += o_repr.size
        out_irreps_count += len(o_repr.irreps)
        
    # return attributes, basis_ids
    return basis_ids
