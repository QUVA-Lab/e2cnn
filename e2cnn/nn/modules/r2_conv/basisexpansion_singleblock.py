
from e2cnn.kernels import KernelBasis, EmptyBasisException
from .basisexpansion import BasisExpansion

from typing import Callable, Dict, List, Iterable, Union

import torch
import numpy as np

__all__ = ["SingleBlockBasisExpansion", "block_basisexpansion"]


class SingleBlockBasisExpansion(BasisExpansion):
    
    def __init__(self,
                 basis: KernelBasis,
                 points: np.ndarray,
                 basis_filter: Callable[[dict], bool] = None,
                 ):
        r"""
        
        Basis expansion method for a single contiguous block, i.e. for kernels whose input type and output type contain
        only fields of one type.
        
        Args:
            basis (KernelBasis): analytical basis to sample
            points (ndarray): points where the analytical basis should be sampled
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            
        """

        super(SingleBlockBasisExpansion, self).__init__()
        
        self.basis = basis
        
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
            
        if not any(mask):
            raise EmptyBasisException

        attributes = [attr for b, attr in enumerate(basis) if mask[b]]
        
        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in attributes:
            sizes.append(attr["shape"][0])

        # sample the basis on the grid
        sampled_basis = torch.Tensor(basis.sample(points)).permute(2, 0, 1, 3)

        # DEPRECATED FROM PyTorch 1.2
        # PyTorch 1.2 suggests using BoolTensor instead of ByteTensor for boolean indexing
        # but BoolTensor have been introduced only in PyTorch 1.2
        # Hence, for the moment we use ByteTensor
        mask = torch.tensor(mask.astype(np.uint8))

        # filter out the basis elements discarded by the filter
        sampled_basis = sampled_basis[mask, ...]
        
        # normalize the basis
        sizes = torch.tensor(sizes, dtype=sampled_basis.dtype)
        sampled_basis = normalize_basis(sampled_basis, sizes)

        # discard the basis which are close to zero everywhere
        norms = (sampled_basis ** 2).reshape(sampled_basis.shape[0], -1).sum(1) > 1e-2
        if not any(norms):
            raise EmptyBasisException
        sampled_basis = sampled_basis[norms, ...]
        self._mask = mask

        self.attributes = [attr for b, attr in enumerate(attributes) if norms[b]]
        
        # register the bases tensors as parameters of this module
        self.register_buffer('sampled_basis', sampled_basis)
            
        self._idx_to_ids = []
        self._ids_to_idx = {}
        for idx, attr in enumerate(self.attributes):
            id = '({}-{},{}-{})_({}/{})_{}'.format(
                    attr["in_irrep"], attr["in_irrep_idx"],  # name and index within the field of the input irrep
                    attr["out_irrep"], attr["out_irrep_idx"],  # name and index within the field of the output irrep
                    attr["radius"],  # radius of the ring
                    attr["frequency"],  # frequency of the basis element
                    # int(np.abs(attr["frequency"])),  # absolute frequency of the basis element
                    attr["inner_idx"],
                    # index of the basis element within the basis of radially independent kernels between the irreps
                )
            attr["id"] = id
            self._ids_to_idx[id] = idx
            self._idx_to_ids.append(id)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
    
        assert len(weights.shape) == 2 and weights.shape[1] == self.dimension()
    
        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        return torch.einsum('boi...,kb->koi...', self.sampled_basis, weights) #.transpose(1, 2).contiguous()

    def get_basis_names(self) -> List[str]:
        return self._idx_to_ids

    def get_element_info(self, name: Union[str, int]) -> Dict:
        if isinstance(name, str):
            name = self._ids_to_idx[name]
        return self.attributes[name]

    def get_basis_info(self) -> Iterable:
        return iter(self.attributes)

    def dimension(self) -> int:
        return self.sampled_basis.shape[0]

    def __eq__(self, other):
        if isinstance(other, SingleBlockBasisExpansion):
            return (
                    self.basis == other.basis and
                    torch.allclose(self.sampled_basis, other.sampled_basis) and
                    (self._mask == other._mask).all()
            )
        else:
            return False

    def __hash__(self):
        return 10000 * hash(self.basis) + 100 * hash(self.sampled_basis) + hash(self._mask)


# dictionary storing references to already built basis tensors
# when a new filter tensor is built, it is also stored here
# when the same basis is built again (eg. in another layer), the already existing filter tensor is retrieved
_stored_filters = {}


def block_basisexpansion(basis: KernelBasis,
                         points: np.ndarray,
                         basis_filter: Callable[[dict], bool] = None,
                         recompute: bool = False
                         ) -> SingleBlockBasisExpansion:
    r"""


    Args:
        basis (KernelBasis): basis defining the space of kernels
        points (ndarray): points where the analytical basis should be sampled
        basis_filter (callable):
        recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

    """
    
    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
        
        key = (basis, mask.tobytes(), points.tobytes())
        if key not in _stored_filters:
            _stored_filters[key] = SingleBlockBasisExpansion(basis, points, basis_filter)
        
        return _stored_filters[key]
    
    else:
        return SingleBlockBasisExpansion(basis, points, basis_filter)


def normalize_basis(basis: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    r"""

    Normalize the filters in the input tensor.
    The tensor of shape :math:`(B, O, I, ...)` is interpreted as a basis containing ``B`` filters/elements, each with
    ``I`` inputs and ``O`` outputs. The spatial dimensions ``...`` can be anything.

    .. notice ::
        Notice that the method changes the input tensor inplace

    Args:
        basis (torch.Tensor): tensor containing the basis to normalize
        sizes (torch.Tensor): original input size of the basis elements, without the padding and the change of basis

    Returns:
        the normalized basis (the operation is done inplace, so this is ust a reference to the input tensor)

    """
    
    b = basis.shape[0]
    assert len(basis.shape) > 2
    assert sizes.shape == (b,)
    
    # compute the norm of each basis vector
    norms = torch.einsum('bop...,bpq...->boq...', (basis, basis.transpose(1, 2)))
    
    # Removing the change of basis, these matrices should be multiples of the identity
    # where the scalar on the diagonal is the variance
    # in order to find this variance, we can compute the trace (which is invariant to the change of basis)
    # and divide by the number of elements in the diagonal ignoring the padding.
    # Therefore, we need to know the original size of each basis element.
    norms = torch.einsum("bii...->b", norms)
    # norms = norms.reshape(b, -1).sum(1)
    norms /= sizes

    norms[norms < 1e-15] = 0
    
    norms = torch.sqrt(norms)
    
    norms[norms < 1e-6] = 1
    norms[norms != norms] = 1
    
    norms = norms.view(b, *([1] * (len(basis.shape) - 1)))
    
    # divide by the norm
    basis /= norms

    return basis



