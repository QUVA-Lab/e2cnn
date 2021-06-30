

from e2cnn.nn.modules.r2_conv.basisexpansion import BasisExpansion

from collections import defaultdict

import torch
from scipy import stats
import math

__all__ = ["generalized_he_init", "deltaorthonormal_init"]


def _generalized_he_init_variances(basisexpansion: BasisExpansion):
    r"""

    Compute the variances of the weights of a convolutional layer with a generalized He's weight initialization method.

    Args:
        basisexpansion (BasisExpansion): the basis expansion method

    """
    
    vars = torch.ones((basisexpansion.dimension(),))
    
    inputs_count = defaultdict(lambda: set())
    basis_count = defaultdict(int)
    
    basis_info = list(basisexpansion.get_basis_info())
    
    for attr in basis_info:
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        inputs_count[o].add(in_irrep)
        basis_count[(in_irrep, o)] += 1
    
    for o in inputs_count.keys():
        inputs_count[o] = len(inputs_count[o])
    
    for w, attr in enumerate(basis_info):
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        vars[w] = 1. / math.sqrt(inputs_count[o] * basis_count[(in_irrep, o)])
    
    return vars


cached_he_vars = {}


def generalized_he_init(tensor: torch.Tensor, basisexpansion: BasisExpansion, cache: bool = False):
    r"""

    Initialize the weights of a convolutional layer with a generalized He's weight initialization method.

    Because the computation of the variances can be expensive, to save time on consecutive runs of the same model,
    it is possible to cache the tensor containing the variance of each weight, for a specific ```basisexpansion```.
    This can be useful if a network contains multiple convolution layers of the same kind (same input and output types,
    same kernel size, etc.) or if one needs to train the same network from scratch multiple times (e.g. to perform
    hyper-parameter search over learning rate or to repeat an experiment with different random seeds).

    .. note ::
        The variance tensor is cached in memory and therefore is only available to the current process.

    Args:
        tensor (torch.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method
        cache (bool, optional): cache the variance tensor. By default, ```cache=False```

    """
    # Initialization
    
    assert tensor.shape == (basisexpansion.dimension(),)
    
    if cache and basisexpansion not in cached_he_vars:
        cached_he_vars[basisexpansion] = _generalized_he_init_variances(basisexpansion)
    
    if cache:
        vars = cached_he_vars[basisexpansion]
    else:
        vars = _generalized_he_init_variances(basisexpansion)
    
    tensor[:] = vars * torch.randn_like(tensor)


def deltaorthonormal_init(tensor: torch.Tensor, basisexpansion: BasisExpansion):
    r"""
    
    Initialize the weights of a convolutional layer with *delta-orthogonal* initialization.
    
    Args:
        tensor (torch.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method

    """
    # Initialization

    assert tensor.shape == (basisexpansion.dimension(), )
    
    tensor.fill_(0.)
    
    counts = defaultdict(lambda: defaultdict(lambda: []))
    
    for p, attr in enumerate(basisexpansion.get_basis_info()):
        i = attr["in_irrep"]
        o = attr["out_irrep"]
        ip = attr["in_irreps_position"]
        op = attr["out_irreps_position"]
        r = attr["radius"]
        
        if i == o and r == 0.:
            counts[i][(ip, op)].append(p)
    
    def same_content(l):
        l = list(l)
        return all(ll == l[0] for ll in l)
    
    for irrep, count in counts.items():
        assert same_content([len(x) for x in count.values()]), [len(x) for x in count.values()]
        in_c = defaultdict(int)
        out_c = defaultdict(int)
        for ip, op in count.keys():
            in_c[ip] += 1
            out_c[op] += 1
        assert same_content(in_c.values()), count.keys()
        assert same_content(out_c.values()), count.keys()
        
        assert list(in_c.values())[0] == len(out_c.keys())
        assert list(out_c.values())[0] == len(in_c.keys())
        
        s = len(list(count.values())[0])
        i = len(in_c.keys())
        o = len(out_c.keys())
        
        # assert i <= o, (i, o, s, irrep, self._input_size, self._output_size)
        # if i > o:
        #     print("Warning: using delta orthogonal initialization to map to a larger number of channels")
        
        if max(o, i) > 1:
            W = stats.ortho_group.rvs(max(i, o))[:o, :i]
            # W = np.eye(o, i)
        else:
            W = 2 * torch.randint(0, 1, size=(1, 1)) - 1
            # W = np.array([[1]])
        
        w = torch.randn((o, s))
        # w = torch.ones((o, s))
        w *= 5.
        w /= (w ** 2).sum(dim=1, keepdim=True).sqrt()
        
        for i, ip in enumerate(in_c.keys()):
            for o, op in enumerate(out_c.keys()):
                for p, pp in enumerate(count[(ip, op)]):
                    tensor[pp] = w[o, p] * W[o, i]

