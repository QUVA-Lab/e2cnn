
import numpy as np

from .basis import KernelBasis, EmptyBasisException
from .irreps_basis import IrrepBasis

from e2cnn.group import Representation

from typing import Type


class SteerableKernelBasis(KernelBasis):
    
    def __init__(self,
                 irreps_basis: Type[IrrepBasis],
                 in_repr: Representation,
                 out_repr: Representation,
                 **kwargs):
        r"""
        
        Implements a general basis for the vector space of equivariant kernels.
        A :math:`G`-equivariant kernel :math:`\kappa`, mapping between an input field, transforming under
        :math:`\rho_\text{in}` (``in_repr``), and an output field, transforming under  :math:`\rho_\text{out}`
        (``out_repr``), satisfies the following constraint:
        
        .. math ::
            
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1} \qquad \forall g \in G, \forall x \in X
        
        As the kernel constraint is a linear constraint, the space of equivariant kernels is a vector subspace of the
        space of all convolutional kernels. It follows that any equivariant kernel can be expressed in terms of a basis
        of this space.
        
        This class solves the kernel constraint for two arbitrary representations by combining the solutions of the
        kernel constraints associated to their :class:`~e2cnn.group.IrreducibleRepresentation` s.
        In order to do so, it relies on ``irreps_basis`` which solves individual irreps constraints. ``irreps_basis``
        must be a class (subclass of :class:`~e2cnn.kernels.IrrepsBasis`) which builds a basis for equivariant
        kernels associated with irreducible representations when instantiated.
        
        The groups :math:`G` which are currently implemented are origin-preserving isometries (what are called
        structure groups, or sometimes gauge groups, in the language of
        `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ ).
        The origin-preserving isometries of :math:`\R^d` are subgroups of :math:`O(d)`, i.e. reflections and rotations.
        Therefore, equivariance does not enforce any constraint on the radial component of the kernels.
        Hence, this class only implements a basis for the angular part of the kernels.
        
        In order to build a complete basis of kernels, you should combine this basis with a basis which defines the
        radial profile (such as :class:`~e2cnn.kernels.GaussianRadialProfile`) through
        :class:`~e2cnn.kernels.PolarBasis`.
        
        .. math::
            
            \mathcal{B} = \left\{ b_i (r) :=  \exp \left( \frac{ \left( r - r_i \right)^2}{2 \sigma_i^2} \right) \right\}_i
        
        .. warning ::
            
            Typically, the user does not need to manually instantiate this class.
            Instead, we suggest to use the interface provided in :doc:`e2cnn.gspaces`.
        
        Args:
            irreps_basis (class): class defining the irreps basis. This class is instantiated for each pair of irreps to solve all irreps constraints.
            in_repr (Representation): Representation associated with the input feature field
            out_repr (Representation): Representation associated with the output feature field
            **kwargs: additional arguments used when instantiating ``irreps_basis``
            
        """
        
        assert in_repr.group == out_repr.group
        
        self.in_repr = in_repr
        self.out_repr = out_repr
        group = in_repr.group
        self.group = group

        A_inv = np.array(in_repr.change_of_basis_inv, copy=True)
        B = np.array(out_repr.change_of_basis, copy=True)
        
        # A_inv = in_repr.change_of_basis_inv
        # B = out_repr.change_of_basis

        if not np.allclose(A_inv, np.eye(in_repr.size)):
            self.A_inv = A_inv
        else:
            self.A_inv = None
            
        if not np.allclose(B, np.eye(out_repr.size)):
            self.B = B
        else:
            self.B = None

        self.irreps_bases = {}

        # loop over all input irreps
        for i_irrep_name in set(in_repr.irreps):
            # loop over all output irreps
            for o_irrep_name in set(out_repr.irreps):
        
                try:
                    # retrieve the irrep intertwiner basis
                    basis = irreps_basis(group=group,
                                         in_irrep=i_irrep_name,
                                         out_irrep=o_irrep_name,
                                         **kwargs)

                    self.irreps_bases[(i_irrep_name, o_irrep_name)] = basis

                except EmptyBasisException:
                    # if the basis is empty, skip it
                    pass
        
        self.bases = [[None for _ in range(len(out_repr.irreps))] for _ in range(len(in_repr.irreps))]
        
        self.in_sizes = []
        self.out_sizes = []
        # loop over all input irreps
        for ii, i_irrep_name in enumerate(in_repr.irreps):
            self.in_sizes.append(group.irreps[i_irrep_name].size)
            
        # loop over all output irreps
        for oo, o_irrep_name in enumerate(out_repr.irreps):
            self.out_sizes.append(group.irreps[o_irrep_name].size)

        dim = 0
        # loop over all input irreps
        for ii, i_irrep_name in enumerate(in_repr.irreps):
            # loop over all output irreps
            for oo, o_irrep_name in enumerate(out_repr.irreps):
                if (i_irrep_name, o_irrep_name) in self.irreps_bases:
                    self.bases[ii][oo] = self.irreps_bases[(i_irrep_name, o_irrep_name)]
                    dim += self.bases[ii][oo].dim
        
        super(SteerableKernelBasis, self).__init__(dim, (out_repr.size, in_repr.size))
            
    def sample(self, angles: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""

        Sample the continuous basis elements on the discrete set of angles in ``angles``.
        Optionally, store the resulting multidimentional array in ``out``.
        
        A value of ``nan`` is interpreted as the angle of a point placed on the origin of the axes.

        ``angles`` must be an array of shape `(1, N)`, where `N` is the number of points.

        Args:
            angles (~numpy.ndarray): angles where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis
            
        """
        assert len(angles.shape) == 2
        assert angles.shape[0] == 1

        if out is None:
            out = np.zeros((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
        else:
            out.fill(0)
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])
    
        if self.A_inv is None and self.B is None:
            out = self._sample_direct_sum(angles, out=out)
        else:
            samples = self._sample_direct_sum(angles)
            out = self._change_of_basis(samples, out=out)
    
        return out
    
    def _sample_direct_sum(self, angles: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        assert len(angles.shape) == 2
    
        if out is None:
            out = np.zeros((self.shape[0], self.shape[1], self.dim, angles.shape[1]))

        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])
    
        basis_count = 0
        in_position = 0
        for ii, in_size in enumerate(self.in_sizes):
            out_position = 0
            for oo, out_size in enumerate(self.out_sizes):
                if self.bases[ii][oo] is not None:
                    dim = self.bases[ii][oo].dim
                    
                    block = out[
                            out_position:out_position+out_size,
                            in_position:in_position+in_size,
                            basis_count:basis_count+dim,
                            ...
                    ]
                    self.bases[ii][oo].sample(angles, out=block)
                    
                    # out[
                    #     out_position:out_position+out_size,
                    #     in_position:in_position+in_size,
                    #     basis_count:basis_count+dim,
                    #     ...
                    # ] = self.bases[ii][oo].sample(angles)

                    basis_count += dim
                    
                out_position += out_size
            in_position += in_size

        return out
    
    def _change_of_basis(self, samples: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        # multiply by the change of basis matrices to transform the irreps basis in the full representations basis
        
        if self.A_inv is not None and self.B is not None:
            out = np.einsum("no,oibp,ij->njbp", self.B, samples, self.A_inv, out=out)
        elif self.A_inv is not None:
            out = np.einsum("oibp,ij->ojbp", samples, self.A_inv, out=out)
        elif self.B is not None:
            out = np.einsum("no,oibp->nibp", self.B, samples, out=out)
        else:
            out[...] = samples
        
        return out

    def __getitem__(self, idx):
        assert idx < self.dim
        
        count = 0
        for ii in range(len(self.in_sizes)):
            for oo in range(len(self.out_sizes)):
                if self.bases[ii][oo] is not None:
                    dim = self.bases[ii][oo].dim

                    rel_idx = idx - count
                    if rel_idx >= 0 and rel_idx < dim:
                        
                        attr = dict(self.bases[ii][oo][rel_idx])
                        
                        attr["shape"] = self.bases[ii][oo].shape
                        
                        attr["in_irrep"] = self.in_repr.irreps[ii]
                        attr["out_irrep"] = self.out_repr.irreps[oo]
                        
                        attr["in_irrep_idx"] = ii
                        attr["out_irrep_idx"] = oo
                        
                        attr["inner_idx"] = attr["idx"]
                        attr["idx"] = idx
                        

                        return attr
                
                    count += dim

    def __iter__(self):
        idx = 0
        for ii in range(len(self.in_sizes)):
            for oo in range(len(self.out_sizes)):
                if self.bases[ii][oo] is not None:
                    for rel_idx in range(len(self.bases[ii][oo])):
                        attr = dict(self.bases[ii][oo][rel_idx])

                        attr["shape"] = self.bases[ii][oo].shape
                        
                        attr["in_irrep"] = self.in_repr.irreps[ii]
                        attr["out_irrep"] = self.out_repr.irreps[oo]
                    
                        attr["in_irrep_idx"] = ii
                        attr["out_irrep_idx"] = oo
                        
                        attr["inner_idx"] = attr["idx"]
                        attr["idx"] = idx

                        yield attr
                        
                        idx += 1

    def __eq__(self, other):
        if not isinstance(other, SteerableKernelBasis):
            return False
        elif self.in_repr != other.in_repr or self.out_repr != other.out_repr:
            return False
        else:
            sbk1 = sorted(self.irreps_bases.keys())
            sbk2 = sorted(other.irreps_bases.keys())
            if sbk1 != sbk2:
                return False
            
            for irreps, basis in self.irreps_bases.items():
                if basis != other.irreps_bases[irreps]:
                    return False
            
            return True

    def __hash__(self):
        h = hash(self.in_repr) + hash(self.out_repr)
        for basis in self.irreps_bases.items():
            h += hash(basis)
        return h
        
        


