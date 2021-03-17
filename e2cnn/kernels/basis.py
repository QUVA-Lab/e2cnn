
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple


class EmptyBasisException(Exception):
    def __init__(self):
        r"""
        Exception raised when a :class:`~e2cnn.kernels.KernelBasis` with no elements is built.
        
        """
        message = "The KernelBasis you tried to instantiate is empty (dim = 0). You should catch this exception."
        super(EmptyBasisException, self).__init__(message)
        

class KernelBasis(ABC):
    
    def __init__(self, dim: int, shape: Tuple[int, int]):
        r"""
        
        Abstract class for implementing the basis of a kernel space.
        A kernel space is the space of functions in the form:
        
        .. math::
            \mathcal{K} := \{ \kappa: X \to \mathbb{R}^{c_\text{out} \times c_\text{in}} \}
        
        where :math:`X` is the base space on which the kernel is defined.
        For instance, for planar images :math:`X = \R^2`.
        
        Args:
            dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`
            
        Attributes:
            ~.dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            ~.shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`
            
        """
        assert isinstance(dim, int)
        assert isinstance(shape, tuple) and len(shape) == 2
        
        assert dim >= 0
        
        if dim == 0:
            raise EmptyBasisException()
        
        self.dim = dim
        self.shape = shape

    def __len__(self):
        return self.dim
    
    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    @abstractmethod
    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""
        Sample the continuous basis elements on discrete points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``points`` must be an array of shape `(D, N)`, where `D` is the dimensionality of the (parametrization of the)
        base space while `N` is the number of points.

        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass
    
    @abstractmethod
    def __hash__(self):
        pass


class GaussianRadialProfile(KernelBasis):
    
    def __init__(self, radii: List[float], sigma: Union[List[float], float]):
        r"""
        
        Basis for kernels defined over a radius in :math:`\R^+_0`.
        
        Each basis element is defined as a Gaussian function.
        Different basis elements are centered at different radii (``rings``) and can possibly be associated with
        different widths (``sigma``).
        
        More precisely, the following basis is implemented:
        
        .. math::
            
            \mathcal{B} = \left\{ b_i (r) :=  \exp \left( \frac{ \left( r - r_i \right)^2}{2 \sigma_i^2} \right) \right\}_i
        
        In order to build a complete basis of kernels, you should combine this basis with a basis which defines the
        angular profile through :class:`~e2cnn.kernels.PolarBasis`.
        
        
        Args:
            radii (list): centers of each basis element. They should be different and spread to cover all
                domain of interest
            sigma (list or float): widths of each element. Can potentially be different.
        
        
        """
        
        if isinstance(sigma, float):
            sigma = [sigma]*len(radii)
            
        assert len(radii) == len(sigma)
        assert isinstance(radii, list)
        
        for r in radii:
            assert r >= 0.
        
        for s in sigma:
            assert s > 0.
        
        super(GaussianRadialProfile, self).__init__(len(radii), (1, 1))

        self.radii = np.array(radii).reshape(1, 1, -1, 1)
        self.sigma = np.array(sigma).reshape(1, 1, -1, 1)
        
    def sample(self, radii: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""
        
        Sample the continuous basis elements on the discrete set of radii in ``radii``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``radii`` must be an array of shape `(1, N)`, where `N` is the number of points.

        Args:
            radii (~numpy.ndarray): radii where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(radii.shape) == 2
        assert radii.shape[0] == 1
    
        if out is None:
            out = np.empty((self.shape[0], self.shape[1], self.dim, radii.shape[1]))
    
        assert out.shape == (self.shape[0], self.shape[1], self.dim, radii.shape[1])
    
        radii = radii.reshape(1, 1, 1, -1)
        
        d = (self.radii - radii)**2
        
        out = np.exp(-0.5*d/self.sigma**2, out=out)
        
        return out

    def __getitem__(self, r):
        assert r < self.dim
        return {"radius": self.radii[0, 0, r, 0], "sigma": self.sigma[0, 0, r, 0], "idx": r}

    def __eq__(self, other):
        if isinstance(other, GaussianRadialProfile):
            return np.allclose(self.radii, other.radii) and np.allclose(self.sigma, other.sigma)
        else:
            return False
    
    def __hash__(self):
        return hash(self.radii.tobytes()) + hash(self.sigma.tobytes())
    

class PolarBasis(KernelBasis):
    
    def __init__(self, radial: KernelBasis, angular: KernelBasis):
        r"""
        
        Build the tensor product basis of a radial profile basis and an angular profile basis for kernels over the
        plane. Given two bases :math:`A = \{a_i\}_i` and :math:`B = \{b_j\}_j`, this basis is defined as
        
        .. math::
            C = A \otimes B = \left\{ c_{i,j}(x, y) := a_i(r) b_j(\phi) \right\}_{i,j}
        
        
        where :math:`(r, \phi)` is the polar coordinates of the points :math:`(x, y)` on the plane.
        
        Args:
            radial (KernelBasis): the radial basis
            angular (KernelBasis): the angular basis
        
        Attributes:
            ~.radial (KernelBasis): the radial basis
            ~.angular (KernelBasis): the angular basis
        
        """
        super(PolarBasis, self).__init__(radial.dim * angular.dim, (radial.shape[0] * angular.shape[0], radial.shape[1] * angular.shape[1]))
        self.radial = radial
        self.angular = angular
    
    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""

        Sample the continuous basis elements on a discrete set of ``points`` on the plane.
        Optionally, store the resulting multidimensional array in ``out``.

        ``points`` must be an array of shape `(2, N)` containing `N` points on the plane.
        Note that the points are specified in cartesian coordinates :math:`(x, y)`.

        Args:
            points (~numpy.ndarray): points on the plane where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(points.shape) == 2
        assert points.shape[0] == 2

        # computes the polar coordinates
        radii = np.sqrt((points**2).sum(0)).reshape(1, -1)
        angles = np.arctan2(points[1, :], points[0, :]).reshape(1, -1)
        # the angle at the origin is not well defined
        angles[radii < 1e-9] = np.nan

        if out is None:
            out = np.empty((self.shape[0], self.shape[1], self.dim, points.shape[1]))
    
        assert out.shape == (self.shape[0], self.shape[1], self.dim, points.shape[1])
        
        # sample the radial and the angular basis
        o1 = self.radial.sample(radii)
        o2 = self.angular.sample(angles)
        
        m, n, a, p = o1.shape
        q, r, b, p = o2.shape
        
        # build the tensor product
        out = out.reshape((m, q, n, r, a, b, p))
        out = np.einsum("mnap,qrbp->mqnrabp", o1, o2, out=out)
        
        return out.reshape((q*m, n*r, self.dim, p))
    
    def __getitem__(self, idx):
        assert idx < self.dim
        idx1, idx2 = divmod(idx, self.angular.dim)
        attr1 = self.radial[idx1]
        attr2 = self.angular[idx2]
        
        attr = dict()
        # attr.update({"radial_"+k: v for k, v in attr1.items()})
        # attr.update({"angular_"+k: v for k, v in attr2.items()})
        attr.update(attr1)
        attr.update(attr2)
        
        attr["idx"] = idx
        attr["idx1"] = idx1
        attr["idx2"] = idx2
        
        return attr

    def __iter__(self):
        idx = 0
        for attr1 in self.radial:
            for attr2 in self.angular:
                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["idx"] = idx
                attr["idx1"] = attr1["idx"]
                attr["idx2"] = attr2["idx"]

                yield attr
                idx += 1

    def __eq__(self, other):
        if isinstance(other, PolarBasis):
            return self.radial == other.radial and self.angular == other.angular
        else:
            return False
    
    def __hash__(self):
        return hash(self.radial) + hash(self.angular)

