
from dataclasses import dataclass
import itertools
from e2cnn.kernels.basis import EmptyBasisException
import numpy as np
from typing import Iterable, List, Optional, Type, Union, Tuple

from e2cnn.kernels import Basis
from e2cnn.group import SO2, Representation
# we need SteerableDiffopBasis, but importing that directly
# leads to cyclic imports because it also relies on this file.
# So we just import the diffops module instead
import e2cnn.diffops as diffops
from .utils import discretize_homogeneous_polynomial, multiply_polynomials, laplacian_power, display_diffop, transform_polynomial


@dataclass(frozen=True)
class DiscretizationArgs:
    r"""Parameters specifying a discretization procedure for PDOs.
        
        Attributes:
            ~.method (str, optional): the discretization method to use,
                either ``"rbffd"``, ``"fd"`` or ``"gauss"``.
            ~.smoothing (float, optional): ``smoothing`` is the standard
                deviation of the Gaussian used for discretization. Must be set if ``method="gauss"``,
                has no effect otherwise.
            ~.angle_offset (float, optional): if not ``None``, rotate the PDOs by this many radians.
            ~.phi (str, optional): which RBF to use (only relevant for RBF-FD).
                Can be any of the abbreviations `here <https://rbf.readthedocs.io/en/latest/basis.html>`_.
    """
    method: str = "fd"
    smoothing: Optional[float] = None
    angle_offset: Optional[float] = None
    phi: str = "ga"


class DiffopBasis(Basis):

    def __init__(self, 
                 coefficients: List[np.ndarray],
                 discretization: DiscretizationArgs = DiscretizationArgs(),
                 ):
        r"""
        Abstract class for implementing the basis of a space of differential operators.
        Such a space consists of :math:`c_\text{out} \times c_\text{in}` matrices with
        partial differential operators as entries.
        
        Args:
            coefficients (list): a list of ndarrays. Each array describes one element
                of the basis and has shape ``(c_out, c_in, n + 1)``, where ``n``
                is the derivative order of the entries of the matrix.
                PDOs are encoded as the coefficients of :math:`\frac{\partial^n}{\partial x^n}`,
                :math:`\frac{\partial^n}{\partial x^{n - 1}\partial y}`, ...,
                :math:`\frac{\partial^n}{\partial y^n}`.
            discretization (optional): additional parameters specifying parameters for
                the discretization procedure. See :class:`~e2cnn.diffops.DiscretizationArgs`.
            
        Attributes:
            ~.coefficients (list): an analytical description of the PDO basis elements, see above
            ~.dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            ~.shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`
            ~.maximum_order (int): the largest derivative order occuring in the basis

        """
        dim = len(coefficients)
        if dim == 0:
            raise EmptyBasisException
        shape = coefficients[0].shape[:2]
        self.disc = discretization
        self.maximum_order = 0
        for element in coefficients:
            assert element.shape[:2] == shape
            assert len(element.shape) == 3
            # we sometimes get very small coefficients (on the order of 1e-17)
            # through rounding errors, those should be 0
            # this is important to get the derivative orders right for basis filters
            element[np.abs(element) < 1e-8] = 0
            # We want to know the maximum order that appears in this basis.
            # The last axis contains the actual derivative, and has length order + 1
            self.maximum_order = max(self.maximum_order, element.shape[-1] - 1)
        self.coefficients = coefficients
        super().__init__(dim, shape)

    def sample(self,
               points: np.ndarray,
               ) -> np.ndarray:
        r"""
        Discretize the basis on a set of points.
        See :meth:`~e2cnn.diffops.DiffopBasis.sampled_masked` for details.
        """
        return self.sample_masked(points)

    def sample_masked(self,
                      points: np.ndarray,
                      mask: np.ndarray = None,
                      ) -> np.ndarray:
        r"""
        Discretize the basis on a set of points.

        Args:
            points (ndarray): a `2 x N` array with `N` points on which to discretize.
                If FD is used (default), this has to be a flattened version of
                a regular sorted grid (like you would get using np.meshgrid on ranges).
            mask (ndarray, optional): Boolean array of shape (dim, ), where ``dim`` is the number of basis elements.
                True for elements to discretize and False for elements to discard.

        Returns:
            ndarray of with shape `(C_out, C_in, num_basis_elements, n_in)`, where
            `num_basis_elements` are the number of elements after applying the mask, and `n_in` is the number
            of points.

        """
        if mask is None:
            # if no mask is given, we use all basis elements
            mask = np.array([True] * self.dim)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (self.dim, )

        if self.disc.angle_offset is not None:
            so2 = SO2(1)
            # rotation matrix by angle_offset
            matrix = so2.irrep(1)(self.disc.angle_offset)
            # we transform the polynomial with the matrix
            coefficients = (transform_polynomial(element, matrix) for element in self.coefficients)
        else:
            coefficients = self.coefficients

        coefficients = (coeff for coeff, m in zip(coefficients, mask) if m)

        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[0] == 2
        num_points = points.shape[1]

        basis = np.empty((np.sum(mask), ) + self.shape + (num_points, ))
        for k, element in enumerate(coefficients):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    basis[k, i, j] = discretize_homogeneous_polynomial(
                        points,
                        element[i, j],
                        self.disc.smoothing,
                        phi=self.disc.phi,
                        method=self.disc.method,
                    )

        # Finally, we move the len_basis axis to the third position
        basis = basis.transpose(1, 2, 0, 3)

        return basis

    def pretty_print(self) -> str:
        """Return a human-readable representation of all basis elements."""
        out = ""
        for element in self.coefficients:
            out += display_matrix(element)
            out += "\n----------------------------------\n"
        return out


class LaplaceProfile(DiffopBasis):

    def __init__(self, max_power: int, discretization: DiscretizationArgs = DiscretizationArgs()):
        r"""
        Basis for rotationally invariant PDOs.
        Each basis element is defined as a power of a Laplacian.

        In order to build a complete basis of PDOs, you should combine this basis
        with a basis which defines the angular profile through :class:`~e2cnn.diffops.TensorBasis`.


        Args:
            max_power (int): the maximum power of the Laplace operator that will be used.
                The maximum degree (as a differential operator) will be two times this maximum
                power.
            discretization (DiscretizationArgs, optional): additional parameters specifying parameters for
                the discretization procedure. See :class:`~e2cnn.diffops.DiscretizationArgs`.

        """

        assert isinstance(max_power, int)
        assert max_power >= 0

        coefficients = [
            laplacian_power(k).reshape(1, 1, -1) for k in range(max_power + 1)
        ]

        super().__init__(coefficients, discretization)

        self.max_power = max_power

    def __getitem__(self, r):
        assert r < self.dim
        return {"power": r, "order": 2 * r, "idx": r}

    def __eq__(self, other):
        if isinstance(other, LaplaceProfile):
            return self.max_power == other.max_power
        else:
            return False

    def __hash__(self):
        return hash(self.max_power)


class TensorBasis(DiffopBasis):

    def __init__(self,
                 irreps_basis: Type[DiffopBasis],
                 in_repr: Representation,
                 out_repr: Representation,
                 max_power: int,
                 discretization: DiscretizationArgs = DiscretizationArgs(),
                 **kwargs
                 ):
        r"""

        Build the tensor product basis of two PDO bases over the
        plane. Given two bases :math:`A = \{a_i\}_i` and :math:`B = \{b_j\}_j`, this basis is defined as

        .. math::
            C = A \otimes B = \left\{ c_{i,j} := a_i \circ b_j \right\}_{i,j}.

        The arguments are passed on to :class:`~e2cnn.diffops.SteerableDiffopBasis` and
        :class:`~e2cnn.diffops.LaplaceProfile`, see their documentation.

        Attributes:
            ~.basis1 (SteerableDiffopBasis): the first basis
            ~.basis2 (LaplaceProfile): the second basis

        """
        basis1 = diffops.SteerableDiffopBasis(
            irreps_basis, in_repr, out_repr, discretization, **kwargs
        )
        basis2 = LaplaceProfile(max_power, discretization)

        coefficients = []
        for a, b in itertools.product(basis1.coefficients, basis2.coefficients):
            order = a.shape[2] + b.shape[2] - 2
            out = np.empty((a.shape[0], b.shape[0], a.shape[1], b.shape[1], order + 1))
            for i, j, k, l in itertools.product(range(a.shape[0]),
                                             range(b.shape[0]),
                                             range(a.shape[1]),
                                             range(b.shape[1])):
                out[i, j, k, l] = multiply_polynomials(a[i, k], b[j, l])
            out = out.reshape(a.shape[0] * b.shape[0], a.shape[1] * b.shape[1], order + 1)
            coefficients.append(out)
        super().__init__(coefficients, discretization)
        self.basis1 = basis1
        self.basis2 = basis2

    def __getitem__(self, idx):
        assert idx < self.dim
        idx1, idx2 = divmod(idx, self.basis2.dim)
        attr1 = self.basis1[idx1]
        attr2 = self.basis2[idx2]

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["order"] = attr1["order"] + attr2["order"]

        attr["idx"] = idx
        attr["idx1"] = idx1
        attr["idx2"] = idx2

        return attr

    def __iter__(self):
        idx = 0
        for attr1 in self.basis1:
            for attr2 in self.basis2:
                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["order"] = attr1["order"] + attr2["order"]
                attr["idx"] = idx
                attr["idx1"] = attr1["idx"]
                attr["idx2"] = attr2["idx"]

                yield attr
                idx += 1

    def __eq__(self, other):
        if isinstance(other, TensorBasis):
            return self.basis1 == other.basis1 and self.basis2 == other.basis2
        else:
            return False

    def __hash__(self):
        return hash(self.basis1) + hash(self.basis2)


def display_matrix(element):
    out = ""
    for i in range(element.shape[0]):
        for j in range(element.shape[1]):
            out += display_diffop(element[i, j]) + "\t"
        out += "\n"
    return out
