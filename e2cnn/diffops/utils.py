
from typing import List, Union, Iterable, Tuple
import os
import warnings
import pickle

import numpy as np
import scipy.special

try:
    from sympy.calculus.finite_diff import finite_diff_weights
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

try:
    from rbf.pde.fd import weight_matrix # type: ignore
    from rbf.basis import set_symbolic_to_numeric_method, get_rbf # type: ignore

    _RBF_AVAILABLE = True

    set_symbolic_to_numeric_method('lambdify')
    _gaussian = get_rbf("ga")

except ImportError:
    _RBF_AVAILABLE = False

_DIFFOP_CACHE = {}
_1D_KERNEL_CACHE = {}

def load_cache(path: str = "./.e2cnn_cache/diffops.pickle"):
    """
    Load cached PDO discretizations from disk.
    The cache should have been previously created using :func:`~e2cnn.diffops.store_cache`.
    
    Args:
        path (str, optional): the path to the file with discretizations.
        
    """
    if os.path.exists(path):
        print("Loading cached Diffops")
        with open(path, "rb") as f:
            _DIFFOP_CACHE = pickle.load(f)
    else:
        print("Diffop cache not found, skipping")

def store_cache(path: str = "./.e2cnn_cache/diffops.pickle"):
    """
    Cache PDO discretizations on disk.
    The cache can later be loaded using :func:`~e2cnn.diffops.load_cache`.
    This will speed up network instantiation as long as the architecture does not
    change too much.
    
    Args:
        path (str, optional): the path to the file with discretizations.
        
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w+b") as f:
        pickle.dump(_DIFFOP_CACHE, f)


def discretize_homogeneous_polynomial(
        points: np.ndarray,
        coefficients: np.ndarray,
        smoothing: float = None,
        phi: str = "ga",
        method: str = "fd",
) -> np.ndarray:
    r"""Discretize a homogeneous partial differential operator.
    Homogeneous means that all terms have the same derivative order.

    See :meth:`e2cnn.diffops.DiffopBasis.sample` for details.
    
    .. warning::
        The discretization relies on two external packages: `sympy <https://docs.sympy.org/>`_ and `rbf <https://rbf.readthedocs.io>`_.
        If they are not available, an error is raised.

    """
    assert isinstance(points, np.ndarray)
    assert points.shape[0] == 2
    num_points = points.shape[1]

    targets = np.array([[0, 0]])

    key = (points.tobytes(), coefficients.tobytes(), smoothing, phi, method)
    if key in _DIFFOP_CACHE:
        return _DIFFOP_CACHE[key]

    n = len(coefficients) - 1

    nonzero = (coefficients != 0)
    nonzero_indices = [k for k in range(n + 1) if nonzero[k]]

    diffs = [[n - k, k] for k in nonzero_indices]
    if not diffs:
        # we have the zero operator
        return np.zeros(num_points)
    
    if method == "gauss":
        if smoothing is None:
            raise ValueError("smoothing must be set when method = 'gauss'")
        if not _RBF_AVAILABLE:
            raise RuntimeError("Using derivatives of Gaussians for discretization "
                               "requires the RBF package, please install it. "
                               "See https://github.com/treverhines/RBF for instructions.")
        out = gaussian_derivative(coefficients, points, smoothing)

    elif method == "fd":
        if not _SYMPY_AVAILABLE:
            raise RuntimeError("Using finite difference discretization "
                               "requires sympy, please install it.")
        # For FD, points needs to be a regular grid, and we need to
        # extract its projection onto the two axes.
        axis_points = (np.unique(points[0]), np.unique(points[1]))
        # Here, we just check that everything went as expected
        # (i.e. that the input had the expected format).
        # The minus sign is the convention that get_grid_coords
        # uses, which is what the modules in the nn package use.
        check_points = np.stack(
            np.meshgrid(axis_points[0], -axis_points[1], indexing="xy")
        ).reshape(2, -1)
        if not np.array_equal(check_points, points):
            print(axis_points)
            print(points)
            print(check_points)
            raise ValueError("Format of points doesn't match the one needed for FD.")

        kernels = np.stack(
            [
                # type checkers get confused by expanding diff, but we know
                # that it has length 2.
                discretize_2d_monomial(*diff, axis_points) # type: ignore
                for diff in diffs
            ]
        ).reshape(-1, num_points)

        out = np.sum(
            coefficients[nonzero][:, None] * kernels, axis=0
        )

    elif method == "rbffd":
        if not _RBF_AVAILABLE:
            raise RuntimeError("Using RBF-FD for discretization "
                               "requires the RBF package, please install it. "
                               "See https://github.com/treverhines/RBF for instructions.")
        out = weight_matrix(targets, points.T, num_points, diffs, coefficients[nonzero], phi=phi)
        if np.any(np.isnan(out.data)):
            raise Exception(f"NaNs encountered while discretizing diffop {display_diffop(coefficients)} on {num_points} points.\n"
                            f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}")
        if np.all(out.data == 0):
            warnings.warn(f"Zero filter encountered while discretizing diffop {display_diffop(coefficients)} on {num_points} points. "
                        "This might indicate that the kernel size is too small for this differential operator.\n"
                        f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}", RuntimeWarning)
        if np.any(np.abs(out.data) > 1e2):
            warnings.warn(f"Large filter values encountered while discretizing diffop {display_diffop(coefficients)} on {num_points} points. "
                        "A larger kernel size or different RBF might help.\n"
                        f"Max abs filter value: {np.max(np.abs(out))}\n"
                        f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}", RuntimeWarning)

        out = out.toarray().flatten()
    else:
        raise ValueError(f"Discretization method {method} not recognized. "
                         "Choices are 'fd', 'gauss' and 'rbffd'.")

    _DIFFOP_CACHE[key] = out
    return out


def discretize_1d_monomial(n: int, points: List[float]) -> np.ndarray:
    """Discretize the differential operator d^n/dx^n as a convolutional kernel."""
    # calculating the finite difference coefficients using sympy is fast,
    # but given that this function is called extremely often when sampling
    # a basis, it's still a bottleneck. Even with the cache on the level
    # of differential operators (because that one only caches something
    # if exactly the same operator appears multiple times).
    # So we use an additional cache here, which caches single monomials.
    key = (n, tuple(points))
    if key not in _1D_KERNEL_CACHE:
        assert n < len(points), (f"Can't discretize differential operator of order {n} on {len(points)} points, "
                                f"at least {n + 1} points are needed")
        _1D_KERNEL_CACHE[key] = fd_weights(n, points)
    return _1D_KERNEL_CACHE[key]


def fd_weights(n, points):
    weights = finite_diff_weights(n, points, 0)
    # first -1 is because we want the highest order (n),
    # second -1 means we want the most accurate approximation (using all points)
    return np.array(weights[-1][-1], dtype=float)


def discretize_2d_monomial(n_x: int,
                           n_y: int,
                           points: Tuple[List[float], List[float]],
                           ) -> np.ndarray:
    """Discretize the differential operator d^{n_x + n_y}/{dx^n_x dy^n_y}."""
    x_kernel = discretize_1d_monomial(n_x, points[0])
    y_kernel = discretize_1d_monomial(n_y, points[1])
    return x_kernel[:, None] * y_kernel[None, :]


def multiply_polynomials(a: np.ndarray, b: np.ndarray):
    """Multiply two homogeneous polynomials.

    Args:
        a (ndarray): coefficients of x^n, x^{n - 1}y, ..., y^n for the first polynomial
        b (ndarray): coefficients for the second polynomial

    Returns:
        coefficients of the product
    """
    n, m = len(a), len(b)

    # compute the Cauchy product
    return np.array([
        sum(a[l] * b[k - l] for l in range(k + 1) if l < n and (k - l) < m)
        for k in range(n + m - 1)
    ])


def expand_binomial(a, b, exponent):
    """Expand (ax + by)^exponent in terms of monomials and return their coefficients."""
    ks = np.arange(exponent + 1)
    binom_coeffs = scipy.special.binom(exponent, ks)
    return binom_coeffs * a ** (exponent - ks) * b ** ks


def transform_polynomial(coefficients: np.ndarray, matrix: np.ndarray):
    """Calculate the coefficients of P(Ax), where P is the homogeneous polynomial
    defined by the given coefficients and A is the given matrix.
    
    Args:
        coefficients: ndarray of shape(..., n + 1), where the last axis enumerates
            coefficients for x^n, x^{n - 1}y, ..., y^n. The other axes are batch
            dimension.
        matrix: ndarray of shape (2, 2)"""
    n = coefficients.shape[-1] - 1
    batch_shape = coefficients.shape[:-1]
    # the transformed polynomial will have the same degree:
    transformed = np.zeros(batch_shape + (n + 1, ))
    # now we iterate over all n + 1 coefficients:
    for i in range(n + 1):
        # we have a term coeff * x^{n - i} * y^i
        # First, we calculate the transformed versions of the x and y terms:
        # (x')^{n - i} = (A_11 x + A_21 y)^{n - i}
        x_trafo = expand_binomial(matrix[0, 0], matrix[1, 0], n - i)
        # (y')^i = (A_12 x + A_22 y)^i
        y_trafo = expand_binomial(matrix[0, 1], matrix[1, 1], i)
        # then we combine them
        xy_trafo = multiply_polynomials(x_trafo, y_trafo)
        # and finally scale by the coefficient
        # broadcasting: (*batch_shape, 1) * (1, ..., 1, n + 1) -> (*batch_shape, n + 1)
        transformed += coefficients[..., i, None] * xy_trafo[(None, ) * len(batch_shape)]
    return transformed


def gaussian_derivative(coefficients: np.ndarray, points: np.ndarray, std: float = 1.):
    n = len(coefficients) - 1
    num_points = points.shape[1]

    nonzero = (coefficients != 0)
    nonzero_indices = [k for k in range(n + 1) if nonzero[k]]

    diffs = [[n - k, k] for k in nonzero_indices]
    if not diffs:
        # we have the zero operator
        return np.zeros(num_points)
    
    center = np.zeros((1, 2))

    kernels = np.stack(
        [_gaussian(points.T, center, 1/std**2, np.array(diff)).flatten() for diff in diffs]
    )

    return np.sum(
        coefficients[nonzero][:, None] * kernels, axis=0
    )


def homogenized_cheby(n: int, kind: str = "t") -> np.ndarray:
    """Compute the coefficients for the homogenized version of T_n or U'_n.

    Args:
        n: degree of the polynomial. May be negative, in that case the degree will
           be abs(n) but the sign may differ (see notes for exact definition)
        kind: Either 't' for the first kind or 'u' for the second kind

    Returns:
        ndarray with coefficients, ordered in descending order of the power of x,
        i.e. x^n, x^{n - 1}y, ..., y^n."""
    sign = np.sign(n)
    n = abs(n)

    kind = kind.lower()

    result = np.zeros(n + 1)
    if kind == "t":
        q = n // 2
        ks = np.arange(q + 1)
        # these are the coefficients of x^{n - 2k}y^{2k}, from k = 0 to n/2
        result[::2] = scipy.special.binom(n, 2 * ks) * (-1) ** ks
    elif kind == "u":
        q = (n - 1) // 2
        ks = np.arange(q + 1)
        # these are the coefficients of x^{n - 2k - 1}y^{2k + 1}, from k = 0 to (n - 1)/2
        result[1::2] = sign * scipy.special.binom(n, 2 * ks + 1) * (-1) ** ks
    else:
        raise ValueError("kind must be either 'u' or 't'")
    return result


def laplacian_power(k: int):
    """Compute the coefficients of x^n, x^{n - 1}y, ..., y^n for the k-th power of the Laplacian."""
    result = np.zeros(2*k + 1)
    # The k-th power is given by (x^2 + y^2)^k = sum_{i = 0}^k {k choose i} x^{2i}y^{2(k - i)}.
    # So the coefficient of x^{2i}y^{2(k - i)} is the binomial coefficient, and the coefficients
    # at odd indices are zero
    result[::2] = scipy.special.binom(k, np.arange(k + 1))
    return result


def display_diffop(coefficients: np.ndarray):
    """Show a homogeneous differential operator as a pretty string."""
    out = ""
    n = len(coefficients) - 1
    for k, coeff in enumerate(coefficients):
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        if isinstance(coeff, float):
            coeff = round(coeff, 2)
        if coeff == 0:
            continue

        if out == "":
            if coeff == -1:
                out += "-"
            elif coeff != 1:
                out += f"{coeff}"
            if coeff in {1, -1} and n == 0:
                # constant term, we can't drop the 1
                out += "1"
        elif coeff == 1:
            out += " + "
        elif coeff == -1:
            out += " - "
        elif coeff > 0:
            out += f" + {coeff}"
            if not isinstance(coeff, int):
                out += " "
        elif coeff < 0:
            out += f" - {abs(coeff)}"
            if not isinstance(coeff, int):
                out += " "

        if n - k > 0:
            out += "x" + prettify_exponent(n - k)
            if n - k > 3 and k > 0:
                out += " "
        if k > 0:
            out += "y" + prettify_exponent(k)
    if out == "":
        out = "0"
    return out

def prettify_exponent(k):
    if k == 1:
        return ""
    if k == 2:
        return "²"
    if k == 3:
        return "³"
    return f"^{k}"


def eval_polys(coefficients, points):
    """Evaluate homogeneous polynomials on a set of points.

    Args:
        coefficients: list of ndarrays of shape (..., n + 1) with coefficients of x^n, x^{n - 1}y, ..., y^n
        points: ndarray with shape (2, N)

    Returns:
        ndarray with shape (L, ..., N) where N is the number of points and L the length of the input list
    """
    results = []
    for element in coefficients:
        dims = len(element.shape)
        nones = (dims - 1) * (None, )
        n = element.shape[-1] - 1
        ks = np.arange(n + 1).reshape(1, -1)
        xs = points[0].reshape(-1, 1)
        ys = points[1].reshape(-1, 1)
        monomials = xs**(n - ks) * ys**ks
        results.append((element[..., None, :] * monomials[nones]).sum(axis=-1))
    return np.stack(results)

def required_points(order: int, accuracy: int) -> int:
    """Compute the number of points necessary to achieve an approximation of the given accuracy.

    Important note: this function assumes that the points will arranged symmetrically
    around 0. Corollary 7 from https://arxiv.org/pdf/1102.3203.pdf then says that the
    order of accuracy may be boosted by 1 compared to the usual one, which is exploited in
    this function to return a lower number of points when possible.

    Args:
        order (int): order of the differential operator
        accuracy (int): desired accuracy (e.g. 2 for approximation to quadratic order)

    Returns:
        number of required sampling points
    """
    # The usual formula for the required number of points:
    N = order + accuracy
    # The remaining question is whether we can reduce this number by
    # 1 and still retain the desired accuracy, an effect called "boosted order of accuracy".
    # Corollary 7 from https://arxiv.org/pdf/1102.3203.pdf says that
    # boosting happens if the number of points minus the order is odd.
    # We want to check whether N - 1 still has the desired accuracy,
    # i.e. whether (N - 1) - order, or simply accuracy - 1, is odd:
    if (accuracy - 1) % 2:
        return N - 1
    return N

def largest_possible_order(num_points: int, accuracy: int) -> int:
    """Compute the largest diffop order such that the given accuracy is satisfied.

    See ``required_points`` for details."""
    order = num_points - accuracy
    assert order >= 0
    # check if the next-larger order would be boosted:
    if (num_points - (order + 1)) % 2:
        return order + 1
    return order

def guaranteed_accuracy(num_points: int, order: int) -> int:
    """Compute the accuracy that is guaranteed for the given order and number of points.

    See ``required_points`` for details."""
    accuracy = num_points - order
    assert accuracy >= 0
    # check if boosting happens:
    if accuracy % 2:
        return accuracy + 1
    return accuracy

def symmetric_points(n: int, dilation: float = 1) -> List[float]:
    # return e.g. [-1, 0, 1] for n = 3 and [-0.5, 0.5] for n = 2, etc.
    points = range(n)
    offset = (n - 1) / 2
    return [(x - offset) * dilation for x in points]
