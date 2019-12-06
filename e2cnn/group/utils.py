
import numpy as np
import math


def psi(theta: float, k: int = 1, gamma: float = 0.):
    r"""
    Rotation matrix corresponding to the angle :math:`k \theta + \gamma`.
    """
    x = k * theta + gamma
    c, s = np.cos(x), np.sin(x)
    return np.array(([
        [c, -s],
        [s,  c],
    ]))


def chi(s: int):
    #
    r"""
    The orthonormal matrix associated to the reflection along the :math:`y` axis if ``s=1``, the identity otherwise.
    """

    assert s in [0, 1]
    s = (-1 if s else 1)
    # assert s in [-1, 1]
    return np.array(([
        [1, 0],
        [0, s],
    ]))


def psichi(theta: float, s: int, k: int = 1, gamma: float = 0.):
    r"""
    Rotation matrix corresponding to the angle :math:`k \theta + \gamma` if `s=0`.
    Otherwise, it corresponds to the reflection along the axis defined by that angle.
    
    It is equal to::
        
        psi(theta, k, gamma) @ chi(s)
    
    """

    assert s in [0, 1]
    s = (-1 if s else 1)
    # assert s in [-1, 1]
    
    x = k * theta + gamma
    return np.array(([
        [np.cos(x), -s*np.sin(x)],
        [np.sin(x),  s*np.cos(x)],
    ]))


def cycle_isclose(a, b, S, rtol=1e-9, atol=1e-11):
    r"""
    
    Cyclic "isclose" check.
    
    Checks if the numbers ``a`` and ``b`` are close to each other in a cycle of length ``S``,
    i.e. if ``a - b`` is close to a multiple of ``S``.
    
    """
    
    d = (a - b) % S
    
    close_0 = math.isclose(d, 0., rel_tol=rtol, abs_tol=atol) and d >= 0.
    close_S = math.isclose(d, S, rel_tol=rtol, abs_tol=atol) and d <= S
    
    return close_0 or close_S
