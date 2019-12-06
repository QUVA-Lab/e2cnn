
import math
import numpy as np
from typing import Union


__all__ = [
    'psi',
    'chi',
    'psichi',
]


def offset_iterator(base_frequency, N, maximum_offset=None, maximum_frequency=None, non_negative: bool = False):
    if N < 0:
        # assert maximum_offset == 0
        if maximum_frequency is not None and math.fabs(base_frequency) <= maximum_frequency:
            yield 0
    else:
        assert maximum_frequency is not None or maximum_offset is not None
        
        if non_negative:
            minimum_frequency = 0
        else:
            minimum_frequency = -maximum_frequency if maximum_frequency is not None else None

        def round(x):
            if x > 0:
                return int(math.floor(x))
            else:
                return int(math.ceil(x))
            
        if maximum_frequency is not None:
            min_offset = (minimum_frequency - base_frequency) / N
            max_offset = (maximum_frequency - base_frequency) / N
        else:
            min_offset = -10000
            max_offset = 10000
            
        if maximum_offset is not None:
            min_offset = max(min_offset, -maximum_offset)
            max_offset = min(max_offset, maximum_offset)
        
        min_offset = math.ceil(min_offset)
        max_offset = math.floor(max_offset)
        
        for j in range(min_offset, max_offset+1):
            yield j


def psi(theta: Union[np.ndarray, float],
        k: int = 1,
        gamma: float = 0.,
        out: np.ndarray = None) -> np.ndarray:
    
    # rotation matrix of frequency k corresponding to the angle theta

    if isinstance(theta, float):
        theta = np.array(theta)

    k = np.array(k, copy=False).reshape(-1, 1)
    gamma = np.array(gamma, copy=False).reshape(-1, 1)
    theta = theta.reshape(1, -1)

    x = k * theta + gamma

    cos, sin = np.cos(x), np.sin(x)

    if out is None:
        out = np.empty((2, 2, x.shape[0], x.shape[-1]))

    out[0, 0, ...] = cos
    out[0, 1, ...] = -sin
    out[1, 0, ...] = sin
    out[1, 1, ...] = cos
    
    return out


def chi(s: int, out: np.ndarray = None) -> np.ndarray:
    # the orthonormal matrix associated to the flip
    
    s = -1 * (s > 0) + (s <= 0)

    s = np.array(s, copy=False).reshape(-1, 1)

    if out is None:
        out = np.empty((2, 2, s.shape[0], 1))
    
    out[0, 0, ...] = 1
    out[0, 1, ...] = 0
    out[1, 0, ...] = 0
    out[1, 1, ...] = s
    return out


def psichi(theta: Union[np.ndarray, float], s: int, k: int = 1, gamma: float = 0., out: np.ndarray = None) -> np.ndarray:
    # equal to the matrix multiplication of psi(theta, k, gamma) and chi(s) along the first 2 axis
    
    if isinstance(theta, float):
        theta = np.array(theta)
    
    s = -1 * (s > 0) + (s <= 0)
    
    s = np.array(s, copy=False).reshape(-1, 1)
    k = np.array(k, copy=False).reshape(-1, 1)
    gamma = np.array(gamma, copy=False).reshape(-1, 1)
    theta = theta.reshape(1, -1)
    
    x = k * theta + gamma
    
    cos, sin = np.cos(x), np.sin(x)
    
    if out is None:
        out = np.empty((2, 2, x.shape[0], x.shape[-1]))
    
    out[0, 0, ...] = cos
    out[0, 1, ...] = -s*sin
    out[1, 0, ...] = sin
    out[1, 1, ...] = s*cos
    
    return out


if __name__ == "__main__":

    def test(N, b, MF):
        fs = []
        for f in range(-MF, MF+1):
            if (f-b) % N == 0:
                fs.append(f)
            
        fs2 = [b + j*N for j in offset_iterator(b, N, None, MF)]
        
        assert sorted(fs) == sorted(fs2), (N, b, MF, fs, fs2)


    for N in range(1, 30):
        for b in range(N):
            for MF in range(1, 40):
                test(N, b, MF)
    
    #points = np.array([[0., 0.], [0., 1.], [1., 1.], [-1, 0.]]).T
    points = np.random.randn(2, 50)
    angles = np.arctan2(points[1, :], points[0, :])
    gamma = np.array([0., 0., np.pi / 2])
    k = np.array([1, 2, 4])
    s = np.array([1, 0, 0])
    
    p = psi(angles, k, gamma).transpose(2, 3, 0, 1)
    c = chi(s).transpose(2, 3, 0, 1)
    pc = psichi(angles, s, k, gamma).transpose(2, 3, 0, 1)
    
    assert np.allclose(p@c, pc)

