
import numpy as np
import math

from e2cnn.kernels.basis import KernelBasis
from e2cnn.kernels.utils import offset_iterator, psi, psichi, chi

from e2cnn.group import Group, IrreducibleRepresentation
from e2cnn.group import cyclic_group, dihedral_group, so2_group, o2_group
from e2cnn.group import CyclicGroup, DihedralGroup, SO2, O2

from typing import Union, Tuple


class IrrepBasis(KernelBasis):
    
    def __init__(self, group: Group, in_irrep: IrreducibleRepresentation, out_irrep: IrreducibleRepresentation, dim: int):
        r"""
        
        Abstract class for bases implementing the kernel constraint solutions associated to irreducible input and output
        representations.
        
        Args:
            group:
            in_irrep:
            out_irrep:
            dim:
        """
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        
        super(IrrepBasis, self).__init__(dim, (out_irrep.size, in_irrep.size))


class R2FlipsSolution(IrrepBasis):
    
    def __init__(self,
                 group: Group,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 axis: float,
                 max_frequency: int = None,
                 max_offset: int = None,
                 ):
        
        if isinstance(group, int):
            group = cyclic_group(2)
        
        assert isinstance(group, CyclicGroup) and group.order() == 2
        
        assert (max_frequency is not None or max_offset is not None), \
            'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'
        
        self.max_frequency = max_frequency
        self.max_offset = max_offset
        
        assert max_frequency is None or (isinstance(max_frequency, int) and max_frequency >= 0)
        assert max_offset is None or (isinstance(max_offset, int) and max_offset >= 0)
        
        assert isinstance(axis, float)
        self.axis = axis
        
        if isinstance(in_irrep, int):
            in_irrep = group.irrep(in_irrep)
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(f"'in_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        if isinstance(out_irrep, int):
            out_irrep = group.irrep(out_irrep)
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(f"'out_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.N = 1
        
        self.fi = in_irrep.attributes['frequency']
        self.fo = out_irrep.attributes['frequency']
        
        self.ts = []

        self.gamma = ((self.fi + self.fo) % 2) * np.pi / 2
        mus = []
        
        # for each available frequency offset, build the corresponding basis vector
        for t in offset_iterator(0, 1, self.max_offset, self.max_frequency, non_negative=True):
            
            # the current shifted frequency
            mu = t
            
            if self.max_offset is not None:
                assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
            
            if self.max_frequency is not None:
                assert (math.fabs(mu) <= self.max_frequency), (t, mu, self.max_frequency)
            
            if mu > 0 or self.gamma == 0.:
                # don't add sin(0*theta) as a basis since it is zero everywhere
                mus.append(mu)
                self.ts.append(t)
        
        self.mu = np.array(mus).reshape(-1, 1)
        
        self._non_zero_frequencies = self.mu != 0
        self._has_non_zero_frequencies = np.any(self._non_zero_frequencies)
        
        dim = self.mu.shape[0]
        super(R2FlipsSolution, self).__init__(group, in_irrep, out_irrep, dim)
    
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
            out = np.empty((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])
        
        # find points in the origin
        origin = np.isnan(angles)
        angles = angles.copy()
        angles[origin] = 0.
        
        angles -= self.axis
        
        if self.shape[0] == 1 and self.shape[1] == 1:
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if self._has_non_zero_frequencies:
            # In the origin, only 0-frequencies are permitted.
            # Therefore, any non-0 frequency base is set to 0 in the origin
            
            if np.any(origin):
                mask = self._non_zero_frequencies * origin
                out *= 1 - mask
                
        assert not np.isnan(out).any()

        return out
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx, 0]
        attr["gamma"] = self.gamma
        attr["offset"] = self.ts[idx]
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2FlipsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep or self.axis != other.axis:
            return False
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.gamma, other.gamma)

    def __hash__(self):
        return hash(self.in_irrep) + hash(self.out_irrep) + hash(self.mu.tostring()) + hash(self.gamma)


class R2DiscreteRotationsSolution(IrrepBasis):
    
    def __init__(self,
                 group: Union[Group, int],
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 max_frequency: int = None,
                 max_offset: int = None,
                 ):
        
        if isinstance(group, int):
            group = cyclic_group(group)
        
        assert isinstance(group, CyclicGroup)
        
        assert (max_frequency is not None or max_offset is not None), \
            'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'

        self.max_frequency = max_frequency
        self.max_offset = max_offset

        assert max_frequency is None or (isinstance(max_frequency, int) and max_frequency >= 0)
        assert max_offset is None or (isinstance(max_offset, int) and max_offset >= 0)
        
        if isinstance(in_irrep, int):
            in_irrep = group.irrep(in_irrep)
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(f"'in_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.n = in_irrep.attributes['frequency']

        if isinstance(out_irrep, int):
            out_irrep = group.irrep(out_irrep)
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(f"'out_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")

        self.m = out_irrep.attributes['frequency']
        self.N = group.order()

        self.ts = []
        
        if in_irrep.size == 2 and out_irrep.size == 2:
            # m, n > 0
            gammas = []
            mus = []
            ss = []
            for gamma in [0., np.pi / 2]:
                for s in [0, 1]:
                    k = self.m - self.n * (-1) ** s
            
                    # for each available frequency offset, build the corresponding basis vector
                    for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                
                        # the current shifted frequency
                        mu = k + t * self.N
                
                        if self.max_offset is not None:
                            assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                        if self.max_frequency is not None:
                            assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                        
                        gammas.append(gamma)
                        mus.append(mu)
                        ss.append(s)
                        self.ts.append(t)
                        
            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
            self.s = np.array(ss).reshape(-1, 1)

        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert (self.m == 0 or (self.m == self.N//2 and self.N % 2 == 0))
            # n > 0, m = 0 or N/2

            gammas = []
            mus = []
            
            for gamma in [0., np.pi / 2]:
        
                k = self.n + self.m
        
                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
            
                    # the current shifted frequency
                    mu = k + t * self.N

                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)

                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                    
                    gammas.append(gamma)
                    mus.append(mu)
                    self.ts.append(t)

            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)

        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0))
            # m > 0, n = 0 or N/2

            gammas = []
            mus = []

            for gamma in [0., np.pi / 2]:
    
                k = self.n + self.m
    
                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
        
                    # the current shifted frequency
                    mu = k + t * self.N
        
                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
        
                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
        
                    gammas.append(gamma)
                    mus.append(mu)
                    self.ts.append(t)

            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)

        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0))
            assert (self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0))

            gammas = []
            mus = []

            for gamma in [0., np.pi / 2]:
        
                k = self.m - self.n
        
                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency, non_negative=True):

                    # the current shifted frequency
                    mu = k + t * self.N

                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)

                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)

                    if mu > 0 or gamma == 0.:
                        # don't add sin(0*theta) as a basis since it is zero everywhere
                        gammas.append(gamma)
                        mus.append(mu)
                        self.ts.append(t)

            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
        
        self._non_zero_frequencies = self.mu != 0
        self._has_non_zero_frequencies = np.any(self._non_zero_frequencies)
        
        dim = self.gamma.shape[0]
        super(R2DiscreteRotationsSolution, self).__init__(group, in_irrep, out_irrep, dim)

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
            out = np.empty((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
    
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])

        # find points in the origin
        origin = np.isnan(angles)
        angles = angles.copy()
        angles[origin] = 0.

        # the basis vectors depends on the shape of the input and output irreps,
        # while their frequencies depend on the irreps frequencies
        if self.shape[0] == 2 and self.shape[1] == 2:
            out = psichi(angles, s=self.s, k=self.mu, gamma=self.gamma, out=out)

        elif self.shape[0] == 1 and self.shape[1] == 2:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[0, 1, ...] = np.sin(self.mu * angles + self.gamma)

        elif self.shape[0] == 2 and self.shape[1] == 1:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[1, 0, ...] = np.sin(self.mu * angles + self.gamma)
            
        elif self.shape[0] == 1 and self.shape[1] == 1:
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if self._has_non_zero_frequencies:
            # In the origin, only 0-frequencies are permitted.
            # Therefore, any non-0 frequency base is set to 0 in the origin
            
            if np.any(origin):
                mask = self._non_zero_frequencies * origin
                out *= 1 - mask

        return out

    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx, 0]
        attr["gamma"] = self.gamma[idx, 0]
        if hasattr(self, "s"):
            attr["s"] = self.s[idx, 0]

        attr["offset"] = self.ts[idx]
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2DiscreteRotationsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            return False
        elif hasattr(self, "s") and not np.allclose(self.s, other.s):
            return False
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.gamma, other.gamma)

    def __hash__(self):
        return hash(self.in_irrep) + hash(self.out_irrep) + hash(self.mu.tobytes()) + hash(self.gamma.tobytes())


class R2ContinuousRotationsSolution(IrrepBasis):
    
    def __init__(self,
                 group: Group,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 ):
        
        assert isinstance(group, SO2)
        
        if isinstance(in_irrep, int):
            in_irrep = group.irrep(in_irrep)
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(f"'in_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.n = in_irrep.attributes['frequency']
        
        if isinstance(out_irrep, int):
            out_irrep = group.irrep(out_irrep)
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(f"'out_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.m = out_irrep.attributes['frequency']
        
        if in_irrep.size == 2 and out_irrep.size == 2:
            # m, n > 0
            gammas = []
            mus = []
            ss = []
            for gamma in [0., np.pi / 2]:
                for s in [0, 1]:
                    mu = self.m - self.n * (-1) ** s
                    
                    gammas.append(gamma)
                    mus.append(mu)
                    ss.append(s)
            
            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
            self.s = np.array(ss).reshape(-1, 1)
        
        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert self.m == 0
            # n > 0, m = 0
            
            gammas = []
            mus = []
            
            for gamma in [0., np.pi / 2]:
                mu = self.n + self.m
                
                gammas.append(gamma)
                mus.append(mu)
            
            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert self.n == 0
            # m > 0, n = 0
            
            gammas = []
            mus = []
            
            for gamma in [0., np.pi / 2]:
                mu = self.n + self.m
                
                gammas.append(gamma)
                mus.append(mu)
            
            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert self.n == 0 and self.m == 0
            
            gammas = []
            mus = []
            
            for gamma in [0., np.pi / 2]:
                
                mu = self.m - self.n
                
                if mu > 0 or gamma == 0.:
                    # don't add sin(0*theta) as a basis since it is zero everywhere
                    gammas.append(gamma)
                    mus.append(mu)
            
            self.gamma = np.array(gammas).reshape(-1, 1)
            self.mu = np.array(mus).reshape(-1, 1)
        
        self._non_zero_frequencies = self.mu != 0
        self._has_non_zero_frequencies = np.any(self._non_zero_frequencies)
        
        dim = self.gamma.shape[0]
        super(R2ContinuousRotationsSolution, self).__init__(group, in_irrep, out_irrep, dim)
    
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
            out = np.empty((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])

        # find points in the origin
        origin = np.isnan(angles)
        angles = angles.copy()
        angles[origin] = 0.

        # the basis vectors depends on the shape of the input and output irreps,
        # while their frequencies depend on the irreps frequencies
        if self.shape[0] == 2 and self.shape[1] == 2:
            out = psichi(angles, s=self.s, k=self.mu, gamma=self.gamma, out=out)
        
        elif self.shape[0] == 1 and self.shape[1] == 2:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[0, 1, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 2 and self.shape[1] == 1:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[1, 0, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 1 and self.shape[1] == 1:
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if self._has_non_zero_frequencies:
            # In the origin, only 0-frequencies are permitted.
            # Therefore, any non-0 frequency base is set to 0 in the origin
            
            if np.any(origin):
                mask = self._non_zero_frequencies * origin
                out *= 1 - mask

        return out
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx, 0]
        attr["gamma"] = self.gamma[idx, 0]
        if hasattr(self, "s"):
            attr["s"] = self.s[idx, 0]
            
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2ContinuousRotationsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            return False
        elif hasattr(self, "s") and not np.allclose(self.s, other.s):
            return False
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.gamma, other.gamma)

    def __hash__(self):
        return hash(self.in_irrep) + hash(self.out_irrep) + hash(self.mu.tostring()) + hash(self.gamma.tostring())


class R2FlipsDiscreteRotationsSolution(IrrepBasis):
    
    def __init__(self,
                 group: Union[Group, int],
                 in_irrep: Union[str, IrreducibleRepresentation, Tuple[int]],
                 out_irrep: Union[str, IrreducibleRepresentation, Tuple[int, int]],
                 axis: float,
                 max_frequency: int = None,
                 max_offset: int = None,
                 ):
        
        if isinstance(group, int):
            group = dihedral_group(group)
        
        assert isinstance(group, DihedralGroup)
        
        assert (max_frequency is not None or max_offset is not None), \
            'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'
        
        self.max_frequency = max_frequency
        self.max_offset = max_offset
        
        assert max_frequency is None or (isinstance(max_frequency, int) and max_frequency >= 0)
        assert max_offset is None or (isinstance(max_offset, int) and max_offset >= 0)
        
        assert isinstance(axis, float)
        self.axis = axis
        
        if isinstance(in_irrep, tuple):
            in_irrep = group.irrep(in_irrep[0], in_irrep[1])
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(f"'in_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        if isinstance(out_irrep, tuple):
            out_irrep = group.irrep(out_irrep[0], out_irrep[1])
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(f"'out_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.N = group.rotation_order

        self.m = out_irrep.attributes['frequency']
        self.n = in_irrep.attributes['frequency']

        self.fi = in_irrep.attributes['flip_frequency']
        self.fo = out_irrep.attributes['flip_frequency']

        self.ts = []
        
        if in_irrep.size == 2 and out_irrep.size == 2:
            assert (self.m > 0 and self.n > 0 and self.fi == 1 and self.fo == 1)
            # m, n > 0
            mus = []
            ss = []
            
            self.gamma = 0.
            for s in [0, 1]:
                k = self.m - self.n * (-1) ** s
                
                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                     
                     # the current shifted frequency
                     mu = k + t * self.N
                     
                     if self.max_offset is not None:
                         assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                     
                     if self.max_frequency is not None:
                         assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                     
                     mus.append(mu)
                     ss.append(s)
                     self.ts.append(t)
            
            self.mu = np.array(mus).reshape(-1, 1)
            self.s = np.array(ss).reshape(-1, 1)
        
        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert ((self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0)) and (self.fi == 1))
            # n > 0, m = 0 or N/2
            
            self.gamma = self.fo * np.pi / 2
            mus = []
            
            k = self.n + self.m
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                mus.append(mu)
                self.ts.append(t)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert ((self.n == 0 or (self.n == self.N// 2 and self.N % 2 == 0)) and self.fo == 1)
            # m > 0, n = 0 or N/2
            
            self.gamma = self.fi * np.pi/2
            mus = []
            
            k = self.n + self.m
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                mus.append(mu)
                self.ts.append(t)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0))
            assert (self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0))
            
            self.gamma = ((self.fi + self.fo) % 2) * np.pi/2
            mus = []
            
            k = self.m - self.n
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency, non_negative=True):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                if mu > 0 or self.gamma == 0.:
                    # don't add sin(0*theta) as a basis since it is zero everywhere
                    mus.append(mu)
                    self.ts.append(t)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        self._non_zero_frequencies = self.mu != 0
        self._has_non_zero_frequencies = np.any(self._non_zero_frequencies)
        
        dim = self.mu.shape[0]
        super(R2FlipsDiscreteRotationsSolution, self).__init__(group, in_irrep, out_irrep, dim)
    
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
            out = np.empty((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])

        # find points in the origin
        origin = np.isnan(angles)
        angles = angles.copy()
        angles[origin] = 0.
        
        angles -= self.axis
        
        # the basis vectors depends on the shape of the input and output irreps,
        # while their frequencies depend on the irreps frequencies
        if self.shape[0] == 2 and self.shape[1] == 2:
            out = psichi(angles, s=self.s, k=self.mu, gamma=self.gamma, out=out)
        
        elif self.shape[0] == 1 and self.shape[1] == 2:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[0, 1, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 2 and self.shape[1] == 1:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[1, 0, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 1 and self.shape[1] == 1:
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if self._has_non_zero_frequencies:
            # In the origin, only 0-frequencies are permitted.
            # Therefore, any non-0 frequency base is set to 0 in the origin
            
            if np.any(origin):
                mask = self._non_zero_frequencies * origin
                out *= 1 - mask

        return out
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx, 0]
        attr["gamma"] = self.gamma
        if hasattr(self, "s"):
            attr["s"] = self.s[idx, 0]
        
        attr["offset"] = self.ts[idx]
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2FlipsDiscreteRotationsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            return False
        elif hasattr(self, "s") and not np.allclose(self.s, other.s):
            return False
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.gamma, other.gamma)

    def __hash__(self):
        return hash(self.in_irrep) + hash(self.out_irrep) + hash(self.mu.tostring()) + hash(self.gamma)


class R2FlipsContinuousRotationsSolution(IrrepBasis):
    
    def __init__(self,
                 group: Group,
                 in_irrep: Union[str, IrreducibleRepresentation, Tuple[int]],
                 out_irrep: Union[str, IrreducibleRepresentation, Tuple[int, int]],
                 axis: float = 0.,
                 ):
        
        assert isinstance(group, O2)
        
        assert isinstance(axis, float)
        self.axis = axis
        
        if isinstance(in_irrep, tuple):
            in_irrep = group.irrep(in_irrep[0], in_irrep[1])
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(f"'in_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        if isinstance(out_irrep, tuple):
            out_irrep = group.irrep(out_irrep[0], out_irrep[1])
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(f"'out_irrep' should be a non-negative integer, a string or an instance"
                             f" of IrreducibleRepresentation but {in_irrep} found")
        
        self.m = out_irrep.attributes['frequency']
        self.n = in_irrep.attributes['frequency']
        
        self.fi = in_irrep.attributes['flip_frequency']
        self.fo = out_irrep.attributes['flip_frequency']
        
        if in_irrep.size == 2 and out_irrep.size == 2:
            assert (self.m > 0 and self.n > 0 and self.fi == 1 and self.fo == 1)
            # m, n > 0
            mus = []
            ss = []
            
            self.gamma = 0.
            for s in [0, 1]:
                mu = self.m - self.n * (-1) ** s
                
                mus.append(mu)
                ss.append(s)
            
            self.mu = np.array(mus).reshape(-1, 1)
            self.s = np.array(ss).reshape(-1, 1)
        
        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert self.m == 0 and self.fi == 1
            # n > 0, m = 0
            
            self.gamma = self.fo * np.pi / 2
            mus = []
            
            mu = self.n + self.m
            mus.append(mu)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert self.n == 0 and self.fo == 1
            # m > 0, n = 0
            
            self.gamma = self.fi * np.pi / 2
            mus = []
            
            mu = self.n + self.m
            mus.append(mu)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert self.n == 0 and self.m == 0
            
            self.gamma = ((self.fi + self.fo) % 2) * np.pi / 2
            mus = []
            
            mu = self.m - self.n
            if mu > 0 or self.gamma == 0.:
                # don't add sin(0*theta) as a basis since it is zero everywhere
                mus.append(mu)
            
            self.mu = np.array(mus).reshape(-1, 1)
        
        self._non_zero_frequencies = self.mu != 0
        self._has_non_zero_frequencies = np.any(self._non_zero_frequencies)
        
        dim = self.mu.shape[0]
        super(R2FlipsContinuousRotationsSolution, self).__init__(group, in_irrep, out_irrep, dim)
    
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
            out = np.empty((self.shape[0], self.shape[1], self.dim, angles.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, angles.shape[1])

        # find points in the origin
        origin = np.isnan(angles)
        angles = angles.copy()
        angles[origin] = 0.
        
        angles -= self.axis
        
        # the basis vectors depends on the shape of the input and output irreps,
        # while their frequencies depend on the irreps frequencies
        if self.shape[0] == 2 and self.shape[1] == 2:
            out = psichi(angles, s=self.s, k=self.mu, gamma=self.gamma, out=out)
        
        elif self.shape[0] == 1 and self.shape[1] == 2:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[0, 1, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 2 and self.shape[1] == 1:
            
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
            out[1, 0, ...] = np.sin(self.mu * angles + self.gamma)
        
        elif self.shape[0] == 1 and self.shape[1] == 1:
            out[0, 0, ...] = np.cos(self.mu * angles + self.gamma)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if self._has_non_zero_frequencies:
            # In the origin, only 0-frequencies are permitted.
            # Therefore, any non-0 frequency base is set to 0 in the origin
            
            if np.any(origin):
                mask = self._non_zero_frequencies * origin
                out *= 1 - mask

        return out
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx, 0]
        attr["gamma"] = self.gamma
        if hasattr(self, "s"):
            attr["s"] = self.s[idx, 0]
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2FlipsContinuousRotationsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            return False
        elif hasattr(self, "s") and not np.allclose(self.s, other.s):
            return False
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.gamma, other.gamma)

    def __hash__(self):
        return hash(self.in_irrep) + hash(self.out_irrep) + hash(self.mu.tostring()) + hash(self.gamma)


if __name__ == "__main__":
    
    points = np.array([[0., 0.], [0., 1.], [1., 1.], [-1, 0.]]).T
    # points = np.random.randn(2, 50)

    radii = np.sqrt((points ** 2).sum(0)).reshape(1, -1)
    angles = np.arctan2(points[1, :], points[0, :]).reshape(1, -1)
    angles[radii < 1e-9] = np.nan

    basis = R2DiscreteRotationsSolution(8, 2, 3, 3)
    
    sample = basis.sample(angles)
    
    print(sample.shape)
    
    print(sample)
    
    for i in range(basis.dim):
        print(i)
        print(basis[i])
        print(sample[:, :, i, :])
