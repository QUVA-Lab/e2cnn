
import numpy as np
import math

from e2cnn.kernels.utils import offset_iterator

from e2cnn.group import Group, IrreducibleRepresentation
from e2cnn.group import cyclic_group, dihedral_group, so2_group, o2_group
from e2cnn.group import CyclicGroup, DihedralGroup, SO2, O2

from .utils import homogenized_cheby, transform_polynomial
from .basis import DiffopBasis, DiscretizationArgs

from typing import Union, Tuple, Optional


class R2FlipsSolution(DiffopBasis):
    
    def __init__(self,
                 group: Group,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 axis: float,
                 max_frequency: int = None,
                 max_offset: int = None,
                 discretization: DiscretizationArgs = DiscretizationArgs(),
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
        self.mu = []

        self.invert = (self.fi + self.fo) % 2
        
        # for each available frequency offset, build the corresponding basis vector
        for t in offset_iterator(0, 1, self.max_offset, self.max_frequency, non_negative=True):
            
            # the current shifted frequency
            mu = t
            
            if self.max_offset is not None:
                assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
            
            if self.max_frequency is not None:
                assert (math.fabs(mu) <= self.max_frequency), (t, mu, self.max_frequency)
            
            if mu > 0 or self.invert == 0.:
                # don't add sin(0*theta) as a basis since it is zero everywhere
                self.mu.append(mu)
                self.ts.append(t)
        
        self.dim = len(self.mu)
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        # would be set later anyway but we need it now
        self.shape = (out_irrep.size, in_irrep.size)

        coefficients = []

        if self.shape[0] == 1 and self.shape[1] == 1:
            for i in range(self.dim):
                mu = self.mu[i]
                out = homogenized_cheby(mu, "u" if self.invert else "t").reshape(1, 1, -1)
                coefficients.append(out)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")

        if axis != 0:
            so2 = SO2(1)
            # rotation matrix by angle_offset
            matrix = so2.irrep(1)(axis)
            # we transform the polynomial with the matrix
            coefficients = [transform_polynomial(element, matrix) for element in coefficients]

        super().__init__(coefficients, discretization)
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx]
        attr["order"] = abs(self.mu[idx])
        attr["invert"] = self.invert
        attr["offset"] = self.ts[idx]
        attr["idx"] = idx
        return attr

    def __eq__(self, other):
        if not isinstance(other, R2FlipsSolution):
            return False
        elif self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep or self.axis != other.axis:
            return False
        else:
            return np.allclose(self.mu, other.mu) and self.invert == other.invert

    def __hash__(self):
        return (hash(self.in_irrep) + hash(self.out_irrep) + hash(str(self.mu)) + hash(self.invert))


class R2DiscreteRotationsSolution(DiffopBasis):

    def __init__(self,
                 group: Union[Group, int],
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 max_frequency: int = None,
                 max_offset: int = None,
                 discretization: DiscretizationArgs = DiscretizationArgs(),
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
        self.invert = []
        self.mu = []

        if in_irrep.size == 2 and out_irrep.size == 2:
            self.s = []
            # m, n > 0
            for invert in range(2):
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

                        self.invert.append(invert)
                        self.mu.append(mu)
                        self.s.append(s)
                        self.ts.append(t)

        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert (self.m == 0 or (self.m == self.N//2 and self.N % 2 == 0))
            # n > 0, m = 0 or N/2

            for invert in range(2):

                k = self.n + self.m

                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):

                    # the current shifted frequency
                    mu = k + t * self.N

                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)

                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)

                    self.invert.append(invert)
                    self.mu.append(mu)
                    self.ts.append(t)

        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0))
            # m > 0, n = 0 or N/2

            for invert in range(2):

                k = self.n + self.m

                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):

                    # the current shifted frequency
                    mu = k + t * self.N

                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)

                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)

                    self.invert.append(invert)
                    self.mu.append(mu)
                    self.ts.append(t)

        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0))
            assert (self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0))

            for invert in range(2):

                k = self.m - self.n

                # for each available frequency offset, build the corresponding basis vector
                for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency, non_negative=True):

                    # the current shifted frequency
                    mu = k + t * self.N

                    if self.max_offset is not None:
                        assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)

                    if self.max_frequency is not None:
                        assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)

                    if mu > 0 or invert == 0.:
                        # don't add sin(0*theta) as a basis since it is zero everywhere
                        self.invert.append(invert)
                        self.mu.append(mu)
                        self.ts.append(t)

        self.dim = len(self.invert)
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        # would be set later anyway but we need it now
        self.shape = (out_irrep.size, in_irrep.size)

        coefficients = []

        if self.shape[0] == 2 and self.shape[1] == 2:
            for i in range(self.dim):
                invert = self.invert[i]
                s = self.s[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, invert)
                out[0, 1, :] = -(-1)**s * cheby("u", mu, invert)
                out[1, 0, :] = cheby("u", mu, invert)
                out[1, 1, :] = (-1)**s * cheby("t", mu, invert)
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 2:
            for i in range(self.dim):
                invert = self.invert[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**invert * homogenized_cheby(mu, "u" if invert else "t")
                out[0, 1, :] = homogenized_cheby(mu, "t" if invert else "u")
                coefficients.append(out)

        elif self.shape[0] == 2 and self.shape[1] == 1:
            for i in range(self.dim):
                invert = self.invert[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**invert * homogenized_cheby(mu, "u" if invert else "t")
                out[1, 0, :] = homogenized_cheby(mu, "t" if invert else "u")
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 1:
            for i in range(self.dim):
                invert = self.invert[i]
                mu = self.mu[i]
                out = homogenized_cheby(mu, "u" if invert else "t").reshape(1, 1, -1)
                coefficients.append(out)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")

        super().__init__(coefficients, discretization)


    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx]
        attr["order"] = abs(self.mu[idx])
        attr["invert"] = self.invert[idx]
        if hasattr(self, "s"):
            attr["s"] = self.s[idx]

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
            return np.allclose(self.mu, other.mu) and self.invert == other.invert

    def __hash__(self):
        return (hash(self.in_irrep) + hash(self.out_irrep) + hash(str(self.mu)) + hash(str(self.invert)))


class R2FlipsDiscreteRotationsSolution(DiffopBasis):

    def __init__(self,
                 group: Union[Group, int],
                 in_irrep: Union[str, IrreducibleRepresentation, Tuple[int]],
                 out_irrep: Union[str, IrreducibleRepresentation, Tuple[int, int]],
                 axis: float,
                 max_frequency: int = None,
                 max_offset: int = None,
                 discretization: DiscretizationArgs = DiscretizationArgs(),
                 ):

        if isinstance(group, int):
            group = dihedral_group(group)
        
        assert isinstance(group, DihedralGroup)

        assert (max_frequency is not None or max_offset is not None), \
            'Error! Either the maximum frequency or the maximum offset for the frequencies must be set'

        self.max_frequency = max_frequency
        self.max_offset = max_offset

        assert isinstance(axis, float)
        self.axis = axis

        assert max_frequency is None or (isinstance(max_frequency, int) and max_frequency >= 0)
        assert max_offset is None or (isinstance(max_offset, int) and max_offset >= 0)

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
        self.mu = []

        if in_irrep.size == 2 and out_irrep.size == 2:
            assert (self.m > 0 and self.n > 0 and self.fi == 1 and self.fo == 1)
            self.s = []
            # m, n > 0
            self.invert = 0
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
                     
                     self.mu.append(mu)
                     self.s.append(s)
                     self.ts.append(t)

        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert ((self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0)) and (self.fi == 1))
            # n > 0, m = 0 or N/2
            
            self.invert = self.fo
            
            k = self.n + self.m
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                self.mu.append(mu)
                self.ts.append(t)
        
        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert ((self.n == 0 or (self.n == self.N// 2 and self.N % 2 == 0)) and self.fo == 1), (self.n, self.m, self.N, self.fi, self.fo)
            # m > 0, n = 0 or N/2
            
            self.invert = self.fi
            
            k = self.n + self.m
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                self.mu.append(mu)
                self.ts.append(t)
            
        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert (self.n == 0 or (self.n == self.N // 2 and self.N % 2 == 0)), (self.n, self.m, self.N, self.fi, self.fo)
            assert (self.m == 0 or (self.m == self.N // 2 and self.N % 2 == 0)), (self.n, self.m, self.N, self.fi, self.fo)
            
            self.invert = ((self.fi + self.fo) % 2)
            
            k = self.m - self.n
            
            # for each available frequency offset, build the corresponding basis vector
            for t in offset_iterator(k, self.N, self.max_offset, self.max_frequency, non_negative=True):
                
                # the current shifted frequency
                mu = k + t * self.N
                
                if self.max_offset is not None:
                    assert (math.fabs(t) <= self.max_offset), (t, self.max_offset)
                
                if self.max_frequency is not None:
                    assert (math.fabs(mu) <= self.max_frequency), (k, t, mu, self.max_frequency)
                
                if mu > 0 or self.invert == 0:
                    # don't add sin(0*theta) as a basis since it is zero everywhere
                    self.mu.append(mu)
                    self.ts.append(t)
            
        self.dim = len(self.mu)
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        # would be set later anyway but we need it now
        self.shape = (out_irrep.size, in_irrep.size)

        coefficients = []

        if self.shape[0] == 2 and self.shape[1] == 2:
            for i in range(self.dim):
                s = self.s[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, self.invert)
                out[0, 1, :] = -(-1)**s * cheby("u", mu, self.invert)
                out[1, 0, :] = cheby("u", mu, self.invert)
                out[1, 1, :] = (-1)**s * cheby("t", mu, self.invert)
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 2:
            for i in range(self.dim):
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**self.invert * homogenized_cheby(mu, "u" if self.invert else "t")
                out[0, 1, :] = homogenized_cheby(mu, "t" if self.invert else "u")
                coefficients.append(out)

        elif self.shape[0] == 2 and self.shape[1] == 1:
            for i in range(self.dim):
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**self.invert * homogenized_cheby(mu, "u" if self.invert else "t")
                out[1, 0, :] = homogenized_cheby(mu, "t" if self.invert else "u")
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 1:
            for i in range(self.dim):
                mu = self.mu[i]
                out = homogenized_cheby(mu, "u" if self.invert else "t").reshape(1, 1, -1)
                coefficients.append(out)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")

        if axis != 0:
            so2 = SO2(1)
            # rotation matrix by angle_offset
            matrix = so2.irrep(1)(axis)
            # we transform the polynomial with the matrix
            coefficients = [transform_polynomial(element, matrix) for element in coefficients]

        super().__init__(coefficients, discretization)


    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx]
        attr["order"] = abs(self.mu[idx])
        attr["invert"] = self.invert
        if hasattr(self, "s"):
            attr["s"] = self.s[idx]

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
            return np.allclose(self.mu, other.mu) and self.invert == other.invert

    def __hash__(self):
        return (hash(self.in_irrep) + hash(self.out_irrep) + hash(str(self.mu)) + hash(self.invert))

class R2ContinuousRotationsSolution(DiffopBasis):
    def __init__(
        self,
        group: Group,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        discretization: DiscretizationArgs = DiscretizationArgs(),
    ):

        assert isinstance(group, SO2)

        if isinstance(in_irrep, int):
            in_irrep = group.irrep(in_irrep)
        elif isinstance(in_irrep, str):
            in_irrep = group.irreps[in_irrep]
        elif not isinstance(in_irrep, IrreducibleRepresentation):
            raise ValueError(
                f"'in_irrep' should be a non-negative integer, a string or an instance"
                f" of IrreducibleRepresentation but {in_irrep} found"
            )

        self.n = in_irrep.attributes["frequency"]

        if isinstance(out_irrep, int):
            out_irrep = group.irrep(out_irrep)
        elif isinstance(out_irrep, str):
            out_irrep = group.irreps[out_irrep]
        elif not isinstance(out_irrep, IrreducibleRepresentation):
            raise ValueError(
                f"'out_irrep' should be a non-negative integer, a string or an instance"
                f" of IrreducibleRepresentation but {in_irrep} found"
            )

        self.m = out_irrep.attributes["frequency"]

        self.invert = []
        self.mu = []

        if in_irrep.size == 2 and out_irrep.size == 2:
            # m, n > 0
            ss = []
            for invert in range(2):
                for s in [0, 1]:
                    mu = self.m - self.n * (-1) ** s

                    self.invert.append(invert)
                    self.mu.append(mu)
                    ss.append(s)
            self.s = np.array(ss)

        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert self.m == 0
            # n > 0, m = 0

            for invert in range(2):
                mu = self.n + self.m

                self.invert.append(invert)
                self.mu.append(mu)

        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert self.n == 0
            # m > 0, n = 0

            for invert in range(2):
                mu = self.n + self.m

                self.invert.append(invert)
                self.mu.append(mu)

        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert self.n == 0 and self.m == 0
            self.mu.append(0)
            self.invert.append(0)

        self.dim = len(self.invert)
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        # would be set later anyway but we need it now
        self.shape = (out_irrep.size, in_irrep.size)

        coefficients = []

        if self.shape[0] == 2 and self.shape[1] == 2:
            for i in range(self.dim):
                invert = self.invert[i]
                s = self.s[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, invert)
                out[0, 1, :] = -(-1)**s * cheby("u", mu, invert)
                out[1, 0, :] = cheby("u", mu, invert)
                out[1, 1, :] = (-1)**s * cheby("t", mu, invert)
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 2:
            for i in range(self.dim):
                invert = self.invert[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, invert)
                out[0, 1, :] = cheby("u", mu, invert)
                coefficients.append(out)

        elif self.shape[0] == 2 and self.shape[1] == 1:
            for i in range(self.dim):
                invert = self.invert[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, invert)
                out[1, 0, :] = cheby("u", mu, invert)
                coefficients.append(out)

        elif self.shape[0] == 1 and self.shape[1] == 1:
            out = np.array([1]).reshape(1, 1, 1)
            coefficients.append(out)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")

        super().__init__(coefficients, discretization)

    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx]
        attr["order"] = abs(self.mu[idx])
        attr["invert"] = self.invert[idx]
        if hasattr(self, "s"):
            attr["s"] = self.s[idx]

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
            return np.allclose(self.mu, other.mu) and self.invert == other.invert

    def __hash__(self):
        return (hash(self.in_irrep) + hash(self.out_irrep) + hash(str(self.mu)) + hash(str(self.invert)))


class R2FlipsContinuousRotationsSolution(DiffopBasis):
    
    def __init__(self,
                 group: Group,
                 in_irrep: Union[str, IrreducibleRepresentation, Tuple[int]],
                 out_irrep: Union[str, IrreducibleRepresentation, Tuple[int, int]],
                 axis: float = 0.,
                 discretization: DiscretizationArgs = DiscretizationArgs(),
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

        self.mu = []
        
        if in_irrep.size == 2 and out_irrep.size == 2:
            assert (self.m > 0 and self.n > 0 and self.fi == 1 and self.fo == 1)
            self.s = []
            # m, n > 0
            
            self.invert = 0
            for s in [0, 1]:
                mu = self.m - self.n * (-1) ** s
                
                self.mu.append(mu)
                self.s.append(s)
        
        elif in_irrep.size == 2 and out_irrep.size == 1:
            assert self.m == 0 and self.fi == 1
            # n > 0, m = 0
            
            self.invert = self.fo
            
            mu = self.n + self.m
            self.mu.append(mu)
        
        elif in_irrep.size == 1 and out_irrep.size == 2:
            assert self.n == 0 and self.fo == 1
            # m > 0, n = 0
            
            self.invert = self.fi
            
            mu = self.n + self.m
            self.mu.append(mu)
            
        elif in_irrep.size == 1 and out_irrep.size == 1:
            assert self.n == 0 and self.m == 0
            
            self.invert = ((self.fi + self.fo) % 2)
            
            mu = self.m - self.n
            if mu > 0 or self.invert == 0:
                # don't add sin(0*theta) as a basis since it is zero everywhere
                self.mu.append(mu)
        
        self.dim = len(self.mu)
        self.group = group
        self.in_irrep = in_irrep
        self.out_irrep = out_irrep
        # would be set later anyway but we need it now
        self.shape = (out_irrep.size, in_irrep.size)

        coefficients = []
    
        # the basis vectors depends on the shape of the input and output irreps,
        # while their frequencies depend on the irreps frequencies
        if self.shape[0] == 2 and self.shape[1] == 2:
            for i in range(self.dim):
                s = self.s[i]
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = cheby("t", mu, self.invert)
                out[0, 1, :] = -(-1)**s * cheby("u", mu, self.invert)
                out[1, 0, :] = cheby("u", mu, self.invert)
                out[1, 1, :] = (-1)**s * cheby("t", mu, self.invert)
                coefficients.append(out)
        
        elif self.shape[0] == 1 and self.shape[1] == 2:
            for i in range(self.dim):
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**self.invert * homogenized_cheby(mu, "u" if self.invert else "t")
                out[0, 1, :] = homogenized_cheby(mu, "t" if self.invert else "u")
                coefficients.append(out)
        
        elif self.shape[0] == 2 and self.shape[1] == 1:
            for i in range(self.dim):
                mu = self.mu[i]
                out = np.empty((self.shape) + (abs(mu) + 1,))
                out[0, 0, :] = (-1)**self.invert * homogenized_cheby(mu, "u" if self.invert else "t")
                out[1, 0, :] = homogenized_cheby(mu, "t" if self.invert else "u")
                coefficients.append(out)
        
        elif self.shape[0] == 1 and self.shape[1] == 1:
            for i in range(self.dim):
                mu = self.mu[i]
                out = homogenized_cheby(mu, "u" if self.invert else "t").reshape(1, 1, -1)
                coefficients.append(out)
        else:
            raise ValueError(f"Shape {self.shape} not recognized!")
        
        if axis != 0:
            so2 = SO2(1)
            # rotation matrix by angle_offset
            matrix = so2.irrep(1)(axis)
            # we transform the polynomial with the matrix
            coefficients = [transform_polynomial(element, matrix) for element in coefficients]

        super().__init__(coefficients, discretization)
    
    def __getitem__(self, idx):
        assert idx < self.dim
        attr = {}
        attr["frequency"] = self.mu[idx]
        attr["order"] = abs(self.mu[idx])
        attr["invert"] = self.invert
        if hasattr(self, "s"):
            attr["s"] = self.s[idx]

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
            return np.allclose(self.mu, other.mu) and self.invert == other.invert

    def __hash__(self):
        return (hash(self.in_irrep) + hash(self.out_irrep) + hash(str(self.mu)) + hash(self.invert))


def cheby(kind, mu, invert):
    inverter = {"u": "t", "t": "u"}
    if kind == "t" and invert:
        sign = -1
    else:
        sign = 1

    return sign * homogenized_cheby(mu, inverter[kind] if invert else kind)
