from typing import Union
import unittest
from unittest import TestCase

import numpy as np
from numpy.core.numeric import array_equal

from e2cnn.group import *
from e2cnn.diffops import *
from e2cnn.kernels import EmptyBasisException
from e2cnn.diffops.utils import eval_polys

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


class TestSolutionsEquivariance(TestCase):

    def test_trivial(self):
        N = 1
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation

        basis = diffops_Trivial_act_R2(in_rep, out_rep,
                                       max_power=1,
                                       max_frequency=3)
        action = group.irrep(0) + group.irrep(0)
        self._check(basis, group, in_rep, out_rep, action)

    def test_flips(self):
        group = cyclic_group(2)
        # in_rep = group.regular_representation
        # out_rep = group.regular_representation
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        A = 10
        for a in range(A):
            axis = a*np.pi/A
            print(axis)
    
            basis = diffops_Flip_act_R2(in_rep, out_rep,
                                        axis=axis,
                                        max_power=1,
                                        max_frequency=3)
    
            action = directsum([group.irrep(0), group.irrep(1)], psi(axis)[..., 0, 0], f"horizontal_flip_{axis}")
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_odd_regular(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation

        for _ in range(5):
            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=3)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_even_regular(self):
        N = 6
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation

        basis = diffops_CN_act_R2(in_rep, out_rep,
                                  max_power=1,
                                  max_frequency=3)
        action = group.irrep(1)
        self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_mix(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        for _ in range(5):
            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=3)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_output_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = group.regular_representation

        for _ in range(5):
            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=3)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_input_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        for _ in range(5):
            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=3)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_odd_regular(self):
        N = 5
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi / 2

        basis = diffops_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  max_power=1,
                                  max_frequency=3)

        action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

        self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_even_regular(self):
        N = 4
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi/2

        basis = diffops_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  max_power=1,
                                  max_frequency=3)

        action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

        self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_irreps(self):
        N = 8
        group = cyclic_group(N)

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():

                try:
                    basis = diffops_CN_act_R2(in_rep, out_rep,
                                            max_power=1,
                                            max_frequency=3)
                    action = group.irrep(1)
                    self._check(basis, group, in_rep, out_rep, action)
                except EmptyBasisException:
                    continue

    def test_dihedral_irreps(self):
        N = 4
        group = dihedral_group(N)

        for axis in [0., np.pi/2, np.pi/3]:
            for in_rep in group.irreps.values():
                for out_rep in group.irreps.values():
                    basis = diffops_DN_act_R2(in_rep, out_rep,
                                            axis=axis,
                                            max_power=1,
                                            max_frequency=4)

                    action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

                    self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_2_irreps(self):
        N = 2
        group = dihedral_group(N)
        axis = np.pi / 2

        reprs = list(group.irreps.values()) + [directsum(list(group.irreps.values()), name="irreps_sum"), group.regular_representation]

        for in_rep in reprs:
            for out_rep in reprs:
                basis = diffops_DN_act_R2(in_rep, out_rep,
                                          axis=axis,
                                          max_power=1,
                                          max_frequency=3)

                action = change_basis(group.irrep(0, 1) + group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

                self._check(basis, group, in_rep, out_rep, action)

    def test_flips_irreps(self):
        group = cyclic_group(2)

        A = 10
        for axis in range(A):
            axis = axis * np.pi/A
            for in_rep in list(group.irreps.values()) + [group.regular_representation]:
                for out_rep in list(group.irreps.values()) + [group.regular_representation]:
                    basis = diffops_Flip_act_R2(in_rep, out_rep,
                                                axis=axis,
                                                max_power=1,
                                                max_frequency=4)

                    action = directsum([group.irrep(0), group.irrep(1)], psi(axis)[..., 0, 0], f"horizontal_flip_{axis}")

                    self._check(basis, group, in_rep, out_rep, action)

    def test_so2_irreps(self):

        group = so2_group(2)

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                basis = diffops_SO2_act_R2(in_rep, out_rep,
                                           max_power=1
                                           )
                action = group.irrep(1)
                self._check(basis, group, in_rep, out_rep, action)

    def test_o2_irreps(self):

        group = o2_group(10)
        axis = np.pi / 2

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                try:
                    basis = diffops_O2_act_R2(in_rep, out_rep,
                                              axis=axis,
                                              max_power=1
                                              )
                except EmptyBasisException:
                    print(f"DiffopBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")
                self._check(basis, group, in_rep, out_rep, action)

    def _check(self, basis, group, in_rep, out_rep, action):
        if basis is None:
            print("Empty DiffopBasis!")
            return

        P = 100

        points = np.random.rand(2, P)

        for g in group.testing_elements():
            a = action(g)
            if np.array_equal(a, np.eye(1)):
                transformed_points = points
            elif np.array_equal(a, -np.eye(1)):
                transformed_points = -points
            else:
                transformed_points = a @ points

            # We test the equivariance condition on the polynomials themselves,
            # rather than applying discretized kernels to features.
            output1 = eval_polys(basis.coefficients, transformed_points)
            poly2 = eval_polys(basis.coefficients, points)
            output2 = np.einsum("ij,bjkp,kl->bilp", out_rep(g), poly2, np.linalg.inv(in_rep(g)))

            if not np.allclose(output1, output2):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)

                aerr = np.abs(output1 - output2)
                err = aerr.reshape(basis.dim, -1).max(1)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])
                        print(points)
                        print(transformed_points)
                        break

            self.assertTrue(np.allclose(output1, output2), f"Group {group.name}, {in_rep.name} - {out_rep.name},\n"
                                                           f"element {g}, action {a}")
                                                           # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
