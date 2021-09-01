import numpy as np

from e2cnn.group import *
from e2cnn.diffops import *
from e2cnn.kernels import EmptyBasisException
from e2cnn.diffops.utils import required_points
from e2cnn.diffops import DiscretizationArgs

import unittest
from unittest import TestCase


disc = DiscretizationArgs(smoothing=1, method="gauss")

def make_grid(n):
    x = np.arange(-n, n + 1)
    return np.stack(np.meshgrid(x, -x)).reshape(2, -1)


class Test_Gauss_Equivariance(TestCase):

    def check_quarter_rotations(self, basis, points, elements, in_rep, out_rep):
        if basis is None:
            print("Empty DiffopBasis!")
            return
        print(basis, in_rep, out_rep)

        P = points.shape[1]
        B = 100

        features = np.random.randn(B, in_rep.size, P)

        filters = basis.sample(points)
        self.assertFalse(np.isnan(filters).any())
        self.assertFalse(np.allclose(filters, np.zeros_like(filters)))

        a = basis.sample(points)
        b = basis.sample(points)
        # if not np.allclose(a, b):
        #     print(basis.dim)
        #     print(a.reshape(-1, basis.dim, P))
        #     print(b.reshape(-1, basis.dim, P))
        #     print(np.abs(a - b).max())
        #     print(np.abs(a - b).mean())
        #     print(f"{group.name}, {in_rep.name}, {out_rep.name} \n\n\n\n")

        self.assertTrue(np.allclose(a, b))

        output = np.einsum("oifp,bip->bof", filters, features)

        for k, g in enumerate(elements):

            output1 = np.einsum("oi,bif->bof", out_rep(g), output)

            # We want to evaluate the filters at the rotated points.
            # But evaluation at arbitrary points is not supported for FD, so instead we rotate
            # the filters in the opposite direction (that's why we use -k)
            size = int(np.sqrt(P))
            transformed_filters = np.rot90(filters.reshape(filters.shape[:3] + (size, size)), -k, (-2, -1)).reshape(filters.shape)

            transformed_features = np.einsum("oi,bip->bop", in_rep(g), features)
            output2 = np.einsum("oifp,bip->bof", transformed_filters, transformed_features)

            if not np.allclose(output1, output2):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")

                aerr = np.abs(output1 - output2)
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(
                np.allclose(output1, output2),
                f"{in_rep.name} - {out_rep.name}, element {g}"
            )

    def test_so2_irreps(self):
        group = so2_group(2)

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                basis = diffops_SO2_act_R2(in_rep, out_rep, max_power=1, discretization=disc)
                # depending on the order in which tests are run, more than irreps up
                # to n = 2 may already have been built. But we want to skip any
                # bases with order > 6 (because we'd need larger grids to discretize
                # them)
                if basis.maximum_order > 6:
                    continue
                size = required_points(6, 2) // 2
                points = make_grid(size)
                self.check_quarter_rotations(basis, points, [0., np.pi/2, np.pi, 3*np.pi/2], in_rep, out_rep)

    def test_o2_irreps(self):
        group = o2_group(2)

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                try:
                    basis = diffops_O2_act_R2(in_rep, out_rep, max_power=1, axis=np.pi/2, discretization=disc)
                    # depending on the order in which tests are run, more than irreps up
                    # to n = 2 may already have been built. But we want to skip any
                    # bases with order > 6 (because we'd need larger grids to discretize
                    # them)
                    if basis.maximum_order > 6:
                        continue
                    size = required_points(6, 2) // 2
                    points = make_grid(size)
                    self.check_quarter_rotations(basis, points, [(0, 0.), (0, np.pi/2), (0, np.pi), (0, 3*np.pi/2)], in_rep, out_rep)
                except EmptyBasisException:
                    pass

    def test_cyclic_even_regular(self):
        for N in [4, 8, 12]:
            group = cyclic_group(N)
            in_rep = group.regular_representation
            out_rep = group.regular_representation

            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=4,
                                      discretization=disc)
            size = required_points(6, 2) // 2
            points = make_grid(size)
            self.check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

    def test_cyclic_mix(self):
        for N in [4, 8, 12]:
            group = cyclic_group(N)
            in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
            out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=4,
                                      discretization=disc)
            size = required_points(6, 2) // 2
            points = make_grid(size)
            self.check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

    def test_cyclic_changeofbasis(self):
        for N in [4, 8, 12]:
            group = cyclic_group(N)
            in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
            out_rep = group.regular_representation

            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=4,
                                      discretization=disc)
            size = required_points(6, 2) // 2
            points = make_grid(size)
            self.check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

            in_rep = group.regular_representation
            out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

            basis = diffops_CN_act_R2(in_rep, out_rep,
                                      max_power=1,
                                      max_frequency=4)
            self.check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

    def test_cyclic_irreps(self):
        N = 8
        group = cyclic_group(N)

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                try:
                    basis = diffops_CN_act_R2(in_rep, out_rep,
                                              max_power=1,
                                              max_frequency=4,
                                              discretization=disc)
                    size = required_points(6, 2) // 2
                    points = make_grid(size)
                    self.check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)
                except EmptyBasisException:
                    continue


if __name__ == '__main__':
    unittest.main()
