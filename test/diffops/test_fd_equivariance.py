import numpy as np

from e2cnn.group import *
from e2cnn.diffops import *
from e2cnn.kernels import EmptyBasisException
from e2cnn.diffops.utils import required_points, symmetric_points

def check_quarter_rotations(basis, points, elements, in_rep, out_rep):
    if basis is None:
        print("Empty KernelBasis!")
        return
    print(basis, in_rep, out_rep)

    P = len(points) ** 2
    B = 100

    features = np.random.randn(B, in_rep.size, P)

    filters = basis.sample(points)
    assert not np.isnan(filters).any()
    assert not np.allclose(filters, np.zeros_like(filters))

    a = basis.sample(points)
    b = basis.sample(points)
    # if not np.allclose(a, b):
    #     print(basis.dim)
    #     print(a.reshape(-1, basis.dim, P))
    #     print(b.reshape(-1, basis.dim, P))
    #     print(np.abs(a - b).max())
    #     print(np.abs(a - b).mean())
    #     print(f"{group.name}, {in_rep.name}, {out_rep.name} \n\n\n\n")

    assert np.allclose(a, b)

    output = np.einsum("oifp,bip->bof", filters, features)

    for k, g in enumerate(elements):

        output1 = np.einsum("oi,bif->bof", out_rep(g), output)

        # We want to evaluate the filters at the rotated points.
        # But evaluation at arbitrary points is not supported for FD, so instead we rotate
        # the filters in the opposite direction (that's why we use -k)
        transformed_filters = np.rot90(filters.reshape(filters.shape[:3] + (len(points), len(points))), -k, (-2, -1)).reshape(filters.shape)

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

        assert np.allclose(output1, output2), f"{in_rep.name} - {out_rep.name}, element {g}"

def test_so2_irreps():
    group = so2_group(2)

    for in_rep in group.irreps.values():
        for out_rep in group.irreps.values():
            basis = diffops_SO2_act_R2(in_rep, out_rep, max_power=1)
            size = required_points(6, 2)
            points = symmetric_points(size)
            check_quarter_rotations(basis, points, [0., np.pi/2, np.pi, 3*np.pi/2], in_rep, out_rep)

def test_o2_irreps():
    group = o2_group(2)

    for in_rep in group.irreps.values():
        for out_rep in group.irreps.values():
            try:
                basis = diffops_O2_act_R2(in_rep, out_rep, max_power=1, axis=np.pi/2)
                size = required_points(6, 2)
                points = symmetric_points(size)
                check_quarter_rotations(basis, points, [(0, 0.), (0, np.pi/2), (0, np.pi), (0, 3*np.pi/2)], in_rep, out_rep)
            except EmptyBasisException:
                pass

def test_cyclic_even_regular():
    for N in [4, 8, 12]:
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation

        basis = diffops_CN_act_R2(in_rep, out_rep,
                                  max_power=1,
                                  max_frequency=4)
        size = required_points(6, 2)
        points = symmetric_points(size)
        check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

def test_cyclic_mix():
    for N in [4, 8, 12]:
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        basis = diffops_CN_act_R2(in_rep, out_rep,
                                  max_power=1,
                                  max_frequency=4)
        size = required_points(6, 2)
        points = symmetric_points(size)
        check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

def test_cyclic_changeofbasis():
    for N in [4, 8, 12]:
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = group.regular_representation

        basis = diffops_CN_act_R2(in_rep, out_rep,
                                  max_power=1,
                                  max_frequency=4)
        size = required_points(6, 2)
        points = symmetric_points(size)
        check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

        in_rep = group.regular_representation
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        basis = diffops_CN_act_R2(in_rep, out_rep,
                                  max_power=1,
                                  max_frequency=4)
        check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)

def test_cyclic_irreps():
    N = 8
    group = cyclic_group(N)

    for in_rep in group.irreps.values():
        for out_rep in group.irreps.values():
            try:
                basis = diffops_CN_act_R2(in_rep, out_rep,
                                          max_power=1,
                                          max_frequency=4)
                size = required_points(6, 2)
                points = symmetric_points(size)
                check_quarter_rotations(basis, points, [0, N // 4, N // 2, 3 * N // 4], in_rep, out_rep)
            except EmptyBasisException:
                continue
