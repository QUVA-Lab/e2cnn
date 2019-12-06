import unittest
from unittest import TestCase

import numpy as np

from e2cnn.group import *
from e2cnn.kernels import *


class TestSolutionsEquivariance(TestCase):
    
    def test_trivial(self):
        N = 1
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_Trivial_act_R2(in_rep, out_rep,
                                       radii=[0., 1., 2., 5, 10],
                                       sigma=[0.6, 1., 1.3, 2.5, 3.],
                                       max_frequency=9)
        action = group.irrep(0) + group.irrep(0)
        self._check(basis, group, in_rep, out_rep, action)

    def test_flips(self):
        group = cyclic_group(2)
        # in_rep = group.regular_representation
        # out_rep = group.regular_representation
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")

        # axis = 0.
        # for axis in [0., np.pi / 2, np.pi/3]:
        # for axis in [np.pi/2]:
        A = 10
        for a in range(A):
            axis = a*np.pi/A
            print(axis)
    
            basis = kernels_Flip_act_R2(in_rep, out_rep, axis=axis,
                                        radii=[0., 1., 2., 5, 10],
                                        sigma=[0.6, 1., 1.3, 2.5, 3.],
                                        max_frequency=9)
    
            action = directsum([group.irrep(0), group.irrep(1)], psi(axis)[..., 0, 0], f"horizontal_flip_{axis}")
    
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_odd_regular(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.6, 1., 1.3, 2.5, 3.],
                                      max_frequency=9)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)
    
    def test_cyclic_even_regular(self):
        N = 6
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_CN_act_R2(in_rep, out_rep,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  max_frequency=9)
        action = group.irrep(1)
        self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_mix(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")
    
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      max_frequency=9)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_output_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(list(group.irreps.values()), name="irreps_sum")
        out_rep = group.regular_representation

        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      max_frequency=9)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_input_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = directsum(list(group.irreps.values()), name="irreps_sum")
    
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      max_frequency=9)
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_odd_regular(self):
        N = 5
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi / 2
    
        basis = kernels_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  max_frequency=9)
    
        action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")
    
        self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_even_regular(self):
        N = 2
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi/2

        basis = kernels_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  max_frequency=9)

        # action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")
        action = change_basis(group.irrep(0, 1) + group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

        self._check(basis, group, in_rep, out_rep, action)

    def test_cyclic_irreps(self):
        N = 8
        group = cyclic_group(N)
        
        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
    
                basis = kernels_CN_act_R2(in_rep, out_rep,
                                          radii=[0., 1., 2., 5, 10],
                                          sigma=[0.6, 1., 1.3, 2.5, 3.],
                                          max_frequency=9)
                action = group.irrep(1)
                self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_irreps(self):
        N = 4
        group = dihedral_group(N)

        for axis in [0., np.pi/2, np.pi/3]:
            for in_rep in group.irreps.values():
                for out_rep in group.irreps.values():
                    basis = kernels_DN_act_R2(in_rep, out_rep,
                                              axis=axis,
                                              radii=[0., 1., 2., 5, 10],
                                              sigma=[0.6, 1., 1.3, 2.5, 3.],
                                              max_frequency=13)
                
                    action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")

                    self._check(basis, group, in_rep, out_rep, action)

    def test_dihedral_2_irreps(self):
        N = 2
        group = dihedral_group(N)
        axis = np.pi / 2
        
        reprs = list(group.irreps.values()) + [directsum(list(group.irreps.values()), name="irreps_sum"), group.regular_representation]

        for in_rep in reprs:
            for out_rep in reprs:
                basis = kernels_DN_act_R2(in_rep, out_rep,
                                          axis=axis,
                                          radii=[0., 1., 2., 5, 10],
                                          sigma=[0.6, 1., 1.3, 2.5, 3.],
                                          max_frequency=2)
            
                action = change_basis(group.irrep(0, 1) + group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")
            
                self._check(basis, group, in_rep, out_rep, action)

    def test_flips_irreps(self):
        group = cyclic_group(2)

        # for axis in [0., np.pi/3, np.pi/2,]:
        A = 10
        for axis in range(A):
            axis = axis * np.pi/A
            for in_rep in list(group.irreps.values()) + [group.regular_representation]:
                for out_rep in list(group.irreps.values()) + [group.regular_representation]:
                    basis = kernels_Flip_act_R2(in_rep, out_rep,
                                                axis=axis,
                                                radii=[0., 1., 2., 5, 10],
                                                sigma=[0.6, 1., 1.3, 2.5, 3.],
                                                max_frequency=13)
                
                    action = directsum([group.irrep(0), group.irrep(1)], psi(axis)[..., 0, 0], f"horizontal_flip_{axis}")

                    self._check(basis, group, in_rep, out_rep, action)

    def test_so2_irreps(self):
        
        group = so2_group(10)
    
        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                basis = kernels_SO2_act_R2(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.6, 1., 1.3, 2.5, 3.]
                                           )
                action = group.irrep(1)
                self._check(basis, group, in_rep, out_rep, action)

    def test_o2_irreps(self):
    
        group = o2_group(10)
        axis = np.pi / 2

        for in_rep in group.irreps.values():
            for out_rep in group.irreps.values():
                try:
                    basis = kernels_O2_act_R2(in_rep, out_rep,
                                              axis=axis,
                                              radii=[0., 1., 2., 5, 10],
                                              sigma=[0.6, 1., 1.3, 2.5, 3.]
                                              )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue
                    
                action = change_basis(group.irrep(1, 1), psi(axis)[..., 0, 0], "horizontal_flip")
                self._check(basis, group, in_rep, out_rep, action)

    def _check(self, basis, group, in_rep, out_rep, action):
        if basis is None:
            print("Empty KernelBasis!")
            return
        
        P = 9
        B = 100
        
        square_points = np.array([
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.],
            [-1., 1.],
            [-1., 0.],
            [-1., -1.],
            [0., -1.],
            [1., -1.],
        ]).T
        
        random_points = 3 * np.random.randn(2, P - 1)
        
        points = np.concatenate([random_points, square_points], axis=1)
        
        P = points.shape[1]
        
        features = np.random.randn(B, in_rep.size, P)
        
        filter = np.zeros((out_rep.size, in_rep.size, basis.dim, P))
        
        filters = basis.sample(points, out=filter)
        self.assertFalse(np.isnan(filters).any())
        self.assertFalse(np.allclose(filters, np.zeros_like(filter)))
        

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
        
        for g in group.testing_elements():
            
            output1 = np.einsum("oi,bif->bof", out_rep(g), output)

            a = action(g)
            transformed_points = a @ points
            
            transformed_filters = basis.sample(transformed_points)
            
            transformed_features = np.einsum("oi,bip->bop", in_rep(g), features)
            output2 = np.einsum("oifp,bip->bof", transformed_filters, transformed_features)

            if not np.allclose(output1, output2):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)
                
                aerr = np.abs(output1 - output2)
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(np.allclose(output1, output2), f"Group {group.name}, {in_rep.name} - {out_rep.name},\n"
                                                           f"element {g}, action {a}")
                                                           # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
