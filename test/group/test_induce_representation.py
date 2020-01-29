import unittest
from unittest import TestCase

from e2cnn.group import cyclic_group
from e2cnn.group import dihedral_group
from e2cnn.group import o2_group
from e2cnn.group import so2_group
from e2cnn.group import directsum

from collections import defaultdict

import numpy as np


class TestInducedRepresentations(TestCase):
    
    def test_quotient_cyclic_even(self):
        N = 20
        dg = cyclic_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                sg_id = n
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_cyclic_odd(self):
        N = 21
        dg = cyclic_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                sg_id = n
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_dihedral_even(self):
        N = 4
        dg = dihedral_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                for f in range(N//n):
                    sg_id = (f, n)
                    sg, _, _ = dg.subgroup(sg_id)
                    self.check_induction(dg, sg_id, sg.trivial_representation)
                sg_id = (None, n)
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_dihedral_odd(self):
        N = 15
        dg = dihedral_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                for f in range(N//n):
                    sg_id = (f, n)
                    sg, _, _ = dg.subgroup(sg_id)
                    self.check_induction(dg, sg_id, sg.trivial_representation)
                sg_id = (None, n)
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)
                
    def test_quotient_dihedral(self):
        N = 7
        dg = dihedral_group(N)
        
        sg_id = (None, 1)
        sg, _, _ = dg.subgroup(sg_id)
        self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_induce_irreps_dihedral_odd_dihedral_odd(self):
        dg = dihedral_group(9)
        
        for axis in range(3):
            sg_id = (axis, 3)
            
            sg, _, _ = dg.subgroup(sg_id)
            for name, irrep in sg.irreps.items():
                self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_dihedral_odd_cyclic_odd(self):
        dg = dihedral_group(9)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_dihedral_odd_flips(self):
        dg = dihedral_group(11)
        for axis in range(11):
            sg_id = (axis, 1)
            sg, _, _ = dg.subgroup(sg_id)
            for name, irrep in sg.irreps.items():
                self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_cyclic_odd_cyclic_odd(self):
        dg = cyclic_group(9)
        sg_id = 3
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_even_dihedral_even(self):
        dg = dihedral_group(12)
        for axis in range(2):
            sg_id = (axis, 6)

            sg, _, _ = dg.subgroup(sg_id)
            for name, irrep in sg.irreps.items():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic_even(self):
        dg = dihedral_group(12)
        sg_id = (None, 4)
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_dihedral_odd(self):
        dg = dihedral_group(12)
        for axis in range(4):
            sg_id = (axis, 3)
            sg, _, _ = dg.subgroup(sg_id)
            for name, irrep in sg.irreps.items():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic_odd(self):
        dg = dihedral_group(12)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_even_flips(self):
        dg = dihedral_group(12)
        for axis in range(12):
            sg_id = (0, 1)
            sg, _, _ = dg.subgroup(sg_id)
            for name, irrep in sg.irreps.items():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_cyclic_even_cyclic_even(self):
        dg = cyclic_group(8)
        sg_id = 2
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_cyclic_even_cyclic_odd(self):
        dg = cyclic_group(10)
        sg_id = 5
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic(self):
        dg = dihedral_group(12)
        sg_id = (None, 12)
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_odd_cyclic(self):
        dg = dihedral_group(13)
        sg_id = (None, 13)
        sg, _, _ = dg.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_rr_dihedral_even_flips(self):
        dg = dihedral_group(10)
        
        sg_id = (0, 1)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
    
    def test_induce_rr_dihedral_odd_flips(self):
        dg = dihedral_group(11)
        sg_id = (0, 1)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_dihedral_even_dihedral_even(self):
        dg = dihedral_group(12)
        sg_id = (0, 6)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_dihedral_even_dihedral_odd(self):
        dg = dihedral_group(12)
        sg_id = (0, 3)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_dihedral_odd_dihedral_odd(self):
        dg = dihedral_group(9)
        sg_id = (0, 3)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)

    def test_induce_rr_dihedral_even_cyclic_even(self):
        dg = dihedral_group(8)
        sg_id = (None, 4)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_dihedral_even_cyclic_odd(self):
        dg = dihedral_group(12)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)

    def test_induce_rr_dihedral_odd_cyclic_odd(self):
        dg = dihedral_group(9)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_cyclic_even_cyclic_even(self):
        dg = cyclic_group(16)
        sg_id = 8
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)
        
    def test_induce_rr_cyclic_even_cyclic_odd(self):
        dg = cyclic_group(14)
        sg_id = 7
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)

    def test_induce_rr_cyclic_odd_cyclic_odd(self):
        dg = cyclic_group(15)
        sg_id = 5
        sg, _, _ = dg.subgroup(sg_id)
        repr = sg.regular_representation
        self.check_induction(dg, sg_id, repr)

    def test_induce_irreps_so2_o2(self):
        g = o2_group(10)
        sg_id = (None, -1)
        sg, _, _ = g.subgroup(sg_id)
        for name, irrep in sg.irreps.items():
            self.check_induction_so2_o2(g, sg_id, irrep)

    def check_induction(self, group, subgroup_id, repr):
        
        # print("#######################################################################################################")
        
        subgroup, parent, child = group.subgroup(subgroup_id)
        
        assert repr.group == subgroup
    
        induced_repr = group.induced_representation(subgroup_id, repr)
        
        assert induced_repr.group == group
        
        # check the change of basis is orthonormal
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis.T @ induced_repr.change_of_basis, np.eye(induced_repr.size)),
            "Change of Basis not orthonormal"
        )
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis @ induced_repr.change_of_basis.T, np.eye(induced_repr.size)),
            "Change of Basis not orthonormal"
        )
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis, induced_repr.change_of_basis_inv.T),
            "Change of Basis not orthonormal"
        )
        
        restricted_repr = group.restrict_representation(subgroup_id, induced_repr)
        for e in subgroup.testing_elements():
            
            repr_a = repr(e)
            repr_b = induced_repr(parent(e))[:repr.size, :repr.size]
            repr_c = restricted_repr(e)[:repr.size, :repr.size]

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")

            if not np.allclose(repr_c, repr_b):
                print(e, parent(e))
                print(induced_repr.change_of_basis_inv @ induced_repr(parent(e)) @ induced_repr.change_of_basis)
                print(restricted_repr.change_of_basis_inv @ restricted_repr(e) @ restricted_repr.change_of_basis)
                print(induced_repr.irreps)
                print(restricted_repr.irreps)
                
                # print(induced_repr.change_of_basis)
                # print(restricted_repr.change_of_basis)
                print(np.allclose(induced_repr.change_of_basis, restricted_repr.change_of_basis))
                
            self.assertTrue(np.allclose(repr_c, repr_b), msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_c}\ndifferent from\n {repr_b}\n")
            
        quotient_size = int(group.order() / subgroup.order())
        size = repr.size * quotient_size

        # the coset each element belongs to
        cosets = {}

        # map from a representative to the elements of its coset
        representatives = defaultdict(lambda: [])

        for e in group.elements:
            if e not in cosets:
                representatives[e] = []
                for g in subgroup.elements:
                    eg = group.combine(e, parent(g))
                
                    cosets[eg] = e
                
                    representatives[e].append(eg)

        index = {e: i for i, e in enumerate(representatives)}
        
        P = directsum([group.irreps[irr] for irr in induced_repr.irreps], name="irreps")
        
        for g in group.testing_elements():
            repr_g = np.zeros((size, size), dtype=np.float)
            for r in representatives:
            
                gr = group.combine(g, r)
            
                g_r = cosets[gr]
            
                i = index[r]
                j = index[g_r]
            
                hp = group.combine(group.inverse(g_r), gr)
            
                h = child(hp)
                assert h is not None, (g, r, gr, g_r, group.inverse(g_r), hp)
            
                repr_g[j*repr.size:(j+1)*repr.size, i*repr.size:(i+1)*repr.size] = repr(h)
            
            ind_g = induced_repr(g)
            self.assertTrue(np.allclose(repr_g, ind_g), msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{repr_g}\ndifferent from\n {ind_g}\n")
            
            ind_g2 = induced_repr.change_of_basis @ P(g) @ induced_repr.change_of_basis_inv
            self.assertTrue(np.allclose(ind_g2, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{ind_g2}\ndifferent from\n {ind_g}\n")

    def check_induction_so2_o2(self, group, subgroup_id, repr):
    
        # print("#######################################################################################################")
    
        subgroup, parent, child = group.subgroup(subgroup_id)
    
        assert repr.group == subgroup
    
        # induced_repr = build_induced_representation(group, subgroup_id, repr)
        induced_repr = group.induced_representation(subgroup_id, repr)
    
        assert induced_repr.group == group
        
        assert np.allclose(induced_repr.change_of_basis@induced_repr.change_of_basis_inv, np.eye(induced_repr.size))
        assert np.allclose(induced_repr.change_of_basis_inv@induced_repr.change_of_basis, np.eye(induced_repr.size))
    
        restricted_repr = group.restrict_representation(subgroup_id, induced_repr)
        for e in subgroup.testing_elements():
        
            repr_a = repr(e)
            repr_b = induced_repr(parent(e))[:repr.size, :repr.size]
            repr_c = restricted_repr(e)[:repr.size, :repr.size]
        
            np.set_printoptions(precision=2, threshold=2 * repr_a.size ** 2, suppress=True,
                                linewidth=10 * repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
        
            if not np.allclose(repr_c, repr_b):
                print(e, parent(e))
                print(induced_repr.change_of_basis_inv @ induced_repr(parent(e)) @ induced_repr.change_of_basis)
                print(restricted_repr.change_of_basis_inv @ restricted_repr(e) @ restricted_repr.change_of_basis)
                print(induced_repr.irreps)
                print(restricted_repr.irreps)
            
                # print(induced_repr.change_of_basis)
                # print(restricted_repr.change_of_basis)
                print(np.allclose(induced_repr.change_of_basis, restricted_repr.change_of_basis))
        
            self.assertTrue(np.allclose(repr_c, repr_b),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_c}\ndifferent from\n {repr_b}\n")
    
        quotient_size = 2
        size = repr.size * quotient_size
    
        # the coset each element belongs to
        cosets = {}
    
        # map from a representative to the elements of its coset
        representatives = defaultdict(lambda: [])
    
        for e in group.testing_elements():
            flip, rot = e
            cosets[e] = (flip, 0.)
            representatives[(flip, 0.)].append(e)
            
        index = {e: i for i, e in enumerate(representatives)}
    
        P = directsum([group.irreps[irr] for irr in induced_repr.irreps], name="irreps")
    
        for g in group.testing_elements():
            repr_g = np.zeros((size, size), dtype=np.float)
            for r in representatives:
                gr = group.combine(g, r)
            
                g_r = cosets[gr]
            
                i = index[r]
                j = index[g_r]
            
                hp = group.combine(group.inverse(g_r), gr)
            
                h = child(hp)
                assert h is not None, (g, r, gr, g_r, group.inverse(g_r), hp)
            
                repr_g[j * repr.size:(j + 1) * repr.size, i * repr.size:(i + 1) * repr.size] = repr(h)
        
            ind_g = induced_repr(g)
            self.assertTrue(np.allclose(repr_g, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{repr_g}\ndifferent from\n {ind_g}\n")
        
            ind_g2 = induced_repr.change_of_basis @ P(g) @ induced_repr.change_of_basis_inv
            self.assertTrue(np.allclose(ind_g2, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{ind_g2}\ndifferent from\n {ind_g}\n")


if __name__ == '__main__':
    unittest.main()
