import unittest
from unittest import TestCase

from e2cnn.group import CyclicGroup
from e2cnn.group import DihedralGroup

import numpy as np
import scipy.sparse as sparse


class TestComputeRegularRepresentations(TestCase):
    
    def test_dihedral_rr_odd(self):
        dg = DihedralGroup(9)
        self.dihedral_rr_eval(dg, dg.representations['regular'])
    
    def test_dihedral_rr_even(self):
        dg = DihedralGroup(10)
        self.dihedral_rr_eval(dg, dg.representations['regular'])
    
    def test_dihedral_rr_large(self):
        dg = DihedralGroup(16)
        self.dihedral_rr_eval(dg, dg.representations['regular'])

    def test_dihedral_rr_small(self):
        dg = DihedralGroup(2)
        self.dihedral_rr_eval(dg, dg.representations['regular'])

    def test_cyclic_rr_odd(self):
        cg = CyclicGroup(11)
        self.cyclic_rr_eval(cg, cg.representations['regular'])

    def test_cyclic_rr_even(self):
        cg = CyclicGroup(10)
        self.cyclic_rr_eval(cg, cg.representations['regular'])

    def test_cyclic_rr_large(self):
        cg = CyclicGroup(20)
        self.cyclic_rr_eval(cg, cg.representations['regular'])

    def test_cyclic_rr_small(self):
        cg = CyclicGroup(2)
        self.cyclic_rr_eval(cg, cg.representations['regular'])

    def cyclic_rr_eval(self, cg, rr):
        # rr = cg.representations['regular']

        # np.set_printoptions(precision=4, suppress=True)
        # print('Change of Basis')
        # print(rr.change_of_basis)
        # print('Change of Basis Inv')
        # print(rr.change_of_basis_inv)
        # print('RR')
        # n = cg.order
        # for i in range(n):
        #     print(rr(i * 2 * np.pi / n))
        
        D = rr.change_of_basis
        D_inv = rr.change_of_basis_inv
        for i, element in enumerate(cg.elements):
        
            rho_i = np.zeros([cg.order(), cg.order()])
        
            for k in range(cg.order()):
                rho_i[(i + k) % cg.order(), k] = 1.0
        
            # Build the direct sum of the irreps for this element
            blocks = []
            for irrep in rr.irreps:
                repr = cg.irreps[irrep](element)
                blocks.append(repr)
        
            P = sparse.block_diag(blocks, format='csc')
            R = D @ P @ D_inv
            self.assertTrue(np.allclose(R, rho_i), f"{element}:\n{R}\n!=\n{rho_i}\n")
            self.assertTrue(np.allclose(rr(element), rho_i), f"{element}:\n{rr(element)}\n!=\n{rho_i}\n")

    def dihedral_rr_eval(self, dg, rr):
    
        # rr = dg.representations['regular']
    
        # np.set_printoptions(precision=2, suppress=True)
        # print('Change of Basis')
        # print(rr.change_of_basis)
        # print('Change of Basis Inv')
        # print(rr.change_of_basis_inv)
        # print('RR')
        # n = dg.rotation_order
        # for i in range(n):
        #     print(rr((0, i * 2 * np.pi / n)))
        # for i in range(n):
        #     print(rr((1, i * 2 * np.pi / n)))
    
        D = rr.change_of_basis
        D_inv = rr.change_of_basis_inv

        # np.set_printoptions(precision=3, threshold=10*rr.size**2, suppress=True, linewidth=25*rr.size + 5)
        
        for i, element in enumerate(dg.elements):
        
            rho_i = np.zeros([dg.order(), dg.order()])
        
            f = -1 if element[0] else 1
            # r = int(np.round(element[1] * dg.rotation_order / (2 * np.pi)))
            r = element[1]
        
            for k in range(dg.rotation_order):
                rho_i[dg.rotation_order * element[0] + ((r + k * f) % dg.rotation_order), k] = 1.0
            for k in range(dg.rotation_order):
                rho_i[dg.rotation_order * (1 - element[0]) + ((r + k * f) % dg.rotation_order), dg.rotation_order + k] = 1.0
        
            # Build the direct sum of the irreps for this element
            blocks = []
            for irrep in rr.irreps:
                repr = dg.irreps[irrep](element)
                blocks.append(repr)
        
            P = sparse.block_diag(blocks, format='csc')
            R = D @ P @ D_inv
            self.assertTrue(np.allclose(R, rho_i), f"{element}:\n{R}\n!=\n{rho_i}\n")
            self.assertTrue(np.allclose(rr(element), rho_i), f"{element}:\n{rr(element)}\n!=\n{rho_i}\n")


if __name__ == '__main__':
    unittest.main()
