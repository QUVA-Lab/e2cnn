import unittest
from unittest import TestCase

import numpy as np

from e2cnn import group
from e2cnn.kernels import utils as ks


class TestSolutionsEquivariance(TestCase):
    
    def test_psi(self):
        
        for _ in range(100):
            theta = np.random.rand() * 2 * np.pi
            
            for k in range(10):
                for _ in range(5):
                    gamma = np.random.rand() * 2 * np.pi
                    
                    g_psi = group.psi(theta, k, gamma)
                    k_psi = ks.psi(theta, k, gamma).squeeze()
                    
                    self.assertTrue(np.allclose(g_psi, k_psi))
    
    def test_psichi(self):
        
        for _ in range(100):
            theta = np.random.rand()*2*np.pi
            
            for k in range(10):
                for _ in range(5):
                    for s in range(2):
                        gamma = np.random.rand() * 2 * np.pi

                        g_psi = group.psichi(theta, s, k, gamma)
                        k_psi = ks.psichi(theta, s, k, gamma).squeeze()
                        
                        self.assertTrue(np.allclose(g_psi, k_psi))

    def test_chi(self):
    
        for s in range(2):
            g_psi = group.chi(s)
            k_psi = ks.chi(s).squeeze()
        
            self.assertTrue(np.allclose(g_psi, k_psi))


if __name__ == '__main__':
    unittest.main()
