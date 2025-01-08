import unittest
import numpy as np
import sys
import os

# Add the path to the calcpy module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from NumLinEqu import Gauss_elimination

class TestGaussElimination(unittest.TestCase):

    def test_gauss_elimination(self):
        matrix = np.array([[3, 2, -4],
                           [2, 3, 3],
                           [5, -3, 1]], dtype=float)
        vector = np.array([3, 15, 14], dtype=float)
        expected_solution = np.array([3, 1, 2], dtype=float)

        solution = Gauss_elimination(matrix, vector)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)

    def test_singular_matrix(self):
        matrix = np.array([[1, 2, 3],
                           [2, 4, 6],
                           [1, 5, 9]], dtype=float)
        vector = np.array([6, 12, 15], dtype=float)

        with self.assertRaises(ValueError):
            Gauss_elimination(matrix, vector)

    def test_zero_matrix(self):
        matrix = np.zeros((3, 3), dtype=float)
        vector = np.array([0, 0, 0], dtype=float)

        with self.assertRaises(ValueError):
            Gauss_elimination(matrix, vector)

if __name__ == '__main__':
    unittest.main()
