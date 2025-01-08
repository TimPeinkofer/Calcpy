import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from NumEigenv import matrix_vector, Eigenvalue_calc, Eigenvalues, Aitken, eigenvalue_calc_aitken, Eigenvalues_Aitken

class TestEigenvalueCalculations(unittest.TestCase):

    def test_matrix_vector(self):
        mat = np.array([[2, 1], [1, 3]])
        vec = np.array([1, 2])
        expected_result = np.array([4, 7])

        result = matrix_vector(2, mat, vec)
        np.testing.assert_array_almost_equal(result, expected_result, err_msg="Matrix-vector multiplication failed.")

    def test_Eigenvalue_calc(self):
        mat = np.array([[2, 0], [0, 3]])
        vec = np.array([1, 1])
        iteration = 10

        eigenvalue, eigenvector = Eigenvalue_calc(mat, vec, iteration)

        # Check dominant eigenvalue and normalized eigenvector
        self.assertAlmostEqual(eigenvalue, 3, places=1, msg="Eigenvalue calculation failed.")
        np.testing.assert_almost_equal(np.linalg.norm(eigenvector), 1, err_msg="Eigenvector normalization failed.")

    def test_Eigenvalues(self):
        mat = np.array([[2, 0], [0, 3]])
        vec = np.array([1, 1])
        iteration = 10

        eigenvalues, eigenvectors = Eigenvalues(mat, vec, iteration)

        # Check dominant eigenvalue and normalized eigenvector
        self.assertAlmostEqual(eigenvalues[0], 3, places=1, msg="Eigenvalue computation failed.")
        np.testing.assert_almost_equal(np.linalg.norm(eigenvectors[0]), 1, err_msg="Eigenvector normalization failed.")

    def test_Aitken(self):
        eigenvalues = [2.9, 2.99, 2.999]
        accelerated_value = Aitken(eigenvalues)

        # Expected value should converge to 3
        self.assertAlmostEqual(accelerated_value, 3, places=6, msg="Aitken acceleration failed.")

    def test_eigenvalue_calc_aitken(self):
        mat = np.array([[2, 0], [0, 3]])
        vec = np.array([1, 1])
        iteration = 10

        eigenvalue, eigenvector = eigenvalue_calc_aitken(mat, vec, iteration)

        # Check dominant eigenvalue and normalized eigenvector
        self.assertAlmostEqual(eigenvalue, 3, places=1, msg="Eigenvalue calculation with Aitken failed.")
        np.testing.assert_almost_equal(np.linalg.norm(eigenvector), 1, err_msg="Eigenvector normalization failed.")

    def test_Eigenvalues_Aitken(self):
        mat = np.array([[2, 0], [0, 3]])
        vec = np.array([1, 1])
        iteration = 10

        eigenvalues, eigenvectors = Eigenvalues_Aitken(mat, vec, iteration)

        # Check dominant eigenvalue and normalized eigenvector
        self.assertAlmostEqual(eigenvalues[0], 3, places=1, msg="Eigenvalue computation with Aitken failed.")
        np.testing.assert_almost_equal(np.linalg.norm(eigenvectors[0]), 1, err_msg="Eigenvector normalization failed.")

if __name__ == "__main__":
    unittest.main()
