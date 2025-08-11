import unittest
import numpy as np
import sys
import os

# Add the path to the calcpy module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from NumLinEqu import Gauss_elimination, gauss_seidel, inverse_matrix, Gauss_Jordan, overrelaxation, overrelax_calc, determinant

class TestOverrelaxCalc(unittest.TestCase):
    def test_convergence(self):
        A = np.array([[4,1],[2,3]], dtype=float)
        b = np.array([1,2], dtype=float)
        x = overrelax_calc(A, b, iterations=100, w=1.1, tol=1e-6)
        expected = np.linalg.solve(A, b)
        self.assertIsNotNone(x)
        np.testing.assert_allclose(x, expected, rtol=5e-5)

    def test_zero_diagonal_raises(self):
        A = np.array([[0,1],[2,3]], dtype=float)
        b = np.array([1,2], dtype=float)
        with self.assertRaises(ValueError):
            overrelax_calc(A, b, iterations=10, w=1.0)

    def test_no_convergence(self):
        # Matrix that may not converge quickly
        A = np.array([[1,2],[2,1]], dtype=float)
        b = np.array([3,3], dtype=float)
        x = overrelax_calc(A, b, iterations=5, w=1.5, tol=1e-12)
        self.assertIsNone(x)

class TestOverrelaxation(unittest.TestCase):
    def test_returns_best_solution(self):
        A = np.array([[4,1],[2,3]], dtype=float)
        b = np.array([1,2], dtype=float)
        result = overrelaxation(A, b, Max_iterations=50)
        self.assertIsNotNone(result)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

class TestDeterminant(unittest.TestCase):
    def test_known_determinant(self):
        A = np.array([[1,2],[3,4]], dtype=float)
        det = determinant(A)
        expected = np.linalg.det(A)
        self.assertAlmostEqual(det, expected, places=6)

    def test_singular_matrix(self):
        A = np.array([[1,2],[2,4]], dtype=float)
        det = determinant(A)
        self.assertAlmostEqual(det, 0.0, places=10)

    def test_identity_matrix(self):
        I = np.eye(5)
        det = determinant(I)
        self.assertAlmostEqual(det, 1.0, places=12)

    def test_swap_effect(self):
        # Swapping rows changes sign of determinant
        A = np.array([[0,1],[1,0]], dtype=float)
        det = determinant(A)
        expected = np.linalg.det(A)
        self.assertAlmostEqual(det, expected, places=12)


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



class TestGaussSeidel(unittest.TestCase):
    def test_simple_system(self):
        A = np.array([[4,1],[2,3]], dtype=float)
        b = np.array([1,2], dtype=float)
        x0 = [0,0]
        x = gauss_seidel(A, b, x0, max_iter=100)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-5)

    def test_zero_diagonal(self):
        A = np.array([[0,1],[2,3]], dtype=float)
        b = np.array([1,2], dtype=float)
        x0 = [0,0]
        with self.assertRaises(ValueError):
            gauss_seidel(A, b, x0, max_iter=10)

    def test_no_convergence(self):
        A = np.array([[1,2],[2,1]], dtype=float)
        b = np.array([3,3], dtype=float)
        x0 = [0,0]
        with self.assertRaises(RuntimeError):
            gauss_seidel(A, b, x0, max_iter=5, tol=1e-12)

class TestGaussJordan(unittest.TestCase):
    def test_identity(self):
        A = np.eye(3)
        result = Gauss_Jordan(A.copy())
        np.testing.assert_allclose(result, np.eye(3), atol=1e-12)

    def test_known_matrix(self):
        A = np.array([[2.,1.],[1.,3.]])
        aug = np.hstack((A, np.eye(2)))
        reduced = Gauss_Jordan(aug.copy())
        # Die linke Seite muss die Einheitsmatrix sein
        np.testing.assert_allclose(reduced[:, :2], np.eye(2), atol=1e-12)

    def test_singular_matrix(self):
        A = np.array([[1,2],[2,4]], dtype=float)
        aug = np.hstack((A, np.eye(2)))
        with self.assertRaises(ValueError):
            Gauss_Jordan(aug)

class TestInverseMatrix(unittest.TestCase):
    def test_inverse_of_identity(self):
        I = np.eye(4)
        inv = inverse_matrix(I)
        np.testing.assert_allclose(inv, I, atol=1e-12)

    def test_inverse_known_matrix(self):
        A = np.array([[4,7],[2,6]], dtype=float)
        inv = inverse_matrix(A)
        expected = np.linalg.inv(A)
        np.testing.assert_allclose(inv, expected, rtol=1e-5)

    def test_non_square_matrix(self):
        A = np.array([[1,2,3],[4,5,6]])
        with self.assertRaises(ValueError):
            inverse_matrix(A)

    def test_singular_matrix(self):
        A = np.array([[1,2],[2,4]], dtype=float)
        with self.assertRaises(ValueError):
            inverse_matrix(A)

if __name__ == '__main__':
    unittest.main()
