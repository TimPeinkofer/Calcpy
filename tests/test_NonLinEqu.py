import unittest
import numpy as np

import sys
import os

# Add the path to the numint module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))
from NonLinEq import newtons_method, linear_interpolation, solve_fixed_point, bisection_method, jacobi_method, Gauss_elimination_pivoted

class TestRootFindingMethods(unittest.TestCase):

    def test_newtons_method(self):
        # Define the function and its derivative
        f = lambda x: x**2 - 2
        df = lambda x: 2 * x

        # Test Newton's method
        root = newtons_method(f, df, x0=1.0, max_iter=100, epsilon=1e-6)
        self.assertIsNotNone(root, "Newton's method did not converge.")
        self.assertAlmostEqual(root, np.sqrt(2), places=5, msg="Newton's method returned an incorrect root.")

    def test_linear_interpolation(self):
        # Define the function
        f = lambda x: x**2 - 2

        # Test linear interpolation
        root = linear_interpolation(f, x0=1.0, x1=2.0, max_iter=100, epsilon=1e-6)
        self.assertIsNotNone(root, "Linear interpolation did not converge.")
        self.assertAlmostEqual(root, np.sqrt(2), places=5, msg="Linear interpolation returned an incorrect root.")

    def test_solve_fixed_point(self):
        # Define the functions for the system
        f1 = lambda y: np.sqrt(2 - y)
        f2 = lambda x: np.sqrt(2 - x)

        # Test fixed-point iteration
        x, y = solve_fixed_point(f1, f2, x_init=1.0, y_init=1.0, max_iter=100, tol=1e-6)
        self.assertIsNotNone(x, "Fixed-point iteration did not converge.")
        self.assertIsNotNone(y, "Fixed-point iteration did not converge.")
        self.assertAlmostEqual(x, 1.0, places=5, msg="Fixed-point iteration returned an incorrect x.")
        self.assertAlmostEqual(y, 1.0, places=5, msg="Fixed-point iteration returned an incorrect y.")

def f_test(x):
    return x**2 - 4  # Nullstellen bei x = -2 und x = 2

class TestBisectionMethod(unittest.TestCase):
    def test_root_positive(self):
        root = bisection_method(0, 5, f_test, max_iter=100, eps=1e-6)
        self.assertIsNotNone(root, "Root should not be None")
        self.assertAlmostEqual(root, 2.0, places=6)

    def test_root_negative(self):
        root = bisection_method(-5, 0, f_test, max_iter=100, eps=1e-6)
        self.assertIsNotNone(root, "Root should not be None")
        self.assertAlmostEqual(root, -2.0, places=6)

    def test_no_root(self):
        root = bisection_method(3, 5, f_test, max_iter=100, eps=1e-6)
        self.assertIsNone(root, "Expected None for interval without a root")


class TestJacobiMethod(unittest.TestCase):
    def test_simple_system(self):
        # Ein diagonaldominantes System, das sicher konvergiert
        A = np.array([[4, 1], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        init_val = [0, 0]
        x = jacobi_method(A, b, init_val, max_iter=100)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-5)

    def test_zero_diagonal(self):
        # Wenn eine Null auf der Diagonalen steht, muss der Algorithmus abbrechen
        A = np.array([[0, 1], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        init_val = [0, 0]
        with self.assertRaises(ValueError):
            jacobi_method(A, b, init_val, max_iter=10)

    def test_no_convergence(self):
        # Ein System, das für Jacobi nicht konvergiert
        A = np.array([[1, 2], [2, 1]], dtype=float)
        b = np.array([3, 3], dtype=float)
        init_val = [0, 0]
        with self.assertRaises(RuntimeError):
            jacobi_method(A, b, init_val, max_iter=5, tol=1e-12)


class TestGaussEliminationPivoted(unittest.TestCase):
    def test_standard_system(self):
        matrix = np.array([[3, 2, -4],
                           [2, 3, 3],
                           [5, -3, 1]], dtype=float)
        vector = np.array([3, 15, 14], dtype=float)
        expected_solution = np.linalg.solve(matrix, vector)

        solution = Gauss_elimination_pivoted(matrix, vector)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)

    def test_needs_pivoting(self):
        # Ein System mit einer Null auf der Diagonalen.
        # Ohne Pivotisierung würde der Standard-Gauß hier stolpern oder ungenau werden.
        matrix = np.array([[0, 2, 1],
                           [1, -2, -3],
                           [-1, 1, 2]], dtype=float)
        vector = np.array([-8, 0, 3], dtype=float)
        expected_solution = np.linalg.solve(matrix, vector)
        
        solution = Gauss_elimination_pivoted(matrix, vector)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)

    def test_singular_matrix(self):
        # Matrix ohne eindeutige Lösung
        matrix = np.array([[1, 2, 3],
                           [2, 4, 6],
                           [1, 5, 9]], dtype=float)
        vector = np.array([6, 12, 15], dtype=float)

        with self.assertRaises(ValueError):
            Gauss_elimination_pivoted(matrix, vector)


if __name__ == "__main__":
    unittest.main()
