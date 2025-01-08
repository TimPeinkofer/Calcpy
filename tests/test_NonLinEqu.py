import unittest
import numpy as np

import sys
import os

# Add the path to the numint module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))
from NonLinEq import newtons_method, linear_interpolation, solve_fixed_point

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

if __name__ == "__main__":
    unittest.main()
