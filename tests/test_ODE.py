import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

# Assuming the functions are in a module named 'ode_solver'
from ODE import precalc, Heun, adam_predictor, adam_corrector, adam_ode_int, Adam, runge_kutta


class TestOdeSolver(unittest.TestCase):

    def test_precalc(self):
        x0 = 0.0
        xm = 1.0
        n = 10
        x, y, h = precalc(xm, x0, n)
        
        # Check that the length of x and y arrays is correct
        self.assertEqual(len(x), n + 1)
        self.assertEqual(len(y), n + 1)
        
        # Check that the step size h is correct
        self.assertEqual(h, (xm - x0) / n)
        
        # Check that x values are equally spaced
        self.assertTrue(np.allclose(np.diff(x), h))
        
    def test_Adam(self):
        def f(x, y): return x + y  # Simple test function
        x0 = 0.0
        xm = 1.0
        y0 = 1.0
        n = 10
        x, y = Adam(x0, xm, y0, n, f, plotchoose=False)
        
        # Check that the solution array y has the expected length
        self.assertEqual(len(y), n + 1)
        
        # Check that the first value of y is equal to the initial condition
        self.assertEqual(y[0], y0)
        
    def test_Heun(self):
        def f(x, y): return x - y  # Another simple test function
        x0 = 0.0
        xm = 1.0
        y0 = 1.0
        n = 10
        x, y = Heun(x0, xm, y0, n, f, plotchoose=False)
        
        # Check that the solution array y has the expected length
        self.assertEqual(len(y), n + 1)
        
        # Check that the first value of y is equal to the initial condition
        self.assertEqual(y[0], y0)
        
    def test_runge_kutta(self):
        def f(x, y): return x * y  # Another simple test function
        x_start = 0.0
        x_end = 1.0
        y_0 = 1.0
        n = 10
        x_values, y_values = runge_kutta(x_start, x_end, y_0, n, f, plotchoose=False)
        
        # Check that the solution array y_values has the expected length
        self.assertEqual(len(y_values), n + 1)
        
        # Check that the first value of y_values is equal to the initial condition
        self.assertEqual(y_values[0], y_0)
        
    def test_adam_predictor(self):
        def f(x, y): return np.sin(x) * y  # Another simple test function
        y_0 = 1.0
        x_0 = 0.0
        x_m = 1.0
        n = 10
        x, y_values = adam_predictor(f, y_0, x_0, x_m, n, plotchoose=False)
        
        # Check that the solution array y_values has the expected length
        self.assertEqual(len(y_values), n + 1)
        
        # Check that the first value of y_values is equal to the initial condition
        self.assertEqual(y_values[0], y_0)
        
    def test_adams_corrector(self):
        def f(x, y): return np.cos(x) * y  # Another simple test function
        x = np.linspace(0, 1, 11)
        y_values = np.ones(11)  # Dummy initial values
        n = 10
        h = (1.0 - 0.0) / n
        y_values_corrected = adam_corrector(f, y_values, x, n, h, plotchoose=False)
        
        # Check that the corrected y_values has the same length as the input
        self.assertEqual(len(y_values_corrected), len(y_values))
        
    def test_adams_ode_int(self):
        def f(x, y): return np.exp(x) * y  # Another simple test function
        y_0 = 1.0
        x0 = 0.0
        xm = 1.0
        n = 10
        plotchoose = False
        y_corr, y_pred = adam_ode_int(f, y_0, x0, xm, n, plotchoose)
        
        # Check that the corrected solution y_corr has the expected length
        self.assertEqual(len(y_corr), n + 1)
        
        # Check that the predicted solution y_pred has the expected length
        self.assertEqual(len(y_pred), n + 1)


if __name__ == '__main__':
    unittest.main()
