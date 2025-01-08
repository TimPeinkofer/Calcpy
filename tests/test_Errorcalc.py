import unittest
import numpy as np
import sys
import os

# Add the path to the numint module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from Errorcalc import IntegrationandDiffMethodError, error

def dummy_integration_method(n, a, b, func):
    """
    Dummy integration method for testing purposes.
    Approximates the integral using the midpoint rule.

    Args:
        n (int): Number of intervals
        a (float): Lower bound
        b (float): Upper bound
        func (function): Function to integrate

    Returns:
        float: Approximation of the integral
    """
    h = (b - a) / n
    result = 0
    for i in range(n):
        mid_point = a + (i + 0.5) * h
        result += func(mid_point)
    return result * h

class TestIntegrationandDiffMethodError(unittest.TestCase):

    def test_integration_and_error(self):
        # Define the function to integrate
        func = lambda x: x**2

        # Exact integral of x^2 from 0 to 1 is 1/3
        a, b = 0, 1
        exact_value = 1 / 3

        # Use the IntegrationandDiffMethodError class
        integration_error = IntegrationandDiffMethodError(dummy_integration_method, a, b, func)

        # Calculate the integral using the calculate_exact method
        approx_value = integration_error.calculate_exact()

        # Assert the approximation is close to the exact value
        self.assertAlmostEqual(approx_value, exact_value, places=3, msg="Integration result is incorrect.")

        # Calculate the error
        error_value = integration_error.calculate_error(approx_value)

        # Assert the error is near zero
        self.assertAlmostEqual(error_value, 0, places=6, msg="Error calculation is incorrect.")

    def test_error_function(self):
        # Define the function to integrate
        func = lambda x: np.sin(x)

        # Exact integral of sin(x) from 0 to pi is 2
        a, b = 0, np.pi
        exact_value = 2

        # Use the dummy integration method
        approx_value = dummy_integration_method(1000, a, b, func)

        # Use the error function
        calculated_error = error(dummy_integration_method, a, b, func, approx_value)

        # Assert the error is near zero
        self.assertAlmostEqual(calculated_error, 0, places=6, msg="Error function calculation is incorrect.")

if __name__ == "__main__":
    unittest.main()
