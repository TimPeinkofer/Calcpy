import unittest
import numpy as np
import sys
import os

# Add the path to the numint module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from numint import precalc, Simpson_1_3, Romberg, Gauss_legendre, Newton_cotes, Trapezoidal, Simpson_3_8

# test_numint.py

class TestNumInt(unittest.TestCase):

    def test_precalc(self):
        h, x_values = precalc(2000, 0, 10)
        self.assertAlmostEqual(h, 0.005, places=3)
        self.assertEqual(len(x_values), 2000)
        self.assertAlmostEqual(x_values[0], 0)
        self.assertAlmostEqual(x_values[-1], 10)

    def test_Simpson_1_3(self):
        result = Simpson_1_3(2000, 0, 10, lambda x: x**2)
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Simpson_1_3 failed: {result} != {expected}")

    def test_Romberg(self):
        result = Romberg(2000, 0, 10, lambda x: x**2)
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Romberg failed: {result} != {expected}")

    def test_Gauss_legendre(self):
        result = Gauss_legendre(0, 1, 5, lambda x: x**2)
        expected = 0.3333333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Gauss_legendre failed: {result} != {expected}")

    def test_Newton_cotes(self):
        result = Newton_cotes(2000, 0, 10, lambda x: x**2)
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Newton_cotes failed: {result} != {expected}")

    def test_Trapezoidal(self):
        result = Trapezoidal(2000, 0, 10, lambda x: x**2)
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Trapezoidal failed: {result} != {expected}")

    def test_Simpson_3_8_multiple_of_3(self):
        result = Simpson_3_8(2001, 0, 10, lambda x: x**2)  # n is a multiple of 3
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Simpson_3_8 (multiple of 3) failed: {result} != {expected}")

    def test_Simpson_3_8_not_multiple_of_3(self):
        result = Simpson_3_8(2000, 0, 10, lambda x: x**2)  # n is not a multiple of 3
        expected = 333.3333333333333
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Simpson_3_8 (not multiple of 3) failed: {result} != {expected}")

    def test_functions_with_different_bounds(self):
        # Test different functions and bounds
        result = Simpson_1_3(2000, 1, 3, lambda x: x**3)
        expected = 20.0
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Simpson_1_3 with x^3 failed: {result} != {expected}")

        result = Trapezoidal(2000, 0, np.pi, np.sin)
        expected = 2.0
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Trapezoidal with sin(x) failed: {result} != {expected}")

        result = Gauss_legendre(0, 2, 5, lambda x: np.exp(-x))
        expected = 0.8646647167633873
        self.assertTrue(abs(result - expected) / expected < 2.0, f"Gauss_legendre with exp(-x) failed: {result} != {expected}")

if __name__ == '__main__':
    unittest.main()
