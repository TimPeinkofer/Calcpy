import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NumKit')))

# Importiere die neuen Monte-Carlo-Methoden (Pfad anpassen, falls sie woanders liegen)
from MC import Monte_Carlo, Monte_Carlo_nD

class TestMonteCarloMethods(unittest.TestCase):
    
    def test_monte_carlo_1d(self):
        n, a, b = 100000, 0, 10
        expected = 333.3333333333333
        
        # Seed setzen, damit der Test reproduzierbar bleibt
        np.random.seed(42)
        
        result, error = Monte_Carlo(n, a, b, lambda x: x**2)
        
        # assertAlmostEqual mit delta (Toleranz), da Monte Carlo immer leicht schwankt
        # Wir erlauben hier eine Abweichung von 5% vom Erwartungswert
        toleranz = expected * 0.05
        
        self.assertAlmostEqual(result, expected, delta=toleranz, msg=f"Integralwert weicht zu stark ab: {np.abs(result-expected)}")
        self.assertGreater(error, 0, "Statistischer Fehler sollte größer als 0 sein.")

    
    def test_monte_carlo_nd(self):
        n = 100000
        bounds = [(0, 2), (0, 2)]
        expected = 32 / 3  # ca. 10.666...
        
        # Testfunktion: f(x, y) = x^2 + y^2
        def func_2d(vars):
            return vars[0]**2 + vars[1]**2
        
        np.random.seed(42)
        
        result, error = Monte_Carlo_nD(n, bounds, func_2d)
        
        toleranz = expected * 0.05
        
        self.assertAlmostEqual(result, expected, delta=toleranz, msg=f"2D-Integralwert weicht zu stark ab: {np.abs(result-expected)}")
        self.assertGreater(error, 0, "Statistischer Fehler sollte größer als 0 sein.")


if __name__ == "__main__":
    unittest.main()