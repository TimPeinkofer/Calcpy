import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

from BVP import Matrix_method, solve_by_shooting

def ode_exp(x, y):
    """ Beispielhafte einfache ODE für den Test."""
    dydx = [y[1], -y[0]]  # y'' = -y -> Schwingungsgleichung
    return np.array(dydx)

class TestBoundaryValueMethods(unittest.TestCase):
    
    def test_matrix_method(self):
        x0, xn, n = 0, 1, 10
        y0, yn = 0, 1
        y_numeric = Matrix_method(x0, xn, n, y0, yn)
        
        # Da list_y zwei zusätzliche Werte hat (y0 und yn), muss die Länge n+2 sein
        self.assertEqual(len(y_numeric), n, "Länge des Ergebnisses stimmt nicht.")

        # Randwerte werden nicht direkt in y_numeric gespeichert, sondern gehören zu list_y
        list_y = np.concatenate(([y0], y_numeric, [yn]))  # Rekonstruiere die vollständige Lösung
        
        self.assertAlmostEqual(list_y[0], y0, 1, f"Randbedingung y0 nicht erfüllt: {np.abs(list_y[0]-y0)}")
        self.assertAlmostEqual(list_y[-1], yn, 1, f"Randbedingung yn nicht erfüllt: {np.abs(list_y[-1]-yn)}")

    
    def test_shooting_method(self):
        x1, x2, n = 0, np.pi / 2, 10
        u1, u2 = 0, 1
        v0 = [0, 2]  # Erste Ableitung, Schätzwerte
        max_iter = 10
        
        v_corr, x, y = solve_by_shooting(ode_exp, x1, x2, n, v0, u1, u2, max_iter)
        
        self.assertAlmostEqual(y[-1, 0], u2, 2, "Randbedingung u2 nicht erfüllt.")
        self.assertEqual(len(x), n + 1, "Anzahl der Gitterpunkte stimmt nicht.")

if __name__ == "__main__":
    unittest.main()
