import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NumKit')))

# Assuming the functions are in a module named 'ode_solver'
from ODE import precalc, Heun, adam_predictor, adam_corrector, adam_ode_int, Adam, runge_kutta, velocity_verlet


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

class TestVelocityVerlet(unittest.TestCase):
    
    def test_constant_acceleration(self):
        """ Testet das System unter einer konstanter Kraft (z.B. Schwerkraft). """
        x0, v0 = 0.0, 10.0
        t0, tm, n = 0.0, 2.0, 100
        m = 2.0
        konstante_kraft = -9.81
        
        # F(x) = konst
        def force_func(x):
            return konstante_kraft
            
        t, x, v = velocity_verlet(x0, v0, t0, tm, n, force_func, m=m, plotchoose=False)
        
        # Analytische Erwartungswerte
        a = konstante_kraft / m
        x_expected = x0 + v0 * tm + 0.5 * a * tm**2
        v_expected = v0 + a * tm
        
        # Array-Längen prüfen (Startwert + n Schritte = n + 1)
        self.assertEqual(len(t), n + 1, "Anzahl der Zeitschritte stimmt nicht.")
        self.assertEqual(len(x), n + 1, "Länge des Ortsvektors stimmt nicht.")
        self.assertEqual(len(v), n + 1, "Länge des Geschwindigkeitsvektors stimmt nicht.")
        
        # Werte am Ende der Simulation prüfen
        self.assertAlmostEqual(x[-1], x_expected, places=5, msg="Position bei konstanter Beschleunigung weicht ab.")
        self.assertAlmostEqual(v[-1], v_expected, places=5, msg="Geschwindigkeit bei konstanter Beschleunigung weicht ab.")


    def test_harmonic_oscillator_period(self):
        """ Testet die Energieerhaltung/Periodizität beim harmonischen Oszillator (Federpendel). """
        x0, v0 = 1.0, 0.0  # Startet maximal ausgelenkt aus der Ruhe
        k = 1.0            # Federkonstante
        m = 1.0            # Masse
        
        # Kreisfrequenz und Periodendauer
        omega = np.sqrt(k / m)
        T = 2 * np.pi / omega  # Exakt EINE volle Schwingung
        
        t0, tm, n = 0.0, T, 1000
        
        # F(x) = -k * x (Hookesches Gesetz)
        def force_func(x):
            return -k * x
            
        t, x, v = velocity_verlet(x0, v0, t0, tm, n, force_func, m=m, plotchoose=False)
        
        # Nach exakt einer Periode T muss das System wieder exakt im Ausgangszustand sein
        # Wir prüfen auf 2 Nachkommastellen, da die numerische Integration (aufgrund der endlichen Schrittweite) minimale Rundungsfehler hat.
        self.assertAlmostEqual(x[-1], x0, places=2, msg="Position nach exakt einer Schwingungsperiode nicht erhalten.")
        self.assertAlmostEqual(v[-1], v0, places=2, msg="Geschwindigkeit nach exakt einer Schwingungsperiode nicht erhalten.")


if __name__ == '__main__':
    unittest.main()
