import unittest
import numpy as np
import sys
import os

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NumKit')))


from MC import Monte_Carlo, Monte_Carlo_nD

# Stelle sicher, dass du velocity_verlet hier importierst (je nachdem, wo du es eingefügt hast)
# from ODE import velocity_verlet

class TestVelocityVerlet(unittest.TestCase):
    
    def test_constant_acceleration(self):
        """ Testet das System unter einer konstanten Kraft (z.B. Schwerkraft). """
        x0, v0 = 0.0, 10.0
        t_start, t_end, n = 0.0, 2.0, 100
        m = 2.0
        konstante_kraft = -9.81
        
        # F(x) = konst
        def force_func(x):
            return konstante_kraft
            
        t, x, v = velocity_verlet(x0, v0, t_start, t_end, n, force_func, m=m, plotchoose=False)
        
        # Analytische Erwartungswerte
        a = konstante_kraft / m
        x_expected = x0 + v0 * t_end + 0.5 * a * t_end**2
        v_expected = v0 + a * t_end
        
        # Array-Längen prüfen (Startwert + n Schritte = n + 1)
        self.assertEqual(len(t), n + 1, "Anzahl der Zeitschritte stimmt nicht.")
        self.assertEqual(len(x), n + 1, "Länge des Ortsvektors stimmt nicht.")
        self.assertEqual(len(v), n + 1, "Länge des Geschwindigkeitsvektors stimmt nicht.")
        
        # Werte prüfen
        self.assertAlmostEqual(x[-1], x_expected, places=5, msg="Position bei konstanter Beschleunigung weicht ab.")
        self.assertAlmostEqual(v[-1], v_expected, places=5, msg="Geschwindigkeit bei konstanter Beschleunigung weicht ab.")


    def test_harmonic_oscillator_period(self):
        """ Testet die Periodizität beim harmonischen Oszillator (Feder). """
        x0, v0 = 1.0, 0.0  # Startet maximal ausgelenkt, aus der Ruhe
        k = 1.0            # Federkonstante
        m = 1.0            # Masse
        
        # Kreisfrequenz und Periodendauer
        omega = np.sqrt(k / m)
        T = 2 * np.pi / omega  # Exakt EINE volle Schwingung
        
        t_start, t_end, n = 0.0, T, 1000
        
        # F(x) = -k * x
        def force_func(x):
            return -k * x
            
        t, x, v = velocity_verlet(x0, v0, t_start, t_end, n, force_func, m=m, plotchoose=False)
        
        # Nach exakt einer Periode T muss das System wieder im Ausgangszustand sein
        # Orte prüfen wir auf 2 Nachkommastellen, da numerische Integration minimale Rundungsfehler hat
        self.assertAlmostEqual(x[-1], x0, places=2, msg="Position nach einer Schwingungsperiode nicht erhalten.")
        self.assertAlmostEqual(v[-1], v0, places=2, msg="Geschwindigkeit nach einer Schwingungsperiode nicht erhalten.")

if __name__ == "__main__":
    unittest.main()