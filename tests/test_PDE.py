import unittest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add the path to the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calcpy')))

import PDE

class TestPDESolvers(unittest.TestCase):

    def setUp(self):
        # Alle Plotfunktionen mocken, damit keine Fenster aufgehen
        patcher1 = patch.object(PDE, "mesh_plot_2D")
        patcher2 = patch.object(PDE, "mesh_plot_3D")
        patcher3 = patch.object(PDE, "plot_x_u_for_different_t")
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        patcher1.start()
        patcher2.start()
        patcher3.start()

    def test_elliptic_solver_2D_constant_bc(self):
        bc = [1, 1, 1, 1]
        h = 0.5
        u, x, y = PDE.elliptic_solver_laplace_2D(
            bc, h, [0, 1], [0, 1], maxiter=5
        )
        self.assertEqual(u.shape, (3, 3))
        self.assertTrue(np.allclose(u[0, :], 1))
        self.assertTrue(np.allclose(u[-1, :], 1))
        self.assertTrue(np.allclose(u[:, 0], 1))
        self.assertTrue(np.allclose(u[:, -1], 1))

    def test_elliptic_solver_3D_constant_bc(self):
        bc = [1, 1, 1, 1, 1, 1]
        h = 0.5
        u, x, y, z = PDE.elliptic_solver_laplace_3D(
            bc, h, [0, 1], [0, 1], [0, 1], maxiter=3
        )
        self.assertEqual(u.shape, (3, 3, 3))
        self.assertTrue(np.allclose(u[0, :, :], 1))
        self.assertTrue(np.allclose(u[-1, :, :], 1))

    def test_parabolic_explicit_solver(self):
        def initial(x): return np.sin(np.pi * x)
        bc = [0, 0]
        u, x, t = PDE.parabolic_explicit_solver(
            [0, 1], [0, 0.1], bc, initial, h=0.5, alpha=1
        )
        self.assertEqual(u.shape[1], len(x))
        self.assertEqual(u.shape[0], len(t))
        self.assertAlmostEqual(u[0, 0], 0)

    def test_hyperbolic_solver(self):
        def initial(x): return np.sin(np.pi * x)
        bc = [0, 0]
        u, x, t = PDE.hyperbolic_solver(
            [0, 1], [0, 0.1], bc, initial, h=0.5, alpha=1
        )
        self.assertEqual(u.shape[1], len(x))
        self.assertEqual(u.shape[0], len(t))
        self.assertAlmostEqual(u[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
