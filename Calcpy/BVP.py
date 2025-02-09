import numpy as np
import matplotlib.pyplot as plt
from ODE import runge_kutta
from NonLinEq import linear_interpolation

def Matrix(h: float, n: int) -> np.ndarray:
    """
    Constructs a tridiagonal matrix for the finite difference method.
    
    Args:
        h (float): Step size.
        n (int): Number of unknowns.
    
    Returns:
        np.ndarray: Tridiagonal matrix of size (n, n).
    """
    diag_main = -(2 + 2 * h**2) * np.ones(n)
    diag_upper = np.ones(n - 1)
    diag_lower = np.ones(n - 1)
    A = np.diag(diag_main) + np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
    return A

def Vector(n: int, y0: float, yn: float) -> np.ndarray:
    """
    Constructs the right-hand side vector for the finite difference method.
    
    Args:
        n (int): Number of unknowns.
        y0 (float): Boundary condition at x0.
        yn (float): Boundary condition at xn.
    
    Returns:
        np.ndarray: Right-hand side vector.
    """
    b = np.zeros(n)
    b[0] -= y0
    b[-1] -= yn
    return b

def Matrix_method(x0: float, xn: float, n: int, y0: float, yn: float) -> np.ndarray:
    """
    Solves a boundary value problem using the finite difference method.
    
    Args:
        x0 (float): Left boundary of the domain.
        xn (float): Right boundary of the domain.
        n (int): Number of interior grid points.
        y0 (float): Boundary condition at x0.
        yn (float): Boundary condition at xn.
    
    Returns:
        np.ndarray: Solution values at the grid points.
    """
    h = (xn - x0) / (n + 1)
    list_x = np.linspace(x0, xn, n + 2)
    list_y = np.zeros(n + 2)
    list_y[0] = y0
    list_y[-1] = yn

    A = Matrix(h, n)
    b = Vector(n, y0, yn)
    y = np.linalg.solve(A, b)

    for i in range(1, n + 1):
        list_y[i] = y[i - 1]

    # Plot solution
    plt.figure(figsize=(8, 6))
    plt.plot(list_x, list_y, color='b', label='Numerical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return y

def solve_by_shooting(ode: callable, x_1: float, x_2: float, n: int, v_0: list, u_1: float, u_2: float, max_iter: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solves a boundary value problem using the shooting method.
    
    Args:
        ode (callable): Function defining the ODE system.
        x_1 (float): Left boundary of the domain.
        x_2 (float): Right boundary of the domain.
        n (int): Number of interior grid points.
        v_0 (list): Initial guesses for the derivative at x_1.
        u_1 (float): Boundary condition at x_1.
        u_2 (float): Boundary condition at x_2.
        max_iter (int): Maximum number of iterations for root finding.
    
    Returns:
        tuple: (Corrected initial derivative, solution grid points, solution values)
    """
    def difference(v):
        y_0 = [u_1, v]
        _, y_values = runge_kutta(x_1, x_2, y_0, n, ode, False)
        return y_values[-1, 0] - u_2

    # Debug: Check function values at initial guesses
    print(f"difference({v_0[0]}) = {difference(v_0[0])}")
    print(f"difference({v_0[1]}) = {difference(v_0[1])}")

    v_corr = linear_interpolation(difference, v_0[0], v_0[1], max_iter)

    if v_corr is None:
        raise RuntimeError("Root finding failed.")

    y_0 = [u_1, v_corr]
    x, y = runge_kutta(x_1, x_2, y_0, n, ode, False)
    return v_corr, x, y
