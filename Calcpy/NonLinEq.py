import numpy as np

def newtons_method(f, df, x0: float, max_iter: int, epsilon=1e-6) -> float:
    """
    Find the root of a function using Newton's method.

    Args:
        f (function): Function for which the root is to be found.
        df (function): Derivative of the function.
        x0 (float): Initial guess.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Convergence criterion.

    Returns:
        float: Approximation of the root, or None if no convergence.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if np.abs(fx) < epsilon:
            return x

        if dfx == 0:
            print("Derivative is zero. Cannot proceed.")
            return None

        x -= fx / dfx

    print("Did not converge.")
    return None


def linear_interpolation(f, x0: float, x1: float, max_iter: int, epsilon=1e-6) -> float:
    """
    Find the root of a function using linear interpolation.

    Args:
        f (function): Function for which the root is to be found.
        x0 (float): Initial guess.
        x1 (float): Second guess.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Convergence criterion.

    Returns:
        float: Approximation of the root, or None if no convergence.
    """
    for _ in range(max_iter):
        f0, f1 = f(x0), f(x1)

        if f1 == f0:  # Prevent division by zero
            print("Division by zero encountered.")
            return None

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if np.abs(x2 - x1) < epsilon:
            print(f"Root found: {f(x2)}")
            return x2

        x0, x1 = x1, x2

    print("Did not converge.")
    return None


def solve_fixed_point(f1, f2, x_init: float, y_init: float, max_iter: int, tol=1e-8):
    """
    Solve a system of equations using fixed-point iteration.

    Args:
        f1 (function): Function for the first equation.
        f2 (function): Function for the second equation.
        x_init (float): Initial guess for x.
        y_init (float): Initial guess for y.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence criterion.

    Returns:
        tuple: Approximations of the roots (x, y), or (None, None) if no convergence.
    """
    x, y = x_init, y_init

    for i in range(max_iter):
        x_new = f1(y)
        y_new = f2(x)

        if np.abs(x_new - x) < tol and np.abs(y_new - y) < tol:
            print(f"Converged in {i + 1} iterations.")
            return x_new, y_new

        x, y = x_new, y_new

    print("Did not converge.")
    return None, None
