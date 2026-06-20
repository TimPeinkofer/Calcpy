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
        x (float): Approximation of the root, or None if no convergence.
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
        x2 (float): Approximation of the root, or None if no convergence.
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


def solve_fixed_point(f1, f2, x_init: float, y_init: float, max_iter: int, tol=1e-8) -> tuple[float,float]:
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

def newton_halley(f, df, d2f, x0: float, max_iter: int, epsilon=1e-6) -> float:
    """
    Find the root of a function using Newton-Halley method.

    Args:
        f (function): Function for which the root is to be found.
        df (function): First derivative of the function.
        d2f (function): Second derivative of the function.
        x0 (float): Initial guess.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Convergence criterion.

    Returns:
        x (float): Approximation of the root, or None if no convergence.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)

        if np.abs(fx) < epsilon:
            return x

        if dfx == 0:
            print("Derivative is zero. Cannot proceed.")
            return None

        x -= (2*fx*dfx)/(2*dfx**2-d2fx*fx)

    print("Did not converge.")
    return None


def bisection_method(x1: float, x2: float, f, max_iter: int, eps=1e-6) -> float:
    """
    Solving nonlinear equations with bisection method.

    Args:
        x1 (float): Lower bound of the interval.
        x2 (float): Upper bound of the interval.
        f (function): Function for which the root is to be found.
        max_iter (int): Maximum number of iterations.
        eps (float): Convergence criterion.

    Returns:
        float: Approximation of the root, or None if no convergence.
    """
    x_new = (x1+x2)/2
    for _ in range(max_iter):

        if f(x_new)*f(x1) < 0:
            x2 = x_new
        
        elif f(x_new)*f(x2) < 0:
            x1 = x_new
        
        x_new = (x1+x2)/2

        if np.abs(f(x_new)) < eps:
            return x_new

    print("Did not converge.")
    return None

import numpy as np

def jacobi_method(A: np.ndarray, b: np.ndarray, init_val: list, max_iter: int, tol=1e-6) -> np.ndarray:
    """
    Solve a linear system using the iterative Jacobi method.
    
    In contrast to Gauss-Seidel, the Jacobi method uses only the values 
    from the previous iteration step to calculate the new approximations.

    Args:
        A (numpy.ndarray): Coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        init_val (list): Initial guess for the solution vector x.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Solution vector if convergence is achieved.
    """
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Coefficient matrix A must be square.")
        
    x_old = np.array(init_val, dtype=float)
    x_new = np.zeros_like(x_old)

    for iteration in range(max_iter):
        for i in range(rows):
            if A[i, i] == 0:
                raise ValueError("Diagonal element is zero, cannot proceed with Jacobi method.")
            
            # Sum over all j != i
            s = 0.0
            for j in range(cols):
                if j != i:
                    s += A[i, j] * x_old[j]
            
            x_new[i] = (b[i] - s) / A[i, i]

        # Check for convergence (using infinity norm)
        if np.linalg.norm(x_new - x_old, ord=np.inf) < tol:
            return x_new

        # Update x_old for the next iteration
        x_old = x_new.copy()

    raise RuntimeError("Jacobi method did not converge within the maximum number of iterations.")


def Gauss_elimination_pivoted(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Function for Gaussian elimination using partial pivoting to reduce numerical errors.
    
    Swaps rows to ensure the largest absolute value in the current column 
    becomes the pivot element, minimizing floating-point inaccuracies.

    Args:
        matrix (numpy.ndarray): Coefficient matrix.
        vector (numpy.ndarray): Right-hand side vector.

    Returns:
        x (numpy.ndarray): Solution vector.
    """
    U_Matrix = np.copy(matrix).astype(float)
    U_vector = np.copy(vector).astype(float)
    rows, columns = U_Matrix.shape
    x = np.zeros(rows)
    
    # Forward elimination with partial pivoting
    for i in range(rows - 1):
        # --- NEW: Partial Pivoting ---
        # Find the row with the largest absolute value in the current column i
        max_row_index = i + np.argmax(np.abs(U_Matrix[i:rows, i]))
        
        if max_row_index != i:
            # Swap the current row i with the row containing the maximum value
            U_Matrix[[i, max_row_index]] = U_Matrix[[max_row_index, i]]
            U_vector[[i, max_row_index]] = U_vector[[max_row_index, i]]
        # -----------------------------
            
        if np.isclose(U_Matrix[i, i], 0):
            raise ValueError("Matrix is singular or nearly singular.")

        for j in range(i + 1, rows):
            factor = U_Matrix[j, i] / U_Matrix[i, i]
            U_Matrix[j] -= factor * U_Matrix[i]
            U_vector[j] -= factor * U_vector[i]

    # Back substitution (same as before)
    for i in range(rows - 1, -1, -1):
        if np.isclose(U_Matrix[i, i], 0):
            raise ValueError("Matrix is singular or nearly singular.")
        x[i] = (U_vector[i] - np.dot(U_Matrix[i, i + 1:], x[i + 1:])) / U_Matrix[i, i]
    
    return x