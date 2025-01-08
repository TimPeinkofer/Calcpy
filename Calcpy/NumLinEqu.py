import numpy as np

def Gauss_elimination(matrix, vector): 
    """
    Function for Gaussian elimination.

    Args:
        matrix (numpy.ndarray): Coefficient matrix.
        vector (numpy.ndarray): Right-hand side vector.

    Returns:
        numpy.ndarray: Solution vector.
    """
    # Create copies of the input matrix and vector to avoid modifying the originals
    U_Matrix = np.copy(matrix)
    U_vector = np.copy(vector)

    rows, columns = matrix.shape
    x = np.zeros(rows)  # Initialize the solution vector

    # Forward elimination
    for i in range(rows - 1):
        if U_Matrix[i, i] == 0:
            for k in range(i + 1, rows):
                if U_Matrix[k, i] != 0:
                    # Swap rows in both U_Matrix and U_vector
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    U_vector[[i, k]] = U_vector[[k, i]]
                    break

        # Eliminate entries below the pivot
        for j in range(i + 1, rows):
            if U_Matrix[i, i] != 0:
                factor = U_Matrix[j, i] / U_Matrix[i, i]
                U_Matrix[j] -= factor * U_Matrix[i]
                U_vector[j] -= factor * U_vector[i]

    # Back substitution
    for i in range(rows - 1, -1, -1):
        if U_Matrix[i, i] == 0:
            raise ValueError("Matrix is singular or nearly singular.")
        x[i] = (U_vector[i] - np.dot(U_Matrix[i, i + 1:], x[i + 1:])) / U_Matrix[i, i]

    return x


def overrelaxation(A, b, Max_iterations):
    """
    Perform over-relaxation for solving a linear system.

    Args:
        A (numpy.ndarray): Coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        Max_iterations (int): Maximum number of iterations.

    Prints:
        Solution vectors for different relaxation factors.
    """
    for factor in np.arange(1.0, 2.0, 0.1):
        result = overrelax_calc(A, b, Max_iterations, factor)
        if result is not None:
            print(f"Solution vector after over-relaxation (w = {factor:.1f}):")
            print(result)
            print(" ")


def overrelax_calc(m, vector, iterations, w, tol=1e-3):
    """
    Perform over-relaxation calculation for a given relaxation factor.

    Args:
        m (numpy.ndarray): Coefficient matrix.
        vector (numpy.ndarray): Right-hand side vector.
        iterations (int): Maximum number of iterations.
        w (float): Relaxation factor.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Solution vector if convergence is achieved, otherwise None.
    """
    rows, columns = m.shape
    x = np.zeros_like(vector, dtype=float)  # Initialize solution vector with zeros

    for j in range(iterations):
        x_old = x.copy()
        for i in range(rows):
            if m[i, i] == 0:
                raise ValueError("Diagonal element is zero, cannot proceed with over-relaxation.")
            
            # Compute sums for the iteration
            sum1 = np.dot(m[i, :i], x[:i])
            sum2 = np.dot(m[i, i + 1:], x_old[i + 1:])

            # Update x based on the relaxation factor
            x[i] = (1 - w) * x_old[i] + (w / m[i, i]) * (vector[i] - sum1 - sum2)

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Convergence achieved after {j + 1} iterations with w = {w:.1f}")
            return x

    print(f"No convergence after {iterations} iterations with w = {w:.1f}")
    return None
