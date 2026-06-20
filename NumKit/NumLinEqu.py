import numpy as np

def Gauss_elimination(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Function for Gaussian elimination.

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
    
    # Forward elimination
    for i in range(rows - 1):
        if np.isclose(U_Matrix[i, i], 0):
            for k in range(i + 1, rows):
                if not np.isclose(U_Matrix[k, i], 0):
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    U_vector[[i, k]] = U_vector[[k, i]]
                    break
            else:
                raise ValueError("Matrix is singular or nearly singular.")

        for j in range(i + 1, rows):
            factor = U_Matrix[j, i] / U_Matrix[i, i]
            U_Matrix[j] -= factor * U_Matrix[i]
            U_vector[j] -= factor * U_vector[i]

    # Back substitution
    for i in range(rows - 1, -1, -1):
        if np.isclose(U_Matrix[i, i], 0):
            raise ValueError("Matrix is singular or nearly singular.")
        x[i] = (U_vector[i] - np.dot(U_Matrix[i, i + 1:], x[i + 1:])) / U_Matrix[i, i]
    
    return x

def overrelaxation(A: np.ndarray, b: np.ndarray, Max_iterations: int):
    """
    Perform over-relaxation for solving a linear system.

    Args:
        A (numpy.ndarray): Coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        Max_iterations (int): Maximum number of iterations.

    Prints:
        Solution vectors for different relaxation factors.
    """
    best_solution = None
    best_w = None
    
    for w in np.arange(1.0, 2.0, 0.1):
        result = overrelax_calc(A, b, Max_iterations, w)
        if result is not None:
            print(f"Solution vector after over-relaxation (w = {w:.1f}):\n{result}\n")
            best_solution = result
            best_w = w
    
    if best_solution is not None:
        print(f"Best solution found with w = {best_w:.1f}")
    return best_solution

def overrelax_calc(m: np.ndarray, vector: np.ndarray, iterations: int, w: float, tol=1e-6) -> np.ndarray:
    """
    Perform over-relaxation calculation for a given relaxation factor.

    Args:
        m (numpy.ndarray): Coefficient matrix.
        vector (numpy.ndarray): Right-hand side vector.
        iterations (int): Maximum number of iterations.
        w (float): Relaxation factor.
        tol (float): Convergence tolerance.

    Returns:
        x (numpy.ndarray): Solution vector if convergence is achieved, otherwise None.
    """
    rows, _ = m.shape
    x = np.zeros_like(vector, dtype=float)
    
    for j in range(iterations):
        x_old = x.copy()
        for i in range(rows):
            if np.isclose(m[i, i], 0):
                raise ValueError("Diagonal element is zero, cannot proceed with over-relaxation.")
            sum1 = np.dot(m[i, :i], x[:i])
            sum2 = np.dot(m[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - w) * x_old[i] + (w / m[i, i]) * (vector[i] - sum1 - sum2)
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Convergence achieved after {j + 1} iterations with w = {w:.1f}")
            return x
    
    print(f"No convergence after {iterations} iterations with w = {w:.1f}")
    return None

def determinant(matrix: np.ndarray) -> float:
    """
    Function for calculating the determinant.

    Args:
        matrix (numpy.ndarray): Base matrix.

    Returns:
        det (float): Value of the determinant.
    """
    U_Matrix = np.copy(matrix).astype(float)
    rows, _ = U_Matrix.shape
    det = 1.0
    swaps = 0
    
    for i in range(rows - 1):
        if np.isclose(U_Matrix[i, i], 0):
            for k in range(i + 1, rows):
                if not np.isclose(U_Matrix[k, i], 0):
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    swaps += 1
                    break
            else:
                return 0  # Determinant is zero
        
        for j in range(i + 1, rows):
            factor = U_Matrix[j, i] / U_Matrix[i, i]
            U_Matrix[j] -= factor * U_Matrix[i]
    
    for i in range(rows):
        det *= U_Matrix[i, i]
    
    return det * (-1) ** swaps


def gauss_seidel(m: np.ndarray, vector: np.ndarray, init_val: list,max_iter: int, tol=1e-6) -> np.ndarray:
    """
    Solve a linear system using the Gauss-Seidel method.

    Args:
        m (numpy.ndarray): Coefficient matrix.
        vector (numpy.ndarray): Right-hand side vector.
        init_val (list): Initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Solution vector if convergence is achieved, otherwise raises an error.
    """
    rows, _ = m.shape
    x_old = np.array(init_val, dtype=float)

    for _ in range(max_iter):
        x_new = x_old.copy()

        for i in range(rows):
            if m[i, i] == 0:
                raise ValueError("Diagonal element is zero, cannot proceed with Gauss-Seidel.")

            x_new[i] = (vector[i] 
                        - np.dot(m[i, :i], x_new[:i]) 
                        - np.dot(m[i, i+1:], x_old[i+1:])) / m[i, i]

        x_new = np.where(np.isnan(x_new), 0, x_new)  # NaN durch 0 ersetzen

        if np.linalg.norm(x_new - x_old, ord=np.inf) < tol:
            return x_new

        x_old = x_new

    raise RuntimeError("Gauss-Seidel did not converge within the maximum number of iterations.")


def Gauss_Jordan(A: np.ndarray) -> np.ndarray:
    """
    Perform Gauss-Jordan elimination on a matrix.

    Args:
        A (numpy.ndarray): The augmented matrix to be reduced.

    Returns:
        numpy.ndarray: The reduced row echelon form of the matrix.
    """
    rows, _ = A.shape
    for i in range(rows):
        # Make the diagonal contain all 1s
        diag = A[i, i]
        if np.isclose(diag, 0):
            raise ValueError("Matrix is singular or nearly singular.")
        A[i] = A[i] / diag
        
        # Make the other rows contain 0s in the current column
        for j in range(rows):
            if j != i:
                A[j] -= A[j, i] * A[i]
    
    return A



def inverse_matrix(A: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a matrix using the Gauss-Jordan elimination method.

    Args:
        A (numpy.ndarray): The input matrix to invert.

    Returns:
        numpy.ndarray: The inverse of the input matrix.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to compute its inverse.")
    
    aug = np.hstack((A.astype(float), np.eye(n)))
    

    result = Gauss_Jordan(aug)
    

    return result[:, n:]
