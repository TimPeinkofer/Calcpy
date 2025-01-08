import numpy as np

def matrix_vector(N, Matrix, Vector):
    """
    Multiply a matrix by a vector.

    Args:
        N (int): Size of the matrix and vector.
        Matrix (ndarray): Input matrix.
        Vector (ndarray): Input vector.

    Returns:
        ndarray: Resulting vector after multiplication.
    """
    Solution = np.zeros(N)
    for i in range(N):
        for j in range(N):
            Solution[i] += Matrix[i][j] * Vector[j]
    return Solution

def Eigenvalue_calc(mat, vec, Iteration):
    """
    Compute an eigenvalue and eigenvector using power iteration.

    Args:
        mat (ndarray): Input matrix.
        vec (ndarray): Initial vector.
        Iteration (int): Number of iterations.

    Returns:
        tuple: Eigenvalue and eigenvector.
    """
    size = mat.shape[0]

    for _ in range(Iteration):
        vec = matrix_vector(size, mat, vec)
        vec = vec / np.linalg.norm(vec)  # Normalize the vector to prevent overflow

    eigenvalue = np.dot(matrix_vector(size, mat, vec), vec) / np.dot(vec, vec)
    eigenvector = vec
    return eigenvalue, eigenvector

def Eigenvalues(mat, vec, Iteration):
    """
    Compute eigenvalues and eigenvectors of a matrix and its inverse.

    Args:
        mat (ndarray): Input matrix.
        vec (ndarray): Initial vector.
        Iteration (int): Number of iterations.

    Returns:
        tuple: List of eigenvalues and eigenvectors.
    """
    eigenvalues = []
    eigenvectors = []

    try:
        inv_mat = np.linalg.inv(mat)
        for matrix in [mat, inv_mat]:
            eigenvalue, eigenvector = Eigenvalue_calc(matrix, vec, Iteration)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
    except np.linalg.LinAlgError:
        eigenvalue, eigenvector = Eigenvalue_calc(mat, vec, Iteration)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

    return eigenvalues, eigenvectors

def Aitken(Eigenvalues):
    """
    Apply Aitken's delta-squared process for accelerated convergence.

    Args:
        Eigenvalues (list): List of computed eigenvalues.

    Returns:
        float: Accelerated eigenvalue.
    """
    if len(Eigenvalues) < 3:
        return Eigenvalues[-1]
    else:
        x_n, x_n1, x_n2 = Eigenvalues[-3], Eigenvalues[-2], Eigenvalues[-1]
        return x_n - (x_n1 - x_n) ** 2 / (x_n2 - 2 * x_n1 + x_n)

def eigenvalue_calc_aitken(mat, vec, Iteration):
    """
    Compute an eigenvalue and eigenvector using Aitken's method.

    Args:
        mat (ndarray): Input matrix.
        vec (ndarray): Initial vector.
        Iteration (int): Number of iterations.

    Returns:
        tuple: Eigenvalue and eigenvector.
    """
    EVArray = []
    size = mat.shape[0]

    for i in range(Iteration):
        vec = matrix_vector(size, mat, vec)
        vec = vec / np.linalg.norm(vec)
        eigenvalue = np.dot(matrix_vector(size, mat, vec), vec) / np.dot(vec, vec)
        EVArray.append(eigenvalue)

        Aitken_Eigenvalue = Aitken(EVArray)

        if i >= 3 and abs(Aitken_Eigenvalue - EVArray[-1]) < 1e-8:
            eigenvalue = Aitken_Eigenvalue
            break

    return eigenvalue, vec

def Eigenvalues_Aitken(mat, vec, Iteration):
    """
    Compute eigenvalues and eigenvectors using Aitken's method.

    Args:
        mat (ndarray): Input matrix.
        vec (ndarray): Initial vector.
        Iteration (int): Number of iterations.

    Returns:
        tuple: List of eigenvalues and eigenvectors.
    """
    eigenvalues = []
    eigenvectors = []

    try:
        inv_mat = np.linalg.inv(mat)
        for matrix in [mat, inv_mat]:
            eigenvalue, eigenvector = eigenvalue_calc_aitken(matrix, vec, Iteration)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
    except np.linalg.LinAlgError:
        eigenvalue, eigenvector = eigenvalue_calc_aitken(mat, vec, Iteration)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

    return eigenvalues, eigenvectors
