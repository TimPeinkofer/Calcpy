import numpy as np

def Gauss_elimination(matrix, vector): # Function for gauß elimination
    
    # Generate a copy of the vector and the matrix for our gauß algorithm
    U_Matrix = np.copy(matrix)
    U_vector = np.copy(vector)
    

    rows, columns = matrix.shape
    x = np.zeros((rows, 1)) # Generate a solution vektor based on the number of rows of our matrix
    
    for i in range(rows - 1):

        if U_Matrix[i][i] == 0:
            for k in range(i + 1, rows):
                if U_Matrix[k][i] != 0:
                    # Swap the rows in both U_Matrix and U_vector if the a_ii component is zero
                    U_Matrix[[i, k]] = U_Matrix[[k, i]]
                    U_vector[[i, k]] = U_vector[[k, i]]
                    break
        
        # Continue with elimination if a_ii != 0
        for j in range(i + 1, rows):
            if U_Matrix[i][i] != 0:
                factor = U_Matrix[j][i] / U_Matrix[i][i]
                U_Matrix[j] = U_Matrix[j] - factor * U_Matrix[i]
                U_vector[j] = U_vector[j] - factor * U_vector[i]
    
    
    for i in range(rows):
        index = rows - i - 1
        b_new = U_vector[index] / U_Matrix[index, index]
        
        for r in range(index + 1, rows):
            b_new -= U_Matrix[index, r] * x[r] / U_Matrix[index, index]
        
        x[index] = b_new
    
    return x
    

# Calculate the function with different overrelaxation factors
def overrelaxation(A,b,Max_iterations):
    for factor in np.arange(1, 2, 0.1):  
        result = overrelax_calc(A, b, Max_iterations, factor)
        if result is not None:
            print(f"Solution vector after over-relaxation (w = {factor}):")
            print(result)
            print(" ")

def overrelax_calc(m, vector, iterations, w, tol=1e-3):  # Overrelaxation function
    rows, columns = m.shape
    x = vector.copy()  # Use a copy of the initial guess to avoid modifying the original
    for j in range(iterations):
        x_old = x.copy()
        for i in range(rows):
            if m[i][i] != 0:
                # Calculate the factor for iteration
                factor = w / m[i][i]
                
                # Calculate the sums for the iteration part
                sum1 = sum(m[i][l] * x[l] for l in range(i))
                sum2 = sum(m[i][l] * x_old[l] for l in range(i, rows))
                
                # Update x based on our sums and the factor
                x[i] = x_old[i] + factor * (vector[i] - sum1 - sum2)

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Convergence achieved after {j + 1} iterations with w = {w}")
            return x

    print(f"No convergence after {iterations} iterations with w = {w}")
    return None

