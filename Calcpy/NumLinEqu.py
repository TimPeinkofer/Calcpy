import numpy as np

def gauss(matrix, vector): # Function for gauß elimination
    
    # Generate a copy of the vector and the matrix for our gauß algorithm
    U_Matrix = np.copy(matrix)
    U_vector = np.copy(vector)
    x = np.zeros((rows, 1)) # Generate a solution vektor based on the number of rows of our matrix

    rows, columns = matrix.shape
    
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
    
