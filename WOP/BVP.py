import numpy as np
import matplotlib.pyplot as plt

def Matrix(h, n):
    diag_main = -(2 + 2 * h**2) * np.ones(n)  
    diag_upper = np.ones(n - 1)  
    diag_lower = np.ones(n - 1)  
    A = np.diag(diag_main) + np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
    return A

def Vector(n, y0, yn):
    b = np.zeros(n)
    b[0] -= y0  
    b[-1] -= yn  
    return b


def Matrix_method(x0,xn,n,y0,yn):

    h = (xn-x0)/ (n+1)
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
    plt.plot(list_x, list_y, color='b')
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return y




