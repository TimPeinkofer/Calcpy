import numpy as np

def newtons_method(f, df, x0, max_iter, epsilon=1e-6):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if np.abs(fx) < epsilon:
            return x
        
        x = x - fx / dfx
    
    print("Did not converge.")
    return None

def linear_interpolation(f, x0, x1, max_iter, epsilon=1e-6):
    for _ in range(max_iter):
        f0, f1 = f(x0), f(x1)

        if f1 == f0:  # Prevent division by zero
            print("Division by zero encountered.")
            return None

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if np.abs(x2 - x1) < epsilon:
            print(f(x2))
            return x2
        
        x0, x1 = x1, x2
    
    print("Did not converge.")
    return None

"""
def solve_fixed_point(f1, f2, x_init, y_init, max_iter, tol=1e-6):
    # Initial guesses for x and y
    x, y = x_init, y_init
    for i in range(max_iter):
        # Update x and y using the fixed-point iterations
        x_new = f1(y)
        y_new = f2(x)
        
        # Check for convergence
        if np.abs(x_new - x) < tol and np.abs(y_new - y) < tol:
            print(f"Converged in {i+1} iterations.")
            return x_new, y_new
        
        x, y = x_new, y_new

    print("Did not converge.")
    return None, None
    """