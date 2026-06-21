from Interpolation import cubic_splines, h
from typing import Callable, Optional
import numpy as np

def Central_diff_first_deri(x:float, h:float, func: Callable) -> float:
    """
    Function to calculate the first derivative via Central differences.

    Args:
        x (float): Point were the derivative needs to be calculated
        h (float): Stepsize
        func (function): Function that needed to be evaluated
    
    Return:
        value (float): Value of the derivative

    """
    # Get the value of the function
    y1 = func(x + h)
    y_1 = func(x - h)
    y_2 = func(x - 2*h)
    y2 = func(x + 2*h)
    
    # Apply the formula for central differnces
    value = (y_2 - y2 + 8*(y1-y_1)) / (12 * h)
    return value




def F(h, x, func):
    return 1/(2*h)*(func(x + h) - func(x - h)) 

# Define function for Richardson extrapolation
def psi(h, x, func):
    return (4 * F(h / 2, x, func) - F(h, x, func)) / 3


# Richardson Extrapolation
def Richardson(x:float, h:float, func: Callable) -> float:
    """
    Function to calculate the derivative via Richardson extrapolation.

    Args:
        x (float): Point were the derivative needs to be calculated
        h (float): Stepsize
        func (function): Function that needed to be evaluated
    
    Return:
        value (float): Value of the derivative

    """

    value = psi(h, x, func)
    return value


def spline_derivative(i: int, x_val: float, x: np.ndarray, func: Callable) -> Optional[float]:
    """
    Computes the first derivative of a function using cubic spline interpolation.

    Args:
        i (int): Index of the left boundary of the interval [x[i], x[i+1]].
        x_val (float): The point at which to evaluate the derivative.
        x (np.ndarray): Array of grid points.
        func (Callable): The function to be interpolated.

    Returns:
        float: The derivative of the spline at x_val, or None if evaluation fails.
    """
    n = len(x) - 1
    _, _, sol = cubic_splines(n, x, func)
    
    # KORREKTUR: Wir rechnen nur weiter, wenn wir EINE Lösung (nicht None) haben!
    if sol is not None:
        values = func(x)
        
        if x[i] <= x_val <= x[i + 1]:
            h_i = h(i, x)  # Schrittweite zwischen x[i] und x[i + 1]
            
            # Berechnung der Koeffizienten a, b, c für das Intervall [x[i], x[i+1]]
            a = (sol[i + 1] - sol[i]) / (6 * h_i)
            b = sol[i] / 2
            c = (values[i + 1] - values[i]) / h_i - (h_i / 6) * (2 * sol[i] + sol[i + 1])
            
            # Berechnung der ersten Ableitung des Splines an der Stelle x_val
            derivative = 3 * a * (x_val - x[i]) ** 2 + 2 * b * (x_val - x[i]) + c
            return float(derivative)
        else:
            print(f"x_val = {x_val} liegt außerhalb des Intervalls [{x[i]}, {x[i + 1]}]")
            return None
    
    else:
        print("No solution found")
        return None
