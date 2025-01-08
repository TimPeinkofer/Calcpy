import numpy as np

def Central_diff_first_deri(x, h, func):
    """
    F
    
    """
    # Get the value of the function
    y1 = func(x + h)
    y_1 = func(x - h)
    y_2 = func(x - 2*h)
    y2 = func(x + 2*h)
    
    # Apply the formula for central differnces
    value = (y_2 - y2 + 8*(y1-y_1)) / (12 * h)
    return value