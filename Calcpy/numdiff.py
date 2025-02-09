
def Central_diff_first_deri(x:float, h:float, func:function) -> float:
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
def Richardson(x:float, h:float, func:function) -> float:
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