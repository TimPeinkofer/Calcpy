import numpy as np


def Simpson_1_3(n: int,a: float,b: float, func) -> float: 
    """
    Calculating the integral of a function for given bounds

    Args:
        n (int): Number of subintervals
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        func : Function that needs to be integrated
    
    Result:
        result (float): Value of the integration
    """

    h = (b-a)/(n-1) #Caclulating values for integration
    x_values = np.linspace(a,b, n)
    f_values = [func(x_i) for x_i in x_values]

    sum = 0

    sum = f_values[0]+f_values[-1] # Get the sum of the values of the integral limits

    for i in range(1,len(x_values)): # Sum all other values based on the number of steps
        
        if i % 2 == 0: # Multiply all odd index values with 2 and the others with 4 and calculate the sum
            sum += 4*f_values[i]
        
        else:
            sum += 2*f_values[i]
    
    result = h/3*sum # Get the result

    return result

def Romberg(n:int,a:float,b:float,func) -> float:
    """
    Calculating the integral of a function for given bounds

    Args:
        n (int): Number of subintervals
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        func : Function that needs to be integrated
    
    Result:
        value (float): Value of the integration
    """
    I_1 = Simpson_1_3(n,a,b,func) #Calculate two integrals via Simpson
    I_2 = Simpson_1_3(2*n,a,b,func)
    
    value = I_2 + (I_2 - I_1) / (2**4 - 1) #Error correction

    return value

