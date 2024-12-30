import numpy as np



def precalc(n:int, a: int, b:int):
    """
    Calculating the stepsize and the x_values

    Args:
        n (int): Number of subintervals
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
    
    Result:
        h (float): Stepsize
        x_values (list): x values 
    """
    n = int(n)
    h = (b-a)/(n-1) #Caclulating values for integration
    x_values = np.linspace(a,b, n)
    return h, x_values


def Simpson_1_3(n: int,a: float,b: float, func) -> float: 
    """
    Calculating the integral of a function for given bounds via Simpson 1/3

    Args:
        n (int): Number of subintervals
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        func : Function that needs to be integrated
    
    Result:
        result (float): Value of the integration
    """
    h, x_values = precalc(n,a,b)
    
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
    Calculating the integral of a function for given bounds via Romberg

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


# Transformation for our used Intervall
def transform_G_L(x, b, a):
    return 0.5*(b - a) * x + 0.5*(b + a)



gauss_legendre_data = {
    1: {
        "points": [0],
        "weights": [2.0]
    },
    2: {
        "points": [-1 / 3**0.5, 1 / 3**0.5],  # ±1/√3
        "weights": [1.0, 1.0]
    },
    3: {
        "points": [0, -(3 / 5)**0.5, (3 / 5)**0.5],  # 0, ±√3/5
        "weights": [8 / 9, 5 / 9, 5 / 9]
    },
    4: {
        "points": [
            -(3 / 7 - 2 / 7 * (6 / 5)**0.5)**0.5,
            (3 / 7 - 2 / 7 * (6 / 5)**0.5)**0.5,
            -(3 / 7 + 2 / 7 * (6 / 5)**0.5)**0.5,
            (3 / 7 + 2 / 7 * (6 / 5)**0.5)**0.5
        ],  # ±(3/7 ± 2/7 * √6/5)^(1/2)
        "weights": [
            (18 + 30**0.5) / 36,
            (18 + 30**0.5) / 36,
            (18 - 30**0.5) / 36,
            (18 - 30**0.5) / 36
        ]
    },
    5: {
        "points": [
            -1 / 3 * (5 - 2 * (10 / 7)**0.5)**0.5,
            1 / 3 * (5 - 2 * (10 / 7)**0.5)**0.5,
            -1 / 3 * (5 + 2 * (10 / 7)**0.5)**0.5,
            1 / 3 * (5 + 2 * (10 / 7)**0.5)**0.5,
            0
        ],  # ±(1/3 * √(5 ± 2√10/7))
        "weights": [
            (322 + 13 * 70**0.5) / 900,
            (322 + 13 * 70**0.5) / 900,
            (322 - 13 * 70**0.5) / 900,
            (322 - 13 * 70**0.5) / 900,
            128 / 225
        ]
    }
}


# Gauß-Legendre function
def Gauss_legendre(a:float,b:float, n:int, func)-> float:
    """
    Calculating the integral of a function for given bounds via Gauß-Legendre

    Args:
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        n (int): Number of subintervals
        func : Function that needs to be integrated
    
    Result:
        res (float): Value of the integration
    """
    sum = 0

    if n > 5 or n<1:
        return ValueError("Please use a number from 1 to 5")
    
    A = gauss_legendre_data[n]["weights"]
    x_v = gauss_legendre_data[n]["points"]
    x_transformed = [transform_G_L(r, b, a) for r in x_v]  # Transform nodes for use
    
    for i in range(len(x_transformed)):
        sum += A[i] * func(x_transformed[i])  # Calculate the values with the weigths and nodes

    res = 0.5 * (b - a) * sum  # Multiply by the scaling factor

    return res


def Newton_cotes(n:int,a:float,b:float, func)-> float:
    """
    Calculating the integral of a function for given bounds via Newton_Cotes

    Args:
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        n (int): Number of subintervals
        func : Function that needs to be integrated
    
    Result:
        sum_integral (float): Value of the integration
    """

    h, x_values = precalc(n,a,b)
    
    f_v = [func(x_i) for x_i in x_values]
    sum_integral = 0

    for i in range(0, n - 2, 2):  # Calculating the value for every odd step
        sum_1 = f_v[i] + 4 * f_v[i + 1] + f_v[i + 2]
        sum_integral += h/3 * sum_1

    return sum_integral


def Trapezoidal(n:int,a:float,b:float,func)-> float:
    """
    Calculating the integral of a function for given bounds via Trapezoidal rule

    Args:
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        n (int): Number of subintervals
        func : Function that needs to be integrated
    
    Result:
        I (float): Value of the integration
    """

    h, x_values = precalc(n,a,b)
    sum = 0
    y=[func(x_i) for x_i in x_values]

    for i in range(1,n): #Calculating the middle values
        sum += y[i]

    I=h/2*(y[0]+2*sum+y[-1]) #Calculating the integral

    return I


def Simpson_3_8(n:int,a:float,b:float,f) -> float: #noch zu machen

    #divide the intervall n into two, where n1 is multible of 3
    n1 = (n // 3) * 3
    n2 = n - n1
    
    x=np.linspace(a,b,n+1)
    m1,m2,m3=0,0,0
    y=f(x)

    #h for the intervall n1 
    h1 = (x[n1] - a) / n1  
    #h for the intervall n2
    if n2 > 0:
        h2 = (b - x[n1]) / n2   
    else:
        h2 =0

    #calculate m1 
    for i in range(1,n1,3):
        m1 += y[i]
    #calculate m2
    for i in range(2,n1,3):
        m2 += y[i]
    #calculate m3
    for i in range(3,n1,3):
        m3 += y[i]

    I_Simpson=(3*h1/8)*(y[0]+ 3*m1 + 3*m2 + 2*m3 +y[n1])

    # calculate the rest with trapezoidal rule 
    if n2 > 0:
        I_Trapezoid = (h2 / 2) * (y[n1] + y[-1])
    else:
         I_Trapezoid = 0

    return I_Simpson + I_Trapezoid