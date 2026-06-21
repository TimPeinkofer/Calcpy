import numpy as np
from typing import Callable

def Monte_Carlo(n: int, a: float, b: float, func: Callable) -> tuple[float, float]:
    """
    Calculating the integral of a function via Monte Carlo integration.

    Args:
        n (int): Number of random samples
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        func : Function that needs to be integrated
    
    Returns:
        tuple: (result, error)
            result (float): Approximated value of the integration
            error (float): Estimated statistical error (Standard Error of the Mean)
    """
    n = int(n)
    
    x_random = np.random.uniform(a, b, n)
    f_values = func(x_random)
    
    volume = b - a
    result = volume * np.mean(f_values)
    
    # Statistischer Fehler: Volumen * (Standardabweichung / Wurzel(n))
    # ddof=1 für die unvezerrte Stichprobenvarianz
    error = volume * (np.std(f_values, ddof=1) / np.sqrt(n))
    
    return result, error


def Monte_Carlo_nD(n: int, bounds: list, func: Callable) -> tuple[float, float]:
    """
    Calculating the integral of a multi-dimensional function via Monte Carlo integration.

    Args:
        n (int): Number of random samples
        bounds (list of tuples): [(lower, upper), ...] for each dimension
        func : Function to integrate
    
    Returns:
        tuple: (result, error)
    """
    n = int(n)
    dimensions = len(bounds)
    volume = 1.0
    random_points = np.zeros((n, dimensions))
    
    for d in range(dimensions):
        a, b = bounds[d]
        random_points[:, d] = np.random.uniform(a, b, n)
        volume *= (b - a)
        
    f_values = np.array([func(point) for point in random_points])
    
    result = volume * np.mean(f_values)
    error = volume * (np.std(f_values, ddof=1) / np.sqrt(n))
    
    return result, error