import numpy as np

def Monte_Carlo(n: int, a: float, b: float, func) -> float:
    """
    Calculating the integral of a function for given bounds via Monte Carlo integration.

    Args:
        n (int): Number of random samples
        a (float): Lower bound of the integral
        b (float): Upper bound of the integral
        func : Function that needs to be integrated
    
    Returns:
        result (float): Approximated value of the integration
    """
    n = int(n)
    
    # Zufällige Punkte im Intervall [a, b] generieren
    x_random = np.random.uniform(a, b, n)
    
    # Funktion an den zufälligen Punkten auswerten
    f_values = func(x_random)
    
    # Mittelwert der Funktionswerte berechnen und mit der Intervallbreite multiplizieren
    result = (b - a) * np.mean(f_values)
    
    return result


def Monte_Carlo_nD(n: int, bounds: list, func) -> float:
    """
    Calculating the integral of a multi-dimensional function via Monte Carlo integration.

    Args:
        n (int): Number of random samples
        bounds (list of tuples): List containing (lower_bound, upper_bound) for each dimension
                                 e.g., [(0, 1), (0, 2)] for a 2D integral.
        func : Function that takes a 1D array (or list) of coordinates and returns a float.
    
    Returns:
        result (float): Approximated value of the integration
    """
    n = int(n)
    dimensions = len(bounds)
    volume = 1.0
    
    # Array für die zufälligen Punkte vorbereiten (n Punkte, d Dimensionen)
    random_points = np.zeros((n, dimensions))
    
    # Für jede Dimension zufällige Werte generieren und das Hypervolumen berechnen
    for d in range(dimensions):
        a, b = bounds[d]
        random_points[:, d] = np.random.uniform(a, b, n)
        volume *= (b - a)
        
    # Funktion für jeden generierten Punkt auswerten
    # (Nutzt Listen-Abstraktion, falls die übergebene Funktion nicht vektorisiert ist)
    f_values = np.array([func(point) for point in random_points])
    
    # Integral berechnen (Volumen * durchschnittlicher Funktionswert)
    result = volume * np.mean(f_values)
    
    return result
