import numpy as np

def h(i, x):
    return x[i + 1] - x[i]  # Schrittweite zwischen den Punkten

def cubic_splines(n, x, func):
    values = func(x)
    if n < 2:
        print("Nicht genügend Punkte für Splines")
        return None, None, None
    
    matrix = np.zeros((n - 1, n - 1), dtype=float)
    vec = np.zeros(n - 1, dtype=float)

    # Anfangsbedingungen
    matrix[0, 0] = (h(0, x) + h(1, x)) * (h(0, x) + 2 * h(1, x)) / h(1, x)
    if n > 2:
        matrix[0, 1] = (h(1, x) ** 2 - h(0, x) ** 2) / h(1, x)
    
    vec[0] = (values[2] - values[1]) / h(1, x) - (values[1] - values[0]) / h(0, x)

    # Für den Rest der Matrix
    for i in range(1, n - 2):
        matrix[i, i - 1] = h(i, x)
        matrix[i, i] = 2 * (h(i, x) + h(i + 1, x))
        matrix[i, i + 1] = h(i + 1, x)
        vec[i] = (values[i + 2] - values[i + 1]) / h(i + 1, x) - (values[i + 1] - values[i]) / h(i, x)

    # Endbedingungen
    if n > 2:
        matrix[n - 2, n - 3] = (h(n - 3, x) ** 2 - h(n - 2, x) ** 2) / h(n - 3, x)
        matrix[n - 2, n - 2] = (h(n - 2, x) + h(n - 1, x)) * (h(n - 2, x) + 2 * h(n - 1, x)) / h(n - 1, x)
    
    vec[n - 2] = (values[n] - values[n - 1]) / h(n - 1, x) - (values[n - 1] - values[n - 2]) / h(n - 2, x)
    vec *= 6

    # Konditionierung der Matrix überprüfen
    if np.linalg.cond(matrix) > 1e12:
        print("Matrix ist schlecht konditioniert, Lösung könnte ungenau sein.")
        return matrix, vec, None
    
    try:
        # Lösung des linearen Systems
        solution = np.linalg.solve(matrix, vec)
        S_0 = ((h(0, x) + h(1, x)) * solution[0] - h(0, x) * solution[1]) / h(1, x)
        S_n = ((h(n - 2, x) + h(n - 1, x)) * solution[n - 2] - h(n - 1, x) * solution[n - 3]) / h(n - 2, x)
        
        solution = np.append(solution, S_n)
        solution = np.insert(solution, 0, S_0)
        return matrix, vec, solution
    except np.linalg.LinAlgError:
        print("Keine Lösung vorhanden")
        return matrix, vec, None
 