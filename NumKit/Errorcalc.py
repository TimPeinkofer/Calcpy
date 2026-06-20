import numint, numdiff

class IntegrationandDiffMethodError:
    def __init__(self, method, a: float, b: float, func):
        """
        Initialisiert die Integrationsmethode.

        Args:
            method (function): Die Integrationsmethode (z. B. Newton_cotes oder Trapezoidal)
            a (float): Untere Grenze des Integrals
            b (float): Obere Grenze des Integrals
            func (function): Funktion, die integriert werden soll
        """
        self.method = method
        self.n = 1000  # Standardanzahl der Unterintervalle
        self.a = a
        self.b = b
        self.func = func
        self.exact = None

    def calculate_exact(self):
        """
        Führt die numerische Integration durch und speichert das Ergebnis.

        Returns:
            float: Das Ergebnis der numerischen Integration
        """
        self.exact = self.method(self.n, self.a, self.b, self.func)
        return self.exact

    def calculate_error(self, approx_value: float) -> float:
        """
        Berechnet den Fehler der numerischen Integration, falls der exakte Wert bekannt ist.

        Args:
            approx_value (float): Der approximierte Wert des Integrals

        Returns:
            float: Der Fehler der numerischen Integration
        """
        if self.exact is None:
            self.calculate_exact()
        return abs(approx_value - self.exact)


def error(method, a: float, b: float, func, approx_result: float) -> float:
    """
    Berechnet den Fehler der numerischen Integration.

    Args:
        method: Methode, die für die Berechnung des approximierten Werts verwendet wird
        a (float): Untere Grenze des Integrals
        b (float): Obere Grenze des Integrals
        func: Funktion, die integriert werden muss
        approx_result (float): Der approximierte Wert des Integrals
    
    Returns:
        error (float): Fehler der numerischen Integration
    """
    err_func = IntegrationandDiffMethodError(method, a, b, func)
    exact_value = err_func.calculate_exact()  # Calculate exact value
    error_value = err_func.calculate_error(approx_result)  # Calculate the error
    return error_value
