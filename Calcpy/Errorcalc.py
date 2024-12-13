class IntegrationMethodError:
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

    def calculate(self):
        """
        Führt die numerische Integration durch und speichert das Ergebnis.

        Returns:
            float: Das Ergebnis der numerischen Integration
        """
        self.exact = self.method(self.n, self.a, self.b, self.func)
        return self.exact

    def error(self, approx_value: float) -> float:
        """
        Berechnet den Fehler der numerischen Integration, falls der exakte Wert bekannt ist.

        Args:
            approx_value (float): Der approximierte Wert des Integrals

        Returns:
            float: Der Fehler der numerischen Integration
        """
        if self.exact is None:
            self.calculate()
        return abs(approx_value - self.exact)