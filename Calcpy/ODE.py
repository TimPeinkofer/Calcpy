import numpy as np
import matplotlib.pyplot as plt

def precalc(xm: float, x0: float, n: int):
    """
    Pre-calculation of x values and the step size h.

    Args:   
    xm: float: final value of x
    x0: float: initial value of x
    n: int: number of steps

    Returns:
    x: np.array: x values
    y: np.array: y values
    h: float: step size
    """
    h = (xm - x0) / n
    x = np.linspace(x0, xm, n + 1)  # Ensure last point xm is included
    y = np.zeros(len(x))
    return x, y, h

def sol_plot(x: np.array, y: np.array, plotchoose: bool):
    """
    Plot the numerical solution of ODE.

    Args:
      x: np.array: x values
      y: np.array: y values
      plotchoose: bool: choose to plot the solution or not
    """
    if plotchoose:
        plt.figure(figsize=(16, 9))
        plt.plot(x, y)
        plt.title("Numerical solution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

def Heun(x0: float, xm: float, y0: float, n: int, f, plotchoose: bool):
    """
    Numerical solution of ODE via Heun method.

    Args:
      x0: float: initial value of x  
      xm: float: final value of x
      y0: float: initial value of y
      n: int: number of steps
      f: function: derivative function
      plotchoose: bool: choose to plot the solution or not

    Returns:
      x: np.array: x values
      y: np.array: y values
    """
    x, y, h = precalc(xm, x0, n)

    y[0] = y0

    for i in range(1, len(x)):
        ypred = y[i - 1] + h * f(x[i - 1], y[i - 1])
        ycorr = y[i - 1] + h / 2 * (f(x[i - 1], y[i - 1]) + f(x[i], ypred))
        y[i] = ycorr

    sol_plot(x, y, plotchoose)
    return x, y

def adam_predictor(f, y_0: float, x_0: float, x_m: float, n: int, plotchoose: bool):
    """
    Numerical solution of ODE via Adams-Bashforth method (predictor).

    Args:
      f: function: derivative function
      y_0: float: initial value of y
      x_0: float: initial value of x
      x_m: float: final value of x
      n: int: number of steps
      plotchoose: bool: choose to plot the solution or not

    Returns:
      x: np.array: x values
      y_values: np.array: predicted y values
    """
    h = (x_m - x_0) / n  # Step size
    x, y_values, _ = precalc(x_m, x_0, n)

    y_values[0] = y_0

    # Calculate the first three values via Heun method
    for i in range(1, 4):
        _, y = Heun(x[i - 1], x[i], y_values[i - 1], 1, f, False)
        y_values[i] = y[-1]

    # Adams-Bashforth predictor for the rest
    for i in range(3, n):
        y_values[i + 1] = y_values[i] + (h / 24) * (
            55 * f(x[i], y_values[i])
            - 59 * f(x[i - 1], y_values[i - 1])
            + 37 * f(x[i - 2], y_values[i - 2])
            - 9 * f(x[i - 3], y_values[i - 3])
        )

    sol_plot(x, y_values, plotchoose)
    return x, y_values

def adam_corrector(f, y_values: np.array, x: np.array, n: int, h: float, plotchoose: bool):
    """
    Corrector method for Adams-Moulton method.

    Args:
      f: function: derivative function
      y_values: np.array: y values
      x: np.array: x values
      n: int: number of steps
      h: float: step size
      plotchoose: bool: choose to plot the solution or not

    Returns:
      y_values: np.array: corrected y values
    """
    for i in range(3, n):
        y_values[i + 1] = y_values[i] + (h / 24) * (
            9 * f(x[i + 1], y_values[i + 1])
            + 19 * f(x[i], y_values[i])
            - 5 * f(x[i - 1], y_values[i - 1])
            + f(x[i - 2], y_values[i - 2])
        )

    sol_plot(x, y_values, plotchoose)
    return y_values

def adam_ode_int(f, y_0: float, x0: float, xm: float, n: int, plotchoose: bool):
    """
    Numerical solution of ODE via predictor-corrector method.

    Args:
      f: function: derivative function
      y_0: float: initial value of y
      x0: float: initial value of x
      xm: float: final value of x
      n: int: number of steps
      plotchoose: bool: choose to plot the solution or not

    Returns:
      y_corr: np.array: corrected y values
      y_pred: np.array: predicted y values
    """
    x, y_pred = adam_predictor(f, y_0, x0, xm, n, plotchoose)
    h = (xm - x0) / n
    y_corr = adam_corrector(f, y_pred, x, n, h, plotchoose)
    return y_corr, y_pred

def Adam(x0: float, xm: float, y0: float, n: int, f, plotchoose: bool):
    """
    Numerical solution of ODE via Adams method.

    Args:
      x0: float: initial value of x  
      xm: float: final value of x
      y0: float: initial value of y
      n: int: number of steps
      f: function: derivative function
      plotchoose: bool: choose to plot the solution or not

    Returns:
      x: np.array: x values
      y: np.array: y values
    """
    x, y, h = precalc(xm, x0, n)

    y[0] = y0
    y[1] = y[0] + h * f(x[0], y[0])  # First two values via Heun
    y[2] = y[1] + h * f(x[1], y[1])

    for i in range(3, len(x)):  # Adams method
        k = y[i - 1] + (h / 12) * (5 * f(x[i - 1], y[i - 1]) + 8 * f(x[i - 2], y[i - 2]) - f(x[i - 3], y[i - 3]))
        y[i] = k

    sol_plot(x, y, plotchoose)  # Plot the solution
    return x, y

def runge_kutta(x_start: float, x_end: float, y_0: float, n: int, f, plotchoose: bool):
    """
    Numerical solution of ODE via Runge-Kutta 4th order method.

    Args:
      x_start: float: initial value of x
      x_end: float: final value of x
      y_0: float: initial value of y
      n: int: number of steps
      f: function: derivative function
      plotchoose: bool: choose to plot the solution or not

    Returns:
      x_values: np.array: x values
      y_values: np.array: y values
    """
    h = (x_end - x_start) / n  # Step size
    x_values = np.linspace(x_start, x_end, n + 1)
    y_values = [y_0]  # Solution values
    y = y_0

    for i in range(n):
        x_i = x_values[i]
        k_1 = h * f(x_i, y)
        k_2 = h * f(x_i + 0.5 * h, y + 0.5 * k_1)
        k_3 = h * f(x_i + 0.5 * h, y + 0.5 * k_2)
        k_4 = h * f(x_i + h, y + k_3)

        y = y + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        y_values.append(y)

    sol_plot(x_values, np.array(y_values), plotchoose)
    return x_values, np.array(y_values)


def systems_of_ODE(system, y0:list, t:list) -> list:
    """
    ODE System solver based ona Runge-Kutta 4th order algorithm.

    Args:
      system : ODE equation system.
      y0 (list): List of boundary conditions.
      t (list): List of t values
    
    Return:
      
    
    """
    n = len(t)
    h = t[1] - t[0]  # Calculate step size
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        k1 = system(y[i - 1], t[i - 1])
        k2 = system(y[i - 1] + h * k1 / 2, t[i - 1] + h / 2)
        k3 = system(y[i - 1] + h * k2 / 2, t[i - 1] + h / 2)
        k4 = system(y[i - 1] + h * k3, t[i - 1] + h)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y