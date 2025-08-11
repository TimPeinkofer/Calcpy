import numpy as np
import matplotlib.pyplot as plt


# ---------- Plot-Funktionen ----------

def plot_x_u_for_different_t(x_values, t_values, u):
    """
    Plottet u(x,t) als x-u-Diagramm für verschiedene Zeitpunkte.

    Args:
        x_values (ndarray): 1D-Array der räumlichen Koordinaten x.
        t_values (ndarray): 1D-Array der Zeitwerte t.
        u (ndarray): 2D-Array der Lösung u[t_index, x_index].
    """
    plt.figure(figsize=(10, 6))
    for t_index in range(len(t_values)):
        plt.plot(x_values, u[t_index, :], label=f"t = {t_values[t_index]:.2f}")
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('u(x,t) für unterschiedliche Zeitpunkte t')
    plt.legend()
    plt.grid(True)
    plt.show()


def mesh_plot_3D(x_val, y_val, z_val, u_val):
    """
    3D-Scatterplot der Lösung.

    Args:
        x_val (ndarray): 1D-Array der x-Koordinaten.
        y_val (ndarray): 1D-Array der y-Koordinaten.
        z_val (ndarray): 1D-Array der z-Koordinaten.
        u_val (ndarray): 3D-Array der Lösung u[z, y, x].
    """
    X, Y, Z = np.meshgrid(x_val, y_val, z_val, indexing='ij')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=u_val.flatten(), cmap='plasma', marker='o')
    plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Solution")
    plt.show()


def mesh_plot_2D(x_val, y_val, u_val):
    """
    2D-Konturplot der Lösung.

    Args:
        x_val (ndarray): 1D-Array der x-Koordinaten.
        y_val (ndarray): 1D-Array der y/t-Koordinaten.
        u_val (ndarray): 2D-Array der Lösung u[y_index, x_index].
    """
    X, Y = np.meshgrid(x_val, y_val)
    plt.contourf(X, Y, u_val, 20, cmap='plasma')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t / y")
    plt.title("Solution")
    plt.show()


# ---------- Elliptischer Solver 2D ----------

def elliptic_solver_laplace_2D(bc, h, x_bounds, y_bounds, g=lambda x, y: 0, maxiter=100):
    """
    Löser für die Laplace-/Poisson-Gleichung in 2D mit Gauss-Seidel.

    Args:
        bc (list[float]): [unten, oben, links, rechts] - Randwerte.
        h (float): Gitterabstand in x- und y-Richtung.
        x_bounds (list[float]): [x_min, x_max] - Bereich in x-Richtung.
        y_bounds (list[float]): [y_min, y_max] - Bereich in y-Richtung.
        g (callable): Quellterm g(x, y), Standard = 0 (Laplace).
        maxiter (int): Anzahl der Iterationen.

    Returns:
        tuple: (u, x_values, y_values)
            u (ndarray): 2D-Lösung u[y_index, x_index].
            x_values (ndarray): 1D-Gitterpunkte in x.
            y_values (ndarray): 1D-Gitterpunkte in y.
    """
    n_x = int((x_bounds[1] - x_bounds[0]) / h) + 1
    n_y = int((y_bounds[1] - y_bounds[0]) / h) + 1
    x_values = np.linspace(x_bounds[0], x_bounds[1], n_x)
    y_values = np.linspace(y_bounds[0], y_bounds[1], n_y)
    u = np.zeros((n_y, n_x))

    u[0, :] = bc[0]
    u[-1, :] = bc[1]
    u[:, 0] = bc[2]
    u[:, -1] = bc[3]

    for _ in range(maxiter):
        for i in range(1, n_y - 1):
            for j in range(1, n_x - 1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] +
                                  u[i, j+1] + u[i, j-1] -
                                  h**2 * g(x_values[j], y_values[i]))

    mesh_plot_2D(x_values, y_values, u)
    return u, x_values, y_values


# ---------- Elliptischer Solver 3D ----------

def elliptic_solver_laplace_3D(bc, h, x_bounds, y_bounds, z_bounds, g=lambda x, y, z: 0, maxiter=100):
    """
    Löser für die Laplace-/Poisson-Gleichung in 3D mit Gauss-Seidel.

    Args:
        bc (list[float]): [unten, oben, links, rechts, vorne, hinten] - Randwerte.
        h (float): Gitterabstand.
        x_bounds (list[float]): [x_min, x_max] - Bereich in x-Richtung.
        y_bounds (list[float]): [y_min, y_max] - Bereich in y-Richtung.
        z_bounds (list[float]): [z_min, z_max] - Bereich in z-Richtung.
        g (callable): Quellterm g(x, y, z), Standard = 0 (Laplace).
        maxiter (int): Anzahl der Iterationen.

    Returns:
        tuple: (u, x_values, y_values, z_values)
    """
    n_x = int((x_bounds[1] - x_bounds[0]) / h) + 1
    n_y = int((y_bounds[1] - y_bounds[0]) / h) + 1
    n_z = int((z_bounds[1] - z_bounds[0]) / h) + 1
    x_values = np.linspace(x_bounds[0], x_bounds[1], n_x)
    y_values = np.linspace(y_bounds[0], y_bounds[1], n_y)
    z_values = np.linspace(z_bounds[0], z_bounds[1], n_z)
    u = np.zeros((n_z, n_y, n_x))

    u[0, :, :] = bc[0]
    u[-1, :, :] = bc[1]
    u[:, 0, :] = bc[2]
    u[:, -1, :] = bc[3]
    u[:, :, 0] = bc[4]
    u[:, :, -1] = bc[5]

    for _ in range(maxiter):
        for k in range(1, n_z - 1):
            for i in range(1, n_y - 1):
                for j in range(1, n_x - 1):
                    u[k, i, j] = (1/6) * (u[k+1, i, j] + u[k-1, i, j] +
                                          u[k, i+1, j] + u[k, i-1, j] +
                                          u[k, i, j+1] + u[k, i, j-1] -
                                          h**2 * g(x_values[j], y_values[i], z_values[k]))

    mesh_plot_3D(x_values, y_values, z_values, u)
    return u, x_values, y_values, z_values


# ---------- Parabolischer Solver (explizit) ----------

def parabolic_explicit_solver(x_bounds, t_bounds, bc_t, func, h, alpha):
    """
    Expliziter Finite-Differenzen-Löser für die Wärmeleitungsgleichung u_t = alpha^2 u_xx.

    Args:
        x_bounds (list[float]): [x_min, x_max] - Raumintervall.
        t_bounds (list[float]): [t_min, t_max] - Zeitintervall.
        bc_t (list[float|callable]): [linker Rand, rechter Rand] als Wert oder Funktion f(t).
        func (callable): Anfangsprofil u(x, 0).
        h (float): Räumlicher Gitterabstand.
        alpha (float): Diffusionskoeffizient.

    Returns:
        tuple: (u, x_values, t_values)
    """
    n_x = int((x_bounds[1] - x_bounds[0]) / h) + 1
    h_t = h**2 / (2 * 0.9 * alpha**2)
    n_t = int((t_bounds[1] - t_bounds[0]) / h_t) + 1
    x_values = np.linspace(x_bounds[0], x_bounds[1], n_x)
    t_values = np.linspace(t_bounds[0], t_bounds[1], n_t)
    u = np.zeros((n_t, n_x))
    u[0, :] = func(x_values)

    u[:, 0] = bc_t[0](t_values) if callable(bc_t[0]) else bc_t[0]
    u[:, -1] = bc_t[1](t_values) if callable(bc_t[1]) else bc_t[1]

    r = alpha**2 * h_t / h**2
    for n in range(0, n_t - 1):
        for j in range(1, n_x - 1):
            u[n+1, j] = u[n, j] + r * (u[n, j-1] - 2 * u[n, j] + u[n, j+1])

    mesh_plot_2D(x_values, t_values, u)
    return u, x_values, t_values


# ---------- Hyperbolischer Solver ----------

def hyperbolic_solver(x_bounds, t_bounds, bc_t, func, h, alpha):
    """
    Löser für die Wellengleichung u_tt = alpha^2 u_xx.

    Args:
        x_bounds (list[float]): [x_min, x_max] - Raumintervall.
        t_bounds (list[float]): [t_min, t_max] - Zeitintervall.
        bc_t (list[float|callable]): [linker Rand, rechter Rand] als Wert oder Funktion f(t).
        func (callable): Anfangsprofil u(x, 0).
        h (float): Räumlicher Gitterabstand.
        alpha (float): Wellengeschwindigkeit.

    Returns:
        tuple: (u, x_values, t_values)
    """
    n_x = int((x_bounds[1] - x_bounds[0]) / h) + 1
    h_t = h / alpha * 0.9
    n_t = max(2, int((t_bounds[1] - t_bounds[0]) / h_t) + 1)
    x_values = np.linspace(x_bounds[0], x_bounds[1], n_x)
    t_values = np.linspace(t_bounds[0], t_bounds[1], n_t)
    u = np.zeros((n_t, n_x))
    u[0, :] = func(x_values)

    u[:, 0] = bc_t[0](t_values) if callable(bc_t[0]) else bc_t[0]
    u[:, -1] = bc_t[1](t_values) if callable(bc_t[1]) else bc_t[1]

    r = alpha * h_t / h

    for j in range(1, n_x - 1):
        u[1, j] = u[0, j] + 0.5 * r**2 * (u[0, j-1] - 2 * u[0, j] + u[0, j+1])

    for n in range(1, n_t - 1):
        for j in range(1, n_x - 1):
            u[n+1, j] = 2*(1-r**2)*u[n, j] + r**2*(u[n, j-1] + u[n, j+1]) - u[n-1, j]

    mesh_plot_2D(x_values, t_values, u)
    return u, x_values, t_values



    