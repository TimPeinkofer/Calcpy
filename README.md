# Calcpy

[![CI](https://github.com/TimPeinkofer/Calcpy/actions/workflows/ci.yml/badge.svg)](https://github.com/TimPeinkofer/Calcpy/actions/workflows/ci.yml)

A simple Python library for numerical mathematical calculations and its utility functions.

## Installation

Thanks to the modern package configuration, `numpy` and `matplotlib` will be installed automatically as dependencies.

### Option 1: Direct Installation (Recommended for Users)
You can install the library directly from GitHub using `pip`:
```bash
pip install git+[https://github.com/TimPeinkofer/Calcpy.git](https://github.com/TimPeinkofer/Calcpy.git)
```

### Option 2: Local Installation (For Developers)
If you want to modify the code, clone the repository and install it in editable mode:
```bash
git clone [https://github.com/TimPeinkofer/Calcpy.git](https://github.com/TimPeinkofer/Calcpy.git)
cd Calcpy
pip install -e .
```

## Overview
The following types of numerical problems are addressed in this library:

1. Nonlinear equations
2. Systems of linear equations
3. Eigenvalue and Matrix calculations
4. Interpolation
5. Integration
6. Differentiation
7. Ordinary differential equations
8. Boundary and Initial value problems
9. Hyperbolic, Parabolic and Elliptic Partial differential equations

---

## Modules and Features

### 1. Nonlinear Equations
Methods to find the roots of nonlinear functions or systems.

#### `newtons_method(f, df, x0, max_iter, epsilon=1e-6)`
Finds the root of a function using Newton's method.
- **Args:**
  - `f`: Function to find the root for.
  - `df`: Derivative of the function.
  - `x0`: Initial guess.
  - `max_iter`: Maximum number of iterations.
  - `epsilon`: Convergence criterion.
- **Returns:**
  - `float`: Approximation of the root (or None if no convergence).

#### `linear_interpolation(f, x0, x1, max_iter, epsilon=1e-6)`
Finds the root of a function using linear interpolation (Secant method).
- **Args:**
  - `f`: Function to find the root for.
  - `x0`, `x1`: Initial guesses.
  - `max_iter`: Maximum number of iterations.
  - `epsilon`: Convergence criterion.
- **Returns:**
  - `float`: Approximation of the root.

#### `solve_fixed_point(f1, f2, x_init, y_init, max_iter, tol=1e-8)`
Solves a system of two equations using fixed-point iteration.
- **Args:**
  - `f1`, `f2`: Functions for the equations.
  - `x_init`, `y_init`: Initial guesses.
  - `max_iter`: Maximum iterations.
  - `tol`: Convergence tolerance.
- **Returns:**
  - `tuple`: Approximations of the roots `(x, y)`.

#### `newton_halley(f, df, d2f, x0, max_iter, epsilon=1e-6)`
Finds the root of a function using the Newton-Halley method.
- **Args:**
  - `f`: Function to evaluate.
  - `df`, `d2f`: First and second derivatives of the function.
  - `x0`: Initial guess.
  - `max_iter`: Maximum number of iterations.
  - `epsilon`: Convergence criterion.
- **Returns:**
  - `float`: Approximation of the root.

#### `bisection_method(x1, x2, f, max_iter, eps=1e-6)`
Finds the root of a function using the bisection method.
- **Args:**
  - `x1`, `x2`: Lower and upper bounds of the interval.
  - `f`: Function to evaluate.
  - `max_iter`: Maximum number of iterations.
  - `eps`: Convergence criterion.
- **Returns:**
  - `float`: Approximation of the root.

#### `jacobi_method(A, b, init_val, max_iter, tol=1e-6)`
Solves a linear system using the iterative **Jacobi method**.
- **Args:**
  - `A`: Coefficient matrix `(N x N)`.
  - `b`: Right-hand side vector `(N,)`.
  - `init_val`: Initial guess for the solution.
  - `max_iter`: Maximum number of iterations.
  - `tol`: Convergence tolerance.
- **Returns:**
  - `ndarray`: Solution vector `(N,)`.

#### `gauss_seidel(m, vector, init_val, max_iter, tol=1e-6)`
Solves a linear system using the iterative **Gauss-Seidel method**.
- **Args:**
  - `m`: Coefficient matrix `(N x N)`.
  - `vector`: Right-hand side vector `(N,)`.
  - `init_val`: Initial guess.
  - `max_iter`: Maximum number of iterations.
  - `tol`: Convergence tolerance.
- **Returns:**
  - `ndarray`: Solution vector `(N,)`.

---

### 2. Systems of Linear Equations
Direct and iterative methods for solving $Ax = b$ and matrix operations.

#### `Gauss_elimination(matrix, vector)`
Solves a linear system using standard Gaussian elimination.
- **Args:**
  - `matrix`: Coefficient matrix `(N x N)`.
  - `vector`: Right-hand side vector `(N,)`.
- **Returns:**
  - `ndarray`: Solution vector `(N,)`.

#### `Gauss_elimination_pivoted(matrix, vector)`
Solves a linear system using Gaussian elimination with **partial pivoting** to minimize numerical errors.
- **Args:**
  - `matrix`: Coefficient matrix `(N x N)`.
  - `vector`: Right-hand side vector `(N,)`.
- **Returns:**
  - `ndarray`: Solution vector `(N,)`.

#### `Gauss_Jordan(A)`
Performs Gauss-Jordan elimination.
- **Args:**
  - `A`: The augmented matrix to be reduced.
- **Returns:**
  - `ndarray`: Reduced row echelon form of the matrix.

#### `inverse_matrix(A)`
Calculates the inverse of a square matrix using Gauss-Jordan elimination.
- **Args:**
  - `A`: Input matrix.
- **Returns:**
  - `ndarray`: Inverse of the matrix.

#### `determinant(matrix)`
Calculates the determinant of a square matrix.
- **Args:**
  - `matrix`: Base matrix.
- **Returns:**
  - `float`: Value of the determinant.



#### `overrelaxation(A, b, Max_iterations)`
Finds the optimal solution using **Successive Over-Relaxation (SOR)** by evaluating different relaxation factors.
- **Args:**
  - `A`: Coefficient matrix.
  - `b`: Right-hand side vector.
  - `Max_iterations`: Maximum iterations per factor.
- **Returns:**
  - `ndarray`: Best solution vector found.

---

### 3. Eigenvalue and Eigenvector Computation
Methods for computing eigenvalues and eigenvectors using power iteration and Aitken's acceleration.

#### `Eigenvalue_calc(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using the **power iteration method**.
- **Args:**
  - `mat`: Input matrix.
  - `vec`: Initial guess vector.
  - `Iteration`: Number of iterations.
- **Returns:**
  - `tuple`: `(eigenvalue, eigenvector)`.

#### `Eigenvalues(mat, vec, Iteration)`
Computes the largest and smallest eigenvalues (using the matrix inverse).
- **Args:**
  - `mat`: Input matrix.
  - `vec`: Initial guess vector.
  - `Iteration`: Number of iterations.
- **Returns:**
  - `tuple`: List of eigenvalues and list of eigenvectors.

#### `Aitken(Eigenvalues)`
Applies **Aitken’s delta-squared process** to accelerate convergence.
- **Args:**
  - `Eigenvalues`: Sequence of approximated eigenvalues.
- **Returns:**
  - `float`: Accelerated eigenvalue.

#### `eigenvalue_calc_aitken(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using power iteration with Aitken acceleration.
- **Args:**
  - `mat`: Input matrix.
  - `vec`: Initial guess vector.
  - `Iteration`: Number of iterations.
- **Returns:**
  - `tuple`: `(eigenvalue, eigenvector)`.

---

### 4. Interpolation
Algorithms for smooth data interpolation.

#### `cubic_splines(n, x, func)`
Constructs cubic splines for interpolation across a set of points.
- **Args:**
  - `n`: Number of intervals.
  - `x`: Array of grid points.
  - `func`: Function to evaluate at grid points.
- **Returns:**
  - `tuple`: `(matrix, vec, solution)` containing the spline system and second derivatives.

---

### 5. Numerical Integration
Methods to approximate definite integrals.

#### `Simpson_1_3(n, a, b, func)`
Calculates the integral using Simpson's 1/3 rule.
- **Args:**
  - `n`: Number of subintervals.
  - `a`, `b`: Lower and upper bounds.
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

#### `Romberg(n, a, b, func)`
Calculates the integral using Romberg integration.
- **Args:**
  - `n`: Number of subintervals.
  - `a`, `b`: Lower and upper bounds.
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

#### `Gauss_legendre(a, b, n, func)`
Calculates the integral using Gauss-Legendre quadrature (up to 5 points).
- **Args:**
  - `a`, `b`: Lower and upper bounds.
  - `n`: Number of points (1 to 5).
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

#### `Newton_cotes(n, a, b, func)`
Calculates the integral using the Newton-Cotes formula.
- **Args:**
  - `n`: Number of subintervals.
  - `a`, `b`: Lower and upper bounds.
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

#### `Trapezoidal(n, a, b, func)`
Calculates the integral using the Trapezoidal rule.
- **Args:**
  - `n`: Number of subintervals.
  - `a`, `b`: Lower and upper bounds.
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

#### `Simpson_3_8(n, a, b, func)`
Calculates the integral using Simpson's 3/8 rule.
- **Args:**
  - `n`: Number of subintervals.
  - `a`, `b`: Lower and upper bounds.
  - `func`: Function to integrate.
- **Returns:**
  - `float`: Approximated integral.

---

### 6. Numerical Differentiation Methods

#### `Central_diff_first_deri(x, h, func)`
Computes the first derivative using 4th-order central differences.
- **Args:**
  - `x`: Point to evaluate.
  - `h`: Step size.
  - `func`: Function to differentiate.
- **Returns:**
  - `float`: Derivative at `x`.

#### `Richardson(x, h, func)`
Computes the derivative using Richardson extrapolation.
- **Args:**
  - `x`: Point to evaluate.
  - `h`: Step size.
  - `func`: Function to differentiate.
- **Returns:**
  - `float`: Extrapolated derivative.

#### `spline_derivative(i, x_val, x, func)`
Computes the derivative using cubic spline interpolation.
- **Args:**
  - `i`: Index of the interval.
  - `x_val`: Evaluation point.
  - `x`: Grid points.
  - `func`: Function.
- **Returns:**
  - `float`: Derivative of the spline.

---

### 7. Numerical ODE Solvers
Algorithms for solving ordinary differential equations and systems.

#### `Heun(x0, xm, y0, n, f, plotchoose)`
Solves an ODE using Heun’s method (2nd order).
- **Args:**
  - `x0`, `xm`: Initial and final x values.
  - `y0`: Initial y value.
  - `n`: Number of steps.
  - `f`: Derivative function `f(x, y)`.
  - `plotchoose`: Boolean to plot the result.
- **Returns:**
  - `tuple`: `(x_values, y_values)`.

#### `adam_ode_int(f, y_0, x0, xm, n, plotchoose)`
Solves an ODE using the **Adams Predictor-Corrector** method.
- **Args:**
  - `f`: Derivative function.
  - `y_0`: Initial y value.
  - `x0`, `xm`: Initial and final x values.
  - `n`: Number of steps.
  - `plotchoose`: Boolean to plot the result.
- **Returns:**
  - `tuple`: `(y_corrected, y_predicted)`.

#### `Adam(x0, xm, y0, n, f, plotchoose)`
Solves an ODE using the classical Adams method.
- **Args:**
  - `x0`, `xm`: Initial and final x values.
  - `y0`: Initial y value.
  - `n`: Number of steps.
  - `f`: Derivative function.
  - `plotchoose`: Boolean to plot the result.
- **Returns:**
  - `tuple`: `(x_values, y_values)`.

#### `runge_kutta(x_start, x_end, y_0, n, f, plotchoose)`
Solves an ODE using the **Runge-Kutta 4th order** method.
- **Args:**
  - `x_start`, `x_end`: Initial and final x values.
  - `y_0`: Initial y value.
  - `n`: Number of steps.
  - `f`: Derivative function.
  - `plotchoose`: Boolean to plot the result.
- **Returns:**
  - `tuple`: `(x_values, y_values)`.

#### `systems_of_ODE(system, y0, t)`
Solves a system of ODEs using the Runge-Kutta 4th order algorithm.
- **Args:**
  - `system`: ODE equation system function.
  - `y0`: List of initial conditions.
  - `t`: Array/List of time steps.
- **Returns:**
  - `ndarray`: Solution matrix `y`.

---

### 8. Boundary Value Problem Solver

#### `Matrix_method(x0, xn, n, y0, yn)`
Solves a BVP using the **finite difference** (matrix) method.
- **Args:**
  - `x0`, `xn`: Left and right boundaries.
  - `n`: Number of interior points.
  - `y0`, `yn`: Boundary values.
- **Returns:**
  - `ndarray`: Solution values at the grid points.

#### `solve_by_shooting(ode, x_1, x_2, n, v_0, u_1, u_2, max_iter)`
Solves a BVP using the **shooting method**.
- **Args:**
  - `ode`: ODE system function.
  - `x_1`, `x_2`: Domain boundaries.
  - `n`: Number of grid points.
  - `v_0`: Initial guesses for the derivative.
  - `u_1`, `u_2`: Boundary values.
  - `max_iter`: Maximum iterations for root finding.
- **Returns:**
  - `tuple`: `(corrected_v, x_values, y_values)`.

---

### 9. Partial Differential Equations (PDE)

#### `elliptic_solver_laplace_2D(bc, h, x_bounds, y_bounds, g, maxiter)`
Solves the 2D Laplace/Poisson equation using Gauss-Seidel.
- **Args:**
  - `bc`: Boundary conditions `[bottom, top, left, right]`.
  - `h`: Grid spacing.
  - `x_bounds`, `y_bounds`: Domain limits.
  - `g`: Source term function (default=0).
  - `maxiter`: Iteration count.
- **Returns:**
  - `tuple`: `(u, x_values, y_values)`.

#### `elliptic_solver_laplace_3D(bc, h, x_bounds, y_bounds, z_bounds, g, maxiter)`
Solves the 3D Laplace/Poisson equation.
- **Args:**
  - `bc`: Boundary conditions `[bottom, top, left, right, front, back]`.
  - `h`: Grid spacing.
  - `x_bounds`, `y_bounds`, `z_bounds`: Domain limits.
  - `g`: Source term function.
  - `maxiter`: Iteration count.
- **Returns:**
  - `tuple`: `(u, x_values, y_values, z_values)`.

#### `parabolic_explicit_solver(x_bounds, t_bounds, bc_t, func, h, alpha)`
Explicit finite difference solver for the 1D heat equation.
- **Args:**
  - `x_bounds`, `t_bounds`: Spatial and time intervals.
  - `bc_t`: Boundary conditions over time.
  - `func`: Initial profile `u(x, 0)`.
  - `h`: Spatial step size.
  - `alpha`: Diffusion coefficient.
- **Returns:**
  - `tuple`: `(u, x_values, t_values)`.

#### `hyperbolic_solver(x_bounds, t_bounds, bc_t, func, h, alpha)`
Solver for the 1D wave equation.
- **Args:**
  - `x_bounds`, `t_bounds`: Spatial and time intervals.
  - `bc_t`: Boundary conditions over time.
  - `func`: Initial profile `u(x, 0)`.
  - `h`: Spatial step size.
  - `alpha`: Wave speed.
- **Returns:**
  - `tuple`: `(u, x_values, t_values)`.