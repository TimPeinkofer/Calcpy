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

### Eigenvalue and Eigenvector Computation

This project implements numerical methods for computing eigenvalues and eigenvectors of a matrix using **power iteration** and **Aitken's delta-squared acceleration**.  
Both the matrix and its inverse are considered to approximate the **largest** and **smallest** eigenvalues.

#### 1. `matrix_vector(N, Matrix, Vector)`
Multiplies a square matrix by a vector.  
- **Args:** - `N`: Size of the square matrix  
  - `Matrix`: Input matrix `(N x N)`  
  - `Vector`: Input vector `(N,)`  
- **Returns:** - Resulting vector `(N,)`

#### 2. `Eigenvalue_calc(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using the **power iteration method**.  
- **Args:** - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:** - `(eigenvalue, eigenvector)`

#### 3. `Eigenvalues(mat, vec, Iteration)`
Computes eigenvalues and eigenvectors of a matrix and its inverse.  
- **Args:** - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:** - `eigenvalues`: List of eigenvalues  
  - `eigenvectors`: List of eigenvectors  
- **Note:** - For a regular matrix, this gives the **largest** eigenvalue (via `mat`) and the **smallest** eigenvalue (via `inv(mat)`).

#### 4. `Aitken(Eigenvalues)`
Applies **Aitken’s delta-squared process** to accelerate convergence.  
- **Args:** - `Eigenvalues`: Sequence of approximated eigenvalues  
- **Returns:** - Accelerated eigenvalue approximation  

#### 5. `eigenvalue_calc_aitken(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using **power iteration with Aitken acceleration**.  
- **Args:** - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:** - `(eigenvalue, eigenvector)`

#### 6. `Eigenvalues_Aitken(mat, vec, Iteration)`
Computes eigenvalues and eigenvectors of a matrix (and its inverse) using Aitken’s method.  
- **Args:** - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:** - `eigenvalues`: List of eigenvalues  
  - `eigenvectors`: List of eigenvectors  

---

### Cubic Spline Interpolation

This project implements a **cubic spline interpolation** algorithm in Python using **NumPy**.  
The algorithm constructs and solves a linear system to determine the spline coefficients, allowing smooth interpolation of a given function across a set of points.

#### 1. `h(i, x)`
Computes the step size between two adjacent points.  
- **Args:** - `i`: Index  
  - `x`: Array of grid points  
- **Returns:** - Step size `x[i+1] - x[i]`  

#### 2. `cubic_splines(n, x, func)`
Constructs cubic splines for interpolation.  
- **Args:** - `n`: Number of intervals (`len(x) - 1`)  
  - `x`: Array of grid points of length `n+1`  
  - `func`: Function to evaluate at grid points  
- **Returns:** - `matrix`: Coefficient matrix of the linear system  
  - `vec`: Right-hand side vector  
  - `solution`: Approximated spline second derivatives (including boundary conditions)  

**Algorithm Details:**
1. **Initialization:** Evaluate the function values at the given grid points. Construct the system matrix and vector based on spline conditions.
2. **Interior Equations:** Ensure smoothness of the cubic spline (continuity of first and second derivatives).
3. **Boundary Conditions:** Natural spline end conditions are implemented via extended system equations.
4. **Solve System:** Solve `matrix * solution = vec`. Extend solution to include boundary second derivatives `S_0` and `S_n`.
5. **Conditioning Check:** If the system matrix is ill-conditioned (`cond(matrix) > 1e12`), a warning is raised.

---

### Numerical Differentiation Methods

This module implements different methods for approximating the **first derivative** of a function:
- **Central Differences (4th-order accurate)**
- **Richardson Extrapolation**
- **Spline-Based Derivative Calculation**

#### `Central_diff_first_deri(x, h, func)`
Computes the first derivative using the **central difference method** with 4th-order accuracy.
- **Args:** `x` (Point), `h` (Step size), `func` (Function)
- **Returns:** Approximation of the derivative at `x`  

#### `Richardson(x, h, func)`
Computes the derivative using **Richardson extrapolation**, which accelerates convergence by combining results from multiple step sizes.
- **Args:** `x` (Point), `h` (Step size), `func` (Function)
- **Returns:** Richardson-extrapolated derivative at `x`  

#### `spline_derivative(i, x_val, x, func)`
Computes the derivative of a function using **cubic spline interpolation** on the interval `[x[i], x[i+1]]`.
- **Args:** `i` (Index), `x_val` (Point), `x` (Array of nodes), `func` (Function)
- **Returns:** Derivative of the spline at `x_val`  

**Example Usage:**
```python
import numpy as np
from Differentiation import Central_diff_first_deri, Richardson, spline_derivative

f = np.sin
x0 = np.pi / 4
h = 0.01
x_nodes = np.linspace(0, np.pi, 5)

print("Central difference:", Central_diff_first_deri(x0, h, f))
print("Richardson:", Richardson(x0, h, f))
print("Spline derivative:", spline_derivative(1, x0, x_nodes, f))
```

---

### Numerical ODE Solvers

This project provides several numerical algorithms for solving **ordinary differential equations (ODEs)** and systems of ODEs.  

#### Core Methods:
- **`Heun(x0, xm, y0, n, f, plotchoose)`**: Solves an ODE with Heun’s method (2nd order).
- **`adam_predictor(...)`**: Computes ODE solution using Adams–Bashforth predictor.
- **`adam_corrector(...)`**: Corrects predicted values using Adams–Moulton method.
- **`adam_ode_int(...)`**: Predictor–Corrector method combining Adams–Bashforth and Adams–Moulton.
- **`Adam(...)`**: Classical Adams method using previous three steps for prediction.
- **`runge_kutta(x_start, x_end, y_0, n, f, plotchoose)`**: Solves an ODE with Runge–Kutta 4th order method.
- **`systems_of_ODE(system, y0, t)`**: Solves a system of ODEs using Runge–Kutta 4th order.

---

### Boundary Value Problem Solver

This project implements two numerical methods for solving boundary value problems (BVPs) of ordinary differential equations (ODEs):  
- **Finite Difference Method** (Matrix Method)  
- **Shooting Method**

#### 1. `Matrix_method(x0, xn, n, y0, yn)`
Solves a BVP using the finite difference method.  
- **Returns:** Solution vector `y` at interior points and plots the numerical solution.

#### 2. `solve_by_shooting(ode, x_1, x_2, n, v_0, u_1, u_2, max_iter)`
Solves a BVP using the **shooting method**.  
- **Returns:** `(v_corr, x, y)` - Corrected initial derivative, solution grid points, and solution values.