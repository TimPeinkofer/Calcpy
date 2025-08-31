# Calcpy

A simple Python library for numerical mathematical calculations and it's utility functions.

## Installation

### Prerequisites

The following python bibs are needed for the calculation:

- Python 3.x  
- Libraries:  
  ```bash
  pip install numpy matplotlib
  ```

### How to install via command line
```bash
git clone https://github.com/TimPeinkofer/Calcpy.git
cd Calcpy
pip install .
```

## Overview
The following types of problems will be addressed in the bibliography:

1. Nonlinear equations
2. Systems of linear equations
3. Eigenvalue and Matrix calculations
4. Interpolation
5. Integration
6. Differentiation
7. Ordinary differential equations
8. Boundary and Initial value problems
9. Hyperbolic, Parabolic and Elliptic Partial differential equations


## WOP

## Eigenvalue and Eigenvector Computation

This project implements numerical methods for computing eigenvalues and eigenvectors of a matrix using **power iteration** and **Aitken's delta-squared acceleration**.  
Both the matrix and its inverse are considered to approximate the **largest** and **smallest** eigenvalues.

---

### Functions

### 1. `matrix_vector(N, Matrix, Vector)`
Multiplies a square matrix by a vector.  
- **Args:**  
  - `N`: Size of the square matrix  
  - `Matrix`: Input matrix `(N x N)`  
  - `Vector`: Input vector `(N,)`  
- **Returns:**  
  - Resulting vector `(N,)`

---

#### 2. `Eigenvalue_calc(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using the **power iteration method**.  
- **Args:**  
  - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:**  
  - `(eigenvalue, eigenvector)`

---

#### 3. `Eigenvalues(mat, vec, Iteration)`
Computes eigenvalues and eigenvectors of a matrix and its inverse.  
- **Args:**  
  - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:**  
  - `eigenvalues`: List of eigenvalues  
  - `eigenvectors`: List of eigenvectors  
- **Note:**  
  - For a regular matrix, this gives the **largest** eigenvalue (via `mat`) and the **smallest** eigenvalue (via `inv(mat)`).

---

#### 4. `Aitken(Eigenvalues)`
Applies **Aitken’s delta-squared process** to accelerate convergence.  
- **Args:**  
  - `Eigenvalues`: Sequence of approximated eigenvalues  
- **Returns:**  
  - Accelerated eigenvalue approximation  

---

#### 5. `eigenvalue_calc_aitken(mat, vec, Iteration)`
Computes the dominant eigenvalue and eigenvector using **power iteration with Aitken acceleration**.  
- **Args:**  
  - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:**  
  - `(eigenvalue, eigenvector)`

---

#### 6. `Eigenvalues_Aitken(mat, vec, Iteration)`
Computes eigenvalues and eigenvectors of a matrix (and its inverse) using Aitken’s method.  
- **Args:**  
  - `mat`: Input matrix  
  - `vec`: Initial guess vector  
  - `Iteration`: Number of iterations  
- **Returns:**  
  - `eigenvalues`: List of eigenvalues  
  - `eigenvectors`: List of eigenvectors  

---

## Cubic Spline Interpolation

This project implements a **cubic spline interpolation** algorithm in Python using **NumPy**.  
The algorithm constructs and solves a linear system to determine the spline coefficients, allowing smooth interpolation of a given function across a set of points.

---

### Functions

#### 1. `h(i, x)`
Computes the step size between two adjacent points.  
- **Args:**  
  - `i`: Index  
  - `x`: Array of grid points  
- **Returns:**  
  - Step size `x[i+1] - x[i]`  

---

#### 2. `cubic_splines(n, x, func)`
Constructs cubic splines for interpolation.  
- **Args:**  
  - `n`: Number of intervals (`len(x) - 1`)  
  - `x`: Array of grid points of length `n+1`  
  - `func`: Function to evaluate at grid points  
- **Returns:**  
  - `matrix`: Coefficient matrix of the linear system  
  - `vec`: Right-hand side vector  
  - `solution`: Approximated spline second derivatives (including boundary conditions)  

---

### Algorithm Details

1. **Initialization**  
   - Evaluate the function values at the given grid points.  
   - Construct the system matrix and vector based on spline conditions.  

2. **Interior Equations**  
   - Ensure smoothness of the cubic spline (continuity of first and second derivatives).  

3. **Boundary Conditions**  
   - Natural spline end conditions are implemented via extended system equations.  

4. **Solve System**  
   - Solve `matrix * solution = vec`.  
   - Extend solution to include boundary second derivatives `S_0` and `S_n`.  

5. **Conditioning Check**  
   - If the system matrix is ill-conditioned (`cond(matrix) > 1e12`), a warning is raised.  

---

## Numerical Differentiation Methods

This module implements different methods for approximating the **first derivative** of a function:

- **Central Differences (4th-order accurate)**  
- **Richardson Extrapolation**  
- **Spline-Based Derivative Calculation**  

The functions rely on direct function evaluations and cubic spline interpolation for smooth approximations.

---

### Functions

#### `Central_diff_first_deri(x, h, func)`
Computes the first derivative using the **central difference method** with 4th-order accuracy.

**Args:**
- `x (float)`: Point at which the derivative is evaluated  
- `h (float)`: Step size  
- `func (function)`: Function to be differentiated  

**Returns:**
- `float`: Approximation of the derivative at `x`  

---

#### `F(h, x, func)`
Computes the **first-order central difference approximation** of the derivative.  
Used internally for Richardson extrapolation.

---

#### `psi(h, x, func)`
Applies the Richardson extrapolation formula to improve the accuracy of the derivative approximation.  
Used internally by `Richardson`.

---

#### `Richardson(x, h, func)`
Computes the derivative using **Richardson extrapolation**, which accelerates convergence by combining results from multiple step sizes.

**Args:**
- `x (float)`: Point at which the derivative is evaluated  
- `h (float)`: Step size  
- `func (function)`: Function to be differentiated  

**Returns:**
- `float`: Richardson-extrapolated derivative at `x`  

---

#### `spline_derivative(i, x_val, x, func)`
Computes the derivative of a function using **cubic spline interpolation** on the interval `[x[i], x[i+1]]`.

**Args:**
- `i (int)`: Index such that `x[i] <= x_val <= x[i+1]`  
- `x_val (float)`: Point at which the derivative is evaluated  
- `x (ndarray)`: Array of interpolation nodes  
- `func (function)`: Function providing values `f(x)`  

**Returns:**
- `float`: Derivative of the spline at `x_val`  
- `None`: If `x_val` is outside the interval or no spline solution is found  

---

### Example Usage

```python
import numpy as np
from Differentiation import Central_diff_first_deri, Richardson, spline_derivative

# Define test function
f = np.sin
x0 = np.pi / 4
h = 0.01
x_nodes = np.linspace(0, np.pi, 5)

# Central difference derivative
d_central = Central_diff_first_deri(x0, h, f)

# Richardson extrapolation derivative
d_rich = Richardson(x0, h, f)

# Spline derivative on interval [x[1], x[2]]
d_spline = spline_derivative(1, x0, x_nodes, f)

print("Central difference:", d_central)
print("Richardson:", d_rich)
print("Spline derivative:", d_spline)
print("Exact derivative:", np.cos(x0))
```


## Numerical ODE Solvers

This project provides several numerical algorithms for solving **ordinary differential equations (ODEs)** and systems of ODEs.  
Implemented methods include:

- **Heun’s Method** (improved Euler)
- **Adams–Bashforth Predictor**
- **Adams–Moulton Corrector**
- **Adams Predictor–Corrector Combination**
- **Classical Adams Method**
- **Runge–Kutta 4th Order Method**
- **Runge–Kutta for ODE Systems**

---

### Functions

#### 1. `precalc(xm, x0, n)`
Pre-computes grid points and step size.  
- **Args:**  
  - `xm`: Final value of `x`  
  - `x0`: Initial value of `x`  
  - `n`: Number of steps  
- **Returns:**  
  - `x`: Grid points  
  - `y`: Zero-initialized solution array  
  - `h`: Step size  

---

#### 2. `sol_plot(x, y, plotchoose)`
Plots the numerical solution if `plotchoose=True`.  

---

#### 3. `Heun(x0, xm, y0, n, f, plotchoose)`
Solves an ODE with **Heun’s method** (2nd order).  
- **Args:**  
  - `x0`, `xm`: Interval boundaries  
  - `y0`: Initial value of `y`  
  - `n`: Number of steps  
  - `f`: Derivative function `f(x, y)`  
  - `plotchoose`: Whether to plot the solution  
- **Returns:**  
  - `x`: Grid points  
  - `y`: Solution values  

---

#### 4. `adam_predictor(f, y_0, x_0, x_m, n, plotchoose)`
Computes ODE solution using **Adams–Bashforth predictor**.  
- Uses **Heun’s method** to generate the first three values.  
- **Returns:**  
  - `x`: Grid points  
  - `y_pred`: Predicted values  

---

#### 5. `adam_corrector(f, y_values, x, n, h, plotchoose)`
Corrects predicted values using **Adams–Moulton method**.  
- **Returns:**  
  - `y_corr`: Corrected values  

---

#### 6. `adam_ode_int(f, y_0, x0, xm, n, plotchoose)`
Predictor–Corrector method combining **Adams–Bashforth** and **Adams–Moulton**.  
- **Returns:**  
  - `y_corr`: Corrected values  
  - `y_pred`: Predicted values  

---

#### 7. `Adam(x0, xm, y0, n, f, plotchoose)`
Classical **Adams method** using previous three steps for prediction.  
- **Returns:**  
  - `x`: Grid points  
  - `y`: Solution values  

---

#### 8. `runge_kutta(x_start, x_end, y_0, n, f, plotchoose)`
Solves an ODE with **Runge–Kutta 4th order method**.  
- **Returns:**  
  - `x_values`: Grid points  
  - `y_values`: Solution values  

---

#### 9. `systems_of_ODE(system, y0, t)`
Solves a **system of ODEs** using Runge–Kutta 4th order.  
- **Args:**  
  - `system`: Function describing the ODE system  
  - `y0`: Initial conditions (list or array)  
  - `t`: Array of time/grid points  
- **Returns:**  
  - `y`: Solution array with shape `(len(t), len(y0))`  

---


## Boundary Value Problem Solver

This project implements two numerical methods for solving boundary value problems (BVPs) of ordinary differential equations (ODEs):  

- **Finite Difference Method** (Matrix Method)  
- **Shooting Method**

The implementation is based on **NumPy** and **Matplotlib**.  
Additionally, helper functions `runge_kutta` (for solving ODEs) and `linear_interpolation` (for root finding) are imported from external modules (`ODE.py`, `NonLinEq.py`).  

---

### Functions

### 1. `Matrix(h: float, n: int) -> np.ndarray`
Constructs a tridiagonal matrix for the finite difference scheme.  
- **Args:**  
  - `h`: Step size  
  - `n`: Number of unknowns  
- **Returns:**  
  - Tridiagonal matrix `A` of size `(n, n)`

---

#### 2. `Vector(n: int, y0: float, yn: float) -> np.ndarray`
Constructs the right-hand side vector of the linear system.  
- **Args:**  
  - `n`: Number of unknowns  
  - `y0`: Boundary condition at the left end  
  - `yn`: Boundary condition at the right end  
- **Returns:**  
  - Vector `b` of size `(n,)`

---

#### 3. `Matrix_method(x0: float, xn: float, n: int, y0: float, yn: float) -> np.ndarray`
Solves a BVP using the finite difference method.  
- **Args:**  
  - `x0`, `xn`: Interval boundaries  
  - `n`: Number of interior grid points  
  - `y0`, `yn`: Boundary values  
- **Returns:**  
  - Solution vector `y` at interior points  
- **Additionally:**  
  - Plots the numerical solution  

---

#### 4. `solve_by_shooting(ode: callable, x_1: float, x_2: float, n: int, v_0: list, u_1: float, u_2: float, max_iter: int) -> tuple`
Solves a BVP using the **shooting method**.  
- **Args:**  
  - `ode`: Function defining the ODE system  
  - `x_1`, `x_2`: Interval boundaries  
  - `n`: Number of Runge–Kutta steps  
  - `v_0`: Two initial guesses for the derivative at `x_1`  
  - `u_1`, `u_2`: Boundary conditions for `y(x_1)` and `y(x_2)`  
  - `max_iter`: Maximum number of iterations for root finding  
- **Returns:**  
  - `v_corr`: Corrected initial derivative  
  - `x`: Grid points  
  - `y`: Solution values  

---






