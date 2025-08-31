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



