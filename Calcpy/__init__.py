from .numint import Simpson_1_3, Romberg, Gauss_legendre, Newton_cotes, Trapezoidal, Simpson_3_8
from .numdiff import Central_diff_first_deri, Richardson, spline_derivative
from .Errorcalc import error
from .NumLinEqu import Gauss_elimination, overrelaxation, determinant, gauss_seidel, Gauss_Jordan, inverse_matrix
from .NonLinEq import newtons_method, solve_fixed_point, linear_interpolation, newton_halley
from .NumEigenv import Eigenvalues, Eigenvalues_Aitken
from .BVP import solve_by_shooting, Matrix_method
from .ODE import Heun, adam_corrector, adam_ode_int, adam_predictor, Adam, runge_kutta
from .Interpolation import cubic_splines
from .PDE import parabolic_explicit_solver, hyperbolic_solver, elliptic_solver_laplace_2D, elliptic_solver_laplace_3D