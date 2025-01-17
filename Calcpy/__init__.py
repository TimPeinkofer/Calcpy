from .numint import Simpson_1_3, Romberg, Gauss_legendre, Newton_cotes, Trapezoidal, Simpson_3_8
from .numdiff_WOP import Central_diff_first_deri
from .Errorcalc import error
from .NumLinEqu import Gauss_elimination, overrelaxation
from .NonLinEq import newtons_method, solve_fixed_point, linear_interpolation
from .NumEigenv import Eigenvalues, Eigenvalues_Aitken
from .ODE import Heun, Adam, adam_corrector, adam_ode_int, adam_predictor, runge_kutta
