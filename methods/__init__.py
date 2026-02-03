# Numeričke metode
from .root_finding import bisection, newton_raphson, secant
from .integration import trapezoidal, simpson, romberg, gauss_quadrature, integrate_from_table, integrate_with_interpolation
from .differentiation import forward_diff, backward_diff, central_diff, compare_errors, auto_differentiate, differentiate_from_table
from .linear_systems import jacobi, gauss_seidel
from .regression import linear_regression, exponential_regression, polynomial_regression
