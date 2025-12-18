from typing import Optional, Union, Any, Literal, Iterable
from abc import ABC, abstractmethod
import numpy as np
import numpy.polynomial.chebyshev
import quadprog
from cvxopt import matrix, solvers

from base import ForecastResult, DeterministicForecastResult
# from tsmod.base import ModelFit, Process, Domain
# from tsmod.optimization_objectives import OptimizationObjective
# from tsmod.utils import ConstrainedMatrixWrapper

from base import ModelFit, Signal, Domain
from optimization_objectives import OptimizationObjective
from constrained_matrices import ConstrainedMatrix


#
#   Models have the form
#
#   y_t - B u_t = A g(t) + L e_t,    e_t ~ N(0, I)
#
#   where g(t) is deterministic
#

# ------------------------------
# Abstract Classes
# ------------------------------

class DeterministicFit(ModelFit, ABC):

    def __init__(self,
                 domain: "Deterministic",
                 success: bool,
                 model: "DeterministicSignal",
                 series: np.ndarray,
                 **fit_options):

        super().__init__(domain=domain, success=success, model=model, series=series, **fit_options)

    @abstractmethod
    def forecast(self, k: Iterable[int] | int, **kwargs) -> DeterministicForecastResult:
        raise NotImplementedError

    def get_loss(self, loss: OptimizationObjective) -> float:
        if loss.requires_cov:
            raise ValueError("Deterministic Models do not have prediction uncertainty")
        else:
            return loss(self.get_prediction_errors())

    @abstractmethod
    def get_prediction_errors(self):
        raise NotImplementedError

    def forecast_with_uncertainty(self, k: Iterable[int] | int) -> ForecastResult:
        raise NotImplementedError("Deterministic Models do not implement uncertain forecasts")


class DeterministicSignal(Signal, ABC):

    def __init__(self, shape: tuple, **deterministic_definition):
        super().__init__(shape=shape, **deterministic_definition)

    def simulate(self, time_domain: Union[tuple[int, int], int]):
        if isinstance(time_domain, tuple):
            if not all(isinstance(i, int) for i in time_domain):
                raise TypeError("Time domain should be integers or tuple of integers")
            else:
                T = time_domain[1] - time_domain[0]
                if not T > 0:
                    raise ValueError("Time domain should be greater than 0")
        elif not isinstance(time_domain, int):
            raise TypeError("Time domain should be integers or tuple of integers")
        else:
            if not time_domain > 0:
                raise ValueError("Time domain should be greater than 0")

        return self._simulate(time_domain=time_domain)

    @abstractmethod
    def _simulate(self, time_domain: Union[tuple[int, int], int]):
        raise NotImplementedError


class Deterministic(Domain, ABC):

    def __init__(self, shape: tuple, **kwargs):
        super().__init__(shape=shape, **kwargs)

    @abstractmethod
    def _calc_n_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self,
            series: np.ndarray,
            objective: Union[OptimizationObjective, Any],
            exog: Optional[np.ndarray] = None,
            exog_loadings: Optional[Union[np.ndarray, ConstrainedMatrix]] = None, ) -> DeterministicFit:

        raise NotImplementedError


# ------------------------------
# Specialty (less general but widely used)
# ------------------------------

class ConstantProcess(DeterministicSignal):

    def __init__(self, constant: np.ndarray):
        constant = np.squeeze(constant)
        if constant.ndim != 1:
            raise NotImplementedError("Constant should be vector value")

        length = len(constant)
        super().__init__(shape=(length,), constant=constant)

        self._constant = constant

    def _simulate(self, time_domain: Union[tuple[int, int], int]):
        if isinstance(time_domain, tuple):
            T = time_domain[1] - time_domain[0]
        else:
            T = time_domain

        return np.broadcast(self._constant, np.zeros((T, self._constant.size)))


class Constant(Deterministic):

    def __init__(self, length: int):
        super().__init__(shape=(length,))

    def _calc_n_parameters(self):
        return self.shape[0]

    def fit(self,
            series: np.ndarray,
            objective: Union[OptimizationObjective],
            exog: Optional[np.ndarray] = None,
            exog_loadings: Optional[Union[np.ndarray, ConstrainedMatrix]] = None) -> DeterministicFit:
        pass


# ------------------------------
# Polynomial
# ------------------------------


class PolynomialUtils:

    numpy_poly_fits = {
        'polynomial': np.polynomial.polynomial.polyfit,
        'power': np.polynomial.polynomial.polyfit,
        'chebyshev': np.polynomial.chebyshev.chebfit,
        'legendre': np.polynomial.legendre.legfit,
        'laguerre': np.polynomial.laguerre.lagfit,
    }

    numpy_poly_instances = {
        'polynomial': np.polynomial.polynomial.Polynomial,
        'power': np.polynomial.polynomial.Polynomial,
        'chebyshev': np.polynomial.chebyshev.Chebyshev,
        'legendre': np.polynomial.legendre.Legendre,
        'laguerre': np.polynomial.laguerre.Laguerre,
    }

    numpy_vander = {
        'polynomial': numpy.polynomial.polynomial.polyvander,
        'power': numpy.polynomial.polynomial.polyvander,
        'chebyshev': numpy.polynomial.chebyshev.chebvander,
        'legendre': numpy.polynomial.legendre.legvander,
        'laguerre': numpy.polynomial.laguerre.lagvander,
    }


class PolynomialProcess(DeterministicSignal):

    def __init__(self,
                 poly: Literal["Power", "Polynomial", "Chebyshev", "Laguerre", "Legendre"],
                 coeffs: np.ndarray,
                 domain: tuple[int, int]):
        super().__init__(poly=poly, coeffs=coeffs, domain=domain)
        self._poly = poly
        self._coeffs = coeffs
        self._domain = domain

    @property
    def poly(self):
        return self._poly

    @property
    def degree(self):
        return len(self._coeffs) - 1

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def domain(self):
        return self._domain

    def to_numpy(self):
        return PolynomialUtils.numpy_poly_instances[self.poly](coef=self.coeffs, domain=self.domain)

    def _simulate(self, time_domain: Union[tuple[int, int], int]):
        if isinstance(time_domain, tuple):
            x = np.arange(time_domain[0], time_domain[1], 1)
        else:
            x = np.arange(0, time_domain, 1)

        return self.to_numpy()(x)


class Polynomial(Deterministic):


    def __init__(self, poly: Literal["Power", "Polynomial", "Chebyshev", "Laguerre", "Legendre"], degree: int):

        if poly.lower() in PolynomialUtils.numpy_poly_fits.keys():
            raise ValueError("poly not valid")

        if not isinstance(degree, int) or degree < 0:
            raise ValueError(f"Degree '{degree}' is not valid. Must be an non-negative integer")

        self._poly = poly.lower()
        self._degree = degree

        self._constraints = {}

        super().__init__(poly=poly, order=degree)

    @property
    def degree(self):
        return self._degree

    @property
    def poly(self):
        return self._poly

    def set_constraint(self, degree: int, value: float):
        """Set a constraint for a specific degree of the polynomial."""
        if not isinstance(degree, int) or degree < 0:
            raise ValueError(f"Degree '{degree}' is not valid. Must be an non-negative integer")

        self._constraints[degree] = value
        return self

    def fit(self,
            series: np.ndarray,
            objective: Union[OptimizationObjective, Any],
            exog: Optional[np.ndarray] = None,
            exog_loadings: Optional[Union[np.ndarray, ConstrainedMatrix]] = None, ) -> DeterministicFit:

        # y_t - B u_t = A g(t) + [iid N(0,S)]
        x = np.arange(0, series.shape[0])
        X = PolynomialUtils.numpy_vander[self._poly](x, self.degree)

        # y_t = B u_t + A g(t)

        if exog is not None:
            X = np.hstack([exog, X])

    def _fit_OLS(self, series, X, exog_loadings):
        pass

    def _calc_n_parameters(self):
        return self._degree - len(self._constraints)


# if __name__ == "__main__":
    # Define the data (X and Y)
    # n, p, m = 100, 3, 2
    # X = np.random.randn(n, p)
    # Y = np.random.randn(n, m)
    #
    # # Define the equality constraint matrix (A) and vector (b)
    # A = np.ones((1, p))  # Example: sum of coefficients should be 1
    # b = np.array([1])
    #
    # # QP form: minimize (1/2) x^T Q x + c^T x subject to Ax = b
    # Q = 2 * (X.T @ X)
    # c = -2 * X.T @ Y
    #
    # # Solve the QP problem
    # beta_opt = quadprog.solve_qp(Q, c, A.T, b)[0]
    #

# ------------------------------
# Fourier
# ------------------------------

class FourierUtils:

    pass





# ------------------------------
# Deterministic Model Mapping
# ------------------------------

# deterministic_class_mapping = {
#     'polynomial': Polynomial,
#     'fourier': Fourier
# }
#


#
#
#


#
# class DeterministicRes_old(Result):
#
#     def __init__(self,
#                  model: "Deterministic",
#                  success: bool,
#                  series: np.ndarray,
#                  fit_options: dict,
#                  deterministic_object: Any,
#                  in_sample_values: np.ndarray,):
#         super().__init__(model, success, series, fit_options)
#
#         self._deterministic_object = deterministic_object
#         self._in_sample_values = in_sample_values
#
#     def _get_prediction_errors(self):
#         return self.series - self._in_sample_values
#
#     def _get_k_ahead_prediction_errors(self, k: Iterable[int] | int):
#         raise NotImplementedError("Deterministic does not define k_ahead_prediction_errors")
#
#     def _forecast_with_uncertainty(self, k: Iterable[int] | int) -> ForecastResult:
#         raise NotImplementedError("Deterministic does not define uncertainty")
#
#     def _forecast(self, k: Iterable[int] | int):
#
#         _trend_specification = self.model.trend_specification
#         if self.model.trend_type == "Polynomial":
#             poly_class = getattr(np.polynomial, _trend_specification["Poly"].lower())
#             poly_instance = getattr(poly_class, _trend_specification["Poly"])
#             _, simulated_series = poly_instance(self._deterministic_object).linspace(n=T)
#             return simulated_series
#
#         elif self.trend_type == "RBFInterpolator":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             if _trend_specification["apply_padding"]:
#                 xs = np.concatenate([xs[:3] - (xs[3] - xs[2]) * np.arange(3, 0, -1),
#                                      xs,
#                                      xs[-3:] + (xs[-1] - xs[-2]) * np.arange(1, 4)])
#                 if parameters.ndim == 1:
#                     parameters = np.concatenate([parameters[:3][::-1],  # reflect first 3 points
#                                                  parameters,
#                                                  parameters[-3:][::-1]])  # reflect last 3 points
#                 else:
#                     parameters = np.concatenate([parameters[:3][::-1, :],  # reflect first 3 rows
#                                                  parameters,  # original array
#                                                  parameters[-3:][::-1, :]], axis=0)  # reflect last 3 rows
#
#             inter = RBFInterpolator(xs.reshape(-1, 1),
#                                     parameters,
#                                     kernel=_trend_specification["Kernel"],
#                                     epsilon=1 / (T * _trend_specification["Bandwidth"]),
#                                     smoothing=_trend_specification["Smoothing"])
#
#             simulated_series = inter(np.arange(T).reshape(-1, 1))
#             return simulated_series
#
#
#         elif self.trend_type == "Kernel":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             bw = _trend_specification["Bandwidth"]
#             if not isinstance(bw, str):
#                 bw = [T * bw]
#             kr = KernelReg(parameters, xs,
#                            var_type='c',
#                            reg_type=_trend_specification["Type"],
#                            ckertype=_trend_specification["Kernel"],
#                            bw=bw)
#             simulated_series, _ = kr.fit(np.arange(T))
#             return simulated_series
#
#         elif self.trend_type == "NW":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             bw = T * _trend_specification["Bandwidth"]
#             func = kernel_func[_trend_specification["Kernel"]]
#             weights = func(bw, xs, np.arange(T)[:, None])
#             weights = weights / np.sum(weights, axis=1, keepdims=True)
#             simulated_series = weights @ parameters
#             return simulated_series
#
#     def _calc_nll(self) -> float:
#         raise NotImplementedError("Deterministic does not define nll")
#
#
# class Deterministic_old(Model):
#
#     _valid_trend_type = ["polynomial", "RBFInterpolator", "kernel", "NW"]
#
#     def __init__(self,
#                  n_series: int,
#                  trend_type: Literal["polynomial", "RBFInterpolator", "kernel", "NW"],
#                  specification: Union[str, dict],
#                  degree: int):
#
#         if trend_type not in self._valid_trend_type:
#             raise ValueError(
#                 f"Functional form '{trend_type}' is not valid. Options are {self._valid_trend_type}")
#
#         if not isinstance(degree, int) or degree < 1:
#             raise ValueError(f"Degree '{degree}' is not valid. Must be an non-negative integer")
#
#         if specification is None:
#             if specification == "Polynomial":
#                 specification = {"Poly": "Chebyshev"}
#             elif specification == "RBFInterpolator":
#                 specification = {"Kernel": "gaussian", "Bandwidth": 0.4, "Smoothing": 0.2, "apply_padding": False}
#             elif specification == "Kernel":
#                 specification = {"Type": "ll", "Kernel": "gaussian", "Bandwidth": 0.4}
#             elif specification == "NW":
#                 specification = {"Kernel": "gaussian", "Bandwidth": 0.4}
#
#         super().__init__(n_series, trend_type=trend_type, specification=specification, degree=degree)
#
#         self._trend_type = trend_type
#         self._trend_specification = specification
#         self._degree = degree
#
#
#     @property
#     def degree(self) -> int:
#         return self._degree
#
#     @property
#     def trend_type(self):
#         return self._trend_type
#
#     @property
#     def trend_specification(self):
#         return self._trend_specification
#
#     _numpy_poly_fits = {
#         'polynomial': np.polynomial.chebyshev.chebfit,
#         'chebyshev': np.polynomial.chebyshev.chebfit,
#         'legendre': np.polynomial.legendre.legfit,
#         'laguerre': np.polynomial.laguerre.lagfit,
#     }
#
#     def fit(self, series: np.ndarray) -> DeterministicResult:
#         self._validate_series(series)
#
#         T = series.shape[0]
#         x = np.arange(0, T)
#
#         _trend_specification = self._trend_specification
#         if self.trend_type == "Polynomial":
#             fit_func = self._numpy_poly_fits[self._trend_specification["Poly"].lower()]
#             result_params = fit_func(x, series, self.degree)
#
#         elif self.trend_type == "RBFInterpolator":
#
#             if _trend_specification["apply_padding"]:
#                 # i have this for 1D array. i want for
#                 series = np.concatenate([series[:3][::-1],  # reflect first 3 points
#                                              series,
#                                              series[-3:][::-1]])  # reflect last 3 points
#                 x = np.concatenate([x[:3] - (x[3] - x[2]) * np.arange(3, 0, -1),
#                                      x,
#                                      x[-3:] + (x[-1] - x[-2]) * np.arange(1, 4)])
#
#             inter = RBFInterpolator(xs.reshape(-1, 1),
#                                     parameters,
#                                     kernel=_trend_specification["Kernel"],
#                                     epsilon=1 / (T * _trend_specification["Bandwidth"]),
#                                     smoothing=_trend_specification["Smoothing"])
#             simulated_series = inter(np.arange(T).reshape(-1, 1))
#             return simulated_series
#
#
#         elif self.trend_type == "Kernel":
#             xs = np.linspace(0, T - 1, len(parameters))
#             bw = _trend_specification["Bandwidth"]
#             if not isinstance(bw, str):
#                 bw = [T * bw]
#             kr = KernelReg(parameters, xs,
#                            var_type='c',
#                            reg_type=_trend_specification["Type"],
#                            ckertype=_trend_specification["Kernel"],
#                            bw=bw)
#             simulated_series, _ = kr.fit(np.arange(T))
#             return simulated_series
#
#         elif self.trend_type == "NW":
#             xs = np.linspace(0, T - 1, len(parameters))
#             bw = T * _trend_specification["Bandwidth"]
#             func = kernel_func[_trend_specification["Kernel"]]
#             weights = func(bw, xs, np.arange(T)[:, None])
#             weights = weights / np.sum(weights, axis=1, keepdims=True)
#             simulated_series = weights @ parameters
#             return simulated_series
#
#     def _calc_n_parameters(self):
#         raise self.n_series * self.degree
#
#     def simulate(self, parameters: np.ndarray, T: int, xs: Optional[np.ndarray]) -> np.ndarray:
#         self._simulate_validate_parameters(parameters)
#
#         _trend_specification = self._trend_specification
#         if self.trend_type == "Polynomial":
#             poly_class = getattr(np.polynomial, _trend_specification["Poly"].lower())
#             poly_instance = getattr(poly_class, _trend_specification["Poly"])
#             _, simulated_series = poly_instance(parameters).linspace(n=T)
#             return simulated_series
#
#         elif self.trend_type == "RBFInterpolator":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             if _trend_specification["apply_padding"]:
#                 xs = np.concatenate([xs[:3] - (xs[3] - xs[2]) * np.arange(3, 0, -1),
#                                      xs,
#                                      xs[-3:] + (xs[-1] - xs[-2]) * np.arange(1, 4)])
#                 if parameters.ndim == 1:
#                     parameters = np.concatenate([parameters[:3][::-1],  # reflect first 3 points
#                                                  parameters,
#                                                  parameters[-3:][::-1]])  # reflect last 3 points
#                 else:
#                     parameters = np.concatenate([parameters[:3][::-1, :],  # reflect first 3 rows
#                                                     parameters,  # original array
#                                                     parameters[-3:][::-1, :]], axis=0)  # reflect last 3 rows
#
#             inter = RBFInterpolator(xs.reshape(-1, 1),
#                                     parameters,
#                                     kernel=_trend_specification["Kernel"],
#                                     epsilon=1 / (T * _trend_specification["Bandwidth"]),
#                                     smoothing=_trend_specification["Smoothing"])
#
#             simulated_series = inter(np.arange(T).reshape(-1, 1))
#             return simulated_series
#
#
#         elif self.trend_type == "Kernel":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             bw = _trend_specification["Bandwidth"]
#             if not isinstance(bw, str):
#                 bw = [T * bw]
#             kr = KernelReg(parameters, xs,
#                            var_type='c',
#                            reg_type=_trend_specification["Type"],
#                            ckertype=_trend_specification["Kernel"],
#                            bw=bw)
#             simulated_series, _ = kr.fit(np.arange(T))
#             return simulated_series
#
#         elif self.trend_type == "NW":
#             if xs is None:
#                 xs = np.arange(0, T, len(parameters))
#
#             bw = T * _trend_specification["Bandwidth"]
#             func = kernel_func[_trend_specification["Kernel"]]
#             weights = func(bw, xs, np.arange(T)[:, None])
#             weights = weights / np.sum(weights, axis=1, keepdims=True)
#             simulated_series = weights @ parameters
#             return simulated_series
#
#         raise ValueError("invalid trend type specified. Not sure how it even got here")
#
#     def _simulate_validate_parameters(self, parameters: np.ndarray):
#
#         if self.n_series == 1:
#             if parameters.ndim > 1:
#                 if parameters.shape[0] != self.n_series:
#                     raise ValueError("If n_series is 1, parameters must be shape (1, degree + 1) or (degree + 1,)")
#                 parameters = np.squeeze(parameters)
#             if parameters.shape[0] != self.degree:
#                 raise ValueError("If n_series is 1, parameters must be shape (1, degree + 1) or (degree + 1,)")
#
#         else:
#             if not parameters.ndim == 2:
#                 raise ValueError("If n_series > 1, parameters must be shape (n_series, degree + 1)")
#
#             if not parameters.shape[0] == self.n_series or not parameters.shape[1] == self.degree + 1:
#                 raise ValueError("If n_series > 1, parameters must be shape (n_series, degree + 1)")
#
#

