from typing import Tuple, Literal, Optional
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize

import tsmod.state_space.linear.arima
from tsmod.state_space.linear.linear_ssm import (AtomicLinearStateProcess, LinearStateProcessDynamics)

from arfima_utils import (transformed_pacfs_to_coeffs,
                          coeffs_to_transformed_pacfs,
                          pacf_to_coeffs,
                          coeffs_to_pacf,
                          representation_hamilton,
                          representation_ihamilton,
                          representation_harvey,
                          arma_coeffs_to_ma_representation,
                          ARIMAFitFromMA,
                          )
from arfima_utils import estimate_fractional_d_ewl, frac_diff
from fi2arima import FracIEARIMAApproximator

from tsmod.tools.utils import PositiveNatural
from tsmod.utils import softplus


class NARIMA(AtomicLinearStateProcess, ABC):

    class AdvancedOptions(AtomicLinearStateProcess.AdvancedOptions):

        representation: Literal["harvey", "hamilton", "ihamilton"]

        _valid_options = {
            "representation": ["harvey", "hamilton", "ihamilton"],
        }

        _immutable_options = tuple()

        def __init__(
            self,
            correlation_parameterization="hyperspherical",
            representation="harvey",
        ):
            super().__init__(**{k: v for k, v in locals().items() if k != "self" and k != "__class__"})

    def __init__(self,
                 order: Tuple[int, int, int],
                 fix_scale: bool = False,
                 advanced_options: Optional[AdvancedOptions] = None,):

        if not all(i >= 0 for i in order):
            raise ValueError(f"Order must be non-negative, got {order}")

        advanced_options = advanced_options or self.AdvancedOptions()
        super().__init__(1, 1,
                         scale_constrain="identity" if fix_scale else "diagonal",
                         advanced_options=advanced_options)

        self._order = order
        self._ar_coeffs = None if self.order[0] > 0 else np.array([])
        self._ma_coeffs = None if self.order[2] > 0 else np.array([])

    @property
    def order(self):
        return self._order

    @property
    def ar_coeffs(self):
        return self._ar_coeffs

    @property
    def ar_poly(self):
        return np.hstack([[1], -self.ar_coeffs])

    @property
    def ma_coeffs(self):
        return self._ma_coeffs

    @property
    def ma_poly(self):
        return np.hstack([[1], self.ma_coeffs])

    @property
    def is_dynamics_defined(self):
        return (self.ar_coeffs is not None) and (self.ma_coeffs is not None)

    @property
    def state_dim(self) -> int:
        p, k, q = self.order
        if self.advanced_options.representation == "hamilton":
            return max([p + k, q + 1])
        elif self.advanced_options.representation == "ihamilton":
            return max([p, q + 1]) + k
        elif self.advanced_options.representation == "harvey":
            return max(p, q + 1) + k
        else:
            raise ValueError(
                f"Unknown representation choice. Valid choices are "
                f"{self.advanced_options.get_valid_options()['representation']}"
            )

    @property
    def dynamic_representation(self) -> LinearStateProcessDynamics:
        phis = self.ar_coeffs
        thetas = self.ma_coeffs
        k = self.order[1]
        if self.advanced_options.representation == "hamilton":
            M, F, R = representation_hamilton(phis, k, thetas)
        elif self.advanced_options.representation == "ihamilton":
            M, F, R = representation_ihamilton(phis, k, thetas)
        elif self.advanced_options.representation == "harvey":
            M, F, R = representation_harvey(phis, k, thetas)
        else:
            raise ValueError(
                f"Unknown representation choice. Valid choices are "
                f"{self.advanced_options.get_valid_options()['representation']}"
            )

        return LinearStateProcessDynamics(M=M, F=F, R=R)

    @abstractmethod
    def _first_fit_to(self, series: np.ndarray):
        pass

    @abstractmethod
    def _update_dynamic_params(self, params: np.ndarray):
        pass

    @abstractmethod
    def _get_dynamic_params(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def n_dynamic_params(self) -> int:
        pass


class ARIMA(NARIMA):  # exactly the same as linear/arima.ARIMA without the setters

    class AdvancedOptions(NARIMA.AdvancedOptions):

        first_estimation_method: Literal["ma_matching", "two_step"]

        _valid_options = {
            "first_estimation_method": ["ma_matching", "two_step"],
        }

        _immutable_options = tuple()

        def __init__(
            self,
            correlation_parameterization="hyperspherical",
            representation="harvey",
            first_estimation_method="ma_matching",
        ):
            super().__init__(**{k: v for k, v in locals().items() if k != "self" and k != "__class__"})

    def __init__(self,
                 order: Tuple[int, int, int],
                 enforce_stability: bool  = True,
                 enforce_invertibility: bool = True,
                 fix_scale: bool = False,
                 advanced_options: Optional[AdvancedOptions] = None,):

        advanced_options = advanced_options or self.AdvancedOptions()

        super().__init__(order=order,
                         fix_scale=fix_scale,
                         advanced_options=advanced_options)

        self._order = order
        self._enforce_stability = enforce_stability
        self._enforce_invertibility = enforce_invertibility

    def _first_fit_to(self, series: np.ndarray):

        def fit_ar_ols(y, p):
            """
            Fit AR(p) model to y using OLS:
                y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t

            Returns:
                phi : (p,) AR coefficients
                e   : residuals aligned with y[p:]
            """
            y = np.asarray(y)
            T = y.shape[0]

            # Build regression matrix
            X = np.column_stack([y[p - k - 1:T - k - 1] for k in range(p)])  # lagged values
            y_target = y[p:]

            # OLS estimate
            phi = np.linalg.lstsq(X, y_target, rcond=None)[0]

            # Residuals
            residuals = y_target - X @ phi

            return phi, residuals

        p, k ,q = self.order

        differenced_series = np.diff(series[:, 0], n=k)

        # ---- Step 1: AR(p) on y ----
        if p > 0:
            phi, res = fit_ar_ols(differenced_series, p)
            self._ar_coeffs = phi
        else:
            res = differenced_series

        if q > 0:
            theta, _ = fit_ar_ols(res, q)
            self._ma_coeffs = -theta  # AR residual model corresponds to MA in ARMA

        if self.advanced_options.first_estimation_method == "ma_matching":

            phis_, _ = fit_ar_ols(differenced_series, p + q + 1)
            T = series.shape[0]
            ma_representation = arma_coeffs_to_ma_representation(phis_, [], T)

            approx_calc = ARIMAFitFromMA(order=(p, 0, q), time_weighting=False)
            aprox_phi, aprox_thetas = approx_calc.calc_coeffs(ma_representation,
                                                              initial_phis=self.ar_coeffs,
                                                              initial_thetas=self.ma_coeffs)

            self._ar_coeffs = aprox_phi
            self._ma_coeffs = aprox_thetas

        innovs = lfilter(self.ma_poly, self.ar_poly, series[:, 0])
        self.cov = self._cov_to_constrained_cov(np.cov(innovs).reshape(1,1), validate=False)

    @property
    def n_dynamic_params(self) -> int:
        return self.order[0] + self.order[2]

    def _get_ar_params(self) -> np.ndarray:
        if self.order[0] == 0:
            return np.array([])

        if self._enforce_stability:
            return coeffs_to_transformed_pacfs(self.ar_coeffs)
        else:
            return self.ar_coeffs

    def _get_ma_params(self) -> np.ndarray:
        if self.order[2] == 0:
            return np.array([])

        if self._enforce_invertibility:
            return coeffs_to_transformed_pacfs(-self.ma_coeffs)
        else:
            return self.ma_coeffs

    def _update_ar_params(self, values):
        if len(values) != self.order[0]:
            raise ValueError(f"AR parameters must have length {self.order[0]}, got {len(values)}")

        if self.order[0] == 0:
            self._ar_coeffs = np.array([])
            return

        if self._enforce_stability:
            self._ar_coeffs = transformed_pacfs_to_coeffs(values)

        else:
            self._ar_coeffs = values

    def _update_ma_params(self, values):
        if len(values) != self.order[2]:
            raise ValueError(f"AR parameters must have length {self.order[2]}, got {len(values)}")

        if self.order[2] == 0:
            self._ma_coeffs = np.array([])
            return

        if self._enforce_invertibility:
            self._ma_coeffs = - transformed_pacfs_to_coeffs(values)

        else:
            self._ma_coeffs = values

    def _get_dynamic_params(self) -> np.ndarray:
        params = np.empty((self.n_dynamic_params,))
        params[:self.order[0]] = self._get_ar_params()
        params[self.order[0]:] = self._get_ma_params()
        return params

    def _update_dynamic_params(self, params: np.ndarray) -> None:
        self._update_ar_params(params[:self.order[0]])
        self._update_ma_params(params[self.order[0]:])


class ApproximateFI(NARIMA):

    class AdvancedOptions(NARIMA.AdvancedOptions):
        pass

    def __init__(self,
                 approximate_arima_order: Tuple[int, int, int],
                 fix_scale: bool = False,
                 advanced_options: Optional[AdvancedOptions] = None, ):

        advanced_options = advanced_options or self.AdvancedOptions()

        super().__init__(order=approximate_arima_order,
                         fix_scale=fix_scale,
                         advanced_options=advanced_options)

        self._d_limits = (0, 1.8)  # currently the supported d values are [0, 1.8]. This could be extended in the future
        self._d = None
        self._T = None
        self._approximator = FracIEARIMAApproximator(approximate_arima_order)

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        if d < self._d_limits[0] or d > self._d_limits[1]:
            raise ValueError("d out of range")
        self._d = d
        self._update_arima()

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        self._update_arima()

    def _update_arima(self):
        if self.d is not None and self.T is not None:
            self._approximator.set_T(self.T)
            ar_pacfs, ma_pacfs = self._approximator.get_transformed_pacfs(self.d)
            self._ar_coeffs = transformed_pacfs_to_coeffs(ar_pacfs)
            self._ma_coeffs = - transformed_pacfs_to_coeffs(ma_pacfs)

    @property
    def approximator(self):
        return self._approximator

    @property
    def n_dynamic_params(self) -> int:
        return 1

    def _update_dynamic_params(self, params: np.ndarray) -> None:
        if len(params) > 1:
            raise ValueError("params must have exactly 1 element")
        a, b = self._d_limits
        self.d = a + 0.5 * (np.tanh(params[0]) + 1) * (b - a)
        self._update_arima()

    def _get_dynamic_params(self) -> np.ndarray:
        a, b = self._d_limits
        params = np.array([np.arctanh(2 * (self.d - a) / (b - a) - 1)])
        return params

    def _first_fit_to(self, series: np.ndarray):
        self.T = series.shape[0]
        self.d = estimate_fractional_d_ewl(series[:, 0])

        innovs = frac_diff(series[:, 0], self.d)
        self.cov = self._cov_to_constrained_cov(np.cov(innovs).reshape(1, 1), False)
        self._update_arima()


class ExponentialQuadraticNARMA(NARIMA):

    class AdvancedOptions(NARIMA.AdvancedOptions):

        ma_length_for_matching: int

        _valid_options = {
            "first_estimation_method": PositiveNatural,
        }

        _immutable_options = tuple()

        def __init__(
            self,
            correlation_parameterization="hyperspherical",
            representation="harvey",
            ma_length_for_matching=20,
        ):
            super().__init__(**{k: v for k, v in locals().items() if k != "self" and k != "__class__"})

    def __init__(self,
                 approximate_arima_order: Tuple[int, int, int],
                 fix_scale: bool = False,
                 advanced_options: Optional[AdvancedOptions] = None,):

        super().__init__(order=approximate_arima_order,
                         fix_scale=fix_scale,
                         advanced_options=advanced_options)

        self._dynamic_params = None

    def _first_fit_to(self, series: np.ndarray):

        def fit_ar_ols(y, p):
            """
            Fit AR(p) model to y using OLS:
                y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t

            Returns:
                phi : (p,) AR coefficients
                e   : residuals aligned with y[p:]
            """
            y = np.asarray(y)
            T = y.shape[0]

            # Build regression matrix
            X = np.column_stack([y[p - k - 1:T - k - 1] for k in range(p)])  # lagged values
            y_target = y[p:]

            # OLS estimate
            phi = np.linalg.lstsq(X, y_target, rcond=None)[0]

            # Residuals
            residuals = y_target - X @ phi

            return phi, residuals

        p, k, q = self.order

        phis_, _ = fit_ar_ols(series[:, 0], p + q + 1)
        T = series.shape[0]
        ma_representation = arma_coeffs_to_ma_representation(phis_, [], self.advanced_options.ma_length_for_matching)

        # --- Objective function --------------------------------------------------
        def objective(x):
            self._dynamic_params = x
            ma_rep = self._calc_EQ_ma()
            diff = ma_rep - ma_representation
            return np.sum(diff ** 2)

        best_f = np.inf
        best_x = None
        stagnation = 0

        for _ in range(100):
            # randomize missing parts only
            trial = np.random.rand(T) * 10 - 5

            result = minimize(objective, trial, method="L-BFGS-B")
            fval = result.fun

            if abs(fval - best_f) / (best_f + 1e-12) < 0.01:
                stagnation += 1
            else:
                stagnation = 0

            if fval < best_f:
                best_f = fval
                best_x = result.x

            if stagnation > 4:
                break

        self._dynamic_params = best_x

    @property
    def n_dynamic_params(self) -> int:
        return 3

    def _get_dynamic_params(self) -> np.ndarray:
        return self._dynamic_params

    def _update_dynamic_params(self, params: np.ndarray):
        self._dynamic_params = params
        self._calc_ARIMA()

    def _calc_ARIMA(self):
        ma_coeffs = self._calc_EQ_ma()
        approx_calc = ARIMAFitFromMA(order=self.order, time_weighting=False)
        aprox_phi, aprox_thetas = approx_calc.calc_coeffs(ma_coeffs,
                                                          initial_phis=self.ar_coeffs,
                                                          initial_thetas=self.ma_coeffs)

        self._ar_coeffs = aprox_phi
        self._ma_coeffs = aprox_thetas

    def _calc_EQ_ma(self) -> np.ndarray:
        xs = np.arange(self.advanced_options.ma_length_for_matching + 1)
        d = softplus(self._dynamic_params[0])
        a, b = self._dynamic_params[1:]
        return np.exp(-d * xs) * (1 + a * xs + b * xs ** 2)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from scipy.signal import lfilter
    from arfima_utils import generate_arfima_from_coeffs

    xs = np.arange(40)

    d = np.random.rand(1) * 10 - 5
    a = np.random.rand(1) * 10 - 5
    b = np.random.rand(1) * 10 - 5

    d = softplus(d)
    ma_rep = np.exp(-d * xs) * (1 + a * xs + b * xs ** 2)

    p = 0
    k = 0
    q = 3
    arima = tsmod.state_space.linear.arima.ARIMA(order=(p, k, q), fix_scale=False, enforce_stability=False, enforce_invertibility=False)
    arima.ma_poly = ma_rep[:q+1]
    arima.cov = np.array([[1]])

    series = arima.simulate(k=1000, burn=1000)

    plt.plot(series[:, 0])
    plt.show()

