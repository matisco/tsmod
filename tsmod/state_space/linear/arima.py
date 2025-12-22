from typing import Tuple, Literal
import numpy as np

# from base import ModelFit
# from build.lib.tsmod.optimization_objectives import GaussianNLL
from tsmod.state_space.linear.linear_ssm import LinearStateProcess, LinearStateProcessRepresentation

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


class ARIMA(LinearStateProcess):

    class AdvancedOptions(LinearStateProcess.AdvancedOptions):

        representation: Literal["harvey", "hamilton", "ihamilton"]
        first_estimation_method: Literal["ma_matching", "two_step"]

        _valid_options = {
            "representation": ["harvey", "hamilton", "ihamilton"],
            "first_estimation_method": ["ma_matching", "two_step"],
        }

        def __init__(
            self,
            correlation_parameterization="hyperspherical",
            representation="harvey",
            first_estimation_method="ma_matching"
        ):
            super().__init__(**{k: v for k, v in locals().items() if k != "self"})

    def __init__(self,
                 order = Tuple[int, int, int],
                 enforce_stability = True,
                 enforce_invertibility = True,
                 advanced_options: AdvancedOptions = None,):

        advanced_options = advanced_options or self.AdvancedOptions()
        super().__init__(shape=(1,1), advanced_options=advanced_options)

        self._order = order
        self._enforce_stability = enforce_stability
        self._enforce_invertibility = enforce_invertibility

        self._ar_coeffs = None if self.order[0] > 0 else np.array([])
        self._ma_coeffs = None if self.order[2] > 0 else np.array([])

    @property
    def is_dynamics_defined(self):
        return (self.ar_coeffs is not None) and (self.ma_coeffs is not None)

    @property
    def n_dynamic_params(self) -> int:
        return self.order[0] + self.order[2]

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

    @ar_coeffs.setter
    def ar_coeffs(self, coeffs):
        if len(coeffs) != self.order[0]:
            raise ValueError(f"AR coefficients must have length {self.order[0]}, got {len(coeffs)}")

        if self.order[0] == 0:
            self._ar_coeffs = np.array([])
            return

        if self._enforce_stability:
            pacf = coeffs_to_pacf(coeffs)
            if any(abs(i) > 1 for i in pacf):
                raise ValueError("AR coefficients produce an unstable (non-stationary) AR polynomial, and _enforce_stability is True")

        self._ar_coeffs = coeffs

    def set_ar_coeffs(self, coeffs):
        self.ar_coeffs = coeffs
        return self

    @ar_poly.setter
    def ar_poly(self, value):
        value = np.asarray(value, dtype=float)

        # Check that it is 1D
        if value.ndim != 1:
            raise ValueError("AR polynomial must be a 1D array")

        # Check correct length
        expected_length = self.order[0] + 1
        if len(value) != expected_length:
            raise ValueError(f"AR polynomial must have length {expected_length}, got {len(value)}")

        # Check that first coefficient is 1
        if not np.isclose(value[0], 1.0):
            raise ValueError("The first coefficient of AR polynomial must be 1")

        self.ar_coeffs = -value[1:]

    def set_ar_poly(self, poly):
        self.ar_poly = poly
        return self

    @ma_coeffs.setter
    def ma_coeffs(self, coeffs):
        if len(coeffs) != self.order[2]:
            raise ValueError(f"MA coefficients must have length {self.order[2]}, got {len(coeffs)}")

        if self.order[2] == 0:
            self._ma_coeffs = np.array([])
            return

        if self._enforce_stability:
            pacf = coeffs_to_pacf(-coeffs)  # the function takes the coefficients of a polynomial like [1 - a_1 L - a_2 L^2 ...]
            if any(abs(i) > 1 for i in pacf):
                raise ValueError("AR coefficients produce an unstable (non-stationary) AR polynomial, and _enforce_stability is True")

        self._ma_coeffs = coeffs

    def set_ma_coeffs(self, coeffs):
        self.ma_coeffs = coeffs
        return self

    @ma_poly.setter
    def ma_poly(self, value):
        value = np.asarray(value, dtype=float)

        # Check that it is 1D
        if value.ndim != 1:
            raise ValueError("MA polynomial must be a 1D array")

        # Check correct length
        expected_length = self.order[2] + 1
        if len(value) != expected_length:
            raise ValueError(f"MA polynomial must have length {expected_length}, got {len(value)}")

        # Check that first coefficient is 1
        if not np.isclose(value[0], 1.0):
            raise ValueError("The first coefficient of MA polynomial must be 1")

        self.ma_coeffs = value[1:]

    def set_ma_poly(self, poly):
        self.ma_poly = poly
        return self

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
            self.ar_coeffs = values

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
        params = np.empty((self.n_params,))
        params[:self.order[0]] = self._get_ar_params()
        params[self.order[0]:] = self._get_ma_params()
        return params

    def _update_dynamic_params(self, params: np.ndarray) -> None:
        self._update_ar_params(params[:self.order[0]])
        self._update_ma_params(params[self.order[0]:])

    @property
    def state_dim(self) -> int:
        p, k, q = self.order
        if self._advanced_options.representation == "hamilton":
            return max([p + k, q + 1])
        elif self._advanced_options.representation == "ihamilton":
            return max([p, q + 1]) + k
        elif self._advanced_options.representation == "harvey":
            return max(p, q + 1) + k
        else:
            raise ValueError(
                f"Unknown representation choice. Valid choices are "
                f"{self._advanced_options.get_valid_options()['representation']}"
            )

    def representation(self, *args, **kwargs) -> LinearStateProcessRepresentation:
        phis = self.ar_coeffs
        thetas = self.ma_coeffs
        k = self.order[1]
        if self._advanced_options.representation == "hamilton":
            M, F, R = representation_hamilton(phis, k, thetas)
        elif self._advanced_options.representation == "ihamilton":
            M, F, R = representation_ihamilton(phis, k, thetas)
        elif self._advanced_options.representation == "harvey":
            M, F, R = representation_harvey(phis, k, thetas)
        else:
            raise ValueError(
                f"Unknown representation choice. Valid choices are "
                f"{self._advanced_options.get_valid_options()['representation']}"
            )

        return LinearStateProcessRepresentation(M, F, R)

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
            self.ar_coeffs = phi
        else:
            res = differenced_series

        if q > 0:
            theta, _ = fit_ar_ols(res, q)
            self.ma_coeffs = -theta  # AR residual model corresponds to MA in ARMA

        if self._advanced_options.first_estimation_method == "ma_matching":

            phis_, _ = fit_ar_ols(differenced_series, p + q + 1)
            T = series.shape[0]
            ma_representation = arma_coeffs_to_ma_representation(phis_, [], T)

            approx_calc = ARIMAFitFromMA(order=(p, 0, q), time_weighting=False)
            aprox_phi, aprox_thetas = approx_calc.calc_coeffs(ma_representation,
                                                              initial_phis=self.ar_coeffs,
                                                              initial_thetas=self.ma_coeffs)

            self.ar_coeffs = aprox_phi
            self.ma_coeffs = aprox_thetas


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from scipy.signal import lfilter
    from arfima_utils import generate_arfima_from_coeffs

    p = 2
    q = 1
    k = 0

    np.random.seed(0)

    pacfs_AR = np.random.rand(p) * 1.9 - 1
    pacfs_MA = np.random.rand(q) * 1.9 - 1

    if p > 0:
        phis = pacf_to_coeffs(pacfs_AR)
    else:
        phis = np.array([])

    if q > 0:
        thetas = - pacf_to_coeffs(pacfs_MA)
    else:
        thetas = np.array([])

    noise = 0.5 * generate_arfima_from_coeffs([], 0, [], 1000)

    simulated_series = 0.5 * generate_arfima_from_coeffs(phis, k, thetas, 1000) + noise

    arima = ARIMA(order=(p, k, q), enforce_stability=True, enforce_invertibility=True)
    arima.set_advanced_option("first_estimation_method", "ma_matching")

    print(f"phis: {phis}, k: {k}, thetas: {thetas}")


    for rep in ["hamilton", "ihamilton", "harvey"]:
        arima.set_advanced_option("representation", rep)

        arima._first_fit_to(simulated_series.reshape(-1, 1))

        res = arima.fit(simulated_series.reshape(-1, 1),
                        measurement_noise="diagonal")

        print(f"nll: {res.nll}. ar: {arima.ar_coeffs}. ma: {arima.ma_coeffs}, e: {res.model.exposures.matrix}, std: {res.model.measurement_noise_std.matrix}")


    for _ in range(0):
        p = np.random.randint(0, 5)
        q = np.random.randint(0, 5)
        k = np.random.randint(2)

        pacfs_AR = np.random.rand(p) * 2 - 1
        pacfs_MA = np.random.rand(q) * 2 - 1

        if p > 0:
            phis = pacf_to_coeffs(pacfs_AR)
        else:
            phis = np.array([])

        if q > 0:
            thetas = - pacf_to_coeffs(pacfs_MA)
        else:
            thetas = np.array([])

        harvey_size = max(len(phis), len(thetas) + 1) + k
        hamilton_size = np.max([len(phis) + k, len(thetas) + 1])
        ihamilton_size =  np.max([len(phis), len(thetas) + 1]) + k
        sizes = [harvey_size, hamilton_size, ihamilton_size]

        funcs = [representation_harvey, representation_hamilton, representation_ihamilton]
        names = ["Harvey", "H", "IH"]
        for i, fnc in enumerate(funcs):
            M, F, R = fnc(phis, k, thetas)
            if not sizes[i] == F.shape[0]:
                raise ValueError(f"{names[i]} size does not match F.shape[0]")

        T = 1000
        np.random.seed(0)
        e = np.random.normal(size=T)

        AR_poly = np.hstack([1, -phis])
        MA_poly = np.hstack([1, thetas])
        y1 = lfilter(MA_poly, AR_poly, e)
        for _ in range(k):
            y1 = lfilter([1], [1, -1], y1)

        # Harvey
        M, F, R = representation_harvey(phis, k, thetas)
        x = np.zeros((T, F.shape[0]))
        x[0] = R[:, 0] * e[0]
        for t in range(1, T):
            x[t] = F @ x[t - 1] + R[:, 0] * e[t]
        y_harvey = x @ M.T
        y_harvey = y_harvey[:, 0]

        # Integrated Hamilton
        M, F, R = representation_ihamilton(phis, k, thetas)

        x = np.zeros((T, F.shape[0]))
        x[0] = R[:, 0] * e[0]
        for t in range(1, T):
            x[t] = F @ x[t - 1] + R[:, 0] * e[t]
        y_ihamilton = x @ M.T
        y_ihamilton = y_ihamilton[:, 0]

        # Hamilton
        M, F, R = representation_hamilton(phis, k, thetas)

        x = np.zeros((T, F.shape[0]))
        x[0] = R[:, 0] * e[0]
        for t in range(1, T):
            x[t] = F @ x[t - 1] + R[:, 0] * e[t]
        y_hamilton = x @ M.T
        y_hamilton = y_hamilton[:, 0]

        if not np.allclose(y1, y_hamilton, atol=1e-12):
            raise ValueError("Hamilton error")

        if not np.allclose(y1, y_harvey, atol=1e-12):
            raise ValueError("Harvey error")

        if not np.allclose(y1, y_ihamilton, atol=1e-12):
            raise ValueError("Integrated Hamilton error")



        # kf = KalmanFilter().set_endog(y).set_matrices(M, F, R, np.array([[1]]), np.array([[1]]))
        # kf.initialize(initialization_type='SS', x0=np.zeros((F.shape[0],)))
        #
        # filtered = kf.filter()
        #
        # predicted_arma = filtered.get_factors(conditional_on='predicted')
        # # smoothed_arma = filtered.get_factors(conditional_on='smoothed')
        #
        # print(predicted_arma.shape)
        #
        # plt.plot(y)
        # plt.plot(predicted_arma, '.')
        # plt.show()
