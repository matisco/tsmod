"""
At the moment this only implements multivariate OLS but with the framework i have designed.

I wont use this so its ok for now

"""

import numpy as np
from sympy.core.cache import cached_property

from base import ModelFit, Model, DeterministicForecastResult, ForecastResult
from constrained_matrices import ConstrainedMatrix, UnConstrained
from optimization_objectives import OptimizationObjective


class Regression(Model):

    def __init__(self,
                 exog: np.ndarray,
                 exog_loadings:ConstrainedMatrix
                 ):
        super().__init__((None, 1))

        self._exog = exog
        self._exog_loadings = exog_loadings

        if not isinstance(self._exog_loadings, UnConstrained):
            raise NotImplementedError("Regression currently only support unconstrained loadings")

        self._exog_loadings.shape = (None, self._exog.shape[1])

    @property
    def _n_params(self) -> int:
        return self._exog_loadings.n_params

    @property
    def exog(self):
        return self._exog

    @property
    def n_exog(self):
        return self._exog.shape[1]

    @property
    def exog_loadings(self):
        return self._exog_loadings

    @property
    def is_defined(self):
        return self.exog_loadings.is_defined

    def _get_params(self) -> np.ndarray:
        return self.exog_loadings.get_params()

    def _update_params(self, params: np.ndarray) -> None:
        self.exog_loadings.update_params(params)

    def fit(self, series: np.ndarray) -> "RegressionFit":
        n_obs = series.shape[1]
        self._exog_loadings.shape = (n_obs, None)

        if not self._exog_loadings.is_defined and self._exog_loadings.is_frozen:
            raise AttributeError("Exogenous loadings are frozen and undefined")

        if self._exog_loadings.is_frozen:
            return RegressionFit(series, self)

        B_hat, _, _, _ = np.linalg.lstsq(self.exog, series, rcond=None)
        if B_hat.ndim == 1:
            B_hat = B_hat.reshape(-1, 1)
        else:
            B_hat = B_hat.T

        self.exog_loadings.matrix = B_hat
        self.freeze()
        self.exog_loadings.freeze()
        return RegressionFit(series, self)

    def forecast(self):
        raise NotImplementedError()

    def predict(self, exog: np.ndarray) -> np.ndarray:
        if not self._exog_loadings.is_defined:
            raise AttributeError("Exogenous loadings are undefined. Can not predict.")
        else:
            return exog @ self.exog_loadings.matrix.T


class RegressionFit(ModelFit):

    def __init__(self,
                 series: np.ndarray,
                 model: Regression):

        super().__init__(series, model)

    def get_loss(self, loss: OptimizationObjective) -> float:
        e = self.get_prediction_errors
        if loss.requires_cov:
            raise ValueError("Regression model currently does not support loss function which requires cov")

        return loss(e)

    @cached_property
    def get_prediction_errors(self):
        return self.series - self.model.exog @ self.model.exog_loadings.matrix.T

    def forecast_with_uncertainty(self) -> ForecastResult:
        raise NotImplementedError()

    def forecast(self, *args, **kwargs) -> DeterministicForecastResult:
        raise NotImplementedError()

    def predict(self, exog: np.ndarray) -> np.ndarray:
        return self.model.predict(exog)

    def _calc_nll(self) -> float:  # maybe this is wrong, I dont care i wont use it
        # Taken from statsmodels.regression.linear_model.OLS
        nobs2 = self.nobs / 2.0
        nobs = float(self.nobs)
        resid = self.series - self.model.exog @ self.model.exog_loadings.matrix.T
        ssr = np.sum(resid ** 2)
        return -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2



if "__main__" == __name__:
    import numpy as np
    from scipy.signal import lfilter
    from matplotlib import pyplot as plt
    import time


    def generate_ar1(phi, n, sigma=1.0, x0=0.0):
        # white noise
        eps = np.random.normal(scale=sigma, size=n)

        # AR(1) filter coefficients: x_t = phi x_{t-1} + eps_t
        a = [1, -phi]  # AR polynomial
        b = [1]  # MA polynomial (just noise)

        # generate AR(1)
        x = lfilter(b, a, eps)

        # set initial condition if desired
        if x0 != 0:
            x = x + x0 - x[0]
        return x


    n_factors = 5
    T = 1000
    N = 50
    factors = np.empty((T, n_factors))

    for i in range(n_factors):
        factors[:, i] = generate_ar1(np.random.rand() * 0.5 + 0.4, T)


    loadings = UnConstrained((N, n_factors))
    params = np.random.rand(loadings.n_params)
    loadings.update_params(params)

    noise_scale = 0.5

    endog = factors @ loadings.matrix.T + noise_scale * np.random.randn(T, N)

    model = Regression(factors, UnConstrained((None, None)))
    res = model.fit(endog)

    print(res)

    print(res.nll)

    from optimization_objectives import MAE, MSE, LogCosh

    print(res.get_loss(LogCosh()))
    print(res.get_loss(MAE()))
    print(res.get_loss(MSE()))

