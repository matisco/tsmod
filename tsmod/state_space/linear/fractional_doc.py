from typing import Tuple, Literal

import numpy as np

# from base import ModelFit, Signal
from linear_ssm import CompositeLinearStateProcess
# from constrained_matrices import FreeMatrix
from approximate_fi import ApproximateFI
# from arfima_utils import estimate_fractional_d_ewl

# TODO: I wont use this model in the near future and i am lazy. should be quick todo. whatever bye

class FractionalComponents(CompositeLinearStateProcess[Tuple[ApproximateFI, ...]]):

    def __init__(self, n_components):

        processes = tuple(ApproximateFI(approximate_arima_order=(3,1,3)) for _ in range(n_components))
        mixing_matrix = np.eye(n_components)

        super().__init__(processes, mixing_matrix)

    def _first_fit_to(self, series: np.ndarray):
        X = series - series.mean(axis=0)

        cov = np.cov(X, rowvar=False)
        d, V = np.linalg.eigh(cov)

        idx = np.argsort(d)[::-1]
        d = d[idx]
        V = V[:, idx]

        factors = X @ V[:, :self.n_underlying_processes]

        loadings = V[:, :self.n_underlying_processes] * np.sqrt(d[:self.n_underlying_processes])

    def _first_fit_to_factor_model(self,
                                   series: np.ndarray,
                                   include_constant: bool,
                                   measurement_noise: Literal["zero", "diagonal", "free"]):
        raise NotImplementedError()

