import numpy as np

# from base import ModelFit, Signal
from linear_ssm import CompositeLinearStateProcess
from constrained_matrices import UnConstrained as FreeMatrix

from approximate_fi import ApproximateFI
from arfima_utils import estimate_fractional_d_ewl

class FractionalComponents(CompositeLinearStateProcess):

    def __init__(self, n_components):

        processes = [ApproximateFI(approximate_arima_order=(3,1,3)) for _ in range(n_components)]
        mixing_matrix = np.eye(n_components)

        super().__init__(processes, mixing_matrix)

        self._n_components = n_components
        self._exposures = FreeMatrix((None, self._n_components))

    def _first_fit_to(self, series: np.ndarray):
        X = series - series.mean(axis=0)

        cov = np.cov(X, rowvar=False)
        d, V = np.linalg.eigh(cov)

        idx = np.argsort(d)[::-1]
        d = d[idx]
        V = V[:, idx]

        factors = X @ V[:, :self._n_components]

        loadings = V[:, :self._n_components] * np.sqrt(d[:self._n_components])








