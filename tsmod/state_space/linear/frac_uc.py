import numpy as np

from approximate_fi import ApproximateFI
from arima import ARIMA
from linear_ssm import CompositeLinearStateProcess

from arfima_utils import estimate_fractional_d_ewl

class FracUnobservedComponents(CompositeLinearStateProcess):

    def __init__(self):
        approximate_fi = ApproximateFI(approximate_arima_order=(3, 1, 3))
        ar1 = ARIMA(order=(1,0,0))
        mixing_matrix = np.array([[1, 1]])
        super().__init__([approximate_fi, ar1], mixing_matrix)

    @property
    def long_memory_component(self):
        return self._underlying_processes[0]

    @property
    def short_memory_component(self):
        return self._underlying_processes[1]

    def _first_fit_to(self, series: np.ndarray):
        pass




