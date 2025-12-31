import numpy as np

from approximate_fi import ApproximateFI
from arima import ARIMA
from linear_ssm import CompositeLinearStateProcess


class FracUnobservedComponents(CompositeLinearStateProcess):

    def __init__(self):
        approximate_fi = ApproximateFI(approximate_arima_order=(3, 1, 3))
        ar1 = ARIMA(order=(1,0,0))
        mixing_matrix = np.array([[1, 1]])
        super().__init__([approximate_fi, ar1], mixing_matrix)

    @property
    def long_memory_component(self) -> ApproximateFI:
        return self._underlying_processes[0]

    @property
    def short_memory_component(self) -> ARIMA:
        return self._underlying_processes[1]

    def _first_fit_to(self, series: np.ndarray):
        fie: ApproximateFI = self.long_memory_component
        ar1: ARIMA = self.short_memory_component

        fie._first_fit_to(series)

        ar1.ar_coeffs = np.array([0])
        ar1.cov = fie.cov


if __name__ == '__main__':

    from arfima_utils import generate_arfima_from_coeffs

    d = 0.4
    std1 = 0.5
    ar1_coeff = 0.8
    std2 = 1

    fie = ApproximateFI(approximate_arima_order=(3, 1, 3))
    fie.d = d
    fie.std = std1

    ar1 = ARIMA(order=(1, 0, 0))
    ar1.ar_coeffs = np.array([ar1_coeff])
    ar1.std = std2

    series1 = generate_arfima_from_coeffs([], d, [], 1000)
    series2 = ar1.simulate(k=1000, burn=1000)

    series = series1 + series2

    frac_uc = FracUnobservedComponents()
    res = frac_uc.fit(series)

    estimated_fie = frac_uc.long_memory_component
    estimated_ar1 = frac_uc.short_memory_component
    print(f"d: {estimated_fie.d}, std: {estimated_fie.std}, ar1: {estimated_ar1.ar_coeffs}, std: {estimated_ar1.std}")



