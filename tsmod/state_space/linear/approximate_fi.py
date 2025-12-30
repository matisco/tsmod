import numpy as np
from typing import Literal

from arima import ARIMA
from tsmod.state_space.linear.linear_ssm import (LinearStateProcess, LinearStateProcessRepresentation,
                                                 RotationalSymmetry, RepresentationStructure)
from arfima_utils import estimate_fractional_d_ewl, frac_diff
from fi2arima import FracIEARIMAApproximator


class ApproximateFI(LinearStateProcess):

    class AdvancedOptions(LinearStateProcess.AdvancedOptions):

        representation: Literal["harvey", "hamilton", "ihamilton"]

        _valid_options = {
            "representation": ["harvey", "hamilton", "ihamilton"],
        }

        def __init__(
                self,
                correlation_parameterization="hyperspherical",
                representation="hamilton",
        ):
            self._owner = None
            super().__init__(**{k: v for k, v in locals().items() if k != "self" and k != "__class__"})


        def __setattr__(self, name, value):  # overwritten to update underlying_arima.advanced_options.representation
            super().__setattr__(name, value)  # parent validation runs automatically

            # Trigger update on owner if this is 'representation'
            if name == "representation" and self._owner is not None:
                    self._owner._underlying_arima.advanced_options.representation = value

    def __init__(self,
                 approximate_arima_order: tuple[int, int, int],
                 fix_scale: bool = False,
                 advanced_options: AdvancedOptions = None,):

        advanced_options = advanced_options or self.AdvancedOptions()
        advanced_options._owner = self
        super().__init__(shape=(1, 1),
                         innovation_dim=1,
                         scale_constrain="identity" if fix_scale else "diagonal",
                         advanced_options=advanced_options)

        self._d_limits = (0, 1.8)  # currently the supported d values are [0, 1.8]. This could be extended in the future
        self._d = None
        self._T = None
        self._approximator = FracIEARIMAApproximator(approximate_arima_order)
        self._underlying_arima = ARIMA(order=approximate_arima_order, fix_scale=True)
        self._underlying_arima.advanced_options.representation = self.advanced_options.representation

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        if d < self._d_limits[0] or d > self._d_limits[1]:
            raise ValueError("d out of range")
        self._d = d
        self._update_underlying_arima()

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        self._update_underlying_arima()

    def _update_underlying_arima(self):
        if self.is_defined:
            self._approximator.set_T(self.T)
            ar_pacfs, ma_pacfs = self._approximator.get_transformed_pacfs(self.d)
            self._underlying_arima.update_params(np.hstack([ar_pacfs, ma_pacfs]))

    @property
    def approximator(self):
        return self._approximator

    @property
    def is_dynamics_defined(self):
        return (self._d is not None) and (self._T is not None)

    @property
    def n_dynamic_params(self) -> int:
        return 1

    @property
    def state_dim(self) -> int:
        return self._underlying_arima.state_dim

    def _update_dynamic_params(self, params: np.ndarray) -> None:
        if len(params) > 1:
            raise ValueError("params must have exactly 1 element")
        a, b = self._d_limits
        self.d = a + 0.5 * (np.tanh(params[0]) + 1) * (b - a)

    def _get_dynamic_params(self) -> np.ndarray:
        a, b = self._d_limits
        params = np.array([np.atanh(2 * (self.d - a) / (b - a) - 1)])
        return params

    def _first_fit_to(self, series: np.ndarray):
        self.T = series.shape[0]
        self.d = estimate_fractional_d_ewl(series[:, 0])
        if self.scale_constrain == 'diagonal':
            innvos = frac_diff(series[:, 0], self.d)
            std_innov = np.std(innvos)

    @property
    def representation_structure(self) -> RepresentationStructure:
        return self._underlying_arima.representation_structure

    @property
    def rotational_symmetries(self) -> list[RotationalSymmetry]:
        return []

    def representation(self, *args, **kwargs) -> LinearStateProcessRepresentation:
        return self._underlying_arima.representation(*args, **kwargs)




if __name__ == "__main__":

    from matplotlib import pyplot as plt
    # from scipy.signal import lfilter
    from tsmod.state_space.tools.kalman_filter import KalmanFilter, KalmanFilterInitialization
    # from arfima_utils import generate_arfima_from_polys
    from arfima_utils import generate_arfima_from_coeffs
    import time

    estimated_ds = []

    for iter in range(1):
        d = 0.5

        noise = generate_arfima_from_coeffs([], 0, [], 1000)

        simulated_series = generate_arfima_from_coeffs([], d, [], 1000)

        approx_fie = ApproximateFI(approximate_arima_order=(3, 1, 3))


        # approx_fie.d = 0.5
        #
        # approx_fie.set_representation_type('hamilton')
        # res = approx_fie.fit(simulated_series.reshape(-1, 1), measurement_noise="diagonal")
        # print(f"nll: {res.nll}")
        #
        # kf = KalmanFilter().set_endog(simulated_series).set_representation(res.model.representation()).set_initialization(res.model._kf_innit)
        # print(kf.nll())
        #
        # for rep in ['hamilton', 'ihamilton', 'harvey']:
        #     approx_fie.set_representation_type(rep)
        #     kf_innit = KalmanFilterInitialization(initialization_type="ss",
        #                                           x0=np.zeros(res.model._state_process.state_dim, ),
        #                                           P0=None,
        #                                           P_star=None,
        #                                           P_infty=None)
        #
        #     kf = KalmanFilter().set_endog(simulated_series).set_representation(
        #         res.model.representation()).set_initialization(kf_innit)
        #     print(kf.nll())
        #
        # for rep in ['hamilton', 'ihamilton', 'harvey']:
        #     approx_fie.set_representation_type(rep)
        #     approx_fie._first_fit_to(simulated_series.reshape(-1, 1))
        #     res = approx_fie.fit(simulated_series.reshape(-1, 1), measurement_noise="diagonal")
        #     print(f"nll: {res.nll}, d = {approx_fie.d}, e = {res.model.exposures.matrix}, n = {res.model.measurement_noise_std.matrix}")


        for rep in ['hamilton', 'ihamilton', 'harvey']:
            approx_fie.advanced_options.representation = rep

            start_time = time.time()
            res = approx_fie.fit(simulated_series.reshape(-1, 1),
                                 measurement_noise="diagonal")
            end_time = time.time()
            print(f"Iter {iter},  Elapsed time: {end_time - start_time}. Method NLL, with {approx_fie._underlying_arima.advanced_options.representation}, {approx_fie.advanced_options.representation}. NLL: {res.nll}. d = {approx_fie.d}, e = {res.model.exposures}, n = {res.model.measurement_noise_std}")

            approx_fie._first_fit_to(simulated_series.reshape(-1, 1))

            # start_time = time.time()
            # res = approx_fie.fit(simulated_series.reshape(-1, 1),
            #                      objective="EM",
            #                      measurement_noise="diagonal")
            # end_time = time.time()
            # print(f"Iter {iter},  Elapsed time: {end_time - start_time}. Method EM, with {approx_fie._underlying_arima.advanced_options.representation}, {approx_fie.advanced_options.representation}. NLL: {res.nll}. d = {approx_fie.d}, e = {res.model.exposures}, n = {res.model.measurement_noise_std}")


        # estimated_ds.append(approx_fie.d)

    # plt.hist(estimated_ds)
    # plt.show()
