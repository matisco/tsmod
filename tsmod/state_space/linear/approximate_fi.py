import numpy as np

from arima import ARIMA
from build.lib.tsmod.optimization_objectives import GaussianNLL
from state_space.linear.linear_ssm import LinearStateProcessRepresentation
from tsmod.state_space.linear.linear_ssm import  LinearStateProcess
from arfima_utils import estimate_fractional_d_ewl
from fi2arima import FracIEARIMAApproximator



class ApproximateFI(LinearStateProcess):

    def __init__(self,
                 approximate_arima_order: tuple[int, int, int]):

        super().__init__(shape=(1, 1))

        self._d_limits = (0, 1.8)  # currently the supported d values are [0, 1.8]. This could be extended in the future
        self._d = None
        self._T = None
        self._approximator = FracIEARIMAApproximator(approximate_arima_order)
        self._underlying_arima = ARIMA(order=approximate_arima_order)

        self._force_full_approximate_representation = False

    def set_representation_type(self, value):
        self._underlying_arima.set_representation_type(value)
        return self

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

            approx_fie.set_representation_type(rep)

            start_time = time.time()
            res = approx_fie.fit(simulated_series.reshape(-1, 1),
                                 measurement_noise="diagonal")
            end_time = time.time()
            print(f"Iter {iter},  Elapsed time: {end_time - start_time}. Method NLL, with {approx_fie._underlying_arima._representation_choice}. NLL: {res.nll}. d = {approx_fie.d}, e = {res.model.exposures.matrix}, n = {res.model.measurement_noise_std.matrix}")

            approx_fie._first_fit_to(simulated_series.reshape(-1, 1))

            start_time = time.time()
            res = approx_fie.fit(simulated_series.reshape(-1, 1),
                                 objective="EM",
                                 measurement_noise="diagonal")
            end_time = time.time()
            print(f"Iter {iter},  Elapsed time: {end_time - start_time}. Method EM, with {approx_fie._underlying_arima._representation_choice}. NLL: {res.nll}. d = {approx_fie.d}, e = {res.model.exposures.matrix}, n = {res.model.measurement_noise_std.matrix}")


        # estimated_ds.append(approx_fie.d)

    # plt.hist(estimated_ds)
    # plt.show()
