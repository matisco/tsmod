import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy.interpolate import UnivariateSpline
import math

import os
from typing import Literal, Tuple
from pathlib import Path

from arima import ARIMA
import arfima_utils

MODULE_DIR = Path(__file__).resolve().parent


class FracIEARIMAApproximator:

    def __init__(self, approximation_order: Tuple[int, int, int] = (3, 1, 3)):
        self._approximation_order = approximation_order
        self._retriever = None
        self._T = None

    def set_T(self, T: int):
        if T == self._T:
            return self

        if not isinstance(T, int):
            raise TypeError("T must be an integer")

        if T < 1:
            raise ValueError("T must be a positive integer")

        p, k, q = self._approximation_order
        self._T = T

        filename = "ARIMA" + str(p) + str(k) + str(q) + "_N" + str(T) + ".csv"

        if k != 1 and k != 0:
            raise ValueError("Only ready for k = 1 or 0")

        dir_name = f"ARIMA{p}{k}{q}_Results"
        data_dir = MODULE_DIR / "data" / dir_name

        file_path = data_dir / filename  # file path inside the directory
        file_exists = file_path.is_file()  # True only if the file exists
        if not file_exists:
            raise ValueError(" I dont actually want this to run. So precalculate the approximations")
            ds = np.arange(0.00, 1.8, 0.01)
            include_delta = (k > 0)
            results, _, _ = FractionalARIMACalculator.calc_approximations(ds=ds, T=T, p=p, q=q,
                                                                          include_delta=include_delta)

            results = np.hstack([ds.reshape(-1, 1), results])
            np.savetxt(file_path, results, delimiter=',', fmt='%.8f')  # '%.2f' formats floats to 2 decimal places

        self._retriever = FractionalARIMARetriever(file_path, p, k)

        return self

    def get_arma_coeffs(self, d):
        """

        Args:
            d: (float) order of fractional differencing

        Returns:
            phis: (np.array) ar coefficients of the approximating arima process
            thetas: (np.array) ma coefficients of the approximating arima process
            k: (int) integration order the approximating arima process
        """
        if self._retriever is None:
            raise RuntimeError("Set T prior to calling get_arma_coeffs")

        # if not force_full_approximate_representation:
        #     if np.isclose(np.round(d), d):
        #         return np.array([]), np.array([]), int(d)

        return self._retriever.get_arma_coeffs(d=d)

    def get_pacfs(self, d):
        if self._retriever is None:
            raise RuntimeError("Set T prior to calling get_pacfs")
        return self._retriever.get_pacfs(d=d)

    def get_transformed_pacfs(self, d):
        if self._retriever is None:
            raise RuntimeError("Set T prior to calling get_pacfs")
        return self._retriever.get_transformed_pacfs(d=d)

    def get_arima_process(self, d):
        # phis, thetas = self.get_arma_coeffs(d)
        ar_t_pacf, ma_t_pacf = self.get_transformed_pacfs(d)
        p, k, q = self._approximation_order
        arima = ARIMA(order=(p, k, q), enforce_stability=True, enforce_invertibility=True)
        # arima.ar_coeffs = phis
        # arima.ma_coeffs = thetas
        arima._update_ar_params(ar_t_pacf)
        arima._update_ma_params(ma_t_pacf)
        arima.freeze()
        return arima

    def get_representation(self,
                           d: float,
                           representation_type: Literal["hamilton", "harvey", "ihamilton"] = "hamilton"):
        arima = self.get_arima_process(d)
        arima.representation_type = representation_type
        return arima.representation()


class FractionalARIMARetriever:
    """
    Retrieves the AR and MA polynomials which approximate (in terms on MSE of the MA representation)
    a fractionally integrated error sequence.

    The process

    y_t = Delta^{-d} e_t, e_t ~ N(0, 1),

    where Delta = (1-L), with L the backshift operator,
    is approximated by an ARIMA(p, k, q) process, denoted x_t, and where p,k,q are integers:

    Phi(L) Delta^k x_t = Theta(L) u_t, u_t ~ N(0,1)

    This class retrieves the corresponding AR and MA polynomials and the integer differencing order
    k, based on pre-computed approximations produced by `FractionalARIMACalculator`.

    References:

        [1] Hartl, T., & Jucknewitz, R. (2022). Approximate state space modelling of unobserved fractional components. Econometric Reviews, 41(1), 75-98.


    Methods
    -------
    get_arma_coeffs(d, prev_d=None)
        Return the AR and MA polynomial coefficients and the (integer) order of integration.

    getStateSpaceRep(d, prev_d=None)
        Return the state-space representation matrices (F, R, M) corresponding to the approximated process.
    """

    def __init__(self, file, p, k):

        self.p = p
        self.k = k

        data = np.loadtxt(file, delimiter=',')
        ds_data = data[:, 0]
        arctanh_pacf_data = data[:, 1:]

        d_range = (np.min(ds_data), np.max(ds_data))
        self.d_range = d_range

        self.splines = []
        for j in range(arctanh_pacf_data.shape[1]):
            self.splines.append(UnivariateSpline(ds_data, arctanh_pacf_data[:, j], s=0.9 * len(ds_data)))

    def get_pacfs(self, d):
        if not isinstance(d, (int, float)):
            raise TypeError(f"d must be a number, got {type(d).__name__}")

        if not (self.d_range[0] <= d <= self.d_range[1]):
            raise ValueError(
                f"d = {d} is outside the valid range {self.d_range}."
            )

        # target = d if prev_d is None else prev_d
        # idx = next((i for i, interval in enumerate(self.intervals) if interval[0] <= target < interval[1]), None)

        interpolated_arctanh_pacf = np.array([spline(d) for spline in self.splines])
        params = np.tanh(interpolated_arctanh_pacf)
        return params[0:self.p], params[self.p:]

    def get_transformed_pacfs(self, d):
        if not isinstance(d, (int, float)):
            raise TypeError(f"d must be a number, got {type(d).__name__}")

        if not (self.d_range[0] <= d <= self.d_range[1]):
            raise ValueError(
                f"d = {d} is outside the valid range {self.d_range}."
            )

        # target = d if prev_d is None else prev_d
        # idx = next((i for i, interval in enumerate(self.intervals) if interval[0] <= target < interval[1]), None)

        interpolated_arctanh_pacf = np.array([spline(d) for spline in self.splines])
        return interpolated_arctanh_pacf[0:self.p], interpolated_arctanh_pacf[self.p:]

    def get_arma_coeffs(self, d):
        """
        Retrieves the approximate AR and MA coefficients

        """
        if not isinstance(d, (int, float)):
            raise TypeError(f"d must be a number, got {type(d).__name__}")

        if not (self.d_range[0] <= d <= self.d_range[1]):
            raise ValueError(
                f"d = {d} is outside the valid range {self.d_range}."
            )

        # target = d if prev_d is None else prev_d
        # idx = next((i for i, interval in enumerate(self.intervals) if interval[0] <= target < interval[1]), None)

        interpolated_arctanh_pacf = np.array([spline(d) for spline in self.splines])

        params = np.tanh(interpolated_arctanh_pacf)
        phis = arfima_utils.pacf_to_coeffs(params[0:self.p])
        thetas = -arfima_utils.pacf_to_coeffs(params[self.p:])

        return phis, thetas


class FractionalARIMACalculator:
    """
    Implementation of the ARIMA approximation method described in:

        Hartl, T., & Jucknewitz, R. (2022). Approximate state space modelling of unobserved fractional components. Econometric Reviews, 41(1), 75-98.

    This implementation was written independently and is not affiliated with the authors.
    """

    @staticmethod
    def calc_approximations(ds: list | np.ndarray, T: int, p: int = 3, q: int = 3, include_delta: bool = False):

        results = np.zeros((len(ds), p + q))
        res_rmse = np.zeros((len(ds),))

        idx_mid = int(np.floor(len(ds) / 2))
        AR_poly, MA_poly, x_opt, fval = FractionalARIMACalculator.calc_approximation(ds[idx_mid], p, q, T,
                                                                                     include_delta)
        results[idx_mid, :] = x_opt
        res_rmse[idx_mid] = np.sqrt(fval)

        for i in range(idx_mid + 1, len(ds)):
            AR_poly, MA_poly, x_opt, fval = FractionalARIMACalculator.calc_approximation(ds[i], p, q, T, include_delta,
                                                                                         x0=results[i - 1, :])
            results[i, :] = x_opt
            res_rmse[i] = np.sqrt(fval)

        for i in range(idx_mid - 1, -1, -1):
            AR_poly, MA_poly, x_opt, fval = FractionalARIMACalculator.calc_approximation(ds[i], p, q, T, include_delta,
                                                                                         x0=results[i + 1, :])
            results[i, :] = x_opt
            res_rmse[i] = np.sqrt(fval)

        polys = np.zeros((len(ds), p+q+2))
        for i in range(len(ds)):
            params = results[i]
            params = np.tanh(params)
            polys[i, :1+p] = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[:p])])
            polys[i, 1+p:] = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[p:])])

        return results, res_rmse, polys

    @staticmethod
    def approximation_MSE_for_min(params, AR_order, T, frac_diff_MA_coeffs, include_delta):

        params = np.tanh(params)

        AR_poly = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[:AR_order])])
        MA_poly = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[AR_order:])])

        arma_ma_coeffs = arfima_utils.arma_poly_to_ma_representation(AR_poly, MA_poly, T)
        if include_delta:
            arma_ma_coeffs = arfima_utils.arma_poly_to_ma_representation([1, -1], arma_ma_coeffs, T)

        weights = np.arange(T, 0, -1)
        diffs = arma_ma_coeffs - frac_diff_MA_coeffs

        return np.sum(weights * diffs ** 2) / T

    @staticmethod
    def calc_approximation(d, AR_order, MA_order, T, include_delta, x0=None):

        frac_diff_MA_coeffs = arfima_utils.fractional_integrated_errors_to_ma_poly(d, T)

        def objective(x):
            return FractionalARIMACalculator.approximation_MSE_for_min(
                x, AR_order, T, frac_diff_MA_coeffs, include_delta
            )

        if x0 is None:
            best_x_opt = None
            best_fval = 1e6
            lowest_count = 0
            for i in range(100):
                while True:
                    pis = 2 * np.random.rand(AR_order + MA_order) - 1

                    x0 = np.atanh(pis)
                    if objective(x0) < 1e6:
                        break

                result = minimize(
                    fun=objective,  # your scalar-valued objective
                    x0=x0,  # initial guess
                    method='trust-constr',  # closest to MATLAB 'interior-point'
                    options={'disp': False}  # suppress output
                )

                x_opt = result.x  # optimized variables
                fval = result.fun  # objective function value at optimum

                if abs(fval - best_fval) / best_fval < 0.01:
                    lowest_count = lowest_count + 1

                if fval < best_fval:
                    if np.abs(fval - best_fval) / best_fval > 0.01:
                        lowest_count = 0
                    best_fval = fval
                    best_x_opt = x_opt

                if lowest_count > 4:
                    break

        else:
            result = minimize(
                fun=objective,  # your scalar-valued objective
                x0=x0,  # initial guess
                method='trust-constr',  # 'trust-constr' closest to MATLAB 'interior-point'
                options={'disp': False}  # suppress output
            )

            best_x_opt = result.x  # optimized variables
            best_fval = result.fun  # objective function value at optimum

        params = np.tanh(best_x_opt)

        AR_poly = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[:AR_order])])
        MA_poly = np.hstack([1, -arfima_utils.pacf_to_coeffs(params[AR_order:])])

        return AR_poly, MA_poly, best_x_opt, best_fval




