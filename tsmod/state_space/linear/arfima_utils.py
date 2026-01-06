import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize, minimize_scalar
from scipy.fft import fft, ifft, next_fast_len
from scipy.linalg import block_diag
from scipy.special import gammaln

from typing import Tuple, Optional


def pacf_to_coeffs(pi):
    """
    Barndorff-Nielsen, O., & Schou, G. (1973).
    On the parametrization of autoregressive models by partial autocorrelations.

    see also statsmodels.tsa.statespace.tools.constrain_stationary_univariate
             statsmodels.tsa.statespace.tools.unconstrain_stationary_univariate

    Args:
        pi: partial autocorrelations

    Returns:
        coefficient of lag polynomial

    """
    n = pi.shape[0]
    y = np.zeros((n, n))
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] - pi[k] * y[k - 1, k - i - 1]
        y[k, k] = - pi[k]
    return -y[n - 1, :]


def coeffs_to_pacf(coeffs):
    """
            Barndorff-Nielsen, O., & Schou, G. (1973).
    On the parametrization of autoregressive models by partial autocorrelations.

    Args:
        coeffs: coefficients of lag polynomial

    Returns:
        partial autocorrelations

    """

    n = coeffs.shape[0]
    y = np.zeros((n, n))
    y[n - 1:] = -coeffs
    for k in range(n - 1, 0, -1):
        for i in range(k):
            y[k - 1, i] = (y[k, i] - y[k, k] * y[k, k - i - 1]) / (1 - y[k, k] ** 2)
    r = - y.diagonal()
    return r


def coeffs_to_transformed_pacfs(coeffs):
    pacf = coeffs_to_pacf(coeffs)
    # return pacf / ((1 - pacf ** 2)**0.5)
    return np.atanh(pacf)


def transformed_pacfs_to_coeffs(t_pacf):
    # pacf = t_pacf / (1 + t_pacf ** 2)**0.5
    pacf = np.tanh(t_pacf)
    return pacf_to_coeffs(pacf)


def representation_hamilton(phis, k, thetas):
    """
    let f_t represent the arima process

    the statespace representation is

    f_t = M x_t
    x_t = F x_{t-1} + R e_t, e_t ~ N(0,I)

    This function supports orders of integration k > 0.
    The representation is done in the spirit of Hamilton's representation, where the MA coefficients are in
        the matrix M and the state process x_t is ARI.

    No reference for this extension.

    Args:
        phis: AR coefficients, AR polynomial is 1 - phi[0] L - phi[1] L ^2 ...
        k: order of integration
        thetas: MA coefficients, MA polynomial is 1 + theta[0] L + theta[1] L ^2 ...

    Returns:
        M: selection or transformation matrix which transforms the state into the arima process
        F: transition matrix for the state
        R: innovation selection matrix

    """

    p = len(phis)
    q = len(thetas)
    dim = np.max([p + k, q + 1])

    F = np.zeros((dim, dim))

    if k < dim:
        F[k, k:k + p] = phis
        if k + 1 < dim:
            F[k + 1:dim, k:dim - 1] = np.eye(dim - 1 - k)

    if k > 0:
        for idx in range(k):
            F[idx, idx:k] = 1
            F[idx, k:k + p] = phis

    R = np.zeros((dim, 1))
    R[0:k + 1, 0] = 1

    M_to_factors = np.zeros((1, dim))
    M_to_factors[:, 0] = 1
    for i in range(len(thetas)):
        M_to_factors[:, i + 1] = thetas[i]

    if k > 0:
        M_all = np.eye(dim)
        for d in range(1, k + 1):
            shortened_dim = dim - d + 1
            M = -1 * np.ones((shortened_dim, shortened_dim), dtype=int)
            M[:, 0] = 1
            M[np.triu_indices(shortened_dim, k=1)] = 0
            if d > 1:
                M_top = np.eye(dim - shortened_dim)
                M = block_diag(M_top, M)

            M_all = M_all @ M

        M_to_factors = M_to_factors @ M_all

    return M_to_factors, F, R


def representation_ihamilton(phis, k, thetas):
    M, F, R = representation_hamilton(phis, 0, thetas)
    return integrate_representation(k, M, F, R)


def representation_harvey(phis: np.ndarray, k, thetas: np.ndarray):
    """
    let f_t represent the arima process

    the statespace representation is

    f_t = M x_t
    x_t = F x_{t-1} + R e_t, e_t ~ N(0,I)

    Args:
        phis: AR coefficients, AR polynomial is 1 - phi[0] L - phi[1] L ^2 ...
        k: order of integration
        thetas: MA coefficients, MA polynomial is 1 + theta[0] L + theta[1] L ^2 ...

    Returns:
        M: selection or transformation matrix which transforms the state into the arima process
        F: transition matrix for the state
        R: innovation selection matrix

    """

    m = max(len(phis), len(thetas) + 1)

    R = np.pad(thetas, (1, max(0, m - len(thetas) - 1)), mode='constant', constant_values=0).reshape((-1, 1))
    R[0] = 1

    F = np.zeros((m, m))
    F[:-1, 1:] = np.eye(m - 1)
    F[0:len(phis), 0] = phis

    M = np.zeros((1, m))
    M[0, 0] = 1

    if k > 0:
        return integrate_representation(k, M, F, R)

    return M, F, R


def integrate_representation(k: int, M: np.ndarray, F: np.ndarray, R: np.ndarray):
    """
    given the state

    g_t = M x_t
    x_{t} = F x_{t-1} + R e_t, e_t ~ N(0,I)

    the representation for f_t is returned. f_t is given by:

    Delta^k f_t = g_t


    Args:
        k: order of integration
        M: original selection or transformation matrix
        F: original transition matrix
        R: original innovation selection matrix

    Returns:
        M: new selection or transformation matrix, for process f_t = Delta^-k g_t
        F: new transition matrix
        R: new innovation selection matrix
    """

    if M.shape[0] > 1:
        raise NotImplementedError

    if not isinstance(k, int):
        raise ValueError("k must be an integer")

    if k < 0:
        raise ValueError("k must be a non-negative integer")

    if k == 0:
        return M, F, R

    og_dim = F.shape[0]
    new_dim = og_dim + k

    F_new = np.zeros((new_dim, new_dim))
    F_new[-og_dim:, -og_dim:] = F
    MF = M @ F
    for i in range(k):
        F_new[i, i:k] = 1
        F_new[i, k:] = MF

    MR = M @ R

    R_new = np.zeros((new_dim, R.shape[1]))
    R_new[-og_dim:, :] = R
    R_new[:-og_dim,:] = np.vstack([MR] * k)

    M_new = np.zeros((1, new_dim))
    M_new[0,0] = 1

    return M_new, F_new, R_new


def inverse_polynomial(coeffs, N):
    # `coeffs` is the AR polynomial: a[0] + a[1] z^{-1} + ...
    impulse = np.zeros(N)
    impulse[0] = 1.0

    # To invert a polynomial A(z), we want 1 / A(z)
    # Filter the impulse with numerator=1, denominator=coeffs
    return lfilter([1.0], coeffs, impulse)


def fractional_integrated_errors_to_ar_poly(d, N):
    k = np.arange(1, N)
    b = np.concatenate(([1], np.cumprod((k - d - 1) / k)))
    return b


def fractional_integrated_errors_to_ma_poly(d, N):
    k = np.arange(1, N)
    b = np.concatenate(([1], np.cumprod((k + d - 1) / k)))
    return b


def tempered_fractional_integrated_errors_to_ma_poly(d, lam, N):
    """
    MA coefficients for (1 - exp(-lam) L)^(-d)

    Parameters
    ----------
    d : float
        fractional integration parameter (d > 0)
    lam : float
        tempering parameter (lam >= 0)
    N : int
        number of MA coefficients (including lag 0)

    Returns
    -------
    b : ndarray, shape (N,)
        truncated MA coefficients
    """
    k = np.arange(1, N)
    frac = np.cumprod((k + d - 1) / k)
    b = np.empty(N)
    b[0] = 1.0
    b[1:] = frac * np.exp(-lam * k)
    return b


def tempered_fractional_integrated_errors_to_ma_poly_safe(d, lam, N):
    k = np.arange(N)
    log_b = (
        gammaln(k + d)
        - gammaln(d)
        - gammaln(k + 1)
        - lam * k
    )
    return np.exp(log_b)


def tempered_fractional_integrated_errors_to_ar_poly(d, lam, N):
    """
    AR coefficients for (1 - exp(-lam) L)^d

    Parameters
    ----------
    d : float
        fractional differencing parameter
    lam : float
        tempering parameter (lam >= 0)
    N : int
        number of AR coefficients (including lag 0)

    Returns
    -------
    b : ndarray, shape (N,)
        truncated AR coefficients
    """
    k = np.arange(1, N)
    frac = np.cumprod((k - d - 1) / k)
    b = np.empty(N)
    b[0] = 1.0
    b[1:] = frac * np.exp(-lam * k)
    return b


def tempered_fractional_integrated_errors_to_ar_poly_safe(d, lam, N):
    k = np.arange(N)
    log_b = (
        gammaln(k - d)
        - gammaln(-d)
        - gammaln(k + 1)
        - lam * k
    )
    return np.exp(log_b)


def arma_poly_to_ma_representation(ar_poly, ma_poly, n):
    impulse = np.zeros(n)
    impulse[0] = 1.0

    # MA(∞) coefficients = response of filter MA_poly / AR_poly to an impulse
    psi = lfilter(ma_poly, ar_poly, impulse)
    return psi


def arma_coeffs_to_ma_representation(phis, thetas, N):
    """
    Compute first N coefficients of the MA(∞) representation of an ARMA(p, q)
    given parameter vectors phis and thetas.

    AR polynomial = 1 - φ₁ L - φ₂ L² - ...
    MA polynomial = 1 + θ₁ L + θ₂ L² + ...

    Args:
        phis: array-like, AR coefficients φ
        thetas: array-like, MA coefficients θ
        N: number of MA(∞) terms to compute

    Returns:
        psi: length-N array of MA(∞) coefficients
    """
    # Build polynomials
    if len(phis):
        ar_poly = np.concatenate(([1.0], -np.asarray(phis)))
    else:
        ar_poly = np.array([1.])
    if len(thetas):
        ma_poly = np.concatenate(([1.0],  np.asarray(thetas)))
    else:
        ma_poly = np.array([1.])

    impulse = np.zeros(N)
    impulse[0] = 1.0

    # Filter impulse through MA/AR system to get MA(∞) coeffs
    psi = lfilter(ma_poly, ar_poly, impulse)
    return psi


def frac_diff(x, d):
    if x is None:
        return None

    x = np.asarray(x)

    reshape_to_1D = False
    if x.ndim == 1:
        reshape_to_1D = True
        x = x.reshape(-1, 1)

    cap_T, p = x.shape

    np2 = next_fast_len(2 * cap_T - 1)

    k = np.arange(1, cap_T)
    b = np.concatenate(([1], np.cumprod((k - d - 1) / k)))

    dx = np.zeros((cap_T, p))

    for i in range(p):
        b_padded = np.concatenate([b, np.zeros(np2 - len(b))])
        x_padded = np.concatenate([x[:, i], np.zeros(np2 - cap_T)])

        dxi = ifft(fft(b_padded) * fft(x_padded)).real
        dx[:, i] = dxi[:cap_T]

    if reshape_to_1D:
        dx = dx.reshape((-1,))

    return dx


def ewhittle(d, x, m, param=1):
    """
    Extended Whittle likelihood function for estimating fractional differencing.
    Shimotsu & Phillips (2004).
    """
    x = np.asarray(x)
    n = len(x)

    # Mean adjustment as in Shimotsu (2004)
    if param == 1:
        weight = (d <= 0.5) + 0.5 * (1 + np.cos(-2 * np.pi + 4 * np.pi * d)) * (0.5 < d < 0.75)
        myu = weight * np.mean(x) + (1 - weight) * x[0]
        x = x - myu

    # Fractionally difference the series
    dx = frac_diff(x, d)

    # Frequency components
    t = np.arange(n)
    lam = 2 * np.pi * t / n

    # Compute w(dx) via FFT
    wdx = (1 / np.sqrt(2 * np.pi * n)) * np.conj(np.fft.fft(np.conj(dx))) * np.exp(1j * lam)

    # Compute periodogram and the objective
    lam = lam[1:m + 1]
    vx = wdx[1:m + 1]
    Iv = np.abs(vx) ** 2

    g = np.mean(Iv)
    r = np.log(g) - 2 * d * np.mean(np.log(lam))
    return r


def veltaper(d, x, m, p):
    """
    Velasco's tapered local Whittle likelihood estimator.

    Parameters:
        d : float
            Fractional differencing parameter.
        x : array_like
            Time series data.
        m : int
            Truncation bandwidth.
        p : int
            Taper order: 2 = Bartlett, 3 = Kolmogorov.

    Returns:
        r : float
            Value of the tapered local Whittle likelihood function at d.
    """
    x = np.asarray(x)
    n = len(x)
    t = np.arange(n)
    lam = 2 * np.pi * t / n

    # ----- Apply taper -----
    if p == 2:  # Bartlett
        mm = int(np.ceil(n / 2))
        h = 1 - np.abs(t + 1 - mm) / mm
    elif p == 3:  # Kolmogorov (triangular taper)
        pp = (n + 2) // 3
        h = np.ones(pp)
        h2 = np.concatenate((np.arange(1, pp + 1), np.arange(pp - 1, 0, -1)))
        h3 = np.convolve(h, h2)
        h = np.concatenate([h3, np.zeros(n - len(h3))])
    else:
        raise ValueError("Taper order p must be 2 (Bartlett) or 3 (Kolmogorov).")

    x_tapered = x * h

    # ----- Compute FFT and periodogram -----
    wx = (1 / np.sqrt(2 * np.pi * n)) * np.conj(np.fft.fft(np.conj(x_tapered))) * np.exp(1j * lam)

    # ----- Subsample frequencies -----
    ind = np.arange(p, m + 1, p)
    lam_sub = lam[1 + ind]  # match MATLAB 1-indexing offset
    wx_sub = wx[1 + ind]
    Ix = np.abs(wx_sub) ** 2

    # ----- Compute likelihood -----
    g = np.sum((lam_sub ** (2 * d)) * Ix) * p / m
    r = np.log(g) - 2 * d * np.sum(np.log(lam_sub)) * p / m
    return r


def estimate_fractional_d_ewl(x):
    """
    Estimate the fractional differencing parameter d using a two-step procedure:
    1. Preliminary estimate via Velasco's tapered local Whittle estimator
    2. Refined estimate via Extended Whittle likelihood (ELW)

    Returns:
        d_hat : float
            Estimated value of d
    """
    x = np.asarray(x)
    n = len(x)
    m = int(np.floor(n ** 0.6))  # standard bandwidth choice
    p = 3  # Kolmogorov taper

    # Step 1: preliminary estimate using Velasco's tapered local Whittle
    # plt.plot(x)
    # plt.show()
    result1 = minimize_scalar(lambda d: veltaper(d, x, m, p), bounds=(0, 2), method='bounded')
    d0 = result1.x

    # print(f' veltaper = {d0}')

    # Step 2: refine using Extended Whittle likelihood
    result2 = minimize(lambda d: ewhittle(d, x, m), [d0], method='BFGS')

    d_hat = result2.x[0]  # result.x is an array

    return d_hat


def generate_arfima_from_polys(ar_poly, d, ma_poly, T):
    burn = 1000
    N = T + burn
    eta = np.random.randn(N)
    fIE = frac_diff(eta, -d)
    y_full = lfilter(ma_poly, ar_poly, fIE)
    return y_full[-T:]


def generate_arfima_from_coeffs(ar_coeffs, d, ma_coeffs, T):
    ar_poly = np.concatenate(([1.0], -np.asarray(ar_coeffs)))
    ma_poly = np.concatenate(([1.0],  np.asarray(ma_coeffs)))
    return generate_arfima_from_polys(ar_poly, d, ma_poly, T)


class ARIMAFitFromMA:

    def __init__(self,
                 order: Tuple[int, int, int],
                 enforce_stability: bool = True,
                 enforce_invertibility: bool = True,
                 time_weighting: bool = True,):
        self.order = order

        self._enforce_stability = enforce_stability
        self._enforce_invertibility = enforce_invertibility
        self._time_weighting = time_weighting

    @property
    def order(self) -> Tuple[int, int, int]:
        return self._order

    @order.setter
    def order(self, order: Tuple[int, int, int]):
        if not len(order) == 3:
            raise ValueError("Order must have length 3.")
        if any(not isinstance(i, int) for i in order):
            raise TypeError("Order must be an integer.")
        if any(i < 0 for i in order):
            raise ValueError("Order must be non-negative.")

        self._order = order

    def _params_to_coeffs(self, params):
        p, k, q = self.order
        if self._enforce_stability:
            phis = transformed_pacfs_to_coeffs(params[:p]) if p > 0 else np.array([])
        else:
            phis = params[:p] if p > 0 else np.array([])
        if self._enforce_invertibility:
            thetas = - transformed_pacfs_to_coeffs(params[p:]) if q > 0 else np.array([])
        else:
            thetas = params[p:] if q > 0 else np.array([])
        return phis, thetas

    def _random_params(self):

        def random_transformed_pacfs(n):
            pacfs = np.random.uniform(-0.95, 0.95, n)
            return np.arctanh(pacfs)  # convert to unconstrained

        p, k, q = self.order
        if self._enforce_stability:
            phis_params = random_transformed_pacfs(p) if p > 0 else np.array([])
        else:
            phis_params = 5 * np.random.rand(p) - 2.5 if p > 0 else np.array([])
        if self._enforce_invertibility:
            theta_params = random_transformed_pacfs(q) if q > 0 else np.array([])
        else:
            theta_params = 5 * np.random.rand(q) - 2.5 if q > 0 else np.array([])

        return phis_params, theta_params

    def calc_coeffs(self,
                    ma_representation: np.ndarray,
                    initial_phis: Optional[np.ndarray] = None,
                    initial_thetas: Optional[np.ndarray] = None,):

        p, d, q = self.order
        T = len(ma_representation)

        # --- Safety checks -------------------------------------------------------
        for name, arr in (("initial_phis", initial_phis),
                          ("initial_thetas", initial_thetas)):
            if arr is not None:
                if arr.ndim != 1:
                    raise ValueError(f"{name} must be 1D.")
        if initial_phis is not None and len(initial_phis) != p:
            raise ValueError("initial_phis must have length p.")
        if initial_thetas is not None and len(initial_thetas) != q:
            raise ValueError("initial_thetas must have length q.")


        # --- Build initial guess -------------------------------------------------
        params0 = np.zeros(p + q)

        random_phi_params, random_theta_params = self._random_params()
        if initial_phis is not None:
            if self._enforce_stability:
                params0[:p] = coeffs_to_transformed_pacfs(initial_phis)
            else:
                params0[:p] = initial_phis
        else:
            params0[:p] = random_phi_params

        if initial_thetas is not None:
            if self._enforce_invertibility:
                params0[p:] = coeffs_to_transformed_pacfs(- initial_thetas)
            else:
                params0[p:] = initial_thetas
        else:
            params0[p:] = random_theta_params

        # --- Objective function --------------------------------------------------
        def objective(x):
            phis, thetas = self._params_to_coeffs(x)
            if np.any(np.isnan(phis)) or np.any(np.isnan(thetas)):
                return np.inf

            ma_rep = arma_coeffs_to_ma_representation(phis, thetas, T)

            for _ in range(d):
                ma_rep = arma_poly_to_ma_representation([1, -1], ma_rep, T)

            diff = ma_rep - ma_representation

            if self._time_weighting:
                weights = np.arange(T, 0, -1)
                return np.sum(weights * diff ** 2)
            return np.sum(diff ** 2)

        # --- Main optimization ---------------------------------------------------
        run_multi_start = True
        if (initial_phis is not None) and (initial_thetas is not None):
            # One run only
            result = minimize(objective, params0, method="L-BFGS-B")
            if np.any(np.isnan(result.x)) or np.isnan(result.fun):
                initial_phis = None
                initial_thetas = None
            else:
                best_x = result.x
                run_multi_start = False

        if run_multi_start:
            # Multi-start optimization
            best_f = np.inf
            best_x = None
            stagnation = 0

            for _ in range(100):
                # randomize missing parts only
                trial = params0.copy()
                random_phi_params, random_theta_params = self._random_params()
                if initial_phis is None:
                    trial[:p] = random_phi_params
                if initial_thetas is None:
                    trial[p:] = random_theta_params

                result = minimize(objective, trial, method="L-BFGS-B")
                if np.any(np.isnan(result.x)) or np.isnan(result.fun):
                    continue

                fval = result.fun

                if abs(fval - best_f) / (best_f + 1e-8) < 0.01:
                    stagnation += 1
                else:
                    stagnation = 0

                if fval < best_f:
                    best_f = fval
                    best_x = result.x

                if stagnation > 4:
                    break

        # --- Final parameters -----------------------------------------------------
        phis, thetas = self._params_to_coeffs(best_x)
        return phis, thetas



if __name__ == '__main__':

    import time

    start = time.time()
    ma_rep = tempered_fractional_integrated_errors_to_ma_poly_safe(1.4, 0.2, 10000)
    end_time = time.time()
    print(end_time - start)

    start = time.time()
    ma_rep2 = tempered_fractional_integrated_errors_to_ma_poly(1.4, 0.2, 10000)
    end_time = time.time()
    print(end_time - start)

    print(ma_rep)
    print(ma_rep2)

    ma_rep = tempered_fractional_integrated_errors_to_ma_poly_safe(0.4, 0, 10)
    ma_rep2 = fractional_integrated_errors_to_ma_poly(0.4, 10)

    print(ma_rep)
    print(ma_rep2)

    # p = 4
    # q = 3
    # k = 0
    #
    # N = 1000
    #
    # np.random.seed(0)
    # pacfs_AR = np.random.rand(p) * 1.9 - 1
    # pacfs_MA = np.random.rand(q) * 1.9 - 1
    #
    # if p > 0:
    #     phis = pacf_to_coeffs(pacfs_AR)
    # else:
    #     phis = np.array([])
    #
    # if q > 0:
    #     thetas = - pacf_to_coeffs(pacfs_MA)
    # else:
    #     thetas = np.array([])
    #
    # approx_calc = ARIMAFitFromMA(order=(p, k, q), time_weighting=False)
    # ma_representation_true = arma_coeffs_to_ma_representation(phis, thetas, N)
    # # ma_representation_true = fractional_integrated_errors_MAcoefficients(0.7, N)
    #
    # aprox_phi, aprox_thetas = approx_calc.calc_coeffs(ma_representation_true)
    # print(phis)
    # print(aprox_phi)
    # print(thetas)
    # print(aprox_thetas)

