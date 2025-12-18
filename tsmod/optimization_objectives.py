from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
# from scipy.stats import multivariate_t
# from scipy.special import gammaln
from numba import njit, prange

# ------------------------------
# Numba-compiled functions
# ------------------------------


@njit(parallel=True)
def _gaussian_nll(e, cov, prec, use_cov):
    T = e.shape[0]
    nll = 0.0

    if use_cov:
        if cov.ndim == 2:   # stationary cov
            prec = np.linalg.inv(cov)
            log_sigma = np.log(cov)
            for t in prange(T):
                nll += log_sigma + e[t] @ prec @ e[t]
        else:  # full covariance
            for t in prange(T):
                L = np.linalg.cholesky(cov[t])
                y = np.linalg.solve(L, e[t])
                quad = y @ y
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                nll += logdet + quad
    else:
        if prec.ndim == 2:  # stationary prec
            log_prec = np.log(prec)
            for t in prange(T):
                nll += -log_prec + e[t] @ prec @ e[t]
        else:  # TV precision
            for t in prange(T):
                sign, logdet_prec = np.linalg.slogdet(prec[t])
                nll += -logdet_prec + e[t] @ prec[t] @ e[t]

    return 0.5 * nll + 0.5 * T * np.log(2*np.pi)


@njit(parallel=True)
def energy_score_njit(e, cov, prec, n_samples, seed, use_cov):
    T, d = e.shape
    total_es = 0.0
    rng = np.random.RandomState(seed)

    for t in prange(T):
        et = e[t]

        # Get predictive covariance
        if use_cov:
            Sigma = cov[t] if cov.ndim == 3 else cov
        else:
            P = prec[t] if prec.ndim == 3 else prec
            # Invert precision using Cholesky for stability
            L = np.linalg.cholesky(P)
            # Covariance = inv(P) = inv(L.T) @ inv(L)
            Linv = np.linalg.inv(L)
            Sigma = Linv @ Linv.T

        # Cholesky of covariance for sampling
        Lcov = np.linalg.cholesky(Sigma)

        # Draw n_samples from multivariate normal
        samples = np.empty((n_samples, d))
        for i in range(n_samples):
            z = rng.randn(d)
            samples[i, :] = Lcov @ z + et  # shifted by observation

        # Term1: mean norm to observation
        term1 = 0.0
        for i in range(n_samples):
            norm = 0.0
            for j in range(d):
                norm += samples[i, j] ** 2
            term1 += np.sqrt(norm)
        term1 /= n_samples

        # Term2: 1/2 E||X - X'|| over independent pairs
        term2 = 0.0
        for i in range(n_samples):
            idx1 = rng.randint(0, n_samples)
            idx2 = rng.randint(0, n_samples)
            diff_norm = 0.0
            for j in range(d):
                diff = samples[idx1, j] - samples[idx2, j]
                diff_norm += diff ** 2
            term2 += 0.5 * np.sqrt(diff_norm)
        term2 /= n_samples

        total_es += term1 - term2

    return total_es

# ------------------------------
# Class wrappers
# ------------------------------

class OptimizationObjective(ABC):
    requires_cov: bool = False   # Default: no covariance needed
    prefers: str | None = None   # Default: no preference

    """
    Base class for time-series optimization objectives.

    Subclasses must implement __call__, returning a scalar loss.
    """

    @abstractmethod
    def __call__(
        self,
        e: np.ndarray,            # prediction errors: shape (T, dim)
        cov: Optional[np.ndarray] = None,  # shape (T, dim, dim) or (1, dim, dim)
        prec: Optional[np.ndarray] = None
    ) -> float:
        pass

    def __repr__(self):
        return self.__class__.__name__


class GaussianNLL(OptimizationObjective):
    requires_cov = True   # Default: no covariance needed
    prefers = "prec"

    def __call__(self,
                 e: np.ndarray,
                 cov = None,
                 prec = None) -> float:

        if prec is not None:
            return _gaussian_nll(e, np.empty((0, 0)), prec, False)
        elif cov is not None:
            return _gaussian_nll(e, cov, np.empty((0, 0)), True)
        else:
            raise ValueError("Provide either cov or prec")

        return _gaussian_nll(e, cov, prec)


class Huber(OptimizationObjective):
    requires_cov = False   # Default: no covariance needed
    prefers = None

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
        """
        Standard Huber loss applied element-wise.
        Ignores cov/prec (not probabilistic).
        """
        abs_e = np.abs(e)
        quad = 0.5 * (abs_e <= self.delta) * (e**2)
        linear = self.delta * (abs_e > self.delta) * (abs_e - 0.5*self.delta)
        return float(np.sum(quad + linear))

    def __repr__(self):
        return f"Huber(delta={self.delta})"


class MAE(OptimizationObjective):
    requires_cov = False   # Default: no covariance needed
    prefers = None

    def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
        return float(np.sum(np.abs(e)))


# class StudentTNLL(OptimizationObjective):
#     requires_cov = True   # Default: no covariance needed
#     prefers = "prec"
#
#     def __init__(self, df: float):
#         self.df = df
#
#     def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
#         """
#         Scalar student-t for simplicity.
#         Assumes variance (scale) is in cov or precision in prec.
#         """
#
#         if (cov is None) == (prec is None):
#             raise ValueError("Provide exactly one of cov or prec.")
#
#         T = e.shape[0]
#         df = self.df
#         nll = 0.0
#
#         # Cov version
#         if cov is not None:
#             if cov.ndim != 1:
#                 raise ValueError("StudentT currently supports scalar cov only.")
#             for t in range(T):
#                 sigma = cov[t]
#                 nll += (
#                     0.5*np.log(df*np.pi*sigma)
#                     + (df+1)/2 * np.log(1 + (e[t]**2)/(df*sigma))
#                 )
#
#         # Precision version
#         else:
#             if prec.ndim != 1:
#                 raise ValueError("StudentT currently supports scalar prec only.")
#             for t in range(T):
#                 tau = prec[t]
#                 sigma = 1 / tau
#                 nll += (
#                     0.5*np.log(df*np.pi*sigma)
#                     + (df+1)/2 * np.log(1 + (e[t]**2)/(df*sigma))
#                 )
#
#         return float(nll)
#
#     def __repr__(self):
#         return f"StudentTNLL(df={self.df})"
#
#
# class MultivariateStudentTNLL(OptimizationObjective):
#     requires_cov = True   # Default: no covariance needed
#     prefers = "prec"
#
#     def __init__(self, df: float):
#         self.df = df  # degrees of freedom
#
#     def __call__(self, e: np.ndarray, cov: Optional[np.ndarray] = None, prec: Optional[np.ndarray] = None) -> float:
#         """
#         Multivariate Student-t negative log-likelihood.
#
#         e: (T, d) residuals
#         cov: (T, d, d) or (d, d)
#         prec: (T, d, d) or (d, d)
#         """
#         if (cov is None) == (prec is None):
#             raise ValueError("Provide exactly one of cov or prec.")
#
#         T, d = e.shape
#         df = self.df
#         nll = 0.0
#
#         # Loop over time steps
#         for t in range(T):
#             et = e[t]
#
#             # Covariance version
#             if cov is not None:
#                 Sigma = cov[t] if cov.ndim == 3 else cov
#                 sign, logdet = np.linalg.slogdet(Sigma)
#                 if sign <= 0:
#                     raise ValueError("Covariance must be positive definite")
#                 quad = et.T @ np.linalg.solve(Sigma, et)  # (x-mu)^T Σ^{-1} (x-mu)
#
#             # Precision version
#             else:
#                 Prec = prec[t] if prec.ndim == 3 else prec
#                 sign, logdet_prec = np.linalg.slogdet(Prec)
#                 if sign <= 0:
#                     raise ValueError("Precision must be positive definite")
#                 logdet = -logdet_prec  # log|Σ| = - log|Prec|
#                 quad = et.T @ Prec @ et  # no inversion needed
#
#             # Student-t NLL
#             nll += (
#                 gammaln((df + d)/2) - gammaln(df/2)
#                 + 0.5*logdet
#                 + (d/2)*np.log(df * np.pi)
#                 + ((df + d)/2) * np.log(1 + quad/df)
#             )
#
#         return float(nll)


class LogCosh(OptimizationObjective):
    requires_cov = False
    prefers = None

    def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
        """
        Robust log-cosh loss.
        Independent of cov/prec.
        """
        # return float(np.abs(e) + np.log1p(np.exp(-2 * np.abs(e))) - np.log(2))
        # abs_e = np.abs(e)
        # return np.sum(abs_e + np.log1p(np.exp(-2 * abs_e)) - np.log(2))

        return np.sum(np.log(np.cosh(e)))



class MSE(OptimizationObjective):
    requires_cov = False
    prefers = None

    def __init__(self, warmup: int = 0):
        """
        warmup: number of initial steps to ignore (as in ARMA CSS).
        """
        self.warmup = warmup

    def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
        """
        CSS = sum of squared errors after warmup period.
        """
        if self.warmup >= len(e):
            raise ValueError("warmup is larger than sequence length")

        e2 = e[self.warmup:] ** 2
        return float(np.sum(e2, axis=None))

    def __repr__(self):
        return f"CSS(warmup={self.warmup})"


class CauchySchwarzScore(OptimizationObjective):
    requires_cov = False
    prefers = None

    def __call__(self, e: np.ndarray, cov=None, prec=None) -> float:
        """
        Robust Cauchy–Schwarz Score (CSS/CSD-based).
        Defined as:

            L = -log ( (sum exp(-e^2/2))^2 / (sum exp(-e^2)) )

        Works for both scalar and vector errors (sums over all dims).
        """
        e_flat = e.reshape(-1)  # unify scalar/vector errors

        a = np.sum(np.exp(-0.5 * e_flat**2))
        b = np.sum(np.exp(-e_flat**2))

        # avoid division underflow
        return float(-np.log((a * a) / (b + 1e-12) + 1e-12))

    def __repr__(self):
        return "CauchySchwarzScore()"


class EnergyScore(OptimizationObjective):
    requires_cov = True
    prefers = "cov"

    def __init__(self, n_samples: int = 1000, seed: Optional[int] = None):
        self.n_samples = n_samples
        self.seed = seed if seed is not None else np.random.randint(1_000_000)

    def __call__(self, e: np.ndarray, cov: Optional[np.ndarray] = None, prec: Optional[np.ndarray] = None) -> float:

        if cov is not None:
            return float(energy_score_njit(e, cov, np.empty((0, 0)), self.n_samples, self.seed, True))
        elif prec is not None:
            return float(energy_score_njit(e, np.empty((0, 0)), prec, self.n_samples, self.seed, False))
        else:
            raise ValueError("Provide either cov or prec")


# ------------------------------
# String to Class Mapping
# ------------------------------

optimization_objective_class_mapping = {
    'huber': Huber,
    'mae': MAE,
    'logcosh': LogCosh,
    'mse': MSE,
    'energy_score': EnergyScore,
}
