from typing import NamedTuple, Callable
from dataclasses import dataclass

import numpy as np


class UKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

    Default values taken from https://github.com/sbitzer/UKF-exposed

    see dynamax.nonlinear_gaussian_ssm.inference_ukf
    """
    alpha: float = np.sqrt(3)
    beta: int = 2
    kappa: int = 1


@dataclass
class UnscentedKalmanFilterInitialization:
    x0: np.ndarray
    P0: np.ndarray

    def __post_init__(self):
        self._check_initialize()

    def _check_initialize(self):
        x0, P0 = self.x0, self.P0

        if not isinstance(x0, np.ndarray):
            raise ValueError("x0 must be a numpy array")
        if x0.ndim != 1:
            raise ValueError("x0 must be 1D")

        if not isinstance(P0, np.ndarray):
            raise ValueError("P0 must be a numpy array")
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError("P0 must be square 2D numpy array")
        if x0.shape[0] != P0.shape[0]:
            raise ValueError("dimension mismatch between x0 and P0")


class NumpyUKFModelGenerator:
    """
    Generates transition, measurement, and covariance functions for a UKF.

    Supports:
    - njited f(x, u, theta)
    - generator f(theta)(x, u) (NumPy only if theta changes)
    """

    def __init__(self,
                 f_transition,
                 f_measurement,
                 Q_generator,
                 R_generator,
                 njit_compatible: bool = True):
        """
        njit_compatible: True if the functions are njited and can be used in Numba mode
        """
        self.f_transition = f_transition
        self.f_measurement = f_measurement
        self.Q_generator = Q_generator
        self.R_generator = R_generator
        self.njit_compatible = njit_compatible

    def get_transition(self, theta=None):
        if callable(self.f_transition):
            if theta is not None:
                # generator style
                f = self.f_transition(theta)
                return f  # NumPy only if theta changes
            else:
                # direct f(x,u,theta)
                return self.f_transition
        return None


class NumpyUKFInferenceEngine:

    def __init__(self, ):
        self._initialization = None
        self._

