from typing import Optional
from functools import cached_property  # wraps
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from numba import njit

from tsmod.tools.utils import validate_covariance, validate_chol_factor


@njit
def steady_state_P_riccati(Z, H, F, Q, starter_P=None) -> np.ndarray:
    """
    Args:
        Z: design matrix (m * k)
        H: observation noise matrix (m * m)
        F: transition matrix (k * k)
        Q: state cov matrix (k * k)
        starter_P: Optional (k * k) warm start matrix

    Returns:
        P_SS: steadystate state covariance matrix

    """

    if starter_P is not None:
        P = starter_P
    else:
        P = Q.copy()

    for i in range(10000):
        # P_new = F @ P @ F.T + Q - F @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R) @ H @ P @ F.T
        B = Z @ P @ Z.T + H

        FP = F @ P
        FPZT = FP @ Z.T

        P_new = FP @ F.T + Q - FPZT @ np.linalg.solve(B, FPZT.T)

        if i % 10 == 0:
            P = (P + P.T) / 2
            P_new = (P_new + P_new.T) / 2
            if np.allclose(P_new, P, atol=1e-6):
                P = P_new
                break
        P = P_new

    return P


def steady_state_P_lyapunov(Z, H, F, Q, starter_P=None) -> np.ndarray:
    """

    Args:
        Z: design matrix (m * k)
        H: observation noise matrix (m * m)
        F: transition matrix (k * k)
        Q: state cov matrix (k * k)
        starter_P: Optional (k * k) warm start matrix

    Returns:
        P_SS: steadystate state covariance matrix

    """

    if starter_P is not None:
        P = starter_P
    else:
        P = Q.copy()

    I = np.eye(F.shape[0] ** 2)

    for i in range(1000):
        S = Z @ P @ Z.T + H
        pred_error_precision = np.linalg.inv(S)
        K = P @ Z.T @ pred_error_precision

        A_KC = F - K @ Z
        vec_rhs = (K @ H @ K.T + Q).reshape(-1, 1)

        # Solve linear system
        vec_P = np.linalg.solve(I - np.kron(A_KC, A_KC), vec_rhs)
        P_new = vec_P.reshape(F.shape)

        if i % 10 == 0:
            P = (P + P.T) / 2
            P_new = (P_new + P_new.T) / 2
            if np.allclose(P_new, P, atol=1e-6):
                P = P_new
                break
        P = P_new

    return P


def steady_state_P_dare(Z, H, F, Q) -> np.ndarray:

    P = solve_discrete_are(F.T, Z.T, Q, H)

    return P

# ---------
# StateSpace Representation Classes
# ---------

@dataclass(frozen=True)
class LinearStateProcessDynamics:
    M: np.ndarray
    F: np.ndarray
    R: np.ndarray


class LinearStateProcessRepresentation:

    @classmethod
    def from_dynamic_representation(cls,
                                    dynamic_representation: LinearStateProcessDynamics,
                                    LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                                    validate: bool = True) -> "LinearStateProcessRepresentation":
        return cls(M=dynamic_representation.M,
                   F=dynamic_representation.F,
                   R=dynamic_representation.R,
                   LQ=LQ, Q=Q,
                   validate=validate)


    def __init__(self, M, F, R,
                 LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                 validate: bool = True):

        self._M: Optional[np.ndarray]
        self._F: Optional[np.ndarray]
        self._R: Optional[np.ndarray]
        self._LQ: Optional[np.ndarray]
        self._Q: Optional[np.ndarray]

        self._initialize(M, F, R, LQ, Q, validate)

    def _initialize(self, M, F, R,
                    LQ, Q,
                    validate: bool = True):

        if LQ is None and Q is None:
            raise ValueError("At least one of LH or H must be provided")

        self._M = M
        self._F = F
        self._R = R

        if validate:
            for m in [M, F, R]:
                if not isinstance(m, np.ndarray):
                    raise ValueError("M, F, R need to be a numpy arrays")
            if (LQ is not None) and (Q is not None):
                if not np.allclose(LQ @ LQ.T, Q):
                    raise ValueError("LQ needs to be the Cholesky factor of Q")
            self._validate_dimensions(M, F, R, LQ, Q)
            if LQ is not None:
                validate_chol_factor(LQ)
            if Q is not None:
                validate_covariance(Q)

        self._Q = Q
        self._LQ = LQ
        if LQ is None and Q is None:
            self._LQ = np.eye(R.shape[1])
            self._Q = self._LQ


    def _validate_dimensions(self, M, F, R, LQ, Q):
        # TODO: check there is no dimension missmatch between M, F, R, LQ, Q; raise Error otherwise
        pass

    @property
    def M(self):
        return self._M

    @property
    def F(self):
        return self._F

    @property
    def R(self):
        return self._R

    @property
    def Q(self):
        if self._Q is None:
            self._Q = self._LQ @ self._LQ.T
        return self._Q

    @property
    def LQ(self):
        if self._LQ is None:
            self._LQ = np.linalg.cholesky(self._Q)
        return self._LQ

    @property
    def RQRT(self):
        return self.R @ self.Q @ self.R.T


class LinearStateSpaceModelRepresentation:
    # y_t = E M x_t + e_t, e_t ~ N(0, H) = LH N(0, I)
    # x_t = F x_t-1 + R u_t, u_t ~ N(0, Q) = LQ N(0, I)

    @classmethod
    def from_process_representation(cls,
                     process_representation: LinearStateProcessRepresentation,
                     E: np.ndarray,
                     const: np.ndarray,
                     LH: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None,
                     validate: bool = True):

        LQ = process_representation.LQ if process_representation._LQ is not None else None
        Q = process_representation.Q if process_representation._Q is not None else None
        return cls(E, process_representation.M, process_representation.F, process_representation.R, const, LH, H, LQ, Q, validate)

    def __init__(self,
                 E: np.ndarray,
                 M: np.ndarray, F: np.ndarray, R: np.ndarray,
                 const: np.ndarray,
                 LH: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None,
                 LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                 validate: bool = True):

        self._linear_state_process_representation: Optional[LinearStateProcessRepresentation]

        self._E: Optional[np.ndarray] = None
        self._const: Optional[np.ndarray] = None
        self._LH: Optional[np.ndarray] = None
        self._H: Optional[np.ndarray] = None

        self._initialize(E, M, F, R, const, LH, H, LQ, Q, validate)

    def _initialize(self, E, M, F, R, const, LH, H, LQ, Q, validate: bool = True):

        if LH is None and H is None:
            raise ValueError("At least one of LH or H must be provided")

        self._linear_state_process_representation = LinearStateProcessRepresentation(M, F, R, LQ, Q, validate)

        if validate:
            if (LH is not None) and (H is not None):
                if not np.allclose(LH @ LH.T, H):
                    raise ValueError("LH needs to be the Cholesky factor of H")
            for m in [const, E]:
                if not isinstance(m, np.ndarray):
                    raise ValueError("const, E need to be numpy arrays")
            self._validate_dimensions(M, const, E, LH, H)
            if LH is not None:
                validate_chol_factor(LH)
            if H is not None:
                validate_covariance(H)

        self._const = const
        self._E = E
        self._LH = LH
        self._H = H

    def _validate_dimensions(self, M, const, E, LH, H):
        # TODO: check there is no dimension missmatch between M, const, E, LH, H; raise Error otherwise
        pass

    @property
    def linear_state_process_representation(self):
        return self._linear_state_process_representation

    @property
    def const(self):
        return self._const

    @property
    def E(self):
        return self._E

    @property
    def H(self):
        if self._H is None:
            self._H = self._LH @ self._LH.T
        return self._H

    @property
    def LH(self):
        if self._LH is None:
            self._LH = np.linalg.cholesky(self._H)
        return self._LH

    @property
    def M(self):
        return self._linear_state_process_representation.M

    @property
    def F(self):
        return self._linear_state_process_representation.F

    @property
    def R(self):
        return self._linear_state_process_representation.R

    @property
    def Q(self):
        return self._linear_state_process_representation.Q

    @property
    def LQ(self):
        return self._linear_state_process_representation.LQ

    @property
    def RQRT(self):
        return self._linear_state_process_representation.RQRT

    @property
    def params(self):
        return self.const, self.E, self.H, self.M, self.F, self.R, self.Q

    @cached_property
    def steady_state_covariance(self) -> np.ndarray:
        Z = self.E @ self.M
        return solve_discrete_are(self.F.T, Z.T, self.RQRT, self.H)

    @property
    def steady_state_observation_covariance(self) -> np.ndarray:
        Z = self.E @ self.M
        return Z @ self.steady_state_covariance @ Z.T + self.H

    @cached_property
    def steady_state_observation_precision(self) -> np.ndarray:
        return np.linalg.inv(self.steady_state_observation_covariance)

    @property
    def steady_state_kalman_gain(self) -> np.ndarray:
        Z = self.E @ self.M
        return self.steady_state_covariance @ Z.T @ self.steady_state_observation_precision

    @property
    def posterior_steady_state_covariance(self) -> np.ndarray:
        I_d = np.eye(self.F.shape[0])
        Z = self.E @ self.M
        updated_P_SS = (I_d - self.steady_state_kalman_gain @ Z) @ self.steady_state_covariance
        return (updated_P_SS + updated_P_SS.T) / 2


class MutableLinearStateSpaceModelRepresentation(LinearStateSpaceModelRepresentation):

    _steady_state_calculation_opts = ("riccati", "lyapunov", "dare")

    @classmethod
    def from_frozen(cls,
                    frozen_ssm_representation: LinearStateSpaceModelRepresentation,
                    validate: bool = True,):

        LQ = getattr(frozen_ssm_representation.linear_state_process_representation, "_LQ", None)
        Q = getattr(frozen_ssm_representation.linear_state_process_representation, "Q", None)

        return cls(frozen_ssm_representation.E,
                   frozen_ssm_representation.M, frozen_ssm_representation.F, frozen_ssm_representation.R,
                   frozen_ssm_representation.const,
                   frozen_ssm_representation._LH, frozen_ssm_representation._H,
                   LQ, Q,
                   validate)

    def get_frozen(self):
        LQ = getattr(self.linear_state_process_representation, "_LQ", None)
        Q = getattr(self.linear_state_process_representation, "Q", None)
        return LinearStateSpaceModelRepresentation(self.E, self.M, self.F, self.R, self.const, self._LH, self._H,
                                                   LQ, Q, False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steady_state_covariance = None
        self._steady_state_observation_precision = None
        self._steady_state_needs_update = True
        self._steady_state_calculation_method = "dare"


    def _clear_cache(self):
        self._steady_state_covariance = None
        self._steady_state_observation_precision = None
        self._steady_state_needs_update = True

    def _mark_needs_update(self):
        self._steady_state_needs_update = True
        self._steady_state_observation_precision = None

    def update_representation(self,
                              representation: LinearStateSpaceModelRepresentation,
                              validate: bool = True):

        self._initialize(representation.E,
                         representation.M, representation.F, representation.R,
                         representation.const,
                         representation._LH, representation._H,
                         representation.linear_state_process_representation._LQ, representation.linear_state_process_representation._Q,
                         validate)

        self._clear_cache()


    def update_matrices(self,
                        E: np.ndarray,
                        M: np.ndarray, F: np.ndarray, R: np.ndarray,
                        const: np.ndarray,
                        LH: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None,
                        LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                        validate: bool = True):

        self._initialize(E,
                         M, F, R,
                         const,
                         LH, H,
                         LQ, Q,
                         validate)

        self._clear_cache()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(self, "_steady_state_covariance") and name in {"_E", "_F", "_R", "_LH", "_H", "_LQ", "_Q",
                                                                  "_const"}:
            self._clear_cache()

    # overwriting cached_properties
    @property
    def steady_state_covariance(self) -> np.ndarray:
        if self._steady_state_covariance is None or self._steady_state_needs_update:
            self._steady_state_covariance = self._calc_steady_state_covariance()
            self._steady_state_needs_update = False
        return self._steady_state_covariance

    def _calc_steady_state_covariance(self):

        Z = self.E @ self.M
        H = self.H
        F = self.F
        RQRT = self.RQRT

        method = self._steady_state_calculation_method
        starter_P = self._steady_state_covariance
        if method == 'riccati':
            return steady_state_P_riccati(Z, H, F, RQRT, starter_P)

        if method == 'lyapunov':
            return steady_state_P_lyapunov(Z, H, F, RQRT, starter_P)

        if method == "dare":
            return steady_state_P_dare(Z, H, F, RQRT)

        raise ValueError(f"Unknown method {method}")

    @property
    def steady_state_observation_precision(self) -> np.ndarray:
        if self._steady_state_observation_precision is None:
            self._steady_state_observation_precision =  np.linalg.inv(self.steady_state_observation_covariance)
        return self._steady_state_observation_precision

    @property
    def steady_state_calculation_method(self):
        return self._steady_state_calculation_method

    @steady_state_calculation_method.setter
    def steady_state_calculation_method(self, value):
        value = value.lower()
        if value not in self._steady_state_calculation_opts:
            raise ValueError(f"Invalid steady state calculation option: {value}. Options are {self._steady_state_calculation_opts}")
        self._steady_state_calculation_method = value

    def set_steady_state_calculation(self, value):
        self.steady_state_calculation_method = value
        return self

