import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Literal
from functools import cached_property  # wraps

from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

from optimization_objectives import OptimizationObjective, GaussianNLL
# from deterministic import DeterministicProcess, DeterministicModel
from constrained_matrices import (ConstrainedMatrix,
                                  UnitTopRowConstrained as UnitTopRowMatrix,
                                  FreeMatrix as FreeMatrix,
                                  PosDiagonalMatrix as PosDiagonalMatrix,
                                  IdentityMatrix as IdentityMatrix,
                                  DiagonalMatrix as DiagonalMatrix,
                                  ZeroMatrix as ZeroMatrix,
                                  PosDiagLowerTriMatrix as STDMatrix,
                                  CorrelationConstraint as CorrMatrix)


from tsmod.base import Signal, Model, ModelFit, CompositeSignal, CompositeModel, check_is_defined
from base import ForecastResult, DeterministicForecastResult, NormalForecastResult
from state_space.tools.kalman_filter import (KalmanFilter,
                                             KalmanFilterResult,
                                             KalmanFilterInitialization)
from utils import numerical_jacobian

#
# TODO: HIGH PRIORITY
#       1. Add scale to the processes instead of forcing Q = I. I'm not liking this approach anymore,
#               makes using composite model very unwieldy
#

# TODO: MEDIUM PRIORITY
#       1. Add adaptive step to EM optimization (should take very little effort)
#

# TODO: LOW PRIORITY
#           1. Jacobian (score) and Hessian approximations are missing and should speed up minimization
#           2. Concentrating out the observation noise covariance can be added for additional speed up
#   Low priority: I will mostly use models with very few parameters and with univariate series
#   Only after EKF and Score-driven models
#
#

#
# I will bypass the StateSpace Base classes for now.
# After implementing the EKF and Score-Driven I will better integrate with the base class
# from ..base import StateProcess, StateSpaceModelFit, StateSpaceModel, StateProcessRepresentation
#

# ------------------------------
#   STAGE 1: Linear State Space Models
#
#           y_t - A g(t) - B u_t = E f_t + L w_t,           w_t ~ iid N(0,I),
#           f_t                  = M x_t
#           x_t                  = F x_{t-1} + R e_t,       e_t ~ iid N(0,I),
#
#
#   To estimate with a standard kalman-filter:
#
#       y_t - A g(t) - B u_t = E M x_t   + L w_t,     w_t ~ iid N(0,I),
#       x_t                  = F x_{t-1} + R e_t,     e_t ~ iid N(0,I),
#
#
#   f_t is a LinearStateProcess

# ------------------------------

# ---------
# UTILS
# ---------

class LinearSSMUtils:

    @staticmethod
    def validate_covariance(Q):
        """Check that Q is a valid covariance matrix."""
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Covariance matrix Q must be square")
        if not np.allclose(Q, Q.T):
            raise ValueError("Covariance matrix Q must be symmetric")
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(Q)
        if np.any(eigenvalues < -1e-12):  # allow tiny numerical errors
            raise ValueError("Covariance matrix Q must be positive semi-definite")

    @staticmethod
    def validate_chol_factor(LQ):
        """Check that LQ is lower triangular with positive diagonal."""
        if not np.allclose(LQ, np.tril(LQ)):
            raise ValueError("LQ must be lower triangular")
        if np.any(np.diag(LQ) <= 0):
            raise ValueError("Diagonal entries of LQ must be positive")

    @staticmethod
    def simulate_or_forecast(time_domain: Union[tuple[int, int], int],
                             latent_process: "LinearStateProcess",
                             exposures: Optional[np.ndarray] = None,
                             measurement_noise_std: Optional[np.ndarray] = None,
                             burn: int = 0,
                             initial_state: Optional[np.ndarray] = None):

        #       y_t - A g(t) - B u_t = E f_t + L w_t,           w_t ~ iid N(0,I),
        #       f_t                  = M x_t
        #       x_t                  = F x_{t-1} + R e_t,       e_t ~ iid N(0,I),

        if isinstance(time_domain, int):
            if time_domain < 1:
                raise ValueError("time_domain must be a tuple of 2 ints or an int > 0")
            T: int = time_domain  # single int case
        elif isinstance(time_domain, tuple):
            if not (len(time_domain) == 2 and all(isinstance(i, int) for i in time_domain)):
                raise ValueError("time_domain must be a tuple of 2 ints or an int > 0")
            T: int = time_domain[1] - time_domain[0]  # difference of two ints
            if T < 1:
                raise ValueError("time_domain tuple must have a positive difference")
        else:
            raise TypeError("time_domain must be an int or a tuple of 2 ints")

        # ---- Latent Process Representation and Simulation ----
        state_dim = latent_process.state_dim

        M, F, R = latent_process.representation().params

        if initial_state is not None:
            latent_stochastic = np.zeros((T, state_dim))
            eta = np.random.randn(T, state_dim)
            latent_stochastic[0] = F @ initial_state + R @ eta[0]
            for t in range(1, T):
                latent_stochastic[t] = F @ latent_stochastic[t - 1] + R @ eta[t]
            latent_stochastic = latent_stochastic @ M.T
        else:
            latent_stochastic = np.zeros((T + burn, state_dim))
            eta = np.random.randn(T + burn, state_dim)
            latent_stochastic[0] = eta[0]
            for t in range(1, T + burn):
                latent_stochastic[t] = F @ latent_stochastic[t-1] + R @ eta[t]
            latent_stochastic = latent_stochastic @ M.T
            latent_stochastic = latent_stochastic[burn:]

        if exposures is not None:
            obs_dim = exposures.shape[0]
            obs_simulated = latent_stochastic @ exposures.T
        else:
            return latent_stochastic

        # ---- Observation noise ----
        if measurement_noise_std is not None:
            obs_simulated += np.random.randn(T, obs_dim) @ measurement_noise_std.T

        return obs_simulated

# ---------
# Model Fit Class
# ---------

class LinearStateSpaceModelFit(ModelFit):

    def __init__(self,
                 series: np.ndarray,
                 model: "LinearStateSpaceModel"):
        super().__init__(series, model)

        self._kf = KalmanFilter().set_endog(self.series).set_representation(self.model.representation()).set_initialization(self.model._kf_innit)

    @cached_property
    def _filtered(self):
        return self._kf.filter()

    def _calc_nll(self) -> float:
        return self._filtered.nll

    def get_prediction_errors(self):
        return self._filtered.prediction_errors

    def forecast_with_uncertainty(self, k: int) -> ForecastResult:
        forecasted =  self._filtered.forecast(k)
        return NormalForecastResult(forecasted.forecasted_obs, forecasted.forecasted_obs_cov)

    def forecast(self, k: int) -> DeterministicForecastResult:
        forecasted = self._filtered.forecast(k)
        return DeterministicForecastResult(forecasted.forecasted_obs)

    def get_loss(self, loss: OptimizationObjective) -> float:
        if loss.requires_cov:
            if isinstance(loss, GaussianNLL):
                return self.nll()
            raise NotImplementedError("Thus far, losses that require cov are not implemented")
        else:
            return loss(self.get_prediction_errors())

# ---------
# StateSpace Representation Classes
# ---------

class LinearStateProcessRepresentation:

    def __init__(self,M, F, R,
                 LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                 validate: bool = True):

        self._M = M
        self._F = F
        self._R = R

        if validate:
            for m in [M, F, R]:
                if not isinstance(m, np.ndarray):
                    raise ValueError("M, F, R need to be a numpy arrays")
            if LQ is not None:
                LinearSSMUtils.validate_chol_factor(LQ)
            if Q is not None:
                LinearSSMUtils.validate_covariance(Q)

            if (LQ is not None) and (Q is not None):
                if not np.allclose(LQ @ LQ.T, Q):
                    raise ValueError("LQ needs to be the Cholesky factor of Q")

        self._Q = Q
        self._LQ = LQ
        if LQ is None and Q is None:
            self._LQ = np.eye(R.shape[1])
            self._Q = self._LQ

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

    @property
    def params(self):
        return self.M, self.F, self.R


class LinearStateSpaceModelRepresentation(LinearStateProcessRepresentation):

    def __init__(self,
                 E,
                 M, F, R,
                 const: Optional[np.ndarray] = None,
                 LH: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None,
                 LQ: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                 validate: bool = True):

        if LH is None and H is None:
            raise ValueError("At least one of LH or H must be provided")

        super().__init__(M, F, R, LQ, Q, validate)

        if const is None:
            const = np.zeros((E.shape[0],))

        if validate:
            for m in [const, E]:
                if not isinstance(m, np.ndarray):
                    raise ValueError("const, E need to be a numpy arrays")
            if LH is not None:
                LinearSSMUtils.validate_chol_factor(LH)
            if H is not None:
                LinearSSMUtils.validate_covariance(H)

            if (LH is not None) and (H is not None):
                if not np.allclose(LH @ LH.T, H):
                    raise ValueError("LQ needs to be the Cholesky factor of Q")


        self._const = const
        self._E = E
        self._LH = LH
        self._H = H

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
    def params(self):
        return self.const, self.E, self.L, self.M, self.F, self.R

# ---------
# "Latent" process, and composite "latent" process
# ---------

class LinearStateProcess(Model, ABC):

    # _scale_constrain_to_underlying_map = {'free': STDMatrix,
    #                                       'identity': IdentityMatrix,
    #                                       'diagonal': PosDiagonalMatrix,
    #                                       'correlation': CorrMatrix}

    _scale_constrain_options = ["free", "identity", "diagonal", "correlation"]

    def __init__(self, shape: tuple, innovation_dim: int, **kwargs):
        super().__init__(shape, **kwargs)

        self._innovation_dim = innovation_dim
        self._scale_constrain = 'free'
        self._underlying_scale_matrix = STDMatrix((self._innovation_dim, self._innovation_dim))

    @property
    def innovation_dim(self) -> int:
        """Return the dimension of the model innovations."""
        return self._innovation_dim

    @property
    def scale_constrain(self):
        return self._scale_constrain

    @scale_constrain.setter
    def scale_constrain(self, value):
        if value not in self._scale_constrain_options:
            raise ValueError("Scale constraint must be one of: free, identity, diagonal, correlation")

        getattr(self, f'_constrain_scale_to_{value}')(self._scale_constrain)
        self._scale_constrain = value

    def set_scale_constrain(self, value):
        self.scale_constrain = value
        return self

    @property
    def std(self):
        if not self._underlying_scale_matrix.is_defined:
            return None

        if self._scale_constrain == 'free':
            return self._underlying_scale_matrix.matrix
        elif self._scale_constrain == 'identity':
            return self._underlying_scale_matrix.matrix
        elif self._scale_constrain == 'diagonal':
            return self._underlying_scale_matrix.matrix
        elif self._scale_constrain == 'correlation':
            return self._underlying_scale_matrix.matrix
        else:
            raise ValueError("Scale constraint must be one of: free, identity, diagonal, correlation")

    @property
    def cov(self):
        if not self._underlying_scale_matrix.is_defined:
            return None

        if self._scale_constrain == 'free':
            mat = self._underlying_scale_matrix.matrix
            return mat @ mat.T
        elif self._scale_constrain == 'identity':
            return self._underlying_scale_matrix.matrix
        elif self._scale_constrain == 'diagonal':
            return self._underlying_scale_matrix.matrix ** 2
        elif self._scale_constrain == 'correlation':
            return self._underlying_scale_matrix.matrix
        else:
            raise ValueError("Scale constraint must be one of: free, identity, diagonal, correlation")

    def _constrain_scale_to_identity(self, previous_scale_constraint):
        self._underlying_scale_matrix = IdentityMatrix((self._innovation_dim, self._innovation_dim))

    def _constrain_scale_to_diagonal(self, previous_scale_constraint):
        current_std = self.std
        self._underlying_scale_matrix = PosDiagonalMatrix((self._innovation_dim, self._innovation_dim))

    def _constrain_scale_to_correlation(self):
        self._underlying_scale_matrix = CorrMatrix((self._innovation_dim, self._innovation_dim))

    def _constrain_scale_to_free(self):
        self._underlying_scale_matrix = STDMatrix((self._innovation_dim, self._innovation_dim))

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    def is_defined(self) -> bool:
        return self._underlying_scale_matrix.is_defined and self.is_dynamics_defined

    @property
    @abstractmethod
    def is_dynamics_defined(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _update_params(self, params: np.ndarray) -> None:
        self._scale.update_params(params[:self._scale.n_params])
        self._update_params(params[self._scale.n_params:])

    @abstractmethod
    def _get_params(self) -> np.ndarray:
        params = np.empty((self.n_params,))
        params[:self._scale.n_params] = self._scale.get_params()
        params[self._scale.n_params:] = self._get_dynamic_params()
        return params

    @property
    def _n_params(self) -> int:
        return self.n_dynamic_params + self._scale.n_params

    @property
    @abstractmethod
    def n_dynamic_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_dynamic_params(self) -> np.ndarray:
        """
        Return only the concrete subclass parameters as a 1D array.
        Do NOT include scale parameters; the base class will handle those.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_dynamic_params(self, params: np.ndarray):
        """
        Set only the concrete subclass parameters from a 1D array.
        Do NOT include scale parameters; the base class will handle those.
        """
        raise NotImplementedError

    @check_is_defined
    def forecast(self, k: int, initial_state: np.ndarray) -> np.ndarray:
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self,
                                                   initial_state=initial_state)

    @check_is_defined
    def simulate(self, k: int, burn: int = 0):
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self,
                                                   burn=burn)

    def fit(self,
            series: np.ndarray,
            objective: Union[OptimizationObjective, Any] = GaussianNLL(),
            include_constant: bool = False,
            measurement_noise: Optional[Literal["zero", "diagonal", "full"]] = None) -> LinearStateSpaceModelFit:

        n_series = series.shape[1]

        if n_series != self.shape[0]:
            raise ValueError("Series has wrong shape")

        if not self.is_defined:
            self._first_fit_to(series)

        ssm = self._state_space_model(include_constant, measurement_noise)

        if include_constant:
            ssm.constant = np.mean(series).reshape(-1, 1)

        ssm.measurement_noise_std.shape = (n_series, n_series)
        if measurement_noise == "diagonal":
            ssm.measurement_noise_std = np.diag(np.sqrt(np.std(series, axis=0) * 0.5))
        if measurement_noise == "full":
            ssm.measurement_noise_std = np.linalg.cholesky(np.cov(series) * 0.5)

        ssm.exposures = np.eye(n_series)

        return ssm.fit(series, objective)

    @abstractmethod
    def representation(self, *args, **kwargs) -> LinearStateProcessRepresentation:
        raise NotImplementedError

    @abstractmethod
    def _first_fit_to(self, series: np.ndarray):
        """
        fits the model to the series. the series should match the shape of the process.

        for instance, if the process is shape (n, 1), then series should be of size T * n

        Args:
            series: observed time series

        """
        raise NotImplementedError

    def _build_state_space_model(self,
                                 series,
                                 include_constant: bool,
                                 measurement_noise: Optional[Literal["zero", "diagonal", "full"]]) -> "LinearStateSpaceModel":
        """
        Can be overwritten.

        needs to build a state space model of the form

        y_t = E f_t + e_t, e_t ~ N(0,H) = LH N(0, I)
        f_t = M x_t
        x_t = F x_t-1 + R u_t , u_t ~ N(0,Q) = LQ N(0, I)

        see LinearStateSpaceModelRepresentation

        Args:
            series:
            include_constant:
            measurement_noise:

        Returns:

        """

        if series.shape[0] == self.shape[0]:
            if not self.is_defined:
                was_identity_flag = False
                if self._scale_constrain == 'identity':
                    was_identity_flag = True
                    self.constrain_scale_to_diagonal()

                self._first_fit_to(series)

                if was_identity_flag:
                    exposures = DiagonalMatrix((self.shape[0], self.shape[0]))

                else:
                    exposures = IdentityMatrix((self.shape[0], self.shape[0]))



        if include_constant:
            constant = FreeMatrix((None, 1))
        else:
            constant = ZeroMatrix((None, 1))

        if measurement_noise == "zero" or measurement_noise is None:
            measurement_noise = ZeroMatrix((None, None))
        elif measurement_noise == "diagonal":
            measurement_noise = PosDiagonalMatrix((None, None))
        elif measurement_noise == "full":
            measurement_noise = STDMatrix((None, None))
        else:
            raise ValueError("Unknown measurement noise")

        ssm = LinearStateSpaceModel(linear_state_process=self,
                                    exposures=exposures,
                                    constant=constant,
                                    measurement_noise_std=measurement_noise)

        return ssm


class CompositeLinearStateProcess(CompositeModel, LinearStateProcess):

    def _update_dynamic_params(self, params: np.ndarray):
        pass


    def _get_dynamic_params(self) -> np.ndarray:
        pass

    @property
    def n_dynamic_params(self) -> int:
        pass


    def __init__(self, processes: list[LinearStateProcess], mixing_matrix: np.ndarray):
        self._underlying_processes = processes
        self._mixing_matrix = mixing_matrix

        self._check_innit()

        super().__init__((self._mixing_matrix.shape[0], 1))

    def _check_innit(self):
        if not isinstance(self._mixing_matrix, np.ndarray):
            raise ValueError("Mixing matrix must be an np.ndarray.")

        if not self._mixing_matrix.ndim == 2:
            raise ValueError("Mixing matrix must be an 2D np.ndarray.")

        shape_of_underlying = 0
        for row in self.mixing_matrix:
            idxs = row == 0
            shapes = [self._underlying_processes[i].shape[0] for i in idxs]
            if len(set(shapes)) != 1:
                raise ValueError("Shape missmatch. Can not sum processes of different sizes.")
            shape_of_underlying += shapes[0]

        if not self._mixing_matrix.shape[1] == shape_of_underlying:
            raise ValueError("Shape missmatch. Mixing matrix with .")

    @property
    def mixing_matrix(self) -> np.ndarray:
        return self._mixing_matrix

    @property
    def n_underlying_processes(self) -> int:
        return len(self._underlying_processes)

    @property
    def underlying_processes(self) -> list[LinearStateProcess]:
        return self._underlying_processes

    @property
    def _underlying_signals(self) -> list[Signal]:
        return self._underlying_processes

    @property
    def state_dim(self) -> int:
        return sum(i.state_dim for i in self._underlying_processes)

    def representation(self, *args, **kwargs) -> LinearStateProcessRepresentation:
        reps = [process.representation(*args, **kwargs) for process in self._underlying_processes]
        F = block_diag([rep.F for rep in reps])
        R = block_diag([rep.R for rep in reps])
        M = self.mixing_matrix @ block_diag([rep.M for rep in reps])
        return LinearStateProcessRepresentation(M, F, R)

    def _first_fit_to(self, series: np.ndarray):
        raise NotImplementedError(
            "This estimator was initialized with incomplete parameters. "
            "Subclasses must implement _first_estimates() to support "
            "auto-estimation."
        )

# ---------
# "Full" linear state space model
# ---------

class LinearStateSpaceModel(CompositeModel):

    def __init__(self,
                 linear_state_process: "LinearStateProcess",
                 exposures: ConstrainedMatrix,
                 constant: Optional[ConstrainedMatrix] = None,
                 measurement_noise_std: Optional[ConstrainedMatrix]= None):
        super().__init__((None, 1), )

        self._state_process = linear_state_process
        self._exposures = exposures
        self._constant = constant if constant is not None else ZeroMatrix((None, 1))
        self._measurement_noise_std = measurement_noise_std if measurement_noise_std is not None else ZeroMatrix((None, None))

        self._check_shapes()

        self._kf_innit = KalmanFilterInitialization(initialization_type="ss",
                                                    x0=np.zeros(self._state_process.state_dim,),
                                                    P0=None,
                                                    P_star=None,
                                                    P_infty=None)

        for m in [self._exposures, self._constant, self._measurement_noise_std]:
            self.shape = (m.shape[0], 1)


    def _check_shapes(self):

        def require_equal_or_none(values,
                                  require_one: bool = False
                                  ):
            """Check that all non-None values in the list are equal.

            Args:
                values: list of values that may be None
                require_one: if True, at least one value must be non-None
            """
            non_none_values = [v for v in values if v is not None]

            if require_one and not non_none_values:
                raise ValueError("Require one or more non-None values, shape undefined")

            if len(set(non_none_values)) > 1:
                raise ValueError("Incompatible shapes in StateSpaceModel")

            return non_none_values[0]

        process_shape = require_equal_or_none([self.state_process.shape[0], self.exposures.shape[1]], True)
        self.exposures.shape = (None, process_shape)

    @property
    def _underlying_signals(self) -> list[Signal]:
        signals = [self.state_process, self.exposures]
        if self.constant is not None:
            signals.append(self.constant)
        if self.measurement_noise_std is not None:
            signals.append(self.measurement_noise_std)
        return signals

    @property
    def state_process(self) -> "LinearStateProcess":
        return self._state_process

    @property
    def exposures(self) -> Optional[ConstrainedMatrix]:
        return self._exposures

    @exposures.setter
    def exposures(self, value) -> None:
        if isinstance(value, ConstrainedMatrix):
            self._exposures = value
        elif isinstance(value, np.ndarray):
            self._exposures.matrix = value

    @property
    def constant(self) -> Optional[ConstrainedMatrix]:
        return self._constant

    @constant.setter
    def constant(self, value) -> None:
        if isinstance(value, ConstrainedMatrix):
            self._constant = value
        elif isinstance(value, np.ndarray):
            self._constant.matrix = value

    @property
    def measurement_noise_std(self) -> Optional[ConstrainedMatrix]:
        return self._measurement_noise_std

    @measurement_noise_std.setter
    def measurement_noise_std(self, value) -> None:
        if isinstance(value, ConstrainedMatrix):
            self.measurement_noise_std = value
        elif isinstance(value, np.ndarray):
            self._measurement_noise_std.matrix = value

    def representation(self, *args, **kwargs) -> LinearStateSpaceModelRepresentation:
        M, F, R = self.state_process.representation().params
        E = self.exposures.matrix
        L = self.measurement_noise_std.matrix
        const = self.constant.matrix[:, 0]
        return LinearStateSpaceModelRepresentation(const, E, L, M, F, R)

    @check_is_defined
    def forecast(self, k: int, initial_state: np.ndarray) -> np.ndarray:
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self.state_process,
                                                   exposures=self.exposures.matrix,
                                                   measurement_noise_std=self.measurement_noise_std,
                                                   initial_state=initial_state)

    @check_is_defined
    def simulate(self, k: int, burn: int = 0):
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self.state_process,
                                                   exposures=self.exposures.matrix,
                                                   measurement_noise_std=self.measurement_noise_std,
                                                   burn=burn)

    def set_kf_initialization(self,
                              initialization_type: Literal["ss", "ed", "s"],
                              init_state: np.ndarray,
                              P0: Optional[np.ndarray] = None,
                              P_star: Optional[np.ndarray] = None,
                              P_infty: Optional[np.ndarray] = None,):

        self._kf_innit = KalmanFilterInitialization(initialization_type=initialization_type,
                                                    x0=init_state,
                                                    P0=P0,
                                                    P_star=P_star,
                                                    P_infty=P_infty)

    def fit(self,
            series: np.ndarray,
            objective: Union[OptimizationObjective, Literal["EM"]]) -> LinearStateSpaceModelFit:

        n_series = series.shape[1]
        self.constant.shape = (n_series, 1)
        self.exposures.shape = (n_series, self.shape[0])
        self.measurement_noise_std.shape = (n_series, n_series)

        if not self.is_defined:
            self._first_fit_to(series)

        if isinstance(objective, OptimizationObjective):
            self._fit_minimize(series, objective)
        elif objective == "EM":
            self._fit_em(series)
        else:
            raise ValueError(f"Objective must be 'EM' or an instance of OptimizationObjective, got {objective}")

        return LinearStateSpaceModelFit(series, self)

    def _first_fit_to(self, series: np.ndarray):
        raise NotImplementedError(
            "This estimator was initialized with incomplete parameters. "
            "Subclasses must implement _first_fit_to() to support "
            "auto-estimation."
        )

    def _fit_minimize(self, series, objective):
        kf = KalmanFilter().set_endog(series).set_representation(self.representation()).set_initialization(self._kf_innit)
        params0 = self.get_params()

        def obj(x):
            self.update_params(x)
            kf.set_representation(self.representation())
            return kf.loss(objective)

        optimized_result = minimize(obj, x0=np.array(params0), method='L-BFGS-B')

        x_opt = optimized_result.x
        self.update_params(x_opt)

    def _fit_em(self, series):

        for _ in range(500):
            kf = KalmanFilter().set_endog(series).set_representation(self.representation()).set_initialization(self._kf_innit)
            Aj, Bj, Cj, Dj, Ej, Fj = kf.get_em_matrices()

            L = self.measurement_noise_std.matrix
            X = solve_triangular(L, np.eye(L.shape[0]), lower=True)
            H_inv = solve_triangular(L, X, lower=True, trans='T')

            # H_inv = np.linalg.inv(self.measurement_noise_std.matrix @ self.measurement_noise_std.matrix.T)

            next_latent_params = self._calc_em_next_latent_params(Aj, Bj, Cj, Ej, Fj, H_inv)
            next_exposure_params = self._calc_em_next_exposure_params(Ej, Fj, H_inv)

            Z = self.representation().E @ self.representation().M
            measurement_error_vars_new = np.diag((1 / series.shape[0]) * (Dj - Z @ Ej.T - Ej @ Z.T + Z @ Fj @ Z.T))

            self.state_process.update_params(next_latent_params)
            self.exposures.update_params(next_exposure_params)
            self.measurement_noise_std = np.sqrt(measurement_error_vars_new.reshape(self.measurement_noise_std.shape))

    def _calc_em_next_exposure_params(self, Ej, Fj, H_inv):

        current_exposure_params = self.exposures.get_params()
        XI = self.exposures.jacobian_vec_params(order='F')

        M, _, _ = self.state_process.representation().params

        # L = self.measurement_noise_std.matrix
        # X = solve_triangular(L, Ej, lower=True)
        # H_inv_E = solve_triangular(L, X, lower=True, trans='T')

        gj = (H_inv @ Ej @ M.T).reshape((-1,), order='F')

        Gj = np.kron ((M @ Fj @ M.T).T, H_inv)

        step = np.linalg.inv(XI.T @ Gj @ XI) @ XI.T @ (gj - Gj @ self.exposures.get_vectorized(order='F'))

        return current_exposure_params + step

    def _calc_em_next_latent_params(self, Aj, Bj, Cj, Ej, Fj, H_inv):

        M, F, R = self.state_process.representation().params

        # L = self.measurement_noise_std.matrix
        # exposures = self.exposures.matrix
        # B = solve_triangular(L, exposures.T, lower=True, trans='T').T
        # exposures_H_inv = solve_triangular(L, B.T, lower=True).T

        exposures = self.exposures.matrix
        exposures_H_inv = exposures @ H_inv

        g1j = exposures_H_inv @ Ej
        g2j = (R @ R.T) @ Bj
        g3j = np.zeros((R.shape[0] * R.shape[1],))
        gj = np.hstack([g1j.flatten(order='F'), g2j.flatten(order='F'), g3j.flatten(order='F')])

        G1j = np.kron(Fj.T, exposures_H_inv @ exposures)
        G2j = np.kron(Cj.T, R @ R.T)
        Y = Aj - F @ Bj.T - Bj @ F.T + F @ Cj @ F.T
        G3j = np.kron(np.eye(R.shape[1]), Y)
        Gj = block_diag(G1j, G2j, G3j)

        XI = self._dvecMFR_dparams()
        current_vecMFR = np.hstack([m.flatten('F') for m in [M, F, R]])
        step = np.linalg.inv(XI.T @ Gj @ XI) @ XI.T @ (gj - Gj @ current_vecMFR)

        current_latent_params = self.state_process.get_params()
        return current_latent_params + step

    def _dvecMFR_dparams(self):
        current_latent_params = self.state_process.get_params()

        def func(x):
            self.state_process.update_params(x)
            repre = self.state_process.representation().params
            return np.hstack([m.flatten('F') for m in repre])

        jacobian = numerical_jacobian(func, current_latent_params)

        self.state_process.update_params(current_latent_params)

        return jacobian

    def _calc_em_next_latent_and_exposure_params(self, Bj, Cj, Ej, Fj):

        M, F, R = self.state_process.representation().params

        XI = self._dvecFZ_dparams()
        repre = self.representation()
        f_theta1 = np.hstack([m.flatten('F') for m in [repre.F, repre.E @ repre.M]])

        # theta_1, XI, f_theta1 = self.calc_dvecFZ_DexposuresArma(d_FIE, arma_params, exposures)
        H_inv = np.linalg.inv(self.measurement_noise_std.matrix @ self.measurement_noise_std.matrix.T)
        Gj1 = np.kron(Cj.T, R @ R.T)
        Gj2 = np.kron(Fj.T, H_inv)
        Gj = block_diag(Gj1, Gj2)
        g1j = (R @ R.T) @ Bj
        g2j = H_inv @ Ej
        gj = np.hstack([g1j.flatten(order='F'), g2j.flatten(order='F')])
        step = np.linalg.inv(XI.T @ Gj @ XI) @ XI.T @ (gj - Gj @ f_theta1)

        current_latent_params = self.state_process.get_params()
        n_latent = len(current_latent_params)
        current_exposure_params = self.exposures.get_params()

        return current_latent_params + step[:n_latent], current_exposure_params + step[n_latent:]

    def _dvecFZ_dparams(self):

        current_latent_params = self.state_process.get_params()
        n_latent = len(current_latent_params)
        current_exposure_params = self.exposures.get_params()

        current_params = np.hstack([current_latent_params, current_exposure_params])

        def func(x):
            self.state_process.update_params(x[:n_latent])
            self.exposures.update_params(x[n_latent:])
            repre = self.representation()
            return np.hstack([m.flatten('F') for m in [repre.F, repre.E @ repre.M]])

        jacobian = numerical_jacobian(func, current_params)

        self.state_process.update_params(current_latent_params)
        self.exposures.update_params(current_exposure_params)

        return jacobian




# import numpy as np
# from scipy.linalg import solve_triangular, block_diag
#
# class GEM_EM_LM:
#     def __init__(self, state_process, exposures, measurement_noise_std, max_iters=500):
#         self.state_process = state_process
#         self.exposures = exposures
#         self.measurement_noise_std = measurement_noise_std
#         self.max_iters = max_iters
#
#         # LM damping parameters for each block
#         self.lm_lambda_latent = 1.0
#         self.lm_lambda_exposure = 1.0
#
#     def _compute_Q_latent(self, params, Aj, Bj, Cj, Ej, Fj):
#         """Compute expected complete-data Q for latent parameters"""
#         # Extract M, F, R from params
#         M, F, R = self.state_process.unpack_params(params)
#
#         # Use EM matrices to compute Q
#         Y = Aj - F @ Bj.T - Bj @ F.T + F @ Cj @ F.T
#         Q_state = np.sum((R @ R.T) * Y) + np.log(np.linalg.det(R @ R.T))
#         g1j = self.exposures.matrix @ np.linalg.inv(self.measurement_noise_std.matrix) @ Ej
#         Q_meas = np.sum(g1j**2)  # Simplified placeholder, replace with proper formula
#         return Q_state + Q_meas
#
#     def _compute_Q_exposure(self, params, Ej, Fj):
#         """Compute expected complete-data Q for exposures"""
#         # Extract exposure matrix
#         exposure_matrix = self.exposures.unpack_params(params)
#         H_inv = np.linalg.inv(self.measurement_noise_std.matrix @ self.measurement_noise_std.matrix.T)
#         gj = (H_inv @ Ej @ self.state_process.M.T).reshape((-1,), order='F')
#         Gj = np.kron((self.state_process.M @ Fj @ self.state_process.M.T).T, H_inv)
#         Q = np.sum((gj - Gj @ exposure_matrix.flatten('F'))**2)  # proxy Q
#         return Q
#
#     def _lm_update(self, H, g, theta_old, compute_Q, *args):
#         """Levenberg–Marquardt damped step"""
#         lambda_damp = 1.0
#         max_inner_iters = 10
#
#         for _ in range(max_inner_iters):
#             try:
#                 step = np.linalg.solve(H + lambda_damp * np.eye(H.shape[0]), g)
#             except np.linalg.LinAlgError:
#                 lambda_damp *= 10
#                 continue
#
#             theta_new = theta_old + step
#             Q_old = compute_Q(theta_old, *args)
#             Q_new = compute_Q(theta_new, *args)
#
#             # Predicted improvement (Gauss-Newton approximation)
#             predicted_reduction = 0.5 * step.T @ (lambda_damp * step + g)
#             rho = (Q_old - Q_new) / (predicted_reduction + 1e-12)
#
#             if rho > 0:
#                 # Step accepted
#                 lambda_damp *= max(1/3, 1 - (2*rho - 1)**3)
#                 return theta_new, lambda_damp
#             else:
#                 # Step rejected, increase damping
#                 lambda_damp *= 10
#
#         return theta_old, lambda_damp
#
#     def _calc_next_latent_params(self, Aj, Bj, Cj, Ej, Fj):
#         XI = self._dvecMFR_dparams()
#         current_vecMFR = np.hstack([m.flatten('F') for m in self.state_process.get_params()])
#
#         exposures_H_inv = self.exposures.matrix @ np.linalg.inv(self.measurement_noise_std.matrix)
#         g1j = exposures_H_inv @ Ej
#         g2j = (self.state_process.R @ self.state_process.R.T) @ Bj
#         g3j = np.zeros((self.state_process.R.shape[0] * self.state_process.R.shape[1],))
#         gj = np.hstack([g1j.flatten('F'), g2j.flatten('F'), g3j.flatten('F')])
#
#         G1j = np.kron(Fj.T, exposures_H_inv @ self.exposures.matrix)
#         G2j = np.kron(Cj.T, self.state_process.R @ self.state_process.R.T)
#         Y = Aj - self.state_process.F @ Bj.T - Bj @ self.state_process.F.T + self.state_process.F @ Cj @ self.state_process.F.T
#         G3j = np.kron(np.eye(self.state_process.R.shape[1]), Y)
#         Gj = block_diag(G1j, G2j, G3j)
#
#         H = XI.T @ Gj @ XI
#         g = XI.T @ (gj - Gj @ current_vecMFR)
#
#         # LM update
#         updated_vecMFR, self.lm_lambda_latent = self._lm_update(H, g, current_vecMFR, self._compute_Q_latent, Aj, Bj, Cj, Ej, Fj)
#
#         return self.state_process.unpack_vector(updated_vecMFR)
#
#     def _calc_next_exposure_params(self, Ej, Fj):
#         current_exposure_params = self.exposures.get_params()
#         XI = self.exposures.jacobian_vec_params(order='F')
#
#         M = self.state_process.M
#         H_inv = np.linalg.inv(self.measurement_noise_std.matrix @ self.measurement_noise_std.matrix.T)
#
#         gj = (H_inv @ Ej @ M.T).reshape((-1,), order='F')
#         Gj = np.kron((M @ Fj @ M.T).T, H_inv)
#
#         H = XI.T @ Gj @ XI
#         g = XI.T @ (gj - Gj @ self.exposures.get_vectorized(order='F'))
#
#         # LM update
#         updated_exposures, self.lm_lambda_exposure = self._lm_update(H, g, current_exposure_params, self._compute_Q_exposure, Ej, Fj)
#         return updated_exposures
#
#     def fit(self, series):
#         for _ in range(self.max_iters):
#             # ---- E-step ----
#             kf = KalmanFilter().set_endog(series).set_representation(self.state_process.representation()).set_initialization(self.state_process._kf_init)
#             Aj, Bj, Cj, Dj, Ej, Fj = kf.get_em_matrices()
#
#             # ---- M-step: latent parameters ----
#             next_latent_params = self._calc_next_latent_params(Aj, Bj, Cj, Ej, Fj)
#             self.state_process.update_params(next_latent_params)
#
#             # ---- M-step: exposures ----
#             next_exposure_params = self._calc_next_exposure_params(Ej, Fj)
#             self.exposures.update_params(next_exposure_params)
#
#             # ---- M-step: measurement noise ----
#             Z = self.state_process.representation().E @ self.state_process.representation().M
#             measurement_error_vars_new = np.diag(
#                 (1 / series.shape[0]) * (Dj - Z @ Ej.T - Ej @ Z.T + Z @ Fj @ Z.T)
#             )
#             self.measurement_noise_std = np.sqrt(measurement_error_vars_new.reshape(self.measurement_noise_std.shape))
#


# --------
# For later
# -----------

#
# class LinearStateSpaceModel(StateSpaceModel):
#
#
#     def __init__(self,
#                  state_process: "LinearStateProcess",
#                  exposures: Optional[ConstrainedMatrix] = None,
#                  deterministic: Optional[DeterministicProcess] = None,
#                  deterministic_loadings: Optional[ConstrainedMatrix] = None,
#                  exog: Optional[np.ndarray] = None,
#                  exog_loadings: Optional[ConstrainedMatrix] = None,
#                  observation_noise_std: Optional[ConstrainedMatrix] = None):
#
#         # init here adds additional constrain that the latent process be linear
#
#         super().__init__(state_process=state_process,
#                          exposures=exposures,
#                          deterministic=deterministic,
#                          deterministic_loadings=deterministic_loadings,
#                          exog=exog,
#                          exog_loadings=exog_loadings,
#                          observation_noise_std=observation_noise_std)
#
#
#     def fit(self, series: np.ndarray, objective: Union[OptimizationObjective, Any]) -> StateSpaceModelFit:
#         pass
#
#     def simulate(self, *args, **kwargs):
#         pass
#
#     def forecast(self, *args, **kwargs):
#         pass
#
#
# class LinearStateProcessRepresentation(StateProcessRepresentation):
#
#     def __init__(self, M, F, R):
#         self._M = M
#         self._F = F
#         self._R = R
#
#     @property
#     def M(self):
#         return self._M
#
#     @property
#     def F(self):
#         return self._F
#
#     @property
#     def R(self):
#         return self._R
#
#     @property
#     def params(self):
#         return self.M, self.F, self.R
#
#
# class LinearStateProcess(StateProcess, ABC):
#
#     def __init__(self,
#                  shape: tuple[int, int],
#                  **kwargs):
#
#         super().__init__(shape=shape,
#                          **kwargs)
#
#     @property
#     @abstractmethod
#     def state_dim(self) -> int:
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def is_defined(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def representation(self) -> LinearStateProcessRepresentation:
#         raise NotImplementedError
#
#     def _state_space_model(self) -> LinearStateSpaceModel:
#         return LinearStateSpaceModel(self)
#
#     @abstractmethod
#     def _first_fit_to(self, series: np.ndarray):
#         raise NotImplementedError
#
#     @abstractmethod
#     def _get_params(self) -> np.ndarray:
#         raise NotImplementedError
#
#     @abstractmethod
#     def _update_params(self, params: np.ndarray) -> None:
#         raise NotImplementedError
#
#

