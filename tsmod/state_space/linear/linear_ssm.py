# from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Literal, Generic, TypeVar, Tuple
# from typing import List, Sequence
from functools import cached_property  # wraps
# from enum import Enum, auto

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import minimize

from optimization_objectives import OptimizationObjective, GaussianNLL

from constrained_matrices import (ConstrainedMatrix,
                                  FreeMatrix,
                                  IdentityMatrix,
                                  ZeroMatrix,
                                  ConstrainedCovarianceAPI)
                                  # ElementWiseConstrainedMatrix

from base import Model, ModelFit, CompositeMixin, check_is_defined
from base import ForecastResult, DeterministicForecastResult, NormalForecastResult
from tools.utils import validate_covariance, covariance_to_correlation # validate_chol_factor
from state_space.tools.kalman_filter import (KalmanFilter,
                                             KalmanFilterInitialization)
                                             # KalmanFilterResult,

from representation import LinearStateSpaceModelRepresentation, LinearStateProcessRepresentation, LinearStateProcessDynamics

from utils import numerical_jacobian

# TODO: HIGH PRIORITY
#       1. Change EM, currently its wrong because the representation was changed
#       2. Add general "to_identifiable_state_space_model" based on RotationalSymmetry

# TODO: MEDIUM PRIORITY
#       1. Add adaptive step to EM optimization (should take very little effort)
#       2. Add RepresentationStructure and "better EM" which uses it
#       3. Add Gauge Symmetries. Not using right now so I will skip for simplicity

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
        innovation_dim = latent_process.innovation_dim

        rep = latent_process.representation()
        M, F, RLQ = rep.M, rep.F, rep.R @ rep.LQ

        if initial_state is not None:
            latent_stochastic = np.zeros((T, state_dim))
            eta = np.random.randn(T, innovation_dim)
            latent_stochastic[0] = F @ initial_state + RLQ @ eta[0]
            for t in range(1, T):
                latent_stochastic[t] = F @ latent_stochastic[t - 1] + RLQ @ eta[t]
            latent_stochastic = latent_stochastic @ M.T
        else:
            latent_stochastic = np.zeros((T + burn, state_dim))
            eta = np.random.randn(T + burn, innovation_dim)
            latent_stochastic[0] = eta[0]
            for t in range(1, T + burn):
                latent_stochastic[t] = F @ latent_stochastic[t-1] + RLQ @ eta[t]
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
            raise NotImplementedError("Losses that require cov are not implemented")
        else:
            return loss(self.get_prediction_errors())

# ----------------------------------------------------------------------------------------------------
# LinearStateProcess
#
# A LinearStateProcess represents any process f_t with a state-space representation of the form
#
#     f_t = M x_t
#     x_t = F x_{t-1} + R u_t, u_t ~ N(0, Q) = LQ N(0, I)
#
# ----------------------------------------------------------------------------------------------------
class LinearStateProcess(Model, ABC):
    # TODO: put advanced options here and add kalman filter initializer

    def __init__(self, process_dim: int, innovation_dim: int):
        super().__init__(shape=(process_dim, 1))
        self._innovation_dim = innovation_dim

    @property
    def process_dim(self) -> int:
        return self.shape[0]

    @property
    def innovation_dim(self) -> int:
        """Return the dimension of the model innovations."""
        return self._innovation_dim

    @property
    def is_defined(self):
        return self.is_dynamics_defined and self.is_innovation_cov_defined

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def is_dynamics_defined(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_innovation_cov_defined(self) -> bool:
        pass

    @property
    @abstractmethod
    def std(self):
        pass

    @property
    @abstractmethod
    def cov(self):
        pass

    @property
    @abstractmethod
    def dynamic_representation(self) -> LinearStateProcessDynamics:
        raise NotImplementedError

    def representation(self, *args, **kwargs) -> LinearStateProcessRepresentation:
        return LinearStateProcessRepresentation.from_dynamic_representation(self.dynamic_representation,
                                                                            Q=self.cov,
                                                                            validate=True)
    @property
    @abstractmethod
    def _n_params(self) -> int:
        pass

    @abstractmethod
    def _get_params(self) -> np.ndarray:
        pass

    @abstractmethod
    def _update_params(self, params: np.ndarray):
        pass

    def fit(self,
            series: np.ndarray,
            include_constant: bool = False,
            measurement_noise: Literal["zero", "diagonal", "free"] = "zero") -> LinearStateSpaceModelFit:

        if series.shape[1] != self.shape[0]:
            return self.fit_factor_model(series, include_constant=include_constant, measurement_noise=measurement_noise)

        if (not include_constant) and measurement_noise == "zero":
            return self.fit_latent(series)

        return self.fit_observation_model(series, include_constant=include_constant, measurement_noise=measurement_noise)

    def fit_latent(self,
                   series: np.ndarray,
                   objective: Union[OptimizationObjective, Any] = GaussianNLL(),
                   ):
        n_series = series.shape[1]

        if n_series != self.shape[0]:
            raise ValueError("Series has wrong shape")

        if not self.is_defined:
            self._first_fit_to(series)

        ssm = LinearStateSpaceModel(linear_state_process=self,
                                    exposures=IdentityMatrix((n_series, n_series)),
                                    constant=ZeroMatrix((n_series, 1)),
                                    measurement_noise_constrain="zero")

        self._initialize_ssm_kf(ssm)
        return ssm.fit(series, objective)

    @abstractmethod
    def _first_fit_to(self, series: np.ndarray):
        """
        fits the model to the series. the series should match the shape of the process.

        for instance, if the process is shape (n, 1), then series should be of size T * n

        Args:
            series: observed time series

        """
        raise NotImplementedError

    def fit_observation_model(self,
                              series: np.ndarray,
                              objective: Union[OptimizationObjective, Any] = GaussianNLL(),
                              include_constant: bool = False,
                              measurement_noise: Literal["zero", "diagonal", "free"] = "zero"):

        if not include_constant and measurement_noise == "zero":
            return self.fit_latent(series, objective)

        n_series = series.shape[1]

        if n_series != self.shape[0]:
            raise ValueError("Series has wrong shape")

        estimate_1st_constant, estimate_1st_measurement_noise_cov =\
            self._first_fit_to_observation_model(series, include_constant, measurement_noise)

        constant_signal = FreeMatrix((n_series, 1)) if include_constant else ZeroMatrix((n_series, 1))
        ssm = LinearStateSpaceModel(linear_state_process=self,
                                    exposures=IdentityMatrix((n_series, n_series)),
                                    constant=constant_signal,
                                    measurement_noise_constrain=measurement_noise)

        if include_constant:
            ssm.constant = estimate_1st_constant

        if measurement_noise != "zero":
            ssm.measurement_noise_cov = estimate_1st_measurement_noise_cov

        self._initialize_ssm_kf(ssm)
        return ssm.fit(series, objective)

    def _first_fit_to_observation_model(self,
                                        series: np.ndarray,
                                        include_constant: bool,
                                        measurement_noise: Literal["zero", "diagonal", "free"]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Can be overwritten by subclasses.
        """
        self._first_fit_to(series)

        n_series = series.shape[1]

        constant = np.mean(self.cov).reshape(-1, 1) if include_constant else np.zeros((n_series, 1))

        if measurement_noise == "zero":
            measurement_noise_estimate = np.zeros((n_series, n_series))
        else:
            measurement_noise_estimate = np.cov(series, rowvar=False) * 0.5
            if measurement_noise == "diagonal" and measurement_noise_estimate.ndim > 1:
                measurement_noise_estimate = np.diag(np.diag(measurement_noise_estimate))
            else:
                measurement_noise_estimate = measurement_noise_estimate.reshape((n_series, n_series))

        return constant, measurement_noise_estimate

    def fit_factor_model(self,
                         series: np.ndarray,
                         objective: Union[OptimizationObjective, Any] = GaussianNLL(),
                         include_constant: bool = False,
                         measurement_noise: Literal["zero", "diagonal", "free"] = "zero"):

        ssm = self._first_fit_to_factor_model(series, include_constant, measurement_noise)
        self._initialize_ssm_kf(ssm)
        return ssm.fit(series, objective)

    def to_identifiable_state_space_model(self,
                                           include_constant: bool,
                                           measurement_noise: Literal[ "free", "zero", "diagonal"]) -> "LinearStateSpaceModel":
        raise NotImplementedError

    def _first_fit_to_factor_model(self,
                                   series: np.ndarray,
                                   include_constant: bool,
                                   measurement_noise: Literal["zero", "diagonal", "free"]):
        raise NotImplementedError

    def _initialize_ssm_kf(self, ssm: "LinearStateSpaceModel"):
        """
        can be overwritten by subclasses.
        But my idea is to make a KalmanFilterInitializer class and pass that has an advanced option
        """
        ssm.set_kf_initialization(initialization_type='ss',
                                  init_state=np.zeros((self.state_dim, )))

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


class AtomicLinearStateProcess(LinearStateProcess, ABC):

    class AdvancedOptions:
        """Holds advanced, context-specific options."""

        # to provide IDE help:
        correlation_parameterization: Literal["hyperspherical", "log"]

        _valid_options = {
            "correlation_parameterization": ("hyperspherical", "log"),
        }

        # CONSIDER: Allowing re-paremeterization of correlation matrix after init
        _immutable_options = ("correlation_parameterization",)

        def __init__(self, **kwargs):
            # Initialize options, validated through __setattr__
            n_options = len(self.get_valid_options())
            if not len(kwargs) == n_options:
                raise ValueError("Incorrect number of arguments. All options need to be filled")
            for name, value in kwargs.items():
                setattr(self, name, value)

        def __setattr__(self, name, value):
            # immutability check
            if name in self.get_immutable_options() and name in self.__dict__:
                raise ValueError(
                    f"{name} is an immutable option. I.e., it can only be set at init"
                )

            valid = self.get_valid_options().get(name)

            if valid is not None:
                is_valid = False

                for rule in valid:
                    # type rule
                    if isinstance(rule, type):
                        if isinstance(value, rule):
                            is_valid = True
                            break
                    # literal value rule
                    else:
                        if value == rule:
                            is_valid = True
                            break

                if not is_valid:
                    raise ValueError(
                        f"{name} must be one of {valid} or an instance of "
                        f"{tuple(v for v in valid if isinstance(v, type))}"
                    )

            super().__setattr__(name, value)

        def set_option(self, name, value):
            setattr(self, name, value)

        @classmethod
        def get_valid_options(cls):
            # Merge parent _valid_options dynamically
            merged = {}
            for base in reversed(cls.__mro__):
                if hasattr(base, "_valid_options"):
                    merged.update(base._valid_options)
            return merged

        @classmethod
        def get_immutable_options(cls):
            # Merge parent _valid_options dynamically
            merged = tuple()
            for base in reversed(cls.__mro__):
                if hasattr(base, "_immutable_options"):
                    merged += base._immutable_options
            return merged

    _scale_constrain_options = ("free", "identity", "diagonal", "correlation")

    def __init__(self,
                 process_dim: int,
                 innovation_dim: int,
                 scale_constrain: Literal["free", "identity", "diagonal", "correlation"],
                 advanced_options: Optional[AdvancedOptions] = None):

        super().__init__(process_dim, innovation_dim)

        self._advanced_options = advanced_options or self.AdvancedOptions()

        if not scale_constrain in self._scale_constrain_options:
            raise ValueError(f"Invalid scale_constrain: {scale_constrain}")
        self._scale_constrain: Literal["free", "identity", "diagonal", "correlation"] = scale_constrain
        self._cov_api = ConstrainedCovarianceAPI(shape=(self._innovation_dim, self._innovation_dim),
                                                 constrain=self._scale_constrain)

    @property
    def advanced_options(self):
        return self._advanced_options

    def set_advanced_option(self, name: str, value):
        """Set an advanced option by name."""
        if not hasattr(self._advanced_options, name):
            raise AttributeError(f"{name} is not a valid advanced option")
        setattr(self._advanced_options, name, value)

    @property
    def scale_constrain(self):
        return self._scale_constrain

    @property
    def is_innovation_cov_defined(self):
        return self._cov_api.is_defined

    @property
    def std(self):
        if not self._cov_api.is_defined:
            return None

        return self._cov_api.std

    @std.setter
    def std(self, value):
        self._std_setter(value, True)

    @property
    def cov(self):
        if not self._cov_api.is_defined:
            return None

        return self._cov_api.cov

    @cov.setter
    def cov(self, value):
        self._cov_setter(value, True)

    def _std_setter(self, value, validate: bool = True):
        if isinstance(value, float) or isinstance(value, int):
            value = np.array([value]).reshape(1, 1)

        if not isinstance(value, np.ndarray):
            raise TypeError("Std must be a numpy array")

        if value.ndim != 2:
            raise ValueError("Std must be a 2-dimensional array")

        if validate:
            self._cov_api.set_std(value)
        else:
            self._cov_api.set_std_trusted(value)

    def _cov_setter(self, value, validate: bool = True):
        if isinstance(value, float) or isinstance(value, int):
            value = np.array([value]).reshape(1, 1)

        if not isinstance(value, np.ndarray):
            raise TypeError("Std must be a numpy array")

        if value.ndim != 2:
            raise ValueError("Std must be a 2-dimensional array")

        if validate:
            self._cov_api.set_cov(value)
        else:
            self._cov_api.set_cov_trusted(value)

    def _cov_to_constrained_cov(self, cov: np.ndarray, validate: bool) -> np.ndarray:
        """
        Helper function for concrete subclasses to use to set the covariance matrix

        Args:
            cov: covariance matrix
            validate: (bool) flag indicating if the covariance matrix is checked

        Returns:
            covariance matrix conforming to self.scale_constrain

        """
        if validate:
            validate_covariance(cov)

        _scale_constrain_options = ("free", "identity", "diagonal", "correlation")

        if self._scale_constrain == 'identity':
            return np.eye(self._innovation_dim)
        elif self._scale_constrain == 'free':
            return cov
        elif self._scale_constrain == 'diagonal':
            return np.diag(np.diag(cov))
        elif self._scale_constrain == 'correlation':
            return covariance_to_correlation(cov)
        else:
            raise ValueError("Unknown scale constraint")

    @property
    def is_defined(self) -> bool:
        return self._cov_api.is_defined and self.is_dynamics_defined

    @property
    @abstractmethod
    def is_dynamics_defined(self) -> bool:
        raise NotImplementedError

    def _update_params(self, params: np.ndarray) -> None:
        n = self._cov_api.n_params
        self._cov_api.update_params(params[:n])
        self._update_dynamic_params(params[n:])

    def _get_params(self) -> np.ndarray:
        params = np.empty((self.n_params,))
        n = self._cov_api.n_params
        params[:n] = self._cov_api.get_params()
        params[n:] = self._get_dynamic_params()
        return params

    @property
    def _n_params(self) -> int:
        return self.n_dynamic_params + self._cov_api.n_params

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

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError


TProcesses = TypeVar(
    "TProcesses",
    bound=Tuple[LinearStateProcess, ...]
)

class CompositeLinearStateProcess(CompositeMixin[TProcesses], LinearStateProcess, Generic[TProcesses]):

    def __init__(self,
                 processes: TProcesses,
                 mixing_matrix: np.ndarray):


        CompositeMixin.__init__(self, processes)

        self._mixing_matrix = mixing_matrix

        innovation_dim = sum(lsp.innovation_dim for lsp in self._underlying_signals)

        LinearStateProcess.__init__(self,
                                    self._mixing_matrix.shape[0],
                                    innovation_dim,
                                    )

        self._check_innit()

        self._kf_init = None

    def _check_innit(self):
        if not isinstance(self._mixing_matrix, np.ndarray):
            raise ValueError("Mixing matrix must be an np.ndarray.")

        if not self._mixing_matrix.ndim == 2:
            raise ValueError("Mixing matrix must be an 2D np.ndarray.")

        for row in self.mixing_matrix:
            # idxs = row == 0
            # shapes = [self.underlying_processes[i].shape[0] for i in idxs]
            shapes = [
                self.underlying_processes[i].shape[0]
                for i, val in enumerate(row)
                if val == 0
            ]
            if len(set(shapes)) != 1:
                raise ValueError("Shape missmatch. Can not sum processes of different sizes.")

        shape_of_underlying = sum(lsp.shape[0] for lsp in self.underlying_processes)

        if not self._mixing_matrix.shape[1] == shape_of_underlying:
            raise ValueError("Shape missmatch. Mixing matrix with underlying processes")

    @property
    def is_dynamics_defined(self) -> bool:
        return all(lsp.is_dynamics_defined for lsp in self.underlying_processes)

    @property
    def mixing_matrix(self) -> np.ndarray:
        return self._mixing_matrix

    @property
    def n_underlying_processes(self) -> int:
        return len(self.underlying_processes)

    @property
    def underlying_processes(self) -> TProcesses:
        return self._underlying_signals

    @property
    def state_dim(self) -> int:
        return sum(lsp.state_dim for lsp in self.underlying_processes)

    @property
    def is_innovation_cov_defined(self) -> bool:
        return all(lsp.is_innovation_cov_defined for lsp in self.underlying_processes)

    @property
    def cov(self):
        return block_diag(*[lsp.cov for lsp in self.underlying_processes])

    @property
    def std(self):
        return block_diag(*[lsp.std for lsp in self.underlying_processes])

    @property
    def dynamic_representation(self) -> LinearStateProcessDynamics:
        reps = [process.dynamic_representation for process in self.underlying_processes]
        F = block_diag(*[rep.F for rep in reps])
        R = block_diag(*[rep.R for rep in reps])
        M = self.mixing_matrix @ block_diag(*[rep.M for rep in reps])
        return LinearStateProcessDynamics(F=F, R=R, M=M)

    def _first_fit_to(self, series: np.ndarray):
        raise NotImplementedError(
            "This estimator was initialized with incomplete parameters. "
            "Subclasses must implement _first_estimates() to support "
            "auto-estimation."
        )

# -----------------------------------------------------------------------------
# LinearStateSpaceModel
#
# A LinearStateSpaceModel represents a LinearStateProcess after a linear
# transformation A with additive noise e_t. That is, the process:
#
#     y_t = A f_t + e_t
#
# where f_t is a LinearStateProcess, A is a matrix, and e_t is additive noise.
# This is a purely mathematical object and does not bind to data.
#
# The LinearStateProcess is the primary abstraction.
# In typical usage, a LinearStateProcess constructs or returns an appropriate
# LinearStateSpaceModel when a state-space (linear + noise) representation is
# required, e.g. for filtering, smoothing, or likelihood evaluation.
#
# This class is rarely subclassed directly; new behavior should generally be
# introduced by defining new LinearStateProcess classes instead.
# -----------------------------------------------------------------------------
class LinearStateSpaceModel(CompositeMixin, Model):

    _measurement_noise_constraint_options = ("free", "zero", "diagonal")

    def __init__(self,
                 linear_state_process: "LinearStateProcess",
                 exposures: ConstrainedMatrix,
                 constant: ConstrainedMatrix,
                 measurement_noise_constrain: Literal["free", "zero", "diagonal"]):

        self._state_process = linear_state_process
        self._exposures_signal = exposures
        self._constant_signal = constant

        if not measurement_noise_constrain.lower() in self._measurement_noise_constraint_options:
            raise ValueError(f"Invalid measurement_noise_constrain: {measurement_noise_constrain}")
        self._measurement_noise_constrain = measurement_noise_constrain
        self._measurement_noise_signal = ConstrainedCovarianceAPI(shape=(None, None),
                                                                  constrain=self._measurement_noise_constrain)


        CompositeMixin.__init__(self, (linear_state_process, exposures, constant, self._measurement_noise_signal))
        Model.__init__(self, (None, 1),)

        self._check_shapes()

        self._kf_innit = None

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

            if not len(non_none_values):
                return None

            return non_none_values[0]

        process_shape = require_equal_or_none([self._state_process.shape[0], self._exposures_signal.shape[1]], True)
        self._exposures_signal.shape = (None, process_shape)
        self._state_process.shape = (process_shape, None)

        this_shape = require_equal_or_none([self._constant_signal.shape[0],
                                            self._exposures_signal.shape[0],
                                            self._measurement_noise_signal.shape[0],
                                            self._measurement_noise_signal.shape[1]
                                            ], False)

        if this_shape is not None:
            self._constant_signal.shape = (this_shape, 1)
            self._exposures_signal.shape = (this_shape, None)
            self._measurement_noise_signal.shape = (this_shape, this_shape)
            self.shape = (this_shape, 1)

    @property
    def state_process(self) -> "LinearStateProcess":
        return self._state_process

    @property
    def exposures(self) -> np.ndarray:
        return self._exposures_signal.matrix

    @exposures.setter
    def exposures(self, value) -> None:
        self._exposures_signal.matrix = value

    @property
    def constant(self) -> np.ndarray:
        return self._constant_signal.matrix

    @constant.setter
    def constant(self, value) -> None:
        self._constant_signal.matrix = value

    @property
    def measurement_noise_std(self) -> np.ndarray:
        return self._measurement_noise_signal.std

    @measurement_noise_std.setter
    def measurement_noise_std(self, value) -> None:
        self._measurement_noise_signal.set_std(value)

    @property
    def measurement_noise_cov(self) -> np.ndarray:
        return self._measurement_noise_signal.cov

    @measurement_noise_cov.setter
    def measurement_noise_cov(self, value) -> None:
        self._measurement_noise_signal.set_cov(value)

    @check_is_defined
    def representation(self, *args, **kwargs) -> LinearStateSpaceModelRepresentation:
        E = self.exposures
        LH = self.measurement_noise_std
        const = self.constant[:, 0]

        return LinearStateSpaceModelRepresentation.from_process_representation(
            process_representation=self.state_process.representation(),
            E=E, const=const, LH=LH)

    @check_is_defined
    def forecast(self, k: int, initial_state: np.ndarray) -> np.ndarray:
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self.state_process,
                                                   exposures=self.exposures,
                                                   measurement_noise_std=self.measurement_noise_std,
                                                   initial_state=initial_state)

    @check_is_defined
    def simulate(self, k: int, burn: int = 0):
        return LinearSSMUtils.simulate_or_forecast(time_domain=k,
                                                   latent_process=self.state_process,
                                                   exposures=self.exposures,
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
            objective: Union[OptimizationObjective, Literal["EM"]] = GaussianNLL()) -> LinearStateSpaceModelFit:

        n_series = series.shape[1]
        self._constant_signal.shape = (n_series, 1)
        self._exposures_signal.shape = (n_series, self.shape[0])
        self._measurement_noise_signal.shape = (n_series, n_series)

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
        kf = KalmanFilter().set_endog(series).set_representation(self.representation()).set_initialization(
            self._kf_innit)
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

            L = self.measurement_noise_std
            X = solve_triangular(L, np.eye(L.shape[0]), lower=True)
            H_inv = solve_triangular(L, X, lower=True, trans='T')

            # H_inv = np.linalg.inv(self.measurement_noise_std.matrix @ self.measurement_noise_std.matrix.T)

            next_latent_params = self._calc_em_next_latent_params(Aj, Bj, Cj, Ej, Fj, H_inv)
            next_exposure_params = self._calc_em_next_exposure_params(Ej, Fj, H_inv)

            Z = self.representation().E @ self.representation().M
            measurement_error_vars_new = np.diag((1 / series.shape[0]) * (Dj - Z @ Ej.T - Ej @ Z.T + Z @ Fj @ Z.T))

            self.state_process.update_params(next_latent_params)
            self._exposures_signal.update_params(next_exposure_params)
            self._measurement_noise_signal = np.sqrt(measurement_error_vars_new.reshape(self.measurement_noise_std.shape))

    def _calc_em_next_exposure_params(self, Ej, Fj, H_inv):

        current_exposure_params = self._exposures_signal.get_params()
        XI = self._exposures_signal.jacobian_vec_params(order='F')

        M = self.state_process.representation().M

        # L = self.measurement_noise_std.matrix
        # X = solve_triangular(L, Ej, lower=True)
        # H_inv_E = solve_triangular(L, X, lower=True, trans='T')

        gj = (H_inv @ Ej @ M.T).reshape((-1,), order='F')

        Gj = np.kron ((M @ Fj @ M.T).T, H_inv)

        step = np.linalg.inv(XI.T @ Gj @ XI) @ XI.T @ (gj - Gj @ self._exposures_signal.get_vectorized(order='F'))

        return current_exposure_params + step

    def _calc_em_next_latent_params(self, Aj, Bj, Cj, Ej, Fj, H_inv):

        repre = self.state_process.representation()
        M, F, R = repre.M, repre.F, repre.R

        # L = self.measurement_noise_std.matrix
        # exposures = self.exposures.matrix
        # B = solve_triangular(L, exposures.T, lower=True, trans='T').T
        # exposures_H_inv = solve_triangular(L, B.T, lower=True).T

        exposures = self.exposures
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
            repre = self.state_process.representation()
            repre = (repre.M, repre.F, repre.R)
            return np.hstack([m.flatten('F') for m in repre])

        jacobian = numerical_jacobian(func, current_latent_params)

        self.state_process.update_params(current_latent_params)

        return jacobian

    def _calc_em_next_latent_and_exposure_params(self, Bj, Cj, Ej, Fj):

        XI = self._dvecFZ_dparams()
        repre = self.representation()
        f_theta1 = np.hstack([m.flatten('F') for m in [repre.F, repre.E @ repre.M]])

        R = repre.R

        # theta_1, XI, f_theta1 = self.calc_dvecFZ_DexposuresArma(d_FIE, arma_params, exposures)
        H_inv = np.linalg.inv(self.measurement_noise_std @ self.measurement_noise_std.T)
        Gj1 = np.kron(Cj.T, R @ R.T)
        Gj2 = np.kron(Fj.T, H_inv)
        Gj = block_diag(Gj1, Gj2)
        g1j = (R @ R.T) @ Bj
        g2j = H_inv @ Ej
        gj = np.hstack([g1j.flatten(order='F'), g2j.flatten(order='F')])
        step = np.linalg.inv(XI.T @ Gj @ XI) @ XI.T @ (gj - Gj @ f_theta1)

        current_latent_params = self.state_process.get_params()
        n_latent = len(current_latent_params)
        current_exposure_params = self._exposures_signal.get_params()

        return current_latent_params + step[:n_latent], current_exposure_params + step[n_latent:]

    def _dvecFZ_dparams(self):

        current_latent_params = self.state_process.get_params()
        n_latent = len(current_latent_params)
        current_exposure_params = self._exposures_signal.get_params()

        current_params = np.hstack([current_latent_params, current_exposure_params])

        def func(x):
            self.state_process.update_params(x[:n_latent])
            self._exposures_signal.update_params(x[n_latent:])
            repre = self.representation()
            return np.hstack([m.flatten('F') for m in [repre.F, repre.E @ repre.M]])

        jacobian = numerical_jacobian(func, current_params)

        self.state_process.update_params(current_latent_params)
        self._exposures_signal.update_params(current_exposure_params)

        return jacobian

