from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Literal
from functools import cached_property  # wraps
from enum import Enum, auto

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import minimize

from optimization_objectives import OptimizationObjective, GaussianNLL

from constrained_matrices import (ConstrainedMatrix,
                                  UnitTopRowConstrained as UnitTopRowMatrix,
                                  FreeMatrix,
                                  PosDiagonalMatrix,
                                  IdentityMatrix,
                                  DiagonalMatrix,
                                  ZeroMatrix,
                                  STDMatrix,
                                  CorrMatrix,
                                  ElementWiseConstrainedMatrix,
                                  ConstrainedCovarianceAPI)


from base import Signal, Model, ModelFit, CompositeSignal, CompositeModel, check_is_defined
from base import ForecastResult, DeterministicForecastResult, NormalForecastResult
from tools.utils import validate_covariance, validate_chol_factor
from state_space.tools.kalman_filter import (KalmanFilter,
                                             KalmanFilterResult,
                                             KalmanFilterInitialization)

from representation import LinearStateSpaceModelRepresentation, LinearStateProcessRepresentation

from utils import numerical_jacobian

# TODO: HIGH PRIORITY
#       1. Add support for changing the correlation matrix parameterization after it is defined, or dont allow it to happen
#       2. Change EM, currently its wrong because the representation was changed

# TODO: MEDIUM PRIORITY
#       1. Add adaptive step to EM optimization (should take very little effort)
#       2. the representation structure does nothing now. so the design is not super clear.
#       3. Add Gauge Symmetries. Not using right now so I will skip for simplocity
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

        rep = latent_process.representation()
        M, F, RLQ = rep.M, rep.F, rep.R @ rep.LQ

        if initial_state is not None:
            latent_stochastic = np.zeros((T, state_dim))
            eta = np.random.randn(T, state_dim)
            latent_stochastic[0] = F @ initial_state + RLQ @ eta[0]
            for t in range(1, T):
                latent_stochastic[t] = F @ latent_stochastic[t - 1] + RLQ @ eta[t]
            latent_stochastic = latent_stochastic @ M.T
        else:
            latent_stochastic = np.zeros((T + burn, state_dim))
            eta = np.random.randn(T + burn, state_dim)
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
            raise NotImplementedError("Thus far, losses that require cov are not implemented")
        else:
            return loss(self.get_prediction_errors())

# ---------
# "Latent" process, and composite "latent" process
# ---------

@dataclass(frozen=True)
class RotationalSymmetry:
    indices: tuple[int, ...]


class StructuralVariability(Enum):
    FIXED = auto()
    PARAMETRIC = auto()
    FREE = auto()


@dataclass(frozen=True)
class RepresentationStructure:
    M: StructuralVariability
    F: StructuralVariability
    R: StructuralVariability


class LinearStateProcess(Model, ABC):

    class AdvancedOptions:
        """Holds advanced, context-specific options."""

        # to provide IDE help:
        correlation_parameterization: Literal["hyperspherical", "log"]

        _valid_options = {
            "correlation_parameterization": ["hyperspherical", "log"]
        }

        _immutable_options = ("correlation_parameterization",)

        def __init__(self, **kwargs):
            # Initialize options, validated through __setattr__
            n_options = len(self.get_valid_options())
            if not len(kwargs) == n_options:
                raise ValueError("Incorrect number of arguments. All options need to be filled")
            for name, value in kwargs.items():
                setattr(self, name, value)

        def __setattr__(self, name, value):
            # check that attribute is not immutable. second clause allows initialization
            if name in self.get_immutable_options() and name in self.__dict__ :
                raise ValueError(f"{name} is an immutable option. I.e., it can only be set at init")

            # Validate known options
            valid = self.get_valid_options().get(name)
            if valid is not None and value not in valid:
                raise ValueError(f"{name} must be one of {valid}")
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
                 shape: tuple,
                 innovation_dim: int,
                 scale_constrain: Literal["free", "identity", "diagonal", "correlation"],
                 advanced_options: Optional[AdvancedOptions] = None):
        super().__init__(shape)

        self._advanced_options = advanced_options or self.AdvancedOptions()

        self._innovation_dim = innovation_dim
        if not scale_constrain.lower() in self._scale_constrain_options:
            raise ValueError(f"Invalid scale_constrain: {scale_constrain}")
        self._scale_constrain = scale_constrain.lower()
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
    def innovation_dim(self) -> int:
        """Return the dimension of the model innovations."""
        return self._innovation_dim

    @property
    def scale_constrain(self):
        return self._scale_constrain

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
        if validate:
            self._cov_api.set_std(value)
        else:
            self._cov_api.set_std_trusted(value)

    def _cov_setter(self, value, validate: bool = True):
        if validate:
            self._cov_api.set_cov(value)
        else:
            self._cov_api.set_cov_trusted(value)

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    def is_defined(self) -> bool:
        return self._cov_api.is_defined and self.is_dynamics_defined

    @property
    @abstractmethod
    def is_dynamics_defined(self) -> bool:
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def representation_structure(self) -> RepresentationStructure:
        raise NotImplementedError

    def _update_params(self, params: np.ndarray) -> None:
        n = self._cov_api.n_params
        self._cov_api.update_params(params[:n])
        self._update_dynamic_params(params[n:])

    def _get_params(self) -> np.ndarray:
        params = np.empty((self.n_params,))
        n = self._cov_api.n_params
        params[:n] = self._cov_api.get_params()
        params[n.n_params:] = self._get_dynamic_params()
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

    @cached_property
    @abstractmethod
    def rotational_symmetries(self) -> list[RotationalSymmetry]:
        """
        Declare continuous rotational invariances of the latent state.

        Each symmetry must specify the latent block on which
        orthogonal transformations leave the likelihood invariant.
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

        # TODO: add estimation of factor models with to_identifiable_ssm

        n_series = series.shape[1]

        if n_series != self.shape[0]:
            raise ValueError("Series has wrong shape")

        if not self.is_defined:
            self._first_fit_to(series)

        ssm = self.get_identifiable_state_space_model(include_constant=include_constant,
                                                      measurement_noise=measurement_noise)

        if include_constant:
            ssm.constant = np.mean(series).reshape(-1, 1)
        else:
            ssm.constant = np.zeros((n_series, 1))

        if measurement_noise == "diagonal":
            ssm.measurement_noise_std = np.diag(np.sqrt(np.std(series, axis=0) * 0.5))
        elif measurement_noise == "full":
            ssm.measurement_noise_std = np.linalg.cholesky(np.cov(series) * 0.5)
        elif measurement_noise == "zero":
            ssm.measurement_noise_std = np.zeros((n_series, n_series))

        ssm._exposures_signal = IdentityMatrix((n_series, n_series))

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


    def get_identifiable_state_space_model(self,
                             include_constant: bool,
                             measurement_noise: Literal["free", "zero", "diagonal"]) -> "LinearStateSpaceModel":

        # CONSIDER: LinearStateProcess can implement some properties to say that scale or rotation are fixed
        scale_map = {
            'free': (False, False),
            'identity': (True, False),
            'diagonal': (False, True),
            'correlation': (True, False)
        }

        try:
            innovation_noise_fixed_scale, innovation_noise_fixed_rotation = scale_map[self._scale_constrain]
        except KeyError:
            raise NotImplementedError("unknown scale constraint")

        rot_symmetries = self._declare_and_enforce_disjoint_rotational_symmetry # returns a list of RotationalSymmetry objects, each if a tuple of indexe

        # Build exposure constraints
        exposure_constraints = {}

        # Constrain first row if scale is not fixed
        if not innovation_noise_fixed_scale:
            for j in range(self.shape[0]):
                exposure_constraints[(0, j)] = 1

        # Apply rotational symmetry constraints
        if not innovation_noise_fixed_rotation:
            for rot_block in rot_symmetries:
                for i, idx in enumerate(rot_block.indices):
                    # Zero out entries above diagonal
                    for j in range(i):
                        exposure_constraints[(j, idx)] = 0
                    # Set diagonal to 1 if scale is not fixed
                    if not innovation_noise_fixed_scale:
                        exposure_constraints[(i, idx)] = 1

        exposures = (ElementWiseConstrainedMatrix((None, self.shape[0]), exposure_constraints)
                     if len(exposure_constraints) else
                     FreeMatrix((None, self.shape[0])))

        constant = FreeMatrix((None, 1)) if include_constant else ZeroMatrix((None, 1))


        ssm = LinearStateSpaceModel(linear_state_process=self,
                                    exposures=exposures,
                                    constant=constant,
                                    measurement_noise_constrain=measurement_noise)

        return ssm

    @cached_property
    def _declare_and_enforce_disjoint_rotational_symmetry(self):
        """
        Checks that the rotational symmetry blocks declared in
        `_declare_rotational_symmetries` are disjoint.

        Raises:
            ValueError: if any blocks overlap.
        """
        rotational_symmetries = self.rotational_symmetries

        index_to_block = {}

        for rot in rotational_symmetries:
            for idx in rot.indices:
                if idx in index_to_block:
                    first_block = index_to_block[idx]
                    raise ValueError(
                        f"Index {idx} appears in multiple rotational symmetry blocks: "
                        f"{first_block} and {rot.indices}. Blocks must be disjoint."
                    )
                index_to_block[idx] = rot.indices

        return rotational_symmetries

    def _build_state_space_model(self,
                                 series,
                                 include_constant: bool,
                                 measurement_noise: Optional[Literal["zero", "diagonal", "full"]]) -> "LinearStateSpaceModel":

        ssm = self.get_identifiable_state_space_model(include_constant, measurement_noise)


class CompositeLinearStateProcess(CompositeModel, LinearStateProcess):

    def __init__(self,
                 processes: list[LinearStateProcess],
                 mixing_matrix: np.ndarray):
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

    @cached_property
    def rotational_symmetries(self) -> list[RotationalSymmetry]:
        """
        Generally needs to be overwritten.

        Default joins the underlying rotational symmetries, correcting for the index shift

        The default implementation propagates rotational symmetries from underlying latent processes by index shifting.
        This captures all symmetries inherited from individual components.
        Additional symmetries introduced or broken by mixing must be handled by overriding this method.
        """
        joint_rot_syms = []
        latent_dim_shift = 0
        for lsp in self._underlying_processes:
            lsp_rot_syms = lsp.rotational_symmetries
            for rot_sym in lsp_rot_syms:
                indexes = tuple(i + latent_dim_shift for i in rot_sym.indices)
                joint_rot_syms.append(RotationalSymmetry(indexes))
            latent_dim_shift += lsp.state_dim

        return joint_rot_syms

    @property
    def is_dynamics_defined(self) -> bool:
        return all(lsp.is_dynamics_defined for lsp in self._underlying_processes)

    def _update_dynamic_params(self, params: np.ndarray):
        idx_shift = 0
        for lsp in self._underlying_processes:
            n_par = lsp.n_dynamic_params
            lsp._update_dynamic_params(params[idx_shift + n_par])
            idx_shift += n_par

    def _get_dynamic_params(self) -> np.ndarray:
        n_dyn_par = sum(lsp.n_dynamic_params for lsp in self._underlying_processes)
        dyn_params = np.empty((n_dyn_par,))
        idx_shift = 0
        for lsp in self._underlying_processes:
            n_par = lsp.n_dynamic_params
            dyn_params[idx_shift:idx_shift + n_par] = lsp._get_dynamic_params()
            idx_shift += n_par
        return dyn_params

    @property
    def n_dynamic_params(self) -> int:
        return sum(lsp.n_dynamic_params for lsp in self._underlying_processes)

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

    def representation_structure(self) -> RepresentationStructure:
        structs = [lp.representation_structure for lp in self._underlying_processes]

        def composite_struc(structs, attribute):
            if all(getattr(struct, attribute) == StructuralVariability.FIXED for struct in structs):
                return StructuralVariability.FIXED
            elif all(getattr(struct, attribute) == StructuralVariability.FREE for struct in structs):
                return StructuralVariability.FREE
            else:
                return StructuralVariability.PARAMETRIC

        res = RepresentationStructure(
            M=composite_struc(structs, 'M'),
            F=composite_struc(structs, 'F'),
            R=composite_struc(structs, 'R'),
        )

        return res

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
#     y_t = A x_t + e_t
#
# where x_t is a LinearStateProcess, A is a matrix, and e_t is additive noise.
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
class LinearStateSpaceModel(CompositeModel):

    def __init__(self,
                 linear_state_process: "LinearStateProcess",
                 exposures: ConstrainedMatrix,
                 constant: ConstrainedMatrix,
                 measurement_noise_constrain: Literal["free", "zero", "diagonal"]):

        super().__init__((None, 1), )

        self._state_process = linear_state_process
        self._exposures_signal = exposures
        self._constant_signal = constant
        self._measurement_noise_signal = ConstrainedCovarianceAPI(shape=(None, None), constrain=measurement_noise_constrain)

        self._check_shapes()

        self._kf_innit = KalmanFilterInitialization(initialization_type="ss",
                                                    x0=np.zeros(self._state_process.state_dim,),
                                                    P0=None,
                                                    P_star=None,
                                                    P_infty=None)


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
    def _underlying_signals(self) -> list[Signal]:
        return [self._state_process, self._exposures_signal, self._constant_signal, self._measurement_noise_signal]

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
            objective: Union[OptimizationObjective, Literal["EM"]]) -> LinearStateSpaceModelFit:

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

