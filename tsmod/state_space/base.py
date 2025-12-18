from abc import ABC, abstractmethod
import warnings
from functools import wraps
from typing import Optional, Literal, Type, Union, ClassVar, Any
from collections.abc import Iterable
from scipy.stats import norm

import numpy as np
from sympy.core.cache import cached_property

# from tsmod.base import Domain, Process, ModelFit
# from tsmod.optimization_objectives import OptimizationObjective
# from tsmod.deterministic import Deterministic, DeterministicProcess
# from tsmod.utils import ConstrainedMatrixWrapper

from ..base import ModelFit, Model, check_is_defined, CompositeSignal
from ..base import DeterministicForecastResult, ForecastResult

from optimization_objectives import OptimizationObjective
from deterministic import DeterministicProcess
from constrained_matrices import ConstrainedMatrix

# Note: This library differs from statsmodels in 1 core concept:
#       This is library is built on a "signal-first" philosophy:
#           The central object  is not the data; it is the signal process. Three main objects arise:
#
#           1. SignalProcess -> This is a mathematical object purely, no data attached.
#                               It describes the dynamics of a (generally latent) signal
#                               e.g. an ARIMA process with defined AR and MA polynomials and integration order is a SignalProcess
#                               These classes implement simulate()
#           2. Signal        -> These specify the SignalsProcess' hyperparameters and constrains.
#                               In effect, it defines a space of SignalProcesses. A domain in the "SignalProcess space",
#                                   while SignalProcess is a single point in this domain
#                               e.g. an ARIMA(p, i, q) process with defined (p, i, q) \in N_0^3 is a Signal
#                               These classes implement fit() -> "ModelFit"
#                               The short naming "Signal" stems from the fact that a user would likely fit some
#                                   data to a signal in this space, e.g. model = ARIMA(3,1,0); model.fit(data, ...)
#           3. ModelFit      -> These classes contain a SignalProcess and a mapping Signal |-> data
#                               These classes implement forecast()
#
#
#   STAGE 1: Linear State Space Models
#
#           y_t - A g(t) - B u_t = E f_t + L w_t,           w_t ~ iid N(0,I),
#           f_t                  = M x_t,
#           x_t                  = F x_{t-1} + R e_t,       e_t ~ iid N(0,I),
#
#   STAGE 2: Non-Linear and Score-Driven State Space Models
#
#           y_t - A g(t) - B u_t = J(f_t) + D(f_t) + L w_t,   D(f_t) ~ Distribution, w_t ~ iid N(0,I)
#           f_t                  = M x_t,
#           x_t                  = c(x_{:t-1}, w_{:t-1}) +
#                                  + F(x_{:t-1}, w_{:t-1}) x_{t-1} +
#                                  + R(x_{:t-1}, w_{:t-1}) e_t,  e_t ~ iid N(0,I)
#
#   STAGE 3: Generalized Nonlinear State-Space Models (Exogenous and Feedback-Driven)
#
#           y_t - A g(t) - B u_t = J(f_t) + D(f_t) + L w_t,   D(f_t) ~ Distribution, w_t ~ iid N(0,I)
#           f_t                  = M x_t
#           x_t                  = c(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) +
#                                  + F(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) x_{t-1} +
#                                  + R(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) e_t,             e_t ~ iid N(0,I)
#
#                           where h_t, u_t is any information available before observation t
#
#


#-------------------
# SMM base classes
#------------------


class StateSpaceModelFit(ModelFit, ABC):

    def __init__(self,
                 series: np.ndarray,
                 model: "StateSpaceModel"):
        super().__init__(series, model)

    @abstractmethod
    def _calc_nll(self) -> float:
        pass

    @abstractmethod
    def get_prediction_errors(self):
        pass

    @abstractmethod
    def forecast_with_uncertainty(self, *args, **kwargs) -> ForecastResult:
        pass

    @abstractmethod
    def forecast(self, *args, **kwargs) -> DeterministicForecastResult:
        pass

    @abstractmethod
    def get_loss(self, loss: OptimizationObjective) -> float:
        pass


class StateSpaceModel(CompositeSignal):

    #   STAGE 3: Generalized Nonlinear State-Space Models (Exogenous and Feedback-Driven)
    #
    #           y_t - A g(t) - B u_t = E J(f_t) + L w(f_t),   w(f_t) ~ N(0, H(f_t}))
    #           f_t                  = M x_t
    #           x_t                  = c(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) +
    #                                  + F(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) x_{t-1} +
    #                                  + R(x_{:t-1}, y_{:t-1}, h_{:t}, w_{:t-1}) e_t,             e_t ~ iid N(0,I)
    #
    #                           where h_t, u_t is any information available before observation t

    def __init__(self,
                 state_process: "StateProcess",
                 exposures: ConstrainedMatrix,
                 deterministic: Optional[DeterministicProcess] = None,
                 deterministic_loadings: Optional[ConstrainedMatrix] = None,
                 exog: Optional[np.ndarray] = None,
                 exog_loadings: Optional[ConstrainedMatrix] = None,
                 observation_noise_std: Optional[ConstrainedMatrix] = None):

        self._state_process = state_process
        self._exposures = exposures
        self._deterministic = deterministic
        self._deterministic_loadings = deterministic_loadings
        self._exog = exog
        self._exog_loadings = exog_loadings
        self._observation_noise_std = observation_noise_std

        self._check_init()
        obs_shape = self._check_shapes()
        super().__init__((obs_shape, 1),)


    def _check_shapes(self):

        obs_shapes = []

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


        # TODO: its not checking exposures and process. rn i dont care idk what the future of this is

        if self.exog is not None:
            require_equal_or_none([self._exog.shape[1], self.exog_loadings.shape[1]], True)
            obs_shapes.append(self.exog_loadings.shape[0])

        if self.deterministic is not None:
            require_equal_or_none([self.deterministic.shape[0], self.deterministic_loadings.shape[1]], True)
            obs_shapes.append(self.deterministic_loadings.shape[0])

        if self._observation_noise_std is not None:
            obs_shapes.append(self._observation_noise_std.shape[0])

        obs_shapes.append(self.state_process.shape[0])

        obs_shape = require_equal_or_none(obs_shapes, False)

        return obs_shape

    def _check_init(self):

        if (self.deterministic is None) != (self.deterministic_loadings is None):
            raise ValueError(
                "deterministic and deterministic_loadings must both be None or both not None."
            )

        if (self.exog is None) != (self.exog_loadings is None):
            raise ValueError(
                "deterministic and deterministic_loadings must both be None or both not None."
            )

    @property
    def _underlying_signals(self):
        signals = [self.state_process]
        if self.deterministic is not None:
            signals.append(self.deterministic)
            signals.append(self.deterministic_loadings)
        if self.exog is not None:
            signals.append(self.exog_loadings)

        return signals

    @property
    def state_process(self) -> "StateProcess":
        return self._state_process

    @property
    def exposures(self) -> Optional[ConstrainedMatrix]:
        return self._exposures

    @property
    def deterministic(self) -> Optional[DeterministicProcess]:
        return self._deterministic

    @property
    def deterministic_loadings(self) -> Optional[ConstrainedMatrix]:
        return self._deterministic_loadings

    @property
    def exog(self) -> Optional[np.ndarray]:
        return self._exog

    @property
    def exog_loadings(self) -> Optional[ConstrainedMatrix]:
        return self._exog_loadings

    @property
    def state_representation(self):
        return self.state_process.representation

    @abstractmethod
    def forecast(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, series: np.ndarray, objective: Union[OptimizationObjective, Any]) -> StateSpaceModelFit:
        raise NotImplementedError



class StateProcessRepresentation(ABC):
    pass


class StateProcess(Model, ABC):

    def __init__(self, shape: tuple, **kwargs):
        super().__init__(shape, **kwargs)

    @property
    @abstractmethod
    def is_defined(self):
        raise NotImplementedError

    @abstractmethod
    def _update_params(self, params: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_params(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _state_space_model(self) -> StateSpaceModel:
        raise NotImplementedError

    @check_is_defined
    def forecast(self, *args, **kwargs):
        return self._state_space_model().forecast(*args, **kwargs)

    @check_is_defined
    def simulate(self, time_steps: int, burn: int = 0):
        return self._state_space_model().simulate(time_steps, burn)

    @check_is_defined
    def fit(self, series: np.ndarray, objective: Union[OptimizationObjective, Any]) -> StateSpaceModelFit:
        return self._state_space_model().fit(series, objective)

    @abstractmethod
    def representation(self, *args, **kwargs) -> StateProcessRepresentation:
        raise NotImplementedError

    @abstractmethod
    def _first_fit_to(self, series: np.ndarray):
        raise NotImplementedError



