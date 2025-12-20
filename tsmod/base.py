from abc import ABC, abstractmethod
from sympy.core.cache import cached_property
import warnings
from functools import wraps
from typing import Optional, Literal, Type, Union, ClassVar, Any
from collections.abc import Iterable

import numpy as np
from scipy.stats import norm

from optimization_objectives import OptimizationObjective


#-------------------
# Forecast classes
#------------------


def enforce_univariate(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.n_units > 1:
            raise NotImplementedError
        return func(self, *args, **kwargs)
    return wrapper


def check_is_single_horizon(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_single_horizon:
            raise NotImplementedError(f"Property defined only for single horizon forecasts.")
        return func(self, *args, **kwargs)

    return wrapper


class ForecastResult(ABC):
    has_uncertainty: ClassVar[bool]

    def __init__(self, forecast_path: np.ndarray, forecast_horizons: Optional[np.ndarray] = None):
        """
        - forecast_path: Array of forecasted values for each time horizon.
        """
        self._forecast_path = forecast_path  # Shape: (k, N)
        self._forecast_path.setflags(write=False)

        if self.forecast_path.ndim == 1:
            self._n_units = 1
        else:
            self._n_units = self._forecast_path.shape[1]

        if forecast_horizons is None:
            forecast_horizons = np.arange(1, self._forecast_path.shape[0] + 1)
        else:
            if any((not np.issubdtype(type(h), np.integer) or h <= 0) for h in forecast_horizons):
                raise ValueError("Horizon '{}' is not valid.".format(forecast_horizons))
            if len(forecast_horizons) != len(set(forecast_horizons)):
                raise ValueError(f"Forecast horizons contain duplicate values: {forecast_horizons}")
            elif not all(forecast_horizons[i] < forecast_horizons[i + 1] for i in range(len(forecast_horizons) - 1)):
                raise ValueError(f"Forecast horizons are not ordered: {forecast_horizons}")

        self._forecast_horizons = forecast_horizons
        self._forecast_horizons.setflags(write=False)

        forecast_horizon_index_map = {}
        for idx, value in enumerate(forecast_horizons):
            forecast_horizon_index_map[value] = idx
        self._forecast_horizon_index_map = forecast_horizon_index_map

        self._std_paths = None   # (k, N, N)

    # def __init_subclass__(cls, **kwargs):          # commented away cause its runtime check, and i dont like
    #     super().__init_subclass__(**kwargs)
    #     if not hasattr(cls, 'HAS_UNCERTAINTY'):
    #         raise TypeError(f"Class {cls.__name__} must define 'HAS_UNCERTAINTY'")

    def _check_valid_horizon(self, k):
        if isinstance(k, int):
            if k not in self.forecast_horizons:
                raise ValueError(f"Horizon '{k}' is not valid.")
        elif isinstance(k, Iterable):
            for h in k:
                if h not in self.forecast_horizons:
                    raise ValueError(f"Horizon '{h}' is not valid.")
        else:
            raise ValueError("Horizon 'k' must be an int or an iterable of integers.")

    @staticmethod
    def _check_valid_probability(q):
        if not (0 <= q <= 1):
            raise ValueError(f"Probability '{q}' is not valid.")

    def _get_horizon_indexes(self, k: Union[Iterable[int], int]):
        if not hasattr(k, '__iter__'):
            return self._forecast_horizon_index_map[k]
        return [self._forecast_horizon_index_map[value] for value in k]

    @property
    def forecast_path(self):
        return self._forecast_path

    @property
    def std_paths(self):
        if self._std_paths is None:
            self._std_paths = self._calc_std_paths()
        return self._std_paths

    @property
    def forecast_horizons(self) -> np.ndarray:
        return self._forecast_horizons

    @property
    def is_single_horizon(self) -> bool:
        return len(self.forecast_horizons) == 1

    @property
    def n_units(self) -> int:
        """Returns the number of cross-sectional units (N) in the panel data."""
        return self._n_units

    @property
    def is_univariate(self) -> bool:
        return self.n_units == 1

    def get_forecast(self, k: Iterable[int] | int):
        self._check_valid_horizon(k)
        """Returns the forecasted values (mean of the distribution, or the point forecast)."""
        if hasattr(k, '__iter__'):
            k = np.array(k)

        idx = self._get_horizon_indexes(k)
        return self.forecast_path[idx]

    @property
    @check_is_single_horizon
    def mean(self):
        return self.forecast_path[0]

    def get_std(self, k: Iterable[int] | int):
        self._check_valid_horizon(k)
        """Returns the forecasted values (mean of the distribution, or the point forecast)."""
        if hasattr(k, '__iter__'):
            k = np.array(k)

        if self._std_paths is None:
            self._std_paths = self._calc_std_paths()

        idx = self._get_horizon_indexes(k)
        return self._std_paths[idx]

    @property
    @check_is_single_horizon
    def std(self):
        if self._std_paths is None:
            self._std_paths = self._calc_std_paths()
        return self._std_paths[0]

    @abstractmethod
    def _calc_std_paths(self) -> np.ndarray:
        pass

    @enforce_univariate
    def quantile(self, q, k: Iterable[int] | int = 1):
        self._check_valid_probability(q)
        self._check_valid_horizon(k)
        return self._calc_quantile(q, k)

    def ppf(self, q, k: Iterable[int] | int = 1):
        return self.quantile(q, k)

    @abstractmethod
    def _calc_quantile(self, q, k):
        pass

    @enforce_univariate
    def cfd(self, value, k: Iterable[int] | int = 1):
        self._check_valid_horizon(k)
        return self._calc_cfd(value, k)

    @abstractmethod
    def _calc_cfd(self, value, k):
        pass

    def to_horizon(self, k: int) -> "ForecastResult":
        """
        Returns a new ForecastResult object for the specified horizon.
        """
        if k not in self.forecast_horizons:
            raise ValueError(f"Horizon {k} is not valid.")

        return self._to_horizon(k)

    @abstractmethod
    def _to_horizon(self, k: int) -> "ForecastResult":
        """
           Concrete classes must implement this method to transform
           the multi-horizon forecast into a single horizon forecast.
       """
        pass

    def to_unit(self, n: int) -> "ForecastResult":
        if not isinstance(n ,int) or not 0 <= n < self.n_units:
            raise ValueError(f"Unit {n} is not valid.")

        return self._to_unit(n)

    @abstractmethod
    def _to_unit(self, n: int) -> "ForecastResult":
        pass


class DeterministicForecastResult(ForecastResult):
    has_uncertainty = False

    def __init___(self, forecast_path: np.ndarray, forecast_horizons: Optional[np.ndarray] = None):
        super().__init__(forecast_path, forecast_horizons)

    def _to_horizon(self, k: int) -> "DeterministicForecastResult":
        mean = self.get_forecast(k).copy().reshape(1, self.n_units)
        return DeterministicForecastResult(mean, np.array([k]))

    def _calc_cfd(self, value, k):
        raise NotImplementedError("CFD not defined for deterministic forecast")

    def _calc_quantile(self, q, k):
        raise NotImplementedError("Quantile not defined for deterministic forecast")

    def _calc_std_paths(self) -> np.ndarray:
        raise NotImplementedError("Standard Deviation not defined for deterministic forecast")

    def _to_unit(self, n: int) -> "DeterministicForecastResult":
        return DeterministicForecastResult(self.forecast_path[:, n:n+1], self.forecast_horizons)


class NormalForecastResult(ForecastResult):
    has_uncertainty = True

    def __init__(self, forecast_path: np.ndarray, std_paths: np.ndarray, forecast_horizons: Optional[np.ndarray] = None):
        super().__init__(forecast_path, forecast_horizons)
        self._std_paths = std_paths

    def _calc_std_paths(self) -> np.ndarray:
        raise NotImplementedError  # this method should never be called, self._std_paths is defined in __init__

    def _calc_cfd(self, q: float, k: Union[Iterable[int] | int]):
        means = self.get_forecast(k)
        stds = self.get_std(k)
        cdf_values = []
        for i in range(means.shape[0]):
            cdf_values.append(norm.cdf(0, loc=means[i], scale=stds[i]))
        return cdf_values

    def _calc_quantile(self, q: float, k: Union[Iterable[int] | int]):
        idx = self._get_horizon_indexes(k)
        return self.forecast_path[idx] + self._std_paths[idx] * norm.ppf(0.05)

    def _to_horizon(self, k: int) -> "NormalForecastResult":
        mean = self.get_forecast(k).copy().reshape(1, self.n_units)
        std = self.get_std(k).copy().reshape(1, self.n_units, self.n_units)
        return NormalForecastResult(mean, std, np.array([k]))

    def _to_unit(self, n: int) -> "ForecastResult":
        if self.n_units == 1:
            return NormalForecastResult(self.forecast_path, self._std_paths, self.forecast_horizons)

        forecasts = self.forecast_path[:, n:n+1]
        stds = self._std_paths[:, n:n+1, n:n+1]

        return NormalForecastResult(forecasts, stds, self.forecast_horizons)


class BootstrapForecastResult(ForecastResult):
    has_uncertainty = True

    def __init__(self, bootstrap_forecasts: np.ndarray, forecast_horizons: Optional[np.ndarray] = None):
        super().__init__(np.mean(bootstrap_forecasts, axis=2), forecast_horizons)

        if self.is_univariate:
            bootstrap_forecasts = np.sort(bootstrap_forecasts, axis=2)

        self._bootstrap_forecasts = bootstrap_forecasts
        self._n_samples = bootstrap_forecasts.shape[2]

    @property
    def bootstrap_forecasts(self):
        return self._bootstrap_forecasts

    @property
    def n_samples(self):
        return self._n_samples

    def _calc_std_paths(self):
        stds = []
        for i in range(len(self.forecast_horizons)):
            cov_matrix = np.cov(self.bootstrap_forecasts[i, :, :], rowvar=True)
            L = np.linalg.cholesky(cov_matrix)
            stds.append(L)
        return np.array(stds)

    def _calc_cfd(self, value, k):
        idx = self._get_horizon_indexes(k)
        if not hasattr(idx, '__iter__'):
            return np.searchsorted(self.bootstrap_forecasts[idx, 0, :], value, side='right') / self.n_samples

        cfd_values = []
        for i in idx:
            cfd_values.append( np.searchsorted(self.bootstrap_forecasts[i, 0, :], value, side='right') / self.n_samples )
        return np.array(cfd_values)

    def _calc_quantile(self, q, k):
        idx = self._get_horizon_indexes(k)
        if not hasattr(idx, '__iter__'):
            return np.quantile(np.squeeze(self.bootstrap_forecasts[idx, :, :]), q)

        qq = []
        for i in idx:
            qq.append(np.quantile(np.squeeze(self.bootstrap_forecasts[i, :, :]), q))
        return np.array(qq)

    def _to_unit(self, n: int) -> "BootstrapForecastResult":
        print(n)
        return BootstrapForecastResult(self.bootstrap_forecasts[:, n:n+1, :], self.forecast_horizons)

    def _to_horizon(self, k: int) -> "BootstrapForecastResult":
        idx = self._get_horizon_indexes(k)
        return BootstrapForecastResult(self.bootstrap_forecasts[idx:idx+1, :, :], np.array([k]))


#-------------------
# Base Time-Series Model classes
#------------------


def check_is_defined(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_defined:
            warnings.warn("Signal not properly defined", UserWarning)
            return None
        return func(self, *args, **kwargs)
    return wrapper


def check_has_shape(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.has_shape:
            warnings.warn("Signal has no shape", UserWarning)
            return None
        return func(self, *args, **kwargs)
    return wrapper


class ModelFit(ABC):

    def __init__(self,
                 series: np.ndarray,
                 model: "Model",):

        """
        Initializes the model result object.

        Parameters:
        series (ndarray): The time series data associated with the model estimation.
        model (model): the underlying dynamic model
        """

        self._series = series
        self._model = model

    @property
    def series(self):
        return self._series

    @property
    def n_units(self) -> int:
        """Returns the number of cross-sectional units (N) in the panel data."""
        if self.series.ndim == 1:
            return 1
        return self.series.shape[1]

    @property
    def n_time_periods(self) -> int:
        """Returns the number of time periods (T) in the panel data."""
        return self.series.shape[0]

    @property
    def model(self):
        return self._model

    @cached_property
    def nll(self):
        return self._calc_nll()

    @abstractmethod
    def _calc_nll(self) -> float:
        raise NotImplementedError

    @cached_property
    def aic(self):
        """
        Calculates and returns the AIC (Akaike Information Criterion) for the model.
        This value is cached to avoid redundant calculations.
        """
        if self.nll is None:
            return None
        return 2 * self.model.n_params + 2 * self.nll

    @cached_property
    def nobs(self):
        return self.n_units * self.n_time_periods

    @cached_property
    def bic(self):
        """
        Calculates and returns the BIC (Bayesian Information Criterion) for the model.
        This value is cached to avoid redundant calculations.
        """
        if self.nll is None or self.model.n_params is None:
            return None
        return np.log(self.nobs) * self.model.n_params + 2 * self.nll

    @abstractmethod
    def forecast(self, *args, **kwargs) -> DeterministicForecastResult:
        """
        Forecasts future values (ahead by 'k' steps) for the time series.
        Subclasses should implement the specific forecast logic.
        """
        raise NotImplementedError

    @abstractmethod
    def forecast_with_uncertainty(self, *args, **kwargs) -> ForecastResult:
        raise NotImplementedError

    @abstractmethod
    def get_prediction_errors(self):
        """
        Returns the prediction errors for the entire series (or model output).
        Subclasses should implement the error computation logic.
        """
        raise NotImplementedError

    @abstractmethod
    def get_loss(self, loss: OptimizationObjective) -> float:
        raise NotImplementedError


class Signal(ABC):

    def __init__(self, shape: tuple, **signal_init):
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple, got {type(shape)}")
        if len(shape) != 2:
            raise ValueError(f"shape must be length 2, got length {len(shape)}")
        if not all(isinstance(x, (int, type(None))) for x in shape):
            raise TypeError("Each element of shape must be an int or None")

        self._is_frozen = False
        self._shape = shape
        self._signal_init = signal_init

    @property
    def is_frozen(self):
        return self._is_frozen

    def freeze(self):
        self._is_frozen = True

    def unfreeze(self):
        self._is_frozen = False

    # @property
    # def is_frozen(self):
    #     return self._get_is_frozen()
    #
    # @is_frozen.setter
    # def is_frozen(self, value):
    #     self._set_is_frozen(value)
    #
    # def _get_is_frozen(self):
    #     return getattr(self, "_is_frozen", False)
    #
    # def _set_is_frozen(self, value):
    #     super().__setattr__("_is_frozen", bool(value))

    def __setattr__(self, name, value):
        # Skip checks for the internal frozen flag itself
        if name != "_is_frozen" and getattr(self, "is_frozen", False):
            # Allow shape to be set only if unchanged
            if name == "shape" and getattr(self, "shape", None) != value:
                raise RuntimeError("Cannot modify shape of a frozen object.")
            # Block all other attribute changes
            elif name != "shape":
                raise RuntimeError(f"Cannot modify '{name}' of a frozen object.")
        super().__setattr__(name, value)

    @property
    @abstractmethod
    def is_defined(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self._shape

    def check_or_infer_shape(self, proposed_shape: tuple[int | None, int | None]):
        """
        Check the proposed shape against current shape, and infer missing dimensions if possible.

        Parameters
        ----------
        proposed_shape : tuple[int | None, int | None]
            The shape to check/infer.

        Returns
        -------
        tuple[int, int]
            Fully specified shape after inference.

        Raises
        ------
        ValueError
            If the proposed shape conflicts with current shape or cannot be inferred.
        """
        if not isinstance(proposed_shape, tuple) or len(proposed_shape) != 2:
            raise TypeError("proposed_shape must be a tuple of length 2")

        result = []
        for current, new in zip(self.shape, proposed_shape):
            if current is not None and new is not None:
                if current != new:
                    raise ValueError(f"Conflicting values: {current} vs {new}")
                result.append(current)
            else:
                result.append(current if current is not None else new)

        return tuple(result)

    @shape.setter
    def shape(self, value: tuple[int | None, int | None]):
        if not isinstance(value, tuple):
            raise TypeError(f"shape must be a tuple, got {type(value)}")
        if len(value) != 2:
            raise ValueError(f"shape must be length 2, got length {len(value)}")
        if not all(isinstance(x, (int, type(None))) for x in value):
            raise TypeError("Each element of shape must be an int or None")

        shape = self.check_or_infer_shape(value)

        self._shape = shape

    @property
    def has_shape(self):
        return all(i is not None for i in self.shape)

    @property
    def signal_initialization(self):
        return self._signal_init.copy()

    @property
    def n_params(self):
        return 0 if self.is_frozen else self._n_params

    @property
    def n_tot_params(self):
        return self._n_params

    @property
    @abstractmethod
    def _n_params(self) -> int:
        raise NotImplementedError

    @check_is_defined
    def get_params(self) ->  np.ndarray:
        return self._get_params()

    @abstractmethod
    def _get_params(self) -> np.ndarray:
        raise NotImplementedError

    def update_params(self, params: np.ndarray) -> None:
        if self.is_frozen or not self.has_shape:
            return
        self._update_params(params if params.flags.writeable else params.copy())

        # if self.is_frozen:
        #     raise warnings.warn("Object is frozen and can not be updated.", UserWarning)
        # elif not self.has_shape:
        #     raise warnings.warn("Object has no shape and can not be updated.", UserWarning)
        # else:
        #     if not params.flags.writeable:
        #         params = params.copy()
        #     self._update_params(params)

    @abstractmethod
    def _update_params(self, params: np.ndarray) -> None:
        raise NotImplementedError


class Model(Signal, ABC):

    def __init__(self, shape: tuple, **kwargs):
        super().__init__(shape, **kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs) -> ModelFit:
        raise NotImplementedError

    @abstractmethod
    def forecast(self, *args, **kwargs):
        raise NotImplementedError


class CompositeSignal(Signal, ABC):

    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)

    @property
    def is_frozen(self):
        comps = getattr(self, "_underlying_signals", None)
        if comps is None:
            # During initialization, consider the object unfrozen
            return False
        # return all(comp.is_frozen for comp in comps)
        return len(self._underlying_unfrozen_signals) == 0

    def freeze(self):
        for signal in self._underlying_unfrozen_signals:
            signal.freeze()

    def unfreeze(self):
        for signal in self._underlying_signals:
            signal.unfreeze()

    @property
    @abstractmethod
    def _underlying_signals(self) -> list[Signal]:
        raise NotImplementedError

    @property
    def _underlying_unfrozen_signals(self) -> list[Signal]:
        return [i for i in self._underlying_signals if not i.is_frozen]

    def _update_params(self, params: np.ndarray) -> None:
        idx_shift = 0
        for signal in self._underlying_unfrozen_signals:
            signal.update_params(params[idx_shift:idx_shift + signal.n_params])
            idx_shift += signal.n_params

    def _get_params(self) -> np.ndarray:
        params = np.empty((self.n_params,))

        idx_shift = 0
        for signal in self._underlying_unfrozen_signals:
            params[idx_shift:idx_shift + signal.n_params] = signal.get_params()
            idx_shift += signal.n_params
        return params

    @property
    def is_defined(self):
        return all(obj.is_defined for obj in self._underlying_signals)

    @cached_property
    def _n_params(self):
        n_params = 0
        for obj in self._underlying_unfrozen_signals:
            n_params += obj.n_params
        return n_params


class CompositeModel(CompositeSignal, Model, ABC):
    pass

