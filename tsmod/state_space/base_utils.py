from abc import ABC, abstractmethod
import warnings
from functools import wraps
from typing import Optional, Literal, Type, Union, ClassVar, Any
from collections.abc import Iterable
from scipy.stats import norm

import numpy as np
from sympy.core.cache import cached_property

from tools import losses

from base import Signal, SignalProcess, ForecastResult, ModelFit



class DomainSelector:

    def __init__(self,
                 signal: Type[Signal],
                 potential_domains: list[dict],
                 fit_options: dict,
                 series: np.ndarray,
                 exog: Optional[np.ndarray] = None,
                 criteria: Literal["aic", "bic", "forecast"] = "forecast",
                 forecast_settings=None,
                 check_inputs=True):

        if forecast_settings is None:
            forecast_settings = {}

        default_forecastSettings = {
            "ForecastType": "OOS",
            "ForecastHorizon": 1,
            "ForecastTestingRatio": 0.2
        }

        forecast_settings.update(
            {key: forecast_settings.get(key, value) for key, value in default_forecastSettings.items()})

        self.model = signal
        self.model_specifications = potential_domains
        self.fit_options = fit_options
        self.series = series
        self.exog = exog
        self.criteria = criteria
        self.forecast_settings = forecast_settings

    def select(self):

        if self.criteria == "forecast" and self.forecast_settings["ForecastType"] == "OOS":
            model_results, metrics = self._get_oos_forecast_metrics()
        else:
            model_results = []
            metrics = []
            for model_spec in self.model_specifications:
                model = self.model(**model_spec)
                modelRes = model.fit(series=self.series, exog=self.exog, **self.fit_options)
                model_results.append(modelRes)
                metrics.append(self._get_metric(modelRes))

        idx_min = np.argmin(metrics)

        return idx_min, self.model_specifications[idx_min], model_results[idx_min]

    def _get_metric(self, model_result: ModelFit):

        if self.criteria == "aic":
            return model_result.aic
        elif self.criteria == "bic":
            return model_result.bic
        elif self.criteria == "forecast" and self.forecast_settings["ForecastType"] == "IS":
            forecast_horizons = self.forecast_settings["ForecastHorizon"]
            if not isinstance(forecast_horizons, Iterable):
                forecast_horizons = [forecast_horizons]

            sum_Lnorm = sum(losses.squared_error(model_result.get_k_ahead_prediction_errors(k))
                            for k in forecast_horizons)

            return sum_Lnorm
        else:
            warnings.warn("Cannot get metric in Unobserved components model selector, check criteria", UserWarning)
            return np.inf

    def _get_oos_forecast_metrics(self):

        n = self.series.shape[0]
        forecast_testing_ratio = self.forecast_settings["ForecastTestingRatio"]
        forecast_horizons = self.forecast_settings["ForecastHorizon"]
        if not isinstance(forecast_horizons, Iterable):
            forecast_horizons = [forecast_horizons]
        n_test = int(round(n * forecast_testing_ratio))
        n_train = n - n_test

        model_results = []
        metrics = []

        for model_spec in self.model_specifications:
            model = self.model(**model_spec)
            model_res = model.fit(series=self.series, exog=self.exog, **self.fit_options)

            if not model_res.success:
                model_results.append(None)
                metrics.append(np.inf)
                continue

            flag = True
            loss = 0
            for i, idx in enumerate(range(n_train, n - forecast_horizons[-1])):
                model_res = model.fit(series=self.series[0:idx],
                                      exog = self.exog,
                                      first_estimate=model_res,
                                      **self.fit_options)

                if not model_res.success:
                    flag = False
                    break

                y_pred = model_res.forecast(forecast_horizons).forecast_path

                forecasted_idx = np.array(forecast_horizons)
                errors = self.series[idx + forecasted_idx] - y_pred[forecasted_idx - 1]

                loss += losses.squared_error(errors)

            if not flag:
                model_results.append(None)
                metrics.append(np.inf)
                continue

            model_res = model.fit(series=self.series,
                                  first_estimate=model_res,
                                  **self.fit_options)

            model_results.append(model_res)
            metrics.append(loss)

        return model_results, metrics


