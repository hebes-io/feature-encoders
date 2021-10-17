# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from ..utils import as_list, check_X, get_datetime_data

#####################################################################################
# Add new features
#
# All feature generators generate pandas DataFrames
#####################################################################################


class TrendFeatures(TransformerMixin, BaseEstimator):
    """Generate linear time trend features.

    Args:
        ds (str, optional): The name of the input dataframe's column that contains
            datetime information. If None, it is assumed that the datetime information
            is provided by the input dataframe's index. Defaults to None.
        name (str, optional): The name of the generated dataframe's column. Defaults to 'growth'.
        remainder ({'drop', 'passthrough'}, optional): By specifying ``remainder='passthrough'``,
            all the remaining columns of the input dataset will be automatically passed through
            (concatenated with the output of the transformer), otherwise, they will be dropped.
            Defaults to "passthrough".
        replace (bool, optional): Specifies whether replacing an existing column with the same
            name is allowed (applicable when `remainder=passthrough`). Defaults to False.

    Raises:
        ValueError: If ``remainder`` is neither 'drop' nor 'passthrough'.
    """

    def __init__(self, ds=None, name="growth", remainder="passthrough", replace=False):
        if remainder not in ("passthrough", "drop"):
            raise ValueError("Parameter `remainder` should be 'passthrough' or 'drop'")

        self.ds = ds
        self.name = name
        self.remainder = remainder
        self.replace = replace

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            TrendFeatures: Fitted encoder.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
        """
        X = check_X(X)
        dates = X.index.to_frame() if self.ds is None else X[[self.ds]]
        self.t_scaler_ = MinMaxScaler().fit(dates)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dates = X.index.to_frame() if self.ds is None else X[[self.ds]]

        out = pd.DataFrame(
            data=self.t_scaler_.transform(dates),
            columns=[self.name],
            index=X.index,
        )

        if self.remainder == "passthrough":
            if (self.name in X.columns) and not self.replace:
                raise ValueError(f"Found common column name: {self.name}")
            elif self.name in X.columns:
                X = X.drop(self.name, axis=1)
            out = pd.concat((X, out), axis=1)

        return out


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """Generate date and time features.

    Args:
        ds (str, optional): The name of the input dataframe's column that contains
            datetime information. If None, it is assumed that the datetime information
            is provided by the input dataframe's index. Defaults to None.
        remainder ({'drop', 'passthrough'}, optional): By specifying ``remainder='passthrough'``,
            all the remaining columns of the input dataset will be automatically passed through
            (concatenated with the output of the transformer), otherwise, they will be dropped.
            Defaults to "passthrough".
        replace (bool, optional): Specifies whether replacing an existing column with the same
            name is allowed (applicable when `remainder=passthrough`). Defaults to False.
        subset (str or list of str, optional): The names of the features to generate. If
            None, all features will be produced: 'month', 'week', 'dayofyear', 'dayofweek',
            'hour', 'hourofweek'. The last 2 features are generated only if the timestep of
            the input's `ds` (or index if `ds` is None) is smaller than `pandas.Timedelta(days=1)`.
            Defaults to None.

    Raises:
        ValueError: If ``remainder`` is neither 'drop' nor 'passthrough'.
    """

    def __init__(self, ds=None, remainder="passthrough", replace=False, subset=None):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.ds = ds
        self.remainder = remainder
        self.replace = replace
        self.subset = subset

    def _get_all_attributes(self, dt_column):
        attr = ["month", "week", "dayofyear", "dayofweek"]

        dt = dt_column.diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()
        if time_step < pd.Timedelta(days=1):
            attr = attr + ["hour", "hourofweek"]

        if self.subset is not None:
            attr = [i for i in attr if i in as_list(self.subset)]
        return attr

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            DatetimeFeatures: Fitted encoder.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
        """
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.ds)
        self.attr_ = self._get_all_attributes(dt_column)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.ds)

        out = {}
        for n in self.attr_:
            if n == "week":
                out[n] = (
                    dt_column.dt.isocalendar().week.astype(dt_column.dt.day.dtype)
                    if hasattr(dt_column.dt, "isocalendar")
                    else dt_column.dt.week
                )
            elif n == "hourofweek":
                out[n] = None
            else:
                out[n] = getattr(dt_column.dt, n)

        if "hourofweek" in out:
            out["hourofweek"] = 24 * out.get(
                "dayofweek", dt_column.dt.dayofweek
            ) + out.get("hour", dt_column.dt.hour)

        out = pd.DataFrame.from_dict(out)

        if self.remainder == "passthrough":
            common = list(set(X.columns) & set(out.columns))
            if common and not self.replace:
                raise ValueError(f"Found common column names {common}")
            elif common:
                X = X.drop(common, axis=1)
            out = pd.concat((X, out), axis=1)

        return out


class CyclicalFeatures(TransformerMixin, BaseEstimator):
    """Create cyclical (seasonal) features as fourier terms.

    Args:
        seasonality (str): The name of the seasonality. The feature generator can provide
            default values for ``period`` and ``fourier_order`` if ``seasonality`` is one
            of 'daily', 'weekly' or 'yearly'.
        ds (str, optional): The name of the input dataframe's column that contains
            datetime information. If None, it is assumed that the datetime information
            is provided by the input dataframe's index. Defaults to None.
        period (float, optional): Number of days in one period. Defaults to None.
        fourier_order (int, optional): Number of Fourier components to use. Defaults to None.
        remainder ({'drop', 'passthrough'}, optional): By specifying ``remainder='passthrough'``,
            all the remaining columns of the input dataset will be automatically passed through
            (concatenated with the output of the transformer), otherwise, they will be dropped.
            Defaults to "passthrough".
        replace (bool, optional): Specifies whether replacing an existing column with the same
            name is allowed (applicable when `remainder=passthrough`). Defaults to False.

    Raises:
        ValueError: If ``remainder`` is neither 'drop' nor 'passthrough'.
    """

    def __init__(
        self,
        *,
        seasonality,
        ds=None,
        period=None,
        fourier_order=None,
        remainder="passthrough",
        replace=False,
    ):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.seasonality = seasonality
        self.ds = ds
        self.period = period
        self.fourier_order = fourier_order
        self.remainder = remainder
        self.replace = replace

    @staticmethod
    def _fourier_series(dates, period, order):
        # convert to days since epoch
        t = np.array(
            (dates - datetime(2000, 1, 1)).dt.total_seconds().astype(np.float64)
        ) / (3600 * 24.0)

        return np.column_stack(
            [
                fun((2.0 * (i + 1) * np.pi * t / period))
                for i in range(order)
                for fun in (np.sin, np.cos)
            ]
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            CyclicalFeatures: Fitted encoder.

        Raises:
            ValueError: If either ``period`` or ``fourier_order`` is not provided, but
                ``seasonality`` is not one of 'daily', 'weekly' or 'yearly'.
        """
        if self.seasonality not in ["daily", "weekly", "yearly"]:
            if (self.period is None) or (self.fourier_order is None):
                raise ValueError(
                    "When adding custom seasonalities, values for "
                    "`period` and `fourier_order` must be specified."
                )
        if self.seasonality in ["daily", "weekly", "yearly"]:
            if self.period is None:
                self.period = (
                    1
                    if self.seasonality == "daily"
                    else 7
                    if self.seasonality == "weekly"
                    else 365.25
                )
            if self.fourier_order is None:
                self.fourier_order = (
                    4
                    if self.seasonality == "daily"
                    else 3
                    if self.seasonality == "weekly"
                    else 6
                )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If common columns are found and ``replace=False``.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.ds)
        out = self._fourier_series(dt_column, self.period, self.fourier_order)
        out = pd.DataFrame(
            data=out,
            index=X.index,
            columns=[
                f"{self.seasonality}_delim_{i}" for i in range(2 * self.fourier_order)
            ],
        )

        if self.remainder == "passthrough":
            common = list(set(X.columns) & set(out.columns))
            if common and not self.replace:
                raise ValueError(f"Found common column names {common}")
            elif common:
                X = X.drop(common, axis=1)
            out = pd.concat((X, out), axis=1)

        return out
