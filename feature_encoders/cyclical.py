# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import check_X, get_datetime_data


class CyclicalEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self, *, feature=None, seasonality=None, period=None, fourier_order=None
    ):
        """Create cyclical (seasonal) features as fourier terms.

        Parameters
        ----------
        feature : str, optional
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the input
            dataframe's index, by default None
        seasonality : str, optional
            The name of the seasonality. The encoder can provide default values for ``period``
            and ``fourier_order`` if ``seasonality`` is one of 'daily', 'weekly' or 'yearly',
            by default None
        period : float, optional
            Number of days in one period, by default None
        fourier_order : int, optional
            Number of Fourier components to use, by default None

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.
        """
        self.feature = feature
        self.seasonality = seasonality
        self.period = period
        self.fourier_order = fourier_order

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
        """Fit the encoder on the available data.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input dataframe.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.

        Raises
        ------
        ValueError
            If ``period`` or ``fourier_order`` are not specified, while ``seasonality``
            is not one of ("daily", "weekly", "yearly").
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
        self.n_features_out_ = 2 * self.fourier_order
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the feature data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of cyclical features.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.feature)
        return self._fourier_series(dt_column, self.period, self.fourier_order)
