# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.utils.validation import check_is_fitted

from ..compose import FeatureComposer, ModelStructure
from ..encode import CategoricalEncoder, IdentityEncoder
from ..generate import CyclicalFeatures, TrendFeatures
from ..utils import as_list, check_X, check_y


class SeasonalPredictor(BaseEstimator):
    """Time series prediction model based on seasonal decomposition.

    Args:
        ds (str, optional): The name of the input dataframe's column that
            contains datetime information. If None, it is assumed that the
            datetime information is provided by the input dataframe's index.
            Defaults to None.
        add_trend (bool, optional): If True, a linear time trend will be
            added. Defaults to False.
        yearly_seasonality (Union[str, bool, int], optional): Fit yearly
            seasonality. Can be 'auto', True, False, or a number of Fourier
            terms to generate. Defaults to "auto".
        weekly_seasonality (Union[str, bool, int], optional): Fit weekly
            seasonality. Can be 'auto', True, False, or a number of Fourier
            terms to generate. Defaults to "auto".
        daily_seasonality (Union[str, bool, int], optional): Fit daily
            seasonality. Can be 'auto', True, False, or a number of Fourier
            terms to generate. Defaults to "auto".
        min_samples (float ([0, 1]), optional): Minimum number of samples
            chosen randomly from original data by the RANSAC (RANdom SAmple
            Consensus) algorithm. Defaults to 0.5.
        alpha (float, optional): Parameter for the underlying ridge estimator
            (`base_estimator`). It must be a positive float. Regularization
            improves the conditioning of the problem and reduces the variance
            of the estimates. Larger values specify stronger regularization.
            Defaults to 0.01.
    """

    def __init__(
        self,
        ds: str = None,
        add_trend: bool = False,
        yearly_seasonality: Union[str, bool, int] = "auto",
        weekly_seasonality: Union[str, bool, int] = "auto",
        daily_seasonality: Union[str, bool, int] = "auto",
        min_samples=0.5,
        alpha=0.01,
    ):
        self.ds = ds
        self.add_trend = add_trend
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.min_samples = min_samples
        self.alpha = alpha

        # Set during fitting
        self.seasonalities_ = OrderedDict({})
        self.base_estimator_ = RANSACRegressor(
            base_estimator=Ridge(fit_intercept=True, alpha=alpha),
            min_samples=min_samples,
        )

    def add_seasonality(
        self,
        name: str,
        period: float = None,
        fourier_order: int = None,
        condition_name: str = None,
    ):
        """Add a seasonal component with specified period and number of
        Fourier components.

        If `condition_name is provided`, the input dataframe passed to
        `fit` and `predict` should have a column with the specified
        `condition_name` containing booleans that indicate when to apply
        seasonality.

        Args:
            name (str): The name of the seasonality component.
            period (float, optional): Number of days in one period.
                Defaults to None.
            fourier_order (int, optional): Number of Fourier components
                to use. Defaults to None.
            condition_name (str, optional): The name of the seasonality
                condition. Defaults to None.

        Raises:
            Exception: If the method is called after the estimator is fitted.
            ValueError: If either `period` or `fourier_order` are not provided
                and the seasonality is not in ('daily', 'weekly', 'yearly').

        Returns:
            SeasonalPredictor: The updated estimator object.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception("Seasonality must be added prior to model fitting.")

        if name not in ["daily", "weekly", "yearly"]:
            if (period is None) or (fourier_order is None):
                raise ValueError(
                    "When adding custom seasonalities, values for "
                    '"period" and "fourier_order" must be specified.'
                )

        if (period is not None) and (period <= 0):
            raise ValueError("Period must be > 0")
        if (fourier_order is not None) and (fourier_order <= 0):
            raise ValueError("Fourier order must be > 0")

        self.seasonalities_[name] = {
            "period": float(period) if period is not None else None,
            "fourier_order": int(fourier_order) if fourier_order is not None else None,
            "condition_name": condition_name,
        }
        return self

    def _set_seasonalities(self, X):
        dates = X.index.to_series() if self.ds is None else X[self.ds]
        first = dates.min()
        last = dates.max()
        dt = dates.diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        default_params = {"period": None, "fourier_order": None, "condition_name": None}

        # Set yearly seasonality
        if (self.yearly_seasonality is False) or ("yearly" in self.seasonalities_):
            pass
        elif self.yearly_seasonality is True:
            self.seasonalities_["yearly"] = default_params
        elif self.yearly_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if last - first >= pd.Timedelta(days=365):
                self.seasonalities_["yearly"] = default_params
        elif self.yearly_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["yearly"] = dict(
                default_params, fourier_order=self.yearly_seasonality
            )

        # Set weekly seasonality
        if (self.weekly_seasonality is False) or ("weekly" in self.seasonalities_):
            pass
        elif self.weekly_seasonality is True:
            self.seasonalities_["weekly"] = default_params
        elif self.weekly_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if (last - first >= pd.Timedelta(weeks=1)) and (
                time_step < pd.Timedelta(weeks=1)
            ):
                self.seasonalities_["weekly"] = default_params
        elif self.weekly_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["weekly"] = dict(
                default_params, fourier_order=self.weekly_seasonality
            )

        # Set daily seasonality
        if (self.daily_seasonality is False) or ("daily" in self.seasonalities_):
            pass
        elif self.daily_seasonality is True:
            self.seasonalities_["daily"] = default_params
        elif self.daily_seasonality == "auto":
            # Turn on yearly seasonality if there is >=1 years of history
            if (last - first >= pd.Timedelta(days=1)) and (
                time_step < pd.Timedelta(days=1)
            ):
                self.seasonalities_["daily"] = default_params
        elif self.daily_seasonality <= 0:
            raise ValueError("Fourier order must be > 0")
        else:
            self.seasonalities_["daily"] = dict(
                default_params, fourier_order=self.daily_seasonality
            )
        return self

    def _create_composer(self):
        model_structure = ModelStructure()

        if self.add_trend:
            model_structure = model_structure.add_new_feature(
                name="added_trend",
                fgen_type=TrendFeatures(
                    ds=self.ds,
                    name="growth",
                    remainder="passthrough",
                    replace=False,
                ),
            )
            model_structure = model_structure.add_main_effect(
                name="trend",
                enc_type=IdentityEncoder(
                    feature="growth",
                    as_filter=False,
                    include_bias=False,
                ),
            )

        for seasonality, props in self.seasonalities_.items():
            condition_name = props["condition_name"]

            model_structure = model_structure.add_new_feature(
                name=seasonality,
                fgen_type=CyclicalFeatures(
                    seasonality=seasonality,
                    ds=self.ds,
                    period=props.get("period"),
                    fourier_order=props.get("fourier_order"),
                    remainder="passthrough",
                    replace=False,
                ),
            )

            if condition_name is None:
                model_structure = model_structure.add_main_effect(
                    name=seasonality,
                    enc_type=IdentityEncoder(
                        feature=seasonality,
                        as_filter=True,
                        include_bias=False,
                    ),
                )
            else:
                model_structure = model_structure.add_interaction(
                    lenc_name=condition_name,
                    renc_name=seasonality,
                    lenc_type=CategoricalEncoder(
                        feature=condition_name, encode_as="onehot"
                    ),
                    renc_type=IdentityEncoder(
                        feature=seasonality, as_filter=True, include_bias=False
                    ),
                )
        return FeatureComposer(model_structure)

    def _check_input(self, X):
        conditions = [
            props["condition_name"]
            for props in self.seasonalities_.values()
            if props["condition_name"] is not None
        ]

        regressors = as_list(self.ds) + conditions
        X = check_X(X, exists=regressors)

        for condition_name in conditions:
            if not X[condition_name].isin([True, False]).all():
                raise ValueError(f"Found non-boolean in column {condition_name!r}")
        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the estimator with the available data.

        Args:
            X (pandas.DataFrame): Input data.
            y (pandas.DataFrame): Target data.

        Raises:
            Exception: If the estimator is re-fitted. An estimator object can only be
                fitted once.
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If the target data does not pass the checks of `utils.check_y`.

        Returns:
            SeasonalPredictor: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        X = self._check_input(X)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        self._set_seasonalities(X)
        self.composer_ = self._create_composer()

        design_matrix = self.composer_.fit_transform(X, y)
        self.base_estimator_.fit(design_matrix, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using the given input data.

        Args:
            X (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: The prediction.
        """
        check_is_fitted(self, "fitted_")
        X = self._check_input(X)
        design_matrix = self.composer_.transform(X)

        prediction = pd.DataFrame(
            data=self.base_estimator_.predict(design_matrix),
            index=X.index,
            columns=[self.target_name_],
        )
        return prediction
