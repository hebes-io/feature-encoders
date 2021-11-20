# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

from ..compose import FeatureComposer, ModelStructure
from ..utils import check_X, check_y


class LinearPredictor(RegressorMixin, BaseEstimator):
    """A linear regression model with flexible parameterization.

    Args:
        model_structure (ModelStructure): The structure of a linear regression
            model.
        alpha (float, optional): Regularization strength of the underlying ridge
            regression; must be a positive float. Regularization improves the
            conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization. Defaults to 0.01.
        fit_intercept (bool, optional): Whether to fit the intercept for this model.
            If set to false, no intercept will be used in calculations. Defaults to
            False.
    """

    def __init__(
        self, *, model_structure: ModelStructure, alpha=0.01, fit_intercept=False
    ):
        self.model_structure = model_structure
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.composer_ = FeatureComposer(model_structure)

    @property
    def n_parameters(self):
        try:
            self.n_parameters_
        except AttributeError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            return self.n_parameters_

    @property
    def dof(self):
        try:
            self.dof_
        except AttributeError as exc:
            raise ValueError(
                "The degrees of freedom are acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            return self.dof_

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """Fit the estimator with the available data.

        Args:
            X (pandas.DataFrame): Input data.
            y (pandas.Series or pandas.DataFrame): Target data.

        Raises:
            Exception: If the estimator is re-fitted. An estimator object can only be
                fitted once.
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If the target data does not pass the checks of `utils.check_y`.

        Returns:
            LinearPredictor: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        X = check_X(X)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]
        
        design_matrix = self.composer_.fit_transform(X, y)
        self.n_parameters_ = design_matrix.shape[1]
        self.dof_ = np.linalg.matrix_rank(design_matrix)

        if self.alpha is None:
            self.base_estimator_ = LinearRegression(fit_intercept=self.fit_intercept)
        else:
            self.base_estimator_ = Ridge(
                alpha=self.alpha, fit_intercept=self.fit_intercept
            )
        self.base_estimator_ = self.base_estimator_.fit(design_matrix, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        """Predict using the given input data.

        Args:
            X (pandas.DataFrame): Input data.
            include_components (bool, optional): If True, the prediction dataframe will
                include also the individual components' contribution to the predicted
                values. Defaults to False.

        Returns:
            pandas.DataFrame: The prediction.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        design_matrix = self.composer_.transform(X)

        prediction = pd.DataFrame(
            data=self.base_estimator_.predict(design_matrix),
            columns=[self.target_name_],
            index=X.index,
        )

        if include_components:
            components = pd.DataFrame(
                0, index=X.index, columns=self.composer_.component_names_
            )
            feature_cols = self.composer_.component_matrix

            for col in components.columns:
                subset = feature_cols[feature_cols[col] == 1].index.to_list()
                coef = self.base_estimator_.coef_.squeeze()
                pred = np.matmul(design_matrix[:, subset], coef[subset])
                components[col] = components[col] + pred

            prediction = pd.concat((prediction, components), axis=1)

        return prediction
