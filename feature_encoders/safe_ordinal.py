# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .utils import as_list, check_X

UNKNOWN_VALUE = -1


class SafeOrdinalEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, feature=None, unknown_value=None):
        """Encode categorical features as an integer array.

        The encoder converts the features into ordinal integers. This results
        in a single column of integers (0 to n_categories - 1) per feature.

        Parameters
        ----------
        feature : str or list of str, optional
            The names of the columns to encode. If None, all categorical columns will
            be encoded, by default None
        unknown_value : int, optional
            This parameter will set the encoded value for unknown categories. It has
            to be distinct from the values used to encode any of the categories in `fit`.
            If None, the value `-1` is used. During `transform`, unknown categories will
            be replaced using the most frequent value along each column, by default None
        """
        self.feature = feature
        self.unknown_value = unknown_value
        self.features_ = as_list(feature)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on the available data.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input dataframe.
        y : None, optional
            There is no need of a target in the transformer, but the pipeline API
            requires this parameter, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X, categorical_cols, _ = check_X(X, exists=self.features_, return_col_info=True)

        if not self.features_:
            self.features_ = categorical_cols
        else:
            for name in self.features_:
                if pd.api.types.is_float_dtype(X[name]):
                    raise ValueError("The encoder is applied on numerical data")

        self.feature_pipeline_ = Pipeline(
            [
                (
                    "select",
                    ColumnTransformer(
                        [("select", "passthrough", self.features_)], remainder="drop"
                    ),
                ),
                (
                    "encode_ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=self.unknown_value or UNKNOWN_VALUE,
                        dtype=np.int16,
                    ),
                ),
                (
                    "impute_unknown",
                    SimpleImputer(
                        missing_values=self.unknown_value or UNKNOWN_VALUE,
                        strategy="most_frequent",
                    ),
                ),
            ]
        )
        # Fit the pipeline
        self.feature_pipeline_.fit(X)
        self.n_features_out_ = len(self.features_)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the encoder.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input dataframe.

        Returns
        -------
        numpy array, shape (n_samples, n_features_out_)
            The encoded column subset as a numpy array.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.features_)
        return self.feature_pipeline_.transform(X)
