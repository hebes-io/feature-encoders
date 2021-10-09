# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import add_constant, as_list, check_X


class IdentityEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, feature=None, as_filter=False, include_bias=False):
        """Create an encoder that returns what it is fed. It can be used as a linear encoder.

        Parameters
        ----------
        feature : str or list of str, optional
            The name(s) of the input dataframe's column(s) to return. If None, the whole
            input dataframe will be returned, by default None
        as_filter : bool, optional
            If True, the encoder will return all feature labels for which "feature in label == True",
            by default False
        include_bias : bool, optional
            If True, a column of ones is added to the output, by default False

        Raises
        ------
        ValueError
            If `as_filter` is True, `feature` cannot include multiple feature names.
        """
        if as_filter and isinstance(feature, list):
            raise ValueError(
                "If `as_filter` is True, `feature` cannot include multiple feature names"
            )

        self.feature = feature
        self.as_filter = as_filter
        self.include_bias = include_bias
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
        X = check_X(X)

        if self.feature is None:
            n_features_out_ = X.shape[1]
        elif (self.feature is not None) and not self.as_filter:
            n_features_out_ = len(self.features_)
        else:
            n_features_out_ = X.filter(like=self.feature, axis=1).shape[1]

        self.n_features_out_ = int(self.include_bias) + n_features_out_
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
            The selected column subset as a numpy array.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `eensight.utils.check_X`.
        ValueError
            If `include_bias` is True and a column with constant values already
            exists in the returned columns.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)

        if (self.feature is not None) and not self.as_filter:
            X = X[self.features_]
        elif self.feature is not None:
            X = X.filter(like=self.feature, axis=1)

        if self.include_bias:
            X = add_constant(X, has_constant="raise")

        return np.array(X)
