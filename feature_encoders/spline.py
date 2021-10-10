# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from .utils import check_X


class SplineEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        feature,
        n_knots=5,
        degree=3,
        strategy="uniform",
        extrapolation="constant",
        include_bias=True,
        order="C",
    ):
        """Generate univariate B-spline bases for features.

        Generates a new feature matrix consisting of `n_splines=n_knots + degree - 1`
        spline basis functions (B-splines) of polynomial order=`degree` for each feature.

        Parameters
        ----------
        feature : str
            The name of the column to encode.
        n_knots : int, optional
            Number of knots of the splines if `knots` equals one of {'uniform', 'quantile'}.
            Must be larger or equal 2. Ignored if `knots` is array-like., by default 5
        degree : int, optional
            The polynomial degree of the spline basis. Must be a non-negative integer,
            by default 3
        strategy : {'uniform', 'quantile'} or array-like of shape (n_knots, n_features), optional
            Set knot positions such that first knot <= features <= last knot.
            - If 'uniform', `n_knots` number of knots are distributed uniformly
              from min to max values of the features (each bin has the same width).
            - If 'quantile', they are distributed uniformly along the quantiles of
              the features (each bin has the same number of observations).
            - If an array-like is given, it directly specifies the sorted knot
              positions including the boundary knots. Note that, internally,
              `degree` number of knots are added before the first knot, the same
              after the last knot,
            by default "uniform"
        extrapolation : {'error', 'constant', 'linear', 'continue'}, optional
            If 'error', values outside the min and max values of the training features
            raises a `ValueError`. If 'constant', the value of the splines at minimum
            and maximum value of the features is used as constant extrapolation. If
            'linear', a linear extrapolation is used. If 'continue', the splines are
            extrapolated as is, option `extrapolate=True` in `scipy.interpolate.BSpline`,
            by default "constant"
        include_bias : bool, optional
            If False, then the last spline element inside the data range of a feature
            is dropped. As B-splines sum to one over the spline basis functions for each
            data point, they implicitly include a bias term, by default True
        order : {'C', 'F'}, optional
            Order of output array. 'F' order is faster to compute, but may slow
            down subsequent estimators, by default "C"

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.
        encoder_ : sklearn.preprocessing.SplineTransformer
            The underlying encoder that performs the transformation.
        """
        self.feature = feature
        self.n_knots = n_knots
        self.degree = degree
        self.strategy = strategy
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order

    def fit(self, X, y=None, sample_weight=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample. Used to calculate quantiles if
            `strategy="quantile"`. For `strategy="uniform"`, zero weighted
            observations are ignored for finding the min and max of `X`, b
            y default None

        Returns
        -------
        object
            Returns the fitted instance itself.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `utils.check_X`.
        """
        X = check_X(X, exists=self.feature)
        self.encoder_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots=self.strategy,
            extrapolation=self.extrapolation,
            include_bias=self.include_bias,
            order=self.order,
        )

        self.encoder_.fit(X[[self.feature]])
        self.n_features_out_ = self.encoder_.n_features_out_
        self.fitted_ = True
        return self

    def transform(self, X):
        """Transform the feature data to B-splines.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of features, where n_splines is the number of bases
            elements of the B-splines, n_knots + degree - 1.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)
        return self.encoder_.transform(X[[self.feature]])
