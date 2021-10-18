# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_categorical_dtype as is_category
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_object_dtype as is_object
from scipy.stats import skew, wasserstein_distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from ..utils import add_constant, as_list, check_X, check_y, maybe_reshape_2d

logger = logging.getLogger("feature-encoding")

UNKNOWN_VALUE = -1


#####################################################################################
# Encode features
#
# All encoders generate numpy arrays
#####################################################################################


class IdentityEncoder(TransformerMixin, BaseEstimator):
    """Create an encoder that returns what it is fed.

    This encoder can act as a linear feature encoder.

    Args:
        feature (str or list of str, optional): The name(s) of the input dataframe's
            column(s) to return. If None, the whole input dataframe will be returned.
            Defaults to None.
        as_filter (bool, optional): If True, the encoder will return all feature labels
            for which "feature in label == True". Defaults to False.
        include_bias (bool, optional): If True, a column of ones is added to the output.
            Defaults to False.

    Raises:
        ValueError: If `as_filter` is True, `feature` cannot include multiple feature names.
    """

    def __init__(self, feature=None, as_filter=False, include_bias=False):
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

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored.
                Defaults to None.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            IdentityEncoder: Fitted encoder.
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

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input
                dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If `include_bias` is True and a column with constant values
                already exists in the returned columns.

        Returns:
            numpy array of shape: The selected column subset as a numpy array.
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


class SafeOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features as an integer array.

    The encoder converts the features into ordinal integers. This results
    in a single column of integers (0 to n_categories - 1) per feature.

    Args:
        feature (str or list of str, optional): The names of the columns to
            encode. If None, all categorical columns will be encoded. Defaults
            to None.
        unknown_value (int, optional): This parameter will set the encoded value
            for unknown categories. It has to be distinct from the values used to
            encode any of the categories in `fit`. If None, the value `-1` is used.
            During `transform`, unknown categories will be replaced using the most
            frequent value along each column. Defaults to None.
    """

    def __init__(self, feature=None, unknown_value=None):
        self.feature = feature
        self.unknown_value = unknown_value
        self.features_ = as_list(feature)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            SafeOrdinalEncoder: Fitted encoder.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
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

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input
                dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            numpy array of shape: The encoded column subset as a numpy array.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.features_)
        return self.feature_pipeline_.transform(X)


class SafeOneHotEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features in a one-hot form.

    The encoder uses a `SafeOrdinalEncoder`to first encode the feature as an
    integer array and then a `sklearn.preprocessing.OneHotEncoder` to encode
    the features as an one-hot array.

    Args:
        feature (str or list of str, optional): The names of the columns to
            encode. If None, all categorical columns will be encoded. Defaults
            to None.
        unknown_value (int, optional): This parameter will set the encoded value
            of unknown categories. It has to be distinct from the values used to
            encode any of the categories in `fit`. If None, the value `-1` is used.
            During `transform`, unknown categories will be replaced using the most
            frequent value along each column. Defaults to None.
    """

    def __init__(self, feature=None, unknown_value=None):
        self.feature = feature
        self.unknown_value = unknown_value
        self.features_ = as_list(feature)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            SafeOneHotEncoder: Fitted encoder.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If the encoder is applied on numerical (float) data.
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
                    "encode_ordinal",
                    SafeOrdinalEncoder(
                        feature=self.features_,
                        unknown_value=self.unknown_value or UNKNOWN_VALUE,
                    ),
                ),
                ("one_hot", OneHotEncoder(drop=None, sparse=False)),
            ]
        )
        # Fit the pipeline
        self.feature_pipeline_.fit(X)

        self.n_features_out_ = 0
        for category in self.feature_pipeline_["one_hot"].categories_:
            self.n_features_out_ += len(category)

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the encoder.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input
                dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            numpy array of shape: The encoded column subset as a numpy array.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.features_)
        return self.feature_pipeline_.transform(X)


class TargetClusterEncoder(TransformerMixin, BaseEstimator):
    """Encode a categorical feature as clusters of the target's values.

    The purpose of this encoder is to reduce the cardinality of a categorical
    feature. This encoder does not replace unknown values with the most frequent
    one during `transform`. It just assigns them the value of `unknown_value`.

    Args:
        feature (str): The name of the categorical feature to transform. This
            encoder operates on a single feature.
        max_n_categories (int, optional): The maximum number of categories to
            produce. Defaults to None.
        stratify_by (str or list of str, optional): If not None, the encoder
            will first stratify the categorical feature into groups that have
            similar values of the features in `stratify_by`, and then cluster
            based on the relationship between the categorical feature and the
            target. It is used only if the number of unique categories minus
            the `excluded_categories` is larger than `max_n_categories`.
            Defaults to None.
        excluded_categories (str or list of str, optional): The names of the
            categories to be excluded from the clustering process. These categories
            will stay intact by the encoding process, so they cannot have the
            same values as the encoder's results (the encoder acts as an
            ``OrdinalEncoder`` in the sense that the feature is converted into
            a column of integers 0 to n_categories - 1). Defaults to None.
        unknown_value (int, optional): This parameter will set the encoded value of
            unknown categories. It has to be distinct from the values used to encode
            any of the categories in `fit`. If None, the value `-1` is used. Defaults
            to None.
        min_samples_leaf (int, optional): The minimum number of samples required to be
            at a leaf node of the decision tree model that is used for stratifying the
            categorical feature if `stratify_by` is not None. The actual number that will
            be passed to the tree model is `min_samples_leaf` multiplied by the number of
            unique values in the categorical feature to transform. Defaults to 1.
        max_features (int, float or {"auto", "sqrt", "log2"}, optional): The number of
            features that the decision tree considers when looking for the best split:

                - If int, then consider `max_features` features at each split of the decision
                  tree

                - If float, then `max_features` is a fraction and `int(max_features * n_features)`
                  features are considered at each split

                - If "auto", then `max_features=n_features`

                - If "sqrt", then `max_features=sqrt(n_features)`

                - If "log2", then `max_features=log2(n_features)`

                - If None, then `max_features=n_features`

            Defaults to "auto".
        random_state (int or RandomState instance, optional): Controls the randomness of
            the decision tree estimator. To obtain a deterministic behaviour during its
            fitting, ``random_state`` has to be fixed to an integer. Defaults to None.
    """

    def __init__(
        self,
        *,
        feature,
        max_n_categories,
        stratify_by=None,
        excluded_categories=None,
        unknown_value=None,
        min_samples_leaf=5,
        max_features="auto",
        random_state=None,
    ):
        self.feature = feature
        self.max_n_categories = max_n_categories
        self.stratify_by = stratify_by
        self.excluded_categories = excluded_categories
        self.unknown_value = unknown_value
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.stratify_by_ = as_list(stratify_by)
        self.excluded_categories_ = as_list(excluded_categories)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the encoder on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (pandas.DataFrame of shape (n_samples, 1)): The target dataframe.

        Returns:
            TargetClusterEncoder: Fitted encoder.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If the encoder is applied on numerical (float) data.
            ValueError: If any of the values in `excluded_categories` is not found in
                the input data.
            ValueError: If the number of categories left after removing all in
                `excluded_categories` is not larger than `max_n_categories`.
        """
        X = check_X(X, exists=[self.feature] + self.stratify_by_)
        if pd.api.types.is_float_dtype(X[self.feature]):
            raise ValueError("The encoder is applied on numerical data")

        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        X = X.merge(y, left_index=True, right_index=True)

        if self.excluded_categories_:
            unique_vals = X[self.feature].unique()
            for value in self.excluded_categories_:
                if value not in unique_vals:
                    raise ValueError(
                        f"Value {value} of `excluded_categories` not found "
                        f"in the {self.feature} data."
                    )

            mask = X[self.feature].isin(self.excluded_categories_)
            X = X.loc[~mask]
            if len(X) == 0:
                raise ValueError(
                    "No categories left after removing all in `excluded_categories`."
                )
            if X[self.feature].nunique() <= self.max_n_categories:
                raise ValueError(
                    "The number of categories left after removing all in `excluded_categories` "
                    "must be larger than `max_n_categories`."
                )

        if not self.stratify_by_:
            self.mapping_ = self._cluster_without_stratify(X)
        else:
            self.mapping_ = self._cluster_with_stratify(X)

        if self.excluded_categories_:
            for i, cat in enumerate(self.excluded_categories_):
                self.mapping_.update({cat: self.max_n_categories + i})

        self.n_features_out_ = 1
        self.fitted_ = True
        return self

    def _cluster_without_stratify(self, X):
        reference = np.array(X[self.target_name_])
        X = X.groupby(self.feature)[self.target_name_].agg(
            ["mean", "std", skew, lambda x: wasserstein_distance(x, reference)]
        )
        X.fillna(value=1, inplace=True)

        X_to_cluster = StandardScaler().fit_transform(X)
        n_clusters = min(X_to_cluster.shape[0], self.max_n_categories)
        clusterer = KMeans(n_clusters=n_clusters)

        with warnings.catch_warnings(record=True) as warning:
            cluster_labels = pd.Series(
                data=clusterer.fit_predict(X_to_cluster), index=X.index
            )
            for w in warning:
                logger.warning(str(w))
        return cluster_labels.to_dict()

    def _cluster_with_stratify(self, X):
        X_train = None
        for col in self.stratify_by_:
            if (
                is_bool(X[col])
                or is_object(X[col])
                or is_category(X[col])
                or is_integer(X[col])
            ):
                X_train = pd.concat((X_train, pd.get_dummies(X[col])), axis=1)
            else:
                X_train = pd.concat((X_train, X[col]), axis=1)

        y_train = X[self.target_name_]
        n_categories = X[self.feature].nunique()

        min_samples_leaf = n_categories * int(self.min_samples_leaf)
        model = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        model = model.fit(X_train, y_train)
        leaf_ids = model.apply(X_train)
        uniq_ids = np.unique(leaf_ids)
        leaf_samples = [np.where(leaf_ids == id)[0] for id in uniq_ids]

        X_to_cluster = pd.DataFrame(
            index=X[self.feature].unique(), columns=range(len(leaf_samples))
        )
        for i, idx in enumerate(leaf_samples):
            subset = X.iloc[idx][[self.feature, self.target_name_]]
            a = subset.groupby(self.feature)[self.target_name_].mean()
            a = a.reindex(X_to_cluster.index)
            X_to_cluster.iloc[:, i] = a

        X_to_cluster = X_to_cluster.fillna(X_to_cluster.median())
        n_clusters = min(X_to_cluster.shape[0], self.max_n_categories)

        clusterer = KMeans(n_clusters=n_clusters)
        with warnings.catch_warnings(record=True) as warning:
            cluster_labels = pd.Series(
                data=clusterer.fit_predict(X_to_cluster), index=X_to_cluster.index
            )
            for w in warning:
                logger.warning(str(w))
        return cluster_labels.to_dict()

    def transform(self, X: pd.DataFrame):
        """Apply the encoder.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.

        Returns:
            numpy array: The encoded column subset as a numpy array.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)

        return maybe_reshape_2d(
            np.array(
                X[self.feature].map(
                    lambda x: int(
                        self.mapping_.get(x, self.unknown_value or UNKNOWN_VALUE)
                    )
                )
            )
        )


class CategoricalEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features.

    If `max_n_categories` is not `None` and the number of unique values of the
    categorical feature is larger than the `max_n_categories` minus the
    `excluded_categories`, the `TargetClusterEncoder` will be called.

    If `encode_as = 'onehot'`, the result comes from a `TargetClusterEncoder` +
    `SafeOneHotEncoder` pipeline, otherwise from a `TargetClusterEncoder` +
    `SafeOrdinalEncoder` one.

    Args:
        feature (str): The name of the categorical feature to transform. This
            encoder operates on a single feature.
        max_n_categories (int, optional): The maximum number of categories to
            produce. Defaults to None.
        stratify_by (str or list of str, optional): If not None, the encoder
            will first stratify the categorical feature into groups that have
            similar values of the features in `stratify_by`, and then cluster
            based on the relationship between the categorical feature and the
            target. It is used only if the number of unique categories minus
            the `excluded_categories` is larger than `max_n_categories`.
            Defaults to None.
        excluded_categories (str or list of str, optional): The names of the
            categories to be excluded from the clustering process. These categories
            will stay intact by the encoding process, so they cannot have the
            same values as the encoder's results (the encoder acts as an
            ``OrdinalEncoder`` in the sense that the feature is converted into
            a column of integers 0 to n_categories - 1). Defaults to None.
        unknown_value (int, optional): This parameter will set the encoded value of
            unknown categories. It has to be distinct from the values used to encode
            any of the categories in `fit`. If None, the value `-1` is used. Defaults
            to None.
        min_samples_leaf (int, optional): The minimum number of samples required to be
            at a leaf node of the decision tree model that is used for stratifying the
            categorical feature if `stratify_by` is not None. The actual number that will
            be passed to the tree model is `min_samples_leaf` multiplied by the number of
            unique values in the categorical feature to transform. Defaults to 1.
        max_features (int, float or {"auto", "sqrt", "log2"}, optional): The number of
            features that the decision tree considers when looking for the best split:

                - If int, then consider `max_features` features at each split of the decision
                  tree

                - If float, then `max_features` is a fraction and `int(max_features * n_features)`
                  features are considered at each split

                - If "auto", then `max_features=n_features`

                - If "sqrt", then `max_features=sqrt(n_features)`

                - If "log2", then `max_features=log2(n_features)`

                - If None, then `max_features=n_features`

            Defaults to "auto".
        random_state (int or RandomState instance, optional): Controls the randomness of
            the decision tree estimator. To obtain a deterministic behaviour during its
            fitting, ``random_state`` has to be fixed to an integer. Defaults to None.
        encode_as ({'onehot', 'ordinal'}, optional): Method used to encode the transformed
            result.

                - If "onehot", encode the transformed result with one-hot encoding and return a
                  dense array

                - If "ordinal", encode the transformed result as integer values

            Defaults to "onehot".
    """

    def __init__(
        self,
        *,
        feature,
        max_n_categories=None,
        stratify_by=None,
        excluded_categories=None,
        unknown_value=None,
        min_samples_leaf=1,
        max_features="auto",
        random_state=None,
        encode_as="onehot",
    ):
        self.feature = feature
        self.max_n_categories = max_n_categories
        self.stratify_by = stratify_by
        self.excluded_categories = excluded_categories
        self.unknown_value = unknown_value
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.encode_as = encode_as
        self.excluded_categories_ = as_list(excluded_categories)

    def _to_pandas(self, arr: np.ndarray):
        return pd.DataFrame(arr, columns=[self.feature])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Fit the encoder on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.
            y (pandas.DataFrame of shape (n_samples, 1), optional): The target dataframe.
                Defaults to None.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.
            ValueError: If the encoder is applied on numerical (float) data.
            ValueError: If the number of categories minus the `excluded_categories`
                is larger than `max_n_categories` but target values (y) are not
                provided.
            ValueError: If any of the values in `excluded_categories` is not found in
                the input data.

        Returns:
            CategoricalEncoder: Fitted encoder.
        """
        X = check_X(X, exists=self.feature)
        if pd.api.types.is_float_dtype(X[self.feature]):
            raise ValueError("The encoder is applied on numerical data")

        n_categories = X[self.feature].nunique()
        use_target = (self.max_n_categories is not None) and (
            n_categories - len(self.excluded_categories_) > self.max_n_categories
        )

        if use_target and (y is None):
            raise ValueError(
                f"The number of categories to encode: {n_categories - len(self.excluded_categories_)}"
                f" is larger than `max_n_categories`: {self.max_n_categories}. In this case, "
                "the target values must be provided for target-based encoding."
            )

        if not use_target:
            self.feature_pipeline_ = Pipeline(
                [
                    (
                        "encode_features",
                        SafeOneHotEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                    if self.encode_as == "onehot"
                    else (
                        "encode_features",
                        SafeOrdinalEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                ]
            )
        else:
            self.feature_pipeline_ = Pipeline(
                [
                    (
                        "reduce_dimension",
                        TargetClusterEncoder(
                            feature=self.feature,
                            stratify_by=self.stratify_by,
                            max_n_categories=self.max_n_categories,
                            excluded_categories=self.excluded_categories,
                            unknown_value=self.unknown_value,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features,
                            random_state=self.random_state,
                        ),
                    ),
                    (
                        "to_pandas",
                        FunctionTransformer(self._to_pandas),
                    ),
                    (
                        "encode_features",
                        SafeOneHotEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                    if self.encode_as == "onehot"
                    else (
                        "encode_features",
                        SafeOrdinalEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    ),
                ]
            )

        # Fit the pipeline
        self.feature_pipeline_.fit(X, y)
        self.n_features_out_ = self.feature_pipeline_["encode_features"].n_features_out_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the encoder.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The input dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            numpy array: The encoded features as a numpy array.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)
        return self.feature_pipeline_.transform(X)


class SplineEncoder(TransformerMixin, BaseEstimator):
    """Generate univariate B-spline bases for features.

    The encoder generates a matrix consisting of `n_splines=n_knots + degree - 1`
    spline basis functions (B-splines) of polynomial order=`degree` for the given
    feature.

    Args:
        feature (str): The name of the column to encode.
        n_knots (int, optional): Number of knots of the splines if `knots` equals one
            of {'uniform', 'quantile'}. Must be larger or equal 2. Ignored if `knots`
            is array-like. Defaults to 5.
        degree (int, optional): The polynomial degree of the spline basis. Must be a
            non-negative integer. Defaults to 3.
        strategy ({'uniform', 'quantile'} or array-like of shape (n_knots, n_features),
            optional): Set knot positions such that first knot <= features <= last knot.

                - If 'uniform', `n_knots` number of knots are distributed uniformly
                  from min to max values of the features (each bin has the same width)

                - If 'quantile', they are distributed uniformly along the quantiles of
                  the features (each bin has the same number of observations)

                - If an array-like is given, it directly specifies the sorted knot
                  positions including the boundary knots. Note that, internally,
                  `degree` number of knots are added before the first knot, the same
                  after the last knot
              
            Defaults to "uniform".
        extrapolation ({'error', 'constant', 'linear', 'continue'}, optional): If 'error',
            values outside the min and max values of the training features raises a `ValueError`.
            If 'constant', the value of the splines at minimum and maximum value of the features
            is used as constant extrapolation. If 'linear', a linear extrapolation is used. If
            'continue', the splines are extrapolated as is, option `extrapolate=True` in
            `scipy.interpolate.BSpline`.
            Defaults to "constant".
        include_bias (bool, optional): If False, then the last spline element inside the data
            range of a feature is dropped. As B-splines sum to one over the spline basis functions
            for each data point, they implicitly include a bias term. Defaults to True.
        order ({'C', 'F'}, optional): Order of output array. 'F' order is faster to compute, but
            may slow down subsequent estimators. Defaults to "C".
    """

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
        self.feature = feature
        self.n_knots = n_knots
        self.degree = degree
        self.strategy = strategy
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        """Fit the encoder.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The data to fit.
            y (None, optional): Ignored. Defaults to None.
            sample_weight (array-like of shape (n_samples,), optional): Individual
                weights for each sample. Used to calculate quantiles if `strategy="quantile"`.
                For `strategy="uniform"`, zero weighted observations are ignored for finding
                the min and max of `X`. Defaults to None.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            SplineEncoder: Fitted encoder.
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

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The data to transform.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            numpy.ndarray: The B-splines matrix.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)
        return self.encoder_.transform(X[[self.feature]])
