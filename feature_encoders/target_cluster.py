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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .utils import as_list, check_X, check_y, maybe_reshape_2d

logger = logging.getLogger("feature-encoding")

UNKNOWN_VALUE = -1


class TargetClusterEncoder(TransformerMixin, BaseEstimator):
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
        """Encode a categorical feature as clusters of the target's values.

        The purpose of this encoder is to reduce the cardinality of a categorical feature.

        Parameters
        ----------
        feature : str
            The name of the categorical feature to transform. This encoder operates on
            a single feature.
        max_n_categories : int
            The maximum number of categories to produce.
        stratify_by : str or list of str, optional
            If not None, the encoder will first stratify the categorical feature into
            groups that have similar values of the features in `stratify_by`, and then
            cluster based on the relationship between the categorical feature and the target,
            by default None
        excluded_categories : str or list of str, optional
            The names of the categories to be excluded from the clustering process. These categories
            will stay intact by the encoding process, so they cannot have the same values as the
            encoder's results (the encoder acts as an ``OrdinalEncoder`` in the sense that the feature
            is converted into a column of integers 0 to n_categories - 1), by default None
        unknown_value : int, optional
            This parameter will set the encoded value of unknown categories. It has to be distinct
            from the values used to encode any of the categories in `fit`. If None, the value `-1`
            is used, by default None
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node of the decision tree model
            that is used for stratifying the categorical feature if `stratify_by` is not None. The
            actual number that will be passed to the tree model is `min_samples_leaf` multiplied by
            the number of unique values in the categorical feature to transform, by default 5
        max_features : int, float or {"auto", "sqrt", "log2"}, optional
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split of the decision tree.
            - If float, then `max_features` is a fraction and `int(max_features * n_features)`
              features are considered at each split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`,
            by default "auto"
        random_state : int or RandomState instance, optional
            Controls the randomness of the estimator. To obtain a deterministic behaviour during
            fitting, ``random_state`` has to be fixed to an integer, by default None

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.
        mapping_ : dict
            The mapping between the original categories and their clusters.

        Notes
        -----
        This encoder does not replace unknown values with the most frequent one during `transform`.
        It just assigns them the value of `unknown_value`.
        """
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

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input dataframe.
        y : pandas.DataFrame, shape (n_samples, 1)
            The target dataframe.

        Returns
        -------
        object
            Returns the fitted instance itself.

        Raises
        ------
        ValueError
            If the input data do not pass the checks of `utils.check_X`.
        ValueError
            If the encoder is applied on numerical (float) data.
        ValueError
            If a value of `excluded_categories` is not found in the input data.
        ValueError
            If the number of categories left after removing all in `excluded_categories`
            is not larger than `max_n_categories`.
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
            If the input data do not pass the checks of `utils.check_X`.
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
