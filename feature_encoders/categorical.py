# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from .safe_one_hot import SafeOneHotEncoder
from .safe_ordinal import SafeOrdinalEncoder
from .target_cluster import TargetClusterEncoder
from .utils import as_list, check_X


class CategoricalEncoder(TransformerMixin, BaseEstimator):
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
        """Encode categorical features.

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
            the number of unique values in the categorical feature to transform, by default 1
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
        encode_as : str :type: {'onehot', 'ordinal'}, optional
            Method used to encode the transformed result.
            onehot
                Encode the transformed result with one-hot encoding and return a dense array.
            ordinal
                Encode the transformed result as integer values.
            by default "onehot".

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.
        feature_pipeline_ : sklearn.pipeline.Pipeline
            The pipeline that performs the transformation.
        """
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

    def _to_pandas(self, x):
        return pd.DataFrame(x, columns=[self.feature])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Fit the encoder on the available data.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input dataframe.
        y : pandas.DataFrame, shape (n_samples, 1), optional
            The target dataframe, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.

        ValueError
            If the input data do not pass the checks of `utils.check_X`.
        ValueError
            If the encoder is applied on numerical (float) data.
        ValueError
            If the number of categories minus the `excluded_categories` is larger
            than `max_n_categories` but target values (y) are not provided.
        ValueError
            A value of `excluded_categories` is not found in the input data.
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

    def transform(self, X):
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
        return self.feature_pipeline_.transform(X)
