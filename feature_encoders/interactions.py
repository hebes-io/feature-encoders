# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .categorical import CategoricalEncoder
from .identity import IdentityEncoder
from .spline import SplineEncoder
from .utils import tensor_product

# ------------------------------------------------------------------------------------
# Encode pairwise categorical data interactions
# ------------------------------------------------------------------------------------


class ICatEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self, encoder_left: CategoricalEncoder, encoder_right: CategoricalEncoder
    ):
        """Encode the interaction between two categorical features.

        Parameters
        ----------
        encoder_left : CategoricalEncoder
            The encoder for the first of the two features.
        encoder_right : CategoricalEncoder
            The encoder for the second of the two features.

        Raises
        ------
        ValueError
            If any of the two encoders is not a `CategoricalEncoder`.
        ValueError
            If the two encoders do not have the same `encode_as` parameter.

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.

        Notes
        -----
        - Both encoders should have the same `encode_as` parameter.
        - If one or both of the encoders is already fitted, it will not be
          re-fitted during `fit` or `fit_transform`.
        """
        if (not isinstance(encoder_left, CategoricalEncoder)) or (
            not isinstance(encoder_right, CategoricalEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `CategoricalEncoder` encoders"
            )
        if encoder_left.encode_as != encoder_right.encode_as:
            raise ValueError(
                "Both encoders should have the same `encode_as` parameter."
            )

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right
        self.encode_as_ = encoder_left.encode_as

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X, y)

        self.n_features_out_ = (
            self.encoder_left.n_features_out_ * self.encoder_right.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of interaction features.
        """
        check_is_fitted(self, "fitted_")

        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)

        if self.encode_as_ == "onehot":
            return tensor_product(X_left, X_right)
        else:
            X_left = X_left.astype(str, copy=False)
            X_right = X_right.astype(str, copy=False)
            X_left = np.core.defchararray.add(X_left, np.array([":"]))
            return np.core.defchararray.add(X_left, X_right)


# ------------------------------------------------------------------------------------
# Encode pairwise interactions between numerical features
# ------------------------------------------------------------------------------------


class ISplineEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoder_left: SplineEncoder, encoder_right: SplineEncoder):
        """Encode the interaction between two spline-encoded numerical features.

        Parameters
        ----------
        encoder_left : SplineEncoder
            The encoder for the first of the two features.
        encoder_right : SplineEncoder
            The encoder for the second of the two features.

        Raises
        ------
        ValueError
            If any of the two encoders is not a `SplineEncoder`.

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.

        Notes
        -----
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
        """
        if (not isinstance(encoder_left, SplineEncoder)) or (
            not isinstance(encoder_right, SplineEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `SplineEncoder` encoders"
            )

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X)

        self.n_features_out_ = (
            self.encoder_left.n_features_out_ * self.encoder_right.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of interaction features.
        """
        check_is_fitted(self, "fitted_")
        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)
        return tensor_product(X_left, X_right)


class ProductEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoder_left: IdentityEncoder, encoder_right: IdentityEncoder):
        """Encode the interaction between two linear numerical features.

        Parameters
        ----------
        encoder_left : IdentityEncoder
            The encoder for the first of the two features.
        encoder_right : IdentityEncoder
            The encoder for the second of the two features.

        Raises
        ------
        ValueError
            If any of the two encoders is not an `IdentityEncoder`.

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.

        Notes
        -----
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
        """
        if (not isinstance(encoder_left, IdentityEncoder)) or (
            not isinstance(encoder_right, IdentityEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `IdentityEncoder` encoders"
            )

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.

        Raises
        ------
        ValueError
            If any of the two encoders is not a single-feature encoder.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X)
            finally:
                if encoder.n_features_out_ > 1:
                    raise ValueError(
                        "This pairwise interaction encoder supports only single-feature encoders"
                    )

        self.n_features_out_ = 1
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of interaction features.
        """
        check_is_fitted(self, "fitted_")
        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)
        return np.multiply(X_left, X_right)


# ------------------------------------------------------------------------------------
# Encode pairwise interactions of one numerical and one categorical feature
# ------------------------------------------------------------------------------------


class ICatLinearEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self, *, encoder_cat: CategoricalEncoder, encoder_num: IdentityEncoder
    ):
        """Encode the interaction between one categorical and one linear numerical feature.

        Parameters
        ----------
        encoder_cat : CategoricalEncoder
            The encoder for the categorical feature. It must encode features in an one-hot form.
        encoder_num : IdentityEncoder
            The encoder for the numerical feature.

        Raises
        ------
        ValueError
            If `encoder_cat` is not  a `CategoricalEncoder`.
        ValueError
            If `encoder_num` is not  an `IdentityEncoder`.
        ValueError
            If `encoder_cat` is not encoded as one-hot.

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.

        Notes
        -----
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
        """
        if not isinstance(encoder_cat, CategoricalEncoder):
            raise ValueError("`encoder_cat` must be a CategoricalEncoder")

        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if not isinstance(encoder_num, IdentityEncoder):
            raise ValueError("`encoder_num` must be an IdentityEncoder")

        self.encoder_cat = encoder_cat
        self.encoder_num = encoder_num

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.
        """
        try:
            check_is_fitted(self.encoder_cat, "fitted_")
        except NotFittedError:
            self.encoder_cat.fit(X, y)

        try:
            check_is_fitted(self.encoder_num, "fitted_")
        except NotFittedError:
            self.encoder_num.fit(X)

        self.n_features_out_ = (
            self.encoder_cat.n_features_out_ * self.encoder_num.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of interaction features.
        """
        check_is_fitted(self, "fitted_")
        X_cat = self.encoder_cat.transform(X)
        X_num = self.encoder_num.transform(X)
        # guard for single category
        if X_cat.shape[1] == 1:
            return X_num
        else:
            return tensor_product(X_cat, X_num)


class ICatSplineEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        encoder_cat: CategoricalEncoder,
        encoder_num: SplineEncoder,
    ):
        """Encode the interaction between one categorical and one spline-encoded
        numerical feature.

        Parameters
        ----------
        encoder_cat : CategoricalEncoder
            The encoder for the categorical feature. It must encode features in
            an one-hot form.
        encoder_num : SplineEncoder
            The encoder for the numerical feature.

        Raises
        ------
        ValueError
            If `encoder_cat` is not  a `CategoricalEncoder`.
        ValueError
            If `encoder_num` is not  a `SplineEncoder`.
        ValueError
            If `encoder_cat` is not encoded as one-hot.

        Attributes
        ----------
        n_features_out_ : int
            The total number of output features.

        Notes
        -----
        - If the categorical encoder is already fitted, it will not be re-fitted during
          `fit` or `fit_transform`.
        - The numerical encoder will always be (re)fitted (one encoder per level of
          categorical feature).
        """
        if not isinstance(encoder_cat, CategoricalEncoder):
            raise ValueError("`encoder_cat` must be a CategoricalEncoder")

        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if not isinstance(encoder_num, SplineEncoder):
            raise ValueError("`encoder_num` must be a SplineEncoder")

        self.encoder_cat = encoder_cat
        self.encoder_num = encoder_num

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to fit.
        y : None, optional
            Ignored, by default None

        Returns
        -------
        object
            Returns the fitted instance itself.
        """
        try:
            check_is_fitted(self.encoder_cat, "fitted_")
        except NotFittedError:
            self.encoder_cat.fit(X, y)

        encoders = OrderedDict({})
        cat_features = pd.DataFrame(data=self.encoder_cat.transform(X), index=X.index)

        for i, col in enumerate(cat_features.columns):
            mask = cat_features[col] == 1
            enc = clone(self.encoder_num)
            try:
                encoders[i] = enc.fit(X.loc[mask])
            except ValueError:
                encoders[i] = enc

        self.num_encoders_ = encoders
        self.n_features_out_ = (
            len(encoders) * next(iter(encoders.values())).n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform the data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_out_)
            The matrix of interaction features.
        """
        check_is_fitted(self, "fitted_")

        out = None
        cat_features = pd.DataFrame(data=self.encoder_cat.transform(X), index=X.index)

        for i, encoder in self.num_encoders_.items():
            mask = cat_features.loc[:, i] == 1
            subset = X.loc[mask]
            if subset.empty or (not encoder.fitted_):
                trf = pd.DataFrame(
                    data=np.zeros((X.shape[0], encoder.n_features_out_)), index=X.index
                )
            else:
                trf = pd.DataFrame(
                    data=encoder.transform(subset), index=subset.index
                ).reindex(X.index, fill_value=0)
            out = pd.concat((out, trf), axis=1)

        out = np.array(out)
        return out
