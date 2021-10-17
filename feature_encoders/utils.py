# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import scipy
from omegaconf import OmegaConf
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_categorical_dtype as is_category
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_object_dtype as is_object
from sklearn.utils import check_array
from sklearn.utils.validation import column_or_1d

from feature_encoders.settings import CONF_ROOT, PROJECT_PATH


def maybe_reshape_2d(arr: np.ndarray):
    """Reshape an array (if needed) so it's always 2-d and long.

    Args:
        arr (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The reshaped array.
    """
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    return arr


def as_list(val: Any):
    """Cast input as list.

    Helper function, always returns a list of the input value.
    """
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]


def as_series(x: Union[np.ndarray, pd.Series, pd.DataFrame]):
    """Cast an iterable to a Pandas Series object."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    else:
        return pd.Series(column_or_1d(x))


def get_categorical_cols(X: pd.DataFrame, int_is_categorical=True):
    """Return the names of the categorical columns in the input DataFrame.

    Args:
        X (pandas.DataFrame): Input dataframe.
        int_is_categorical (bool, optional): If True, integer types are
            considered categorical. Defaults to True.

    Returns:
        list: The names of categorical columns in the input DataFrame.
    """
    obj_cols = []
    for col in X.columns:
        # check if it is date
        if is_datetime(X[col]):
            continue
        # check if it is bool, object or category
        if is_bool(X[col]) or is_object(X[col]) or is_category(X[col]):
            obj_cols.append(col)
            continue
        # check if it is integer
        if int_is_categorical and is_integer(X[col]):
            obj_cols.append(col)
            continue
    return obj_cols


def get_datetime_data(X: pd.DataFrame, col_name=None):
    """Get datetime information from the input dataframe.

    Args:
        X (pandas.DataFrame): The input dataframe.
        col_name (str, optional): The name of the column that contains
            datetime information. If None, it is assumed that the datetime
            information is provided by the input dataframe's index.
            Defaults to None.

    Returns:
        pandas.Series: The datetime information.
    """
    if col_name is not None:
        dt_column = X[col_name]
    else:
        dt_column = X.index.to_series()

    col_dtype = dt_column.dtype
    if isinstance(col_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        col_dtype = np.datetime64
    if not np.issubdtype(col_dtype, np.datetime64):
        dt_column = pd.to_datetime(dt_column, infer_datetime_format=True)
    return dt_column


def check_X(
    X: pd.DataFrame, exists=None, int_is_categorical=True, return_col_info=False
):
    """Perform a series of checks on the input dataframe.

    Args:
        X (pamdas.DataFrame): The input dataframe.
        exists (str or list of str, optional): Names of columns that must be present
            in the input dataframe. Defaults to None.
        int_is_categorical (bool, optional): If True, integer types are considered
            categorical. Defaults to True.
        return_col_info (bool, optional): If True, the function will return the names
            of the categorical and the names of the numerical columns, in addition to
            the provided dataframe. Defaults to False.

    Raises:
        ValueError: If the input is not a pandas DataFrame.
        ValueError: If any of the column names in `exists` are not found in the input.
        ValueError: If Nan or inf values are found in the provided input data.

    Returns:
        pandas.DataFrame if `return_col_info` is False else (pandas.DataFrame, list, list)
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input values are expected as pandas DataFrames.")

    exists = as_list(exists)
    for name in exists:
        if name not in X:
            raise ValueError(f"Regressor {name} missing from dataframe")

    categorical_cols = get_categorical_cols(X, int_is_categorical=int_is_categorical)
    numeric_cols = X.columns.difference(categorical_cols)

    if (len(categorical_cols) > 0) and X[categorical_cols].isnull().values.any():
        raise ValueError("Found NaN values in input's categorical data")
    if (len(numeric_cols) > 0) and np.any(~np.isfinite(X[numeric_cols])):
        raise ValueError("Found NaN or Inf values in input's numerical data")

    if return_col_info:
        return X, categorical_cols, numeric_cols
    return X


def check_y(y: Union[pd.Series, pd.DataFrame], index=None):
    """Perform a series of checks on the input dataframe.

    The checks are carried out by `sklearn.utils.check_array`.

    Args:
        y (Union[pandas.Series, pandas.DataFrame]): The input dataframe.
        index (Union[pandas.Index, pandas.DatetimeIndex], optional): An index to compare
            with the input dataframe's index. Defaults to None.

    Raises:
        ValueError: If the input is neither a pandas Series nor a pandas DataFrame with
            only a single column.
        ValueError: If the input data has different index than the one that was provided
            for comparison (if `index` is not None).

    Returns:
        pandas.DataFrame: The validated input data.
    """
    if isinstance(y, pd.DataFrame) and (y.shape[1] == 1):
        target_name = y.columns[0]
    elif isinstance(y, pd.Series):
        target_name = y.name or "_target_values_"
    else:
        raise ValueError(
            "This estimator accepts target inputs as "
            "`pd.Series` or `pd.DataFrame` with only a single column."
        )

    if (index is not None) and not y.index.equals(index):
        raise ValueError(
            "Input data has different index than the one "
            "that was provided for comparison"
        )

    y = pd.DataFrame(
        data=check_array(y, ensure_2d=False), index=y.index, columns=[target_name]
    )
    return y


def tensor_product(a: np.ndarray, b: np.ndarray, reshape=True):
    """Compute the tensor product of two matrices.

    If A is (n, m_a), B is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Args:
        a (numpy array of shape (n, m_a)): The first matrix.
        b (numpy array of shape (n, m_b)): The second matrix.
        reshape (bool, optional): Whether to reshape the result to be 2D (n, m_a * m_b)
            or return a 3D tensor (n, m_a, m_b). Defaults to True.

    Raises:
        ValueError: If input arrays are not 2-dimensional.
        ValueError: If both input arrays do not have the same number of samples.

    Returns:
        numpy.ndarray of shape (n, m_a * m_b) if reshape = True else of shape (n, m_a, m_b).
    """
    if (a.ndim != 2) or (b.ndim != 2):
        raise ValueError("Inputs must be 2-dimensional")

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError("Both arguments must have the same number of samples")

    if scipy.sparse.issparse(a):
        a = a.A
    if scipy.sparse.issparse(b):
        b = b.A

    product = a[..., :, None] * b[..., None, :]
    if reshape:
        return product.reshape(na, ma * mb)
    return product


def add_constant(
    data: Union[np.ndarray, pd.Series, pd.DataFrame], prepend=True, has_constant="skip"
):
    """Add a column of ones to an array.

    Args:
        data (array-like): A column-ordered design matrix.
        prepend (bool, optional): If true, the constant is in the first column.
            Else the constant is appended (last column). Defaults to True.
        has_constant ({'raise', 'add', 'skip'}, optional): Behavior if ``data``
            already has a constant. The default will return data without adding
            another constant. If 'raise', will raise an error if any column has a
            constant value. Using 'add' will add a column of 1s if a constant column
            is present. Defaults to "skip".

    Returns:
        numpy.ndarray:  The original values with a constant (column of ones) as
            the first or last column.
    """
    x = np.asanyarray(data)
    ndim = x.ndim
    if ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError("Only implemented for 2-dimensional arrays")

    is_nonzero_const = np.ptp(x, axis=0) == 0
    is_nonzero_const &= np.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == "skip":
            return x
        elif has_constant == "raise":
            if ndim == 1:
                raise ValueError("data is constant.")
            else:
                columns = np.arange(x.shape[1])
                cols = ",".join([str(c) for c in columns[is_nonzero_const]])
                raise ValueError(f"Column(s) {cols} are constant.")

    x = [np.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return np.column_stack(x)


def load_config(model="towt", features="default", merge_multiple=False):
    """Load model configuration and feature generator mapping.

    Given `model` and `features`, the function searches for files in:
    ```python
    conf_path = settings.CONF_ROOT
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(settings.PROJECT_PATH, conf_path)

    model_files = glob.glob(f"{conf_path}/models/{model}.*")
    feature_files = glob.glob(f"{conf_path}/features/{features}.*")
    ```

    Args:
        model (str, optional): The name of the model configuration to load.
            Defaults to "towt".
        features (str, optional): The name of the feature generator mapping to
            load. Defaults to "default".
        merge_multiple (bool, optional): If True and more than one files are found when
            searching for either models or features, the contents of the files will ne merged.
            Otherwise, an exception will be raised. Defaults to False.

    Returns:
        (dict, dict): The model configuration and feature mapping as dictionaries.
    """
    conf_path = CONF_ROOT
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(PROJECT_PATH, conf_path)

    model_conf = None
    model_files = glob.glob(f"{conf_path}/models/{model}.*")
    if len(model_files) == 0:
        raise ValueError("No model configuration files found")
    elif (len(model_files) > 1) and (not merge_multiple):
        raise ValueError("More than one model configuration files found")
    elif len(model_files) > 1:
        model_conf = OmegaConf.merge(
            *[OmegaConf.load(model_file) for model_file in model_files]
        )
    else:
        model_conf = OmegaConf.load(model_files[0])

    feature_conf = None
    feature_files = glob.glob(f"{conf_path}/features/{features}.*")
    if len(feature_files) == 0:
        raise ValueError("No feature generator mapping files found")
    elif (len(feature_files) > 1) and (not merge_multiple):
        raise ValueError("More than one feature generator mapping files found")
    elif len(feature_files) > 1:
        feature_conf = OmegaConf.merge(
            *[OmegaConf.load(feature_file) for feature_file in feature_files]
        )
    else:
        feature_conf = OmegaConf.load(feature_files[0])

    return OmegaConf.to_container(model_conf), OmegaConf.to_container(feature_conf)
