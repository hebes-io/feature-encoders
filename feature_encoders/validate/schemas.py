# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numbers
from typing import List, Optional, Union

from pydantic import BaseModel, validator


class TrendSchema(BaseModel):
    type: str
    ds: Optional[str] = None
    name: str = "growth"
    remainder: str = "passthrough"
    replace: bool = False

    @validator("remainder")
    def check_remainder(cls, data):
        if data not in ("drop", "passthrough"):
            raise ValueError("can be either 'drop' or 'passthrough'")
        return data


class DatetimeSchema(BaseModel):
    type: str
    ds: Optional[str] = None
    remainder: str = "passthrough"
    replace: bool = False
    subset: Optional[Union[str, List[str]]] = None

    @validator("remainder")
    def check_remainder(cls, data):
        if data not in ("drop", "passthrough"):
            raise ValueError("can be either 'drop' or 'passthrough'")
        return data

    @validator("subset")
    def check_subset(cls, data):
        if isinstance(data, str):
            data = [x.strip() for x in data.split(",")]
        if (not isinstance(data, list)) or (not all(isinstance(x, str) for x in data)):
            raise ValueError(
                "must be a list of strings or a string of comma-separated values."
            )
        return data


class CyclicalSchema(BaseModel):
    type: str
    seasonality: str
    ds: Optional[str] = None
    period: Optional[float] = None
    fourier_order: Optional[int] = None
    remainder: str = "passthrough"
    replace: bool = False

    @validator("remainder")
    def check_remainder(cls, data):
        if data not in ("drop", "passthrough"):
            raise ValueError("can be either 'drop' or 'passthrough'")
        return data


class LinearSchema(BaseModel):
    type: str
    feature: str
    as_filter: bool = False
    include_bias: bool = False


class SplineSchema(BaseModel):
    type: str
    feature: str
    n_knots: Optional[int] = 5
    degree: Optional[int] = 3
    strategy: Optional[Union[str, List]] = "uniform"
    extrapolation: Optional[str] = "constant"
    include_bias: bool = False

    @validator("strategy")
    def check_strategy(cls, data):
        if (data is None) or (data in ("uniform", "quantile")):
            return data
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return data
        else:
            raise ValueError(
                "can be one of 'uniform' or 'quantile', or array-like of numbers"
            )

    @validator("extrapolation")
    def check_extrapolation(cls, data):
        if (data is not None) and (
            data not in ("error", "constant", "linear", "continue")
        ):
            raise ValueError("can be one of 'error', 'constant', 'linear', 'continue'")
        return data


class CategoricalSchema(BaseModel):
    type: str
    feature: str
    max_n_categories: Optional[int] = None
    stratify_by: Optional[Union[str, List[str]]] = None
    excluded_categories: Optional[Union[str, List[str]]] = None
    unknown_value: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: Union[str, int, float] = "auto"
    random_state: Optional[int] = None
    encode_as: str = "onehot"

    @validator("stratify_by", "excluded_categories")
    def check_lists(cls, data):
        if isinstance(data, str):
            data = [x.strip() for x in data.split(",")]
        if (not isinstance(data, list)) or (not all(isinstance(x, str) for x in data)):
            raise ValueError(
                "must be a list of strings or a string of comma-separated values."
            )
        return data

    @validator("max_features")
    def check_max_features(cls, data):
        if data in ("auto", "sqrt", "log2"):
            return data
        if data.replace(".", "", 1).isdigit():
            if isinstance(data, numbers.Integral):
                return data
            elif (data > 0) and (data <= 1):
                return data
        raise ValueError(
            "can be int, float between 0 and 1, or one of 'auto', 'sqrt', 'log2'"
        )

    @validator("encode_as")
    def check_encode_as(cls, data):
        if data not in ("onehot", "ordinal"):
            raise ValueError("can be either 'onehot' or 'ordinal'")
        return data
