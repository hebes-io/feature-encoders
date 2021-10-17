# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict, defaultdict
from functools import reduce
from typing import Dict, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..encode import (
    CategoricalEncoder,
    ICatEncoder,
    ICatLinearEncoder,
    ICatSplineEncoder,
    IdentityEncoder,
    ISplineEncoder,
    ProductEncoder,
    SplineEncoder,
)
from ..validate.schemas import CategoricalSchema, LinearSchema, SplineSchema
from ._parse import parse_encoder_definition

################################################################################
# Utilities
################################################################################


def _validate_feature(props, feature_map):
    fgen_type = props.get("type")
    targets = feature_map.get(fgen_type)
    if targets is None:
        raise ValueError(f"Type {fgen_type} not found in provided mapping")
    class_obj = parse_encoder_definition(targets["validate"])
    validated_props = class_obj(**props).dict()
    return validated_props


def _validate_encoder(props):
    enc_type = props.get("type")
    if enc_type == "categorical":
        validated_props = CategoricalSchema(**props).dict()
    elif enc_type == "linear":
        validated_props = LinearSchema(**props).dict()
    elif enc_type == "spline":
        validated_props = SplineSchema(**props).dict()
    else:
        raise ValueError(f"Type {enc_type} not recognized")
    return validated_props


def _interaction_by_types(left_enc, right_enc):
    left_enc_type = type(left_enc).__name__
    right_enc_type = type(right_enc).__name__

    if (left_enc_type, right_enc_type) == ("CategoricalEncoder", "categorical"):
        return ICatEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("CategoricalEncoder", "IdentityEncoder"):
        return ICatLinearEncoder(encoder_cat=left_enc, encoder_num=right_enc)
    elif (left_enc_type, right_enc_type) == ("CategoricalEncoder", "SplineEncoder"):
        return ICatSplineEncoder(encoder_cat=left_enc, encoder_num=right_enc)
    elif (left_enc_type, right_enc_type) == ("IdentityEncoder", "IdentityEncoder"):
        return ProductEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("IdentityEncoder", "CategoricalEncoder"):
        return ICatLinearEncoder(encoder_cat=right_enc, encoder_num=left_enc)
    elif (left_enc_type, right_enc_type) == ("SplineEncoder", "SplineEncoder"):
        return ISplineEncoder(left_enc, right_enc)
    elif (left_enc_type, right_enc_type) == ("SplineEncoder", "CategoricalEncoder"):
        return ICatSplineEncoder(encoder_cat=right_enc, encoder_num=left_enc)
    else:
        raise NotImplementedError(
            f"Interactions between encoder type `{left_enc_type}` "
            f"and encoder type `{right_enc_type}` are not supported"
        )


#######################################################################################
# ModelStructure
#######################################################################################


class ModelStructure:
    """Capture the structure of a linear regression model.

    The class validates and stores the details of a linear regression model: features,
    main effects and interactions.

    Args:
        structure (Dict, optional): A dictionary that includes information about the
            model. Example:
            ::
                {'add_features':
                    {'time':
                        { 'ds': None,
                        'remainder': 'passthrough',
                        'replace': False,
                        'subset': ['month', 'hourofweek']
                        }
                    },
                'main_effects':
                    {'month':
                        { 'feature': 'month',
                        'max_n_categories': None,
                        'encode_as': 'onehot',
                        'interaction_only': False
                        },
                    'tow':
                        { 'feature': 'hourofweek',
                        'max_n_categories': 60,
                        'encode_as': 'onehot',
                        'interaction_only': False
                        },
                    'lin_temperature':
                        { 'feature': 'temperature',
                        'include_bias': False,
                        'interaction_only': False
                        }
                    },
                }
            Defaults to None.
        feature_map (Dict, optional): A mapping between a feature generator name
            and the classes for its validation and creation.
            Example:
            ::
                {'datetime':
                    'validate': 'validate.DatetimeSchema'
                    'generate': 'generate.DatetimeFeatures'
                }
            Defaults to None.
    """

    def __init__(self, structure: Dict = None, feature_map: Dict = None):
        self.structure = structure
        self.feature_map = feature_map
        self.structure_ = (
            {
                "add_features": defaultdict(dict),
                "main_effects": defaultdict(dict),
                "interactions": defaultdict(dict),
            }
            if structure is None
            else structure
        )

    @property
    def components(self):
        return copy.deepcopy(self.structure_)

    @classmethod
    def from_config(cls: Type, config: Dict, feature_map: Dict = None):
        """Create a ModelStructure instance from a configuration file.

        Args:
            config (Dict): A dictionary that includes information about the
                model.
            feature_map (Dict, optional): A mapping between a feature generator
                name and the classes for its validation and creation.
                Defaults to None.

        Returns:
            ModelStructure: A populated ModelStructure instance.
        """
        config = copy.deepcopy(config)
        structure = {
            "add_features": defaultdict(dict),
            "main_effects": defaultdict(dict),
            "interactions": defaultdict(dict),
        }
        temporary = {}

        if "add_features" in config:
            if feature_map is None:
                raise ValueError(
                    "A mapping between feature generator types and classes "
                    "has not been provided."
                )

            for name, props in config["add_features"].items():
                structure["add_features"][name] = _validate_feature(props, feature_map)

        if "regressors" in config:
            for name, props in config["regressors"].items():
                interaction_only = props.pop("interaction_only", False)
                if interaction_only:
                    temporary[name] = props
                else:
                    structure["main_effects"][name] = props

        if "interactions" in config:
            # example of pair_name: temperature, hour
            for pair_name, pair_props in config["interactions"].items():
                pair_name = tuple([x.strip() for x in pair_name.split(",")])
                if len(pair_name) != 2:
                    raise ValueError("Only pairwise interactions are supported.")
                if pair_props is None:
                    pair_props = {}

                for name in pair_name:
                    if name in structure["main_effects"]:
                        props = dict(
                            structure["main_effects"][name],
                            **pair_props.get(name, dict()),
                        )
                        structure["interactions"][pair_name].update({f"{name}": props})

                    elif name in temporary:
                        props = dict(
                            temporary[name],
                            **pair_props.get(name, dict()),
                        )
                        structure["interactions"][pair_name].update({f"{name}": props})

                    elif name in pair_props:
                        structure["interactions"][pair_name].update(
                            {f"{name}": pair_props[name]}
                        )
                    else:
                        raise ValueError(
                            f"The regressor `{name}` has not been added yet and not "
                            "enough information has been provided so that to add it"
                        )
        # validate before store
        for name, props in structure["main_effects"].items():
            structure["main_effects"][name] = _validate_encoder(props)

        for pair_name, pair_props in structure["interactions"].items():
            for name in pair_name:
                structure["interactions"][pair_name][name] = _validate_encoder(
                    pair_props[name]
                )

        return cls(structure=structure, feature_map=feature_map)

    def add_new_feature(
        self, *, name: str, fgen_type: Union[str, BaseEstimator], **kwargs
    ):
        """Add a feature generator.

        Feature generators are applied on the input dataframe with the same order
        that they were added.

        Args:
            name (str): A name for the feature generator.
            fgen_type (str or sklearn-compatible transformer): The feature generator
                to add. If it is a string, the corresponding class will be loaded
                based on the relevant entry in the :attr:`feature_map` dictionary.
            **kwargs: Keyword arguments to be passed during the feature generator
                initialization. Ignored if `fgen` is not a string.

        Raises:
            ValueError: If a feature generator with the same name has already been added.

        Returns:
            ModelStructure: The updated ModelStructure instance.
        """
        if name in self.structure_["add_features"]:
            raise ValueError(f"Feature generator named {name} has already been added")

        self.structure_["add_features"][name].update(
            dict(
                type=fgen_type,
                **kwargs,
            )
        )
        return self

    def add_main_effect(
        self, *, name: str, enc_type: Union[str, BaseEstimator], **kwargs
    ):
        """Add a main effect.

        Args:
            name (str): A name for the main effect.
            enc_type (str or encoder object): The type of the feature encoder to
                apply on the main effect.
            **kwargs: Keyword arguments to be passed during the feature encoder
                initialization. Ignored if `enc_type` is not a string.

        Raises:
            ValueError: If an encoder with the same name has already been added.

        Returns:
            ModelStructure: The updated ModelStructure instance.
        """
        if name in self.structure_["main_effects"]:
            raise ValueError(f"Encoder named {name} has already been added")
        if isinstance(enc_type, str) and (
            enc_type not in ("linear", "spline", "categorical")
        ):
            raise ValueError(f"Encoder type enc_type {enc_type} is not supported")

        self.structure_["main_effects"][name].update(
            dict(
                type=enc_type,
                **kwargs,
            )
        )
        return self

    def add_interaction(
        self,
        *,
        lenc_name: str,
        renc_name: str,
        lenc_type: Union[str, object],
        renc_type: Union[str, object],
        **kwargs,
    ):
        """Add a pairwise interaction.

        Args:
            lenc_name (str): A name for the first part of the interaction pair.
            renc_name (str): A name for the second part of the interaction pair.
            lenc_type (str or encoder object): The type of the feature encoder to
                apply on the first part of the interaction pair.
            renc_type (str or encoder object): The type of the feature encoder to
                apply on the second part of the interaction pair.
            **kwargs: Keyword arguments to be passed during the feature encoders'
                initialization.

        Raises:
            ValueError: If an interaction with the same name `(lenc_name, renc_name)`
                has already been added.

        Returns:
            ModelStructure: The updated ModelStructure instance.

        Example:
        ::
            model = ModelStructure().add_interaction(
                lenc_name="is_Monday",
                renc_name="daily_seasonality",
                lenc_type="categorical",
                renc_type="linear",
                **{
                    is_Monday: {"feature": "is_Monday", "encode_as": "onehot"},
                    daily_seasonality: {"feature": "daily", "as_filter": True},
                },
            )
        """
        if ((lenc_name, renc_name) in self.structure_["interactions"]) or (
            (renc_name, lenc_name) in self.structure_["interactions"]
        ):
            raise ValueError(
                f"Interaction {(lenc_name, renc_name)} has already been added"
            )

        self.structure_["interactions"][(lenc_name, renc_name)][lenc_name] = dict(
            type=lenc_type,
            **kwargs.get(lenc_name, {}),
        )
        self.structure_["interactions"][(lenc_name, renc_name)][renc_name] = dict(
            type=renc_type,
            **kwargs.get(renc_name, {}),
        )
        return self


#######################################################################################
# FeatureComposer
#######################################################################################


class FeatureComposer(TransformerMixin, BaseEstimator):
    """Generate linear features and pairwise interactions.

    Args:
        model_structure (ModelStructure): The structure of a linear regression
            model.
    """

    def __init__(self, model_structure: ModelStructure):
        self.model_structure = model_structure
        self.encoders_ = {
            "main_effects": OrderedDict({}),
            "interactions": OrderedDict({}),
        }
        self.added_features_ = []
        self.train_feature_cols_ = []
        self.component_names_ = []

    def _create_new_features(self):
        for _, props in self.model_structure.components["add_features"].items():
            fgen_type = props.pop("type")
            if isinstance(fgen_type, str):
                if self.model_structure.feature_map is None:
                    raise ValueError(
                        "A mapping between types and classes has not been provided."
                    )

                targets = self.model_structure.feature_map.get(fgen_type)
                if targets is None:
                    raise ValueError(f"Type {fgen_type} not found in provided mapping")

                class_obj = parse_encoder_definition(targets["generate"])
                self.added_features_.append(class_obj(**props))
            else:
                self.added_features_.append(fgen_type)

    def _create_encoders(self):
        for name, props in self.model_structure.components["main_effects"].items():
            enc_type = props.pop("type")
            if isinstance(enc_type, str):
                enc_type = (
                    CategoricalEncoder(**props)
                    if enc_type == "categorical"
                    else SplineEncoder(**props)
                    if enc_type == "spline"
                    else IdentityEncoder(**props)
                )
            self.encoders_["main_effects"][name] = enc_type

        for name, props in self.model_structure.components["interactions"].items():
            left, right = name
            left_enc_type = props[left].pop("type")
            right_enc_type = props[right].pop("type")

            if isinstance(left_enc_type, str):
                left_enc = (
                    CategoricalEncoder(**props[left])
                    if left_enc_type == "categorical"
                    else SplineEncoder(**props[left])
                    if left_enc_type == "spline"
                    else IdentityEncoder(**props[left])
                )
            else:
                left_enc = left_enc_type

            if isinstance(right_enc_type, str):
                right_enc = (
                    CategoricalEncoder(**props[right])
                    if right_enc_type == "categorical"
                    else SplineEncoder(**props[right])
                    if right_enc_type == "spline"
                    else IdentityEncoder(**props[right])
                )
            else:
                right_enc = right_enc_type

            interaction = _interaction_by_types(left_enc, right_enc)
            self.encoders_["interactions"][name] = interaction

    def _main_effects(self, X, y=None, fitting=True):
        for name, encoder in self.encoders_["main_effects"].items():
            if fitting:
                encoder.fit(X, y)
                yield name, encoder.n_features_out_
            else:
                yield name, encoder.transform(X)

    def _interaction_effects(self, X, y=None, fitting=True):
        for name, encoder in self.encoders_["interactions"].items():
            if fitting:
                encoder.fit(X, y)
                yield name, encoder.n_features_out_
            else:
                yield name, encoder.transform(X)

    @property
    def component_matrix(self):
        """Dataframe indicating which columns of the feature matrix correspond
        to which components.

        Returns
        -------
        feature_cols: A binary indicator dataframe. Entry is 1 if that column is used
            in that component.
        """
        if self.train_feature_cols_ is None:
            raise ValueError(
                "The estimator must be fitted before the `component_matrix` can be accessed."
            )

        components = pd.DataFrame(
            {
                "col": np.arange(len(self.train_feature_cols_)),
                "component": [x.split("_delim_")[0] for x in self.train_feature_cols_],
            }
        )
        # Convert to a binary matrix
        feature_cols = pd.crosstab(
            components["col"],
            components["component"],
        ).sort_index(level="col")

        return feature_cols

    def fit(self, X, y=None):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        feature_cols = []
        self._create_new_features()
        # Apply the feature generators
        if self.added_features_:
            X = reduce(
                lambda _df, trans: trans.fit_transform(_df), self.added_features_, X
            )

        self._create_encoders()
        # Fit the main effect encoders
        for name, n_features_out_ in self._main_effects(X, y, fitting=True):
            if "_delim_" in name:
                raise ValueError('The name of the regressor cannot include "_delim_"')
            feature_cols.extend([f"{name}_delim_{i}" for i in range(n_features_out_)])

        # Fit the interaction encoders
        for (left, right), n_features_out_ in self._interaction_effects(
            X, y, fitting=True
        ):
            if ("_delim_" in left) or ("_delim_" in right):
                raise ValueError('The name of the regressor cannot include "_delim_"')
            feature_cols.extend(
                [f"{left}:{right}_delim_{i}" for i in range(n_features_out_)]
            )

        self.train_feature_cols_ = feature_cols
        self.component_names_ = self.component_matrix.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")

        if self.added_features_:
            X = reduce(
                lambda _df, trans: trans.fit_transform(_df), self.added_features_, X
            )

        design_matrix = np.zeros((len(X), len(self.train_feature_cols_)))

        # Add the main effects
        for name, features in self._main_effects(X, fitting=False):
            relevant_cols = self.component_matrix.loc[
                self.component_matrix[name] == 1
            ].index
            design_matrix[:, relevant_cols] = features

        # Add the interactions
        for name, features in self._interaction_effects(X, fitting=False):
            left, right = name
            relevant_cols = self.component_matrix.loc[
                self.component_matrix[f"{left}:{right}"] == 1
            ].index
            design_matrix[:, relevant_cols] = features

        return design_matrix
