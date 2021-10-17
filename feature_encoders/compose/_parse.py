# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import importlib

from ..settings import ADDED_PACKAGES
from ..utils import as_list

_DEFAULT_PACKAGES = ["feature_encoders.", ""] + as_list(ADDED_PACKAGES)


def load_obj(obj_path: str, default_obj_path: str = ""):
    """Extract an object from a given path.

    From https://github.com/quantumblacklabs/kedro/blob/0.17.5/kedro/utils.py

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    try:
        module_obj = importlib.import_module(obj_path)
    except ModuleNotFoundError:
        return None

    if not hasattr(module_obj, obj_name):
        return None
    return getattr(module_obj, obj_name)


def parse_encoder_definition(class_obj: str):
    """Parse and instantiate an encoder class using the configuration provided.

    Args:
        config: Encoder config dictionary. It *must* contain the `type` key
            with fully qualified class name.

    Raises:
        ValueError: If the function fails to parse the configuration provided.

    Returns:
        2-tuple: (Encoder class object, configuration dictionary)
    """
    if isinstance(class_obj, str):
        if len(class_obj.strip(".")) != len(class_obj):
            raise ValueError(
                "`type` class path does not support relative "
                "paths or paths ending with a dot."
            )

        class_paths = [prefix + class_obj for prefix in _DEFAULT_PACKAGES]

        trials = [load_obj(class_path) for class_path in class_paths]
        try:
            class_obj = next(obj for obj in trials if obj is not None)
        except StopIteration as exc:
            raise ValueError(
                f"Class `{class_obj}` not found or one of its dependencies"
                f"has not been installed."
            ) from exc

    return class_obj
