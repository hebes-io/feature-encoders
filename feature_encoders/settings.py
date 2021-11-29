# -*- coding: utf-8 -*-

import importlib.resources

import feature_encoders.config

with importlib.resources.path(feature_encoders.config, "__init__.py") as init_path:
    CONF_PATH = init_path.resolve().parent
