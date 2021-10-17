# -*- coding: utf-8 -*-

import os
from pathlib import Path

# Define the configuration folder.
CONF_ROOT = "conf"

# Define additional packages to search for feature generators.
ADDED_PACKAGES = []

PROJECT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent
