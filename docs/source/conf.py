# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../feature_encoders/'))
sys.path.insert(0, os.path.abspath('../../feature_encoders/compose/'))
sys.path.insert(0, os.path.abspath('../../feature_encoders/encode/'))
sys.path.insert(0, os.path.abspath('../../feature_encoders/generate/'))
sys.path.insert(0, os.path.abspath('../../feature_encoders/models/'))
sys.path.insert(0, os.path.abspath('../../feature_encoders/validate/'))


# -- Project information -----------------------------------------------------

project = 'Feature Encoders'
copyright = '2021, Hebes Intelligence'
author = 'Sotiris Papadelis'

# The full version, including alpha/beta/rc tags
release = 'v0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "myst_parser"
]

source_suffix = ['.rst', '.md']

myst_enable_extensions = ["html_image"]

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#exclude_patterns = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

