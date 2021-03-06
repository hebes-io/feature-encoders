import os
from setuptools import find_packages, setup

from feature_encoders import __version__


here = os.path.abspath(os.path.dirname(__file__))


def dev_extras_require():
    extras = [
        "flake8 >= 3.9.2",
        "flake8-docstrings >= 1.6.0",
        "black >= 21.7b0",
        "pytest >= 6.2.5",
    ]
    return extras


def docs_extras_require():
    extras = [
        'Sphinx >= 3.0.0',  # Force RTD to use >= 3.0.0
        'docutils < 0.18',
        "nbsphinx >= 0.8.7",
        'pylons-sphinx-themes >= 1.0.8',  # Ethical Ads
        "myst-parse >= 0.15.2"
    ]
    return extras


# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)


# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()



configuration = {
    "name": "feature-encoders",
    "version": __version__,
    "python_requires=": ">=3.7",
    "description": (
        "A library for encoding features and their pairwise interactions."
    ),
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    "keywords": "feature encoding interactions",
    "url": "https://github.com/hebes-io/feature-encoders",
    "maintainer": "Sotiris Papadelis",
    "maintainer_email": "spapadel@gmail.com",
    "license": "Apache License, Version 2.0",
    "packages": find_packages(),
    "install_requires": requires,
    "ext_modules": [],
    "cmdclass": {},
    "include_package_data": True,
    "package_data": {'feature_encoders': ['config/features/*.yaml', 'config/models/*.yaml']},
    "extras_require": {"dev": dev_extras_require(), "docs": docs_extras_require()},
}

setup(**configuration)
