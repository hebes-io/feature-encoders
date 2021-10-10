from setuptools import setup
from feature_encoders import __version__


def readme():
    with open("README.md", encoding='utf-8') as readme_file:
        return readme_file.read()


def get_extras_require():
    extras = [
        "flake8 >= 3.9.2",
        "flake8-docstrings >= 1.6.0",
        "black >= 21.7b0",
        "pytest >= 6.2.5",
    ]
    return extras

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)


docs_extras = [
    'Sphinx >= 3.0.0',  # Force RTD to use >= 3.0.0
    'docutils',
    "nbsphinx==0.8.7",
    "numpydoc==1.1.0",
    'pylons-sphinx-themes >= 1.0.8',  # Ethical Ads
    'pylons_sphinx_latesturl',
    "pydata-sphinx-theme==0.7.1",
    'repoze.sphinx.autointerface',
    'sphinxcontrib-autoprogram',
    "sphinx-autodoc-typehints==1.11.1",
]


configuration = {
    "name": "feature_encoders",
    "version": __version__,
    "python_requires=": "==3.8",
    "description": (
        "A library for encoding features and their "
        "pairwise interactions."
    ),
    "long_description_content_type": "text/x-rst",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License, Version 2.0",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "feature encoding interactions",
    "url": "https://github.com/hebes-io/feature-encoders",
    "maintainer": "Sotiris Papadelis",
    "maintainer_email": "spapadel@gmail.com",
    "license": "Apache License, Version 2.0",
    "packages": ["feature_encoders"],
    "install_requires": requires,
    "ext_modules": [],
    "cmdclass": {},
    "tests_require": ["pytest"],
    "data_files": (),
    "extras_require": {"ci": get_extras_require(), "docs": docs_extras},
}

setup(**configuration)