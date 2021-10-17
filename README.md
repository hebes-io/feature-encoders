Feature Encoders
================

Functionality
-------------

`feature-encoders` is a library for encoding categorical and numerical features to create features for linear regression models. In particular, it includes functionality for:

1. Applying custom feature generators to a dataset. Users can add a feature generator to the existing ones by declaring a class for the validation of their inputs and a class for their creation.


2. Encoding categorical and numerical features. The categorical encoder provides the option to reduce the cardinality of a categorical feature by lumping together categories for which the corresponding distibution of the target values is similar.


3. Encoding interactions. Interactions are always pairwise and always between encoders (and not features). The supported interactions are between: (a) categorical and categorical encoders, (b) categorical and linear encoders, (c) categorical and spline encoders, (d) linear and linear encoders, and (e) spline and spline encoders.


4. Composing features for linear regression. `feature-encoders` includes a `ModelStructure` class for aggregating feature generators and encoders into main effect and pairwise interaction terms for linear regression models. A `ModelStructure` instance can get information about additional features and encoders either from YAML files or through its API.


How to use feature-encoders
---------------------------

Please see our [API documentation](https://feature-encoders.readthedocs.io/en/latest/feature_encoders.html) for a complete list of available functions and see our informative [tutorials](https://feature-encoders.readthedocs.io/en/latest/tutorials.html) for more comprehensive example use cases.


Python Version
--------------

`feature-encoders` supports [`Python 3.7+](https://python3statement.org/).


License
-------

Copyright 2021 Hebes Intelligence. Released under the terms of the Apache License, Version 2.0.

<br>
<img align="left" width="500" src="https://github.com/hebes-io/feature-encoders/raw/main/EC_support.png">

