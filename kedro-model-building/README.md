
## Overview

This is a Kedro project, which was generated using Kedro 0.18.8. This Kedro Project was used to clean data and train models used for research and commercial purposes. Some IP elements (Custom Classifier) have been redacted, which however can be easily replaced with LightGBM and refactoring of the code.

Additionally, some modules, that help produce Cusum Plots and split of data by hospital (Research specific) have been provided in the file Additional_tools.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

