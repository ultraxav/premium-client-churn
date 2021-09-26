# Premium Client Churn

## Overview

The focus of this project is to develop a reference implementation example of how to train and register Machine Learning (ML) models intended for predicting client churn, all this applying the latest machine learning techniques and good practices.

The data, which is not provided as part of this release and thus this implementation is only intended to serve as an example, is an extract of anonymized premium clients from a financial institution. Source: https://www.kaggle.com/c/uamds2020ldi1f2

This implementation is built in python and leverages open-source libraries kedro, scikit-learn, MLFlow, and others.

Main libraries considered:

* Kedro - https://kedro.readthedocs.io/en/0.17.4/

> Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. It borrows concepts from software engineering best-practice and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning.

* scikit-learn - https://scikit-learn.org/0.24/

> Simple and efficient tools for predictive data analysis, accessible to everybody, and reusable in various contexts, built on NumPy, SciPy, and matplotlib.

* MLFlow - https://www.mlflow.org/docs/latest/index.html

> MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It tackles four primary functions of Tracking experiments, Packaging ML code, Managing and deploying models, and providing a central model store. For this project an implementation of MLFlow in Kedro - https://kedro-mlflow.readthedocs.io/en/0.7.4/

* Kedro Viz - https://github.com/quantumblacklabs/kedro-viz

> Kedro-Viz is an interactive development tool for building data science pipelines with Kedro, that features complete visualization of a Kedro project and its pipelines.

* Pre-Commit - https://pre-commit.com/

> A framework for managing and maintaining multi-language pre-commit hooks. Git hook scripts are useful for identifying simple issues before submission to code review. We run our hooks on every commit to automatically point out issues in code such as missing semicolons, trailing whitespace, and debug statements. By pointing these issues out before code review, this allows a code reviewer to focus on the architecture of a change while not wasting time with trivial style nitpicks.

This implementation provides examples how to build pipelines for data load and save, data engineering, data science, and summarizing. These pipelines enable scalable, repeatable, and maintainable development of ML models.

## Project relevant structure

    ├── conf                           <- Folder used to store configuration files.
    │   └── README.md                  <- README for developers to configure this project.
    │
    ├── data                           <- Data for a particular model
    │   ├── 01_raw                     <- The original, immutable data dump.
    │   ├── 02_primary                 <- The cleaned, data set for feature engineering.
    │   ├── 02_feature                 <- The final, canonical data set for modeling.
    │   ├── 06_model                   <- Compiled model.
    │   ├── 07_model_output            <- Model predictions.
    │   └── 08_reporting               <- Model performance report.
    │
    ├── mlruns                         <- Folder for model tracking.
    │
    ├── notebooks                      <- Jupyter notebooks.
    │
    ├── src                            <- Source code for use in this project.
    │   ├── premium_client_churn       <- Project source code.
    │   │   └── pipelines              <- Pipelines for data processing, modeling, and reporting.
    │   └── requirements.in            <- The requirements file for reproducing the analysis environment.
    │
    └── README.md                      <- The top-level README for developers using this project.

## General view of the Pipeline

![kedro-pipeline](https://github.com/ultraxav/premium-client-churn/blob/main/docs/kedro-pipeline.png)

## Now... Let's get started!

In this section will be describes the steps to follow to set up your environment and execute the project pipelines:

1. First of all, we strongly recommend creating a virtual environment with Python >=3.6, <3.9 (Version used 3.8.10):

```
python -m venv <name of the environment>
```

2. Install Kedro 0.17.4: 

```
pip install kedro==0.17.4
```

3. In the root of the project build the specific requirements for your execution environment and install them:

```
kedro build-reqs

kedro install
```

4. Initialize mlflow tracking integration into kedro: 

```
kedro mlflow init
```

5. Installing pre-commit hook for code formatting: 

```
pre-commit install
```

## Running the pipeline

To run the complete pipeline run:

```
kedro run --pipeline training
```

If you want to run a specific pipeline run:

```
kedro run --pipeline <name of the pipeline>
```

## See the results

To open the mlflow interface run:

```
kedro mlflow ui
```

To visualize the pipeline run:

```
kedro viz
```

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `kedro install` you will not need to take any extra steps before you use them.

### Jupyter
You can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/05_package_a_project.html)
