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

## Now... Let's get started!

In this section will be describes the steps to follow to set up your environment and execute the project pipelines:

1. First of all, we strongly recommend creating a virtual environment with Python >=3.6, <3.9:

```
python -m venv "name of your choosing"
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

## Overview

This is your new Kedro project, which was generated using `Kedro 0.17.4`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `kedro install` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/05_package_a_project.html)
