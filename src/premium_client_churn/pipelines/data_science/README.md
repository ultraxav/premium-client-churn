# Data Science pipeline

## Overview

This modular pipeline splits the data into 4 datasets (train, validation, test, and leaderboard). Then trains a model, and finally make predictions. The pipeline consists of:

* `split_data` node:
    * Splits the main dataset into train, validation, test, and leaderboard datasets.

* `train_model` node:
    * Finds the best hyperparameters using the training data with 5-fold cross-validation.
    * Trains a final model with all the training data and the hyperparameters found.

* `predict` node:
    * Finds the best cutoff probability and makes predictions for all the datasets.
    * Generates a CSV file ready to upload to kaggle.

## Pipeline inputs

### `feature_data`

| Type | Description |
| ---- | ----------- |
| `pandas.DataFrame` | DataFrame containing train set features |

### `params:data_science`

| Name | Type | Description |
| ---- | ---- | ----------- |
| experiment_dates | `dict` | Dates to split the data |
| months_to_train | `int` | how many months are used for training |
| cols_to_drop | `list` | Columns to drop in training data |
| target_class | `string` | Target class |
| optim | `dict` | Activates the hyperparameter search |
| model_fixed | `dict` | Fixed parameters for the model |
| model_optimized | `dict` | Optimized model hyperparameter |
| pcutoff | `dict` | Cutoff probability |

## Pipeline outputs

### `model_predictions`

| Type | Description |
| ---- | ----------- |
| `pandas.DataFrame` | DataFrame containing predictions |

### `model_metrics`

| Type | Description |
| ---- | ----------- |
| `dict` | Dictionary containing various model metrics |

### `model_study`

| Type | Description |
| ---- | ----------- |
| `optuna.study` | Summary of the hyperparameter search |

### `model_params`

| Type | Description |
| ---- | ----------- |
| `dict` | Final model parameters |

### `DataFrames`

| Name | Type | Description |
| ---- | ---- | ----------- |
| train_data | `pandas.DataFrame` | Train dataset |
| valid_data | `pandas.DataFrame` | Validation dataset |
| test_data | `pandas.DataFrame` | Test dataset |
| leader_data | `pandas.DataFrame` | Leaderboard dataset |