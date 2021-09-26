# Reporting pipeline

## Overview

This modular pipeline collects the data generated from the execution of the training pipeline and elaborates a report.


## Pipeline inputs

### `DataFrames`

| Name | Type | Description |
| ---- | ---- | ----------- |
| train_data | `pandas.DataFrame` | Train dataset |
| valid_data | `pandas.DataFrame` | Validation dataset |
| test_data | `pandas.DataFrame` | Test dataset |
| leader_data | `pandas.DataFrame` | Leaderboard dataset |

### `model_metrics`

| Type | Description |
| ---- | ----------- |
| `dict` | Dictionary containing various model metrics |

### `trained_model`

| Type | Description |
| ---- | ----------- |
| `booster` | Trained model |

### `model_study`

| Type | Description |
| ---- | ----------- |
| `optuna.study` | Summary of the hyperparameter search |

### `model_params`

| Type | Description |
| ---- | ----------- |
| `dict` | Final model parameters |


## Pipeline outputs

### `model_metrics_report`

| Type | Description |
| ---- | ----------- |
| `json` | Model final report |
