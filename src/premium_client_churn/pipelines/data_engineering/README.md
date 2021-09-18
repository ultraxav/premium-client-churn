# Data Engineering pipeline

## Overview

This modular pipeline cleans the raw data from broken features, normalizes the target class and applies a simple feature engineering. The pipeline consists:

* `clean_data` node:
    *  Normalizes the target class.
    * Clean broken features.

* `feat_engineering` node:
    * Cancels the effect of inflation.
    * Calculate lagged values for selected features.


## Pipeline inputs

### `raw_data`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Input data to process into clean and train sets |

### `params:data_engineering`

| Name | Type | Description |
| ---- | ---- | ----------- |
| target_class | `string` | Class to predict |
| lag_qty | `int` | How far is the lag to calculate |
| bool_to_cat | `list` | Features to cast as category |
| day_to_year | `list` | Features to convert to year |
| cols_pesos | `list` | Features in pesos |
| cols_to_lag | `list` | Features to apply lags |

## Pipeline outputs

### `clean_data`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing clean set features |

### `feature_data`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set features |
