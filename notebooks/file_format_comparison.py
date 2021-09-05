# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: PremiumClientChurn
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparation between different file formats to persist data
#
# The objective of this notebook is to compare different file formats to persist the data for the project, the contendants are:
#
# * Compressed CSV files
# * CSV files
# * Python Pickles Binaries
# * Apache Parquet Storage Format
#
# There are three main treats that are important for a data science projects which are:
#
# * Preserving columns data types
# * Read/Write speeds
# * Storage Space
#
# The inspiration for this notebook came from: For more details read it!.
#
# https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
#
# The dataset in question has **7587544 rows × 156 columns**
#
# This notebook only will cover read speed and file sizes

# %%
import pandas as pd

# %%
# %%time
data = pd.read_csv('../data/01_raw/paquete_premium.txt.gz', sep='\t')

# %%
# %%time
data = pd.read_csv('../data/01_raw/paquete_premium.txt', sep='\t')

# %%
catalog.save('pickle_data', data)
catalog.save('parquet_data', data)

# %%
# %%time
data = catalog.load('pickle_data')

# %%
# %%time
data = catalog.load('parquet_data')

# %% [markdown]
# ## Comparison

# %%
df = {
    'file_format': ['Compressed CSV', 'CSV', 'Pickle', 'Parquet'],
    'p_dtypes': ['No', 'No', 'Yes', 'Yes'],
    'load_time': ['1min 44s', '1min 39s', '5.72 s', '5.95 s'],
    'file_size': ['1.00 GB', '3.54 GB', '8.79 GB', '1.08 GB'],
}

df = pd.DataFrame(df).set_index(['file_format'])

df

# %% [markdown]
# For this project the Apache Parquet Storage format was selected for the following reasons:
#
# * Persist data types, therefore no recasting.
# * Has load times similar to Pickle.
# * Has a file size similar to the compressed CSV.
#
# In conclusion Parquet seems to have best of all the other file formats.
#
# During or after the evolution of the project this decision could change.
