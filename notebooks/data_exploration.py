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
# # Dataset exploration

# %%
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
# %%time
data = catalog.load('primary_data')
print(data.shape)
data

# %%
data.loc[data['foto_mes'] == 201701, 'ccajas_consultas']
