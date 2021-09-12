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
data.head(30)

# %%
list(data.columns[120:])  #'mtarjeta_visa_debitos_automaticos'

# %%
median_vector = data.pivot_table(
    index='foto_mes', values='mcaja_ahorro', aggfunc='median'
).sort_index()

# %%
median_vector.columns = ['correction']
median_vector

# %%
median_vector = data[['foto_mes']].merge(median_vector, on='foto_mes')
median_vector

# %%
data = data.sort_values(by=['numero_de_cliente', 'foto_mes']).reset_index(drop=True)

# %%
data.groupby('numero_de_cliente').shift(3)
