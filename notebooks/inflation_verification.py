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

# %%
# libs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
data = catalog.load('feature_data')
print(data.shape)
data.head()

# %%
data = data[['foto_mes', 'mpayroll']]
data

# %%
sueldos = data[data['mpayroll'] > 0]  # .reset_index(drop=True)
sueldos

# %%
plt.figure(figsize=(15, 7.5))
ax = sns.boxplot(x='foto_mes', y='mpayroll', data=sueldos, showfliers=False)
plt.title('mpayroll_median distribution')
plt.xticks(rotation=90)
plt.show()
