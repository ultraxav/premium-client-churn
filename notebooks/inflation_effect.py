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
# # Inflation effect
#
# In this notebooks we will explore the effect of inflation and how it can distort quantities of features that are in argentinian pesos
#
# To explore:
# * How to estimate its effect.
# * Are official inflation numbers fiable.
# * Can we normalize the amount in other ways?
#
#

# %%
# libs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
data = catalog.load('raw_data')
print(data.shape)
data.head()

# %%
sueldos = data[['foto_mes', 'mpayroll']]
sueldos

# %%
sueldos = sueldos[sueldos['mpayroll'] != 0].reset_index(drop=True)
sueldos

# %% [markdown]
# ## Raw Payroll

# %%
plt.figure(figsize=(15, 7.5))
ax = sns.boxplot(x='foto_mes', y='mpayroll', data=sueldos, showfliers=False)
plt.title('mpayroll distribution')
plt.xticks(rotation=90)
plt.show()

# %%
agg_funcs = {
    'month_mean': ('mpayroll', 'mean'),
    'month_median': ('mpayroll', 'median'),
}

# %%
normalizers = sueldos.groupby('foto_mes').agg(**agg_funcs)
normalizers['month_mean'] = normalizers['month_mean'].shift(1)
normalizers['month_median'] = normalizers['month_median'].shift(1)
normalizers

# %%
sueldos = sueldos.merge(normalizers, on='foto_mes', how='left')
sueldos['mpayroll_mean'] = sueldos['mpayroll'] / sueldos['month_mean']
sueldos['mpayroll_median'] = sueldos['mpayroll'] / sueldos['month_median']
results = sueldos[['foto_mes', 'mpayroll', 'mpayroll_mean', 'mpayroll_median']]
sueldos

# %% [markdown]
# ## Payroll normalized by mean

# %%
plt.figure(figsize=(15, 7.5))
ax = sns.boxplot(x='foto_mes', y='mpayroll_mean', data=sueldos, showfliers=False)
plt.title('mpayroll_mean distribution')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ## Payroll normalized by median

# %%
plt.figure(figsize=(15, 7.5))
ax = sns.boxplot(x='foto_mes', y='mpayroll_median', data=sueldos, showfliers=False)
plt.title('mpayroll_median distribution')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ## Analysis
#
# Normalizing the payroll by the meadian seems to be the most effective, but how it was implemented the median did not take into account the effect of utilities and bonuses on regular months like June and December. this months will be handled separately.

# %%
sueldos = data[['foto_mes', 'mpayroll']]
sueldos

# %%
sueldos = sueldos[sueldos['mpayroll'] != 0].reset_index(drop=True)
sueldos['foto_mes'] = sueldos.loc[:, 'foto_mes'].astype('str')
sueldos

# %%
sueldos_ireg = sueldos[
    (sueldos['foto_mes'].str.contains('06')) | (sueldos['foto_mes'].str.contains('12'))
]
sueldos_ireg['foto_mes'] = sueldos_ireg.loc[:, 'foto_mes'].astype('int').copy()
sueldos_ireg

# %%
sueldos_reg = sueldos[
    ~(
        (sueldos['foto_mes'].str.contains('06'))
        | (sueldos['foto_mes'].str.contains('12'))
    )
]
sueldos_reg['foto_mes'] = sueldos_reg.loc[:, 'foto_mes'].astype('int').copy()
sueldos_reg

# %%
agg_funcs = {
    'month_median': ('mpayroll', 'median'),
}

# %%
normalizers_ireg = sueldos_ireg.groupby('foto_mes').agg(**agg_funcs)
normalizers_ireg['month_median'] = normalizers_ireg['month_median'].shift(1)
normalizers_ireg

# %%
normalizers_reg = sueldos_reg.groupby('foto_mes').agg(**agg_funcs)
normalizers_reg['month_median'] = normalizers_reg['month_median'].shift(1)
normalizers_reg

# %%
normalizers = pd.concat([normalizers_reg, normalizers_ireg]).sort_index()
normalizers

# %%
sueldos['foto_mes'] = sueldos.loc[:, 'foto_mes'].astype('int')

# %%
sueldos = sueldos.merge(normalizers, on='foto_mes', how='left')
sueldos['mpayroll_median'] = sueldos['mpayroll'] / sueldos['month_median']
results['mpayroll_median_adjust'] = sueldos['mpayroll_median'].copy()
sueldos

# %% [markdown]
# ## Payroll normalized by median taking into account utilities and bonuses

# %%
plt.figure(figsize=(15, 7.5))
ax = sns.boxplot(x='foto_mes', y='mpayroll_median', data=sueldos, showfliers=False)
plt.title('mpayroll_median distribution')
plt.xticks(rotation=90)
plt.show()

# %%
agg_func = {
    'mpayroll_1_mean': ('mpayroll_mean', 'mean'),
    'mpayroll_1_median': ('mpayroll_mean', 'median'),
    'mpayroll_2_mean': ('mpayroll_median', 'mean'),
    'mpayroll_2_median': ('mpayroll_median', 'median'),
    'mpayroll_3_mean': ('mpayroll_median_adjust', 'mean'),
    'mpayroll_3_median': ('mpayroll_median_adjust', 'median'),
}

# %%
summary = results.groupby('foto_mes').agg(**agg_func)
summary

# %% [markdown]
# ## Results
#
# The option `mpayroll_3` which stands for mpayroll normalized by median and adjusting by utilities and bonuses, since it has the lowest mean and standard deviation.
#
# It was of the outmost importance to use previous month values to avoid leaking values **from the FUTURE!!**

# %%
summary.describe()
